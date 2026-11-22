"""One owner for provider policy, roles, probing and client construction.

The connection service persists secrets and owner-scoped records; this module
turns a kind plus a raw URL into a validated endpoint, carries one resolved
role's selection and optional secret, lists models through an injectable
transport, and builds the validator provider used by a job. Both the probe path
and the job path therefore share the same outbound policy.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse, urlsplit, urlunsplit
from uuid import UUID

import httpx
from edcraft_validator.llm.llm_contracts import (
    ModelProvider,
    TemplateProviderSelection,
)
from edcraft_validator.llm.openai_compatible_provider import (
    OpenAIProvider,
    SocLaasProvider,
)
from openai import DefaultHttpxClient, OpenAI

from edcraft_backend.config.settings import AuthoringProfileSettings
from edcraft_backend.services import provider_host_guard
from edcraft_backend.services.ai_authoring_providers import ManagedProvider
from edcraft_backend.services.provider_errors import (
    ProviderConnectionInvalid,
    ProviderConnectionUnavailable,
)

DEFAULT_BASE_URLS: dict[str, str | None] = {
    "soclaas": None,
    "openai": "https://api.openai.com/v1",
}
MAX_MODELS = 500
PROBE_TIMEOUT_SECONDS = 20.0
DEFAULT_TIMEOUT_SECONDS = {
    "soclaas": 600.0,
    "openai": 120.0,
}


@dataclass(frozen=True, slots=True)
class ResolvedConnection:
    """One role's provider selection, with a BYOK secret when it has one.

    A role without a secret uses a server-configured profile, so the same
    value carries both branches and job wiring never has to branch again.
    """

    selection: TemplateProviderSelection
    connection_id: UUID | None = None
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None

    @classmethod
    def for_profile(cls, profile: AuthoringProfileSettings) -> ResolvedConnection:
        """Resolve a server profile into a role with no per-job secret."""
        return cls(
            selection=TemplateProviderSelection(
                provider=profile.provider, model=profile.model
            )
        )

    @classmethod
    def for_judge(cls, *, provider: str, model: str) -> ResolvedConnection:
        """Resolve the server-configured advisory judge without a per-job secret."""
        return cls(
            selection=TemplateProviderSelection(provider=provider, model=model)
        )

    @classmethod
    def for_connection(
        cls,
        *,
        connection_id: UUID,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
    ) -> ResolvedConnection:
        """Resolve one owner-scoped connection into a role with a BYOK secret."""
        return cls(
            selection=TemplateProviderSelection(provider=provider, model=model),
            connection_id=connection_id,
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def is_byok(self) -> bool:
        """Whether this role carries a per-job secret instead of a profile."""
        return self.api_key is not None

    @property
    def model(self) -> str:
        """The role's model name; every authoring role must name one."""
        model = self.selection.model
        if not isinstance(model, str):
            raise ValueError("The authoring role must name a model.")
        return model

    def open_provider(
        self, *, timeout_seconds: float | None = None, max_retries: int = 1
    ) -> ManagedProvider:
        """Build a fresh provider; the secret stays inside this value."""
        if self.api_key is None:
            raise ValueError("A server profile does not carry a provider secret.")
        endpoint = ProviderEndpoint.parse(self.selection.provider, self.base_url)
        return endpoint.open_provider(
            self.api_key,
            self.model,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else DEFAULT_TIMEOUT_SECONDS.get(self.selection.provider, 120.0)
            ),
            max_retries=max_retries,
        )


@dataclass(frozen=True, slots=True)
class ProviderHttpResponse:
    """Transport-neutral result of one provider HTTP request."""

    status_code: int
    body: bytes


class ModelTransport(Protocol):
    """Outbound HTTP seam used to list provider models."""

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout: float,
    ) -> ProviderHttpResponse: ...


class HttpxModelTransport:
    """Default transport; redirects stay off so a guard cannot be bypassed."""

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout: float,
    ) -> ProviderHttpResponse:
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False
            ) as client:
                response = await client.get(url, headers=headers)
        except (httpx.InvalidURL, ValueError) as exc:
            raise ProviderConnectionInvalid(
                "The provider base URL is not a valid HTTP URL."
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderConnectionUnavailable(
                "Could not reach the provider. Check the base URL and network access."
            ) from exc
        return ProviderHttpResponse(status_code=response.status_code, body=response.content)


@dataclass(frozen=True, slots=True)
class ProviderEndpoint:
    """A validated provider destination shared by probing and job runs."""

    kind: str
    base_url: str

    @classmethod
    def parse(cls, kind: str, raw: str | None) -> ProviderEndpoint:
        """Validate a provider URL and apply the kind's default when omitted."""
        value = _resolve_base_url(kind, raw)
        if len(value) > 500:
            raise ProviderConnectionInvalid("Base URL must be 500 characters or fewer.")
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
            raise ProviderConnectionInvalid("Base URL must not contain control characters.")

        try:
            parsed = urlparse(value)
            # Accessing .port validates malformed and out-of-range port values.
            _ = parsed.port
        except ValueError as exc:
            raise ProviderConnectionInvalid(
                "Base URL has an invalid host or port."
            ) from exc

        if parsed.query or parsed.fragment:
            raise ProviderConnectionInvalid(
                "Base URL must not include a query string or fragment; "
                "remove the '?' and '#' portions."
            )
        if parsed.scheme not in {"http", "https"}:
            raise ProviderConnectionInvalid("Base URL must use http or https.")
        if parsed.username or parsed.password:
            raise ProviderConnectionInvalid("Base URL must not contain credentials.")
        hostname = parsed.hostname
        if not hostname:
            raise ProviderConnectionInvalid("Base URL must include a host.")
        if any(
            char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F
            for char in hostname
        ):
            raise ProviderConnectionInvalid(
                "Base URL host must not contain whitespace or controls."
            )
        if parsed.scheme == "http" and not provider_host_guard.is_loopback(hostname):
            raise ProviderConnectionInvalid(
                "Use HTTPS for remote providers; HTTP is allowed only for localhost."
            )
        provider_host_guard.reject_unsafe_host(hostname)
        return cls(kind=kind, base_url=value.rstrip("/"))

    async def list_models(
        self,
        api_key: str,
        *,
        transport: ModelTransport | None = None,
    ) -> list[str]:
        """List models through the guard and the configured transport."""
        # Resolve immediately before the request so a hostname cannot be
        # rebound to a private address between validation and use.
        await asyncio.to_thread(
            provider_host_guard.assert_public_provider_host, self.base_url
        )
        active = transport or HttpxModelTransport()
        response = await active.get(
            self.models_url,
            {"Authorization": f"Bearer {api_key}"},
            PROBE_TIMEOUT_SECONDS,
        )
        _raise_for_model_response(response.status_code)
        try:
            payload = json.loads(response.body)
        except ValueError as exc:
            raise ProviderConnectionUnavailable(
                "The provider's model list was not valid JSON."
            ) from exc
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ProviderConnectionUnavailable(
                "The provider's model list was not in the expected format."
            )
        return _extract_model_ids(entries)

    def open_provider(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float,
        max_retries: int = 1,
    ) -> ManagedProvider:
        """Build a fresh provider for one job; secrets stay in this closure."""
        # Re-run the resolved-host guard at job time so a stored hostname
        # cannot be rebound to a private address after admission.
        provider_host_guard.assert_public_provider_host(self.base_url)
        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            # Base URLs are user-supplied. A redirect could otherwise point
            # the SDK at a private address after the guard above has passed.
            http_client=DefaultHttpxClient(follow_redirects=False),
        )
        if self.kind == "soclaas":
            delegate: ModelProvider = SocLaasProvider(client=client, model=model)
        elif self.kind == "openai":
            delegate = OpenAIProvider(client=client, model=model)
        else:
            client.close()
            raise ValueError(f"Unsupported provider: {self.kind}")
        return ManagedProvider(
            delegate,
            client,
            provider=self.kind,
            model=model,
        )

    @property
    def models_url(self) -> str:
        """Join the models path onto the base path without string concatenation."""
        parts = urlsplit(self.base_url)
        path = f"{parts.path.rstrip('/')}/models"
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _resolve_base_url(kind: str, raw: str | None) -> str:
    value = (raw or "").strip()
    if not value and kind == "soclaas":
        value = os.getenv("SOCLAAS_BASE_URL", "").strip()
    if not value:
        value = DEFAULT_BASE_URLS.get(kind) or ""
    if not value:
        raise ProviderConnectionInvalid(
            "Enter the SocLaas base URL (for example https://host/v1)."
        )
    return value


def _raise_for_model_response(status_code: int) -> None:
    if status_code in {401, 403}:
        raise ProviderConnectionUnavailable("The provider rejected this API key.")
    if status_code == 404:
        raise ProviderConnectionUnavailable(
            "The provider's model list was not found. Check the base URL."
        )
    if status_code >= 400:
        raise ProviderConnectionUnavailable(
            f"The provider returned HTTP {status_code} while listing models."
        )


def _extract_model_ids(entries: list[object]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            continue
        candidate = model_id.strip()
        if not candidate or len(candidate) > 120 or candidate in seen:
            continue
        seen.add(candidate)
        models.append(candidate)
    return sorted(models, key=str.lower)[:MAX_MODELS]


__all__ = [
    "DEFAULT_BASE_URLS",
    "DEFAULT_TIMEOUT_SECONDS",
    "HttpxModelTransport",
    "MAX_MODELS",
    "ModelTransport",
    "PROBE_TIMEOUT_SECONDS",
    "ProviderEndpoint",
    "ProviderHttpResponse",
    "ResolvedConnection",
]
