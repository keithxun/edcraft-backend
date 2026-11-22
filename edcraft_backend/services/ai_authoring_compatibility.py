"""Probe forced-token decision support with synthetic inputs and bounded work."""

from __future__ import annotations

import asyncio
import hmac
import os
import secrets
import time
from collections import OrderedDict
from contextlib import suppress
from typing import Literal
from uuid import UUID

from edcraft_validator.llm.forced_decision import ClassDecision
from edcraft_validator.llm.llm_contracts import ModelProvider, supports_decisions
from edcraft_validator.llm.llm_errors import DecisionCapabilityError
from edcraft_validator.llm.openai_compatible_provider import OpenAICompatibleProvider
from pydantic import ValidationError

from edcraft_backend.schemas.ai_authoring import JudgeCompatibilityResult
from edcraft_backend.services.ai_authoring_judge import build_plan_provider
from edcraft_backend.services.provider_endpoint import ResolvedConnection

type CheckRole = Literal["difficulty", "relevance"]
type ProbeOutcome = Literal["compatible", "unsupported", "unknown"]
type ProbeKey = tuple[UUID | None, str, str, CheckRole, bytes]

_CHECK_LABELS: dict[CheckRole, tuple[str, ...]] = {
    "difficulty": ("easy", "medium", "hard"),
    "relevance": ("yes", "no"),
}
_REQUEST_TIMEOUT = 8.0
_PROBE_DEADLINE = 9.0
_MAX_INFLIGHT = 8
_CACHE_TTL = 120.0
_MAX_CACHE_ENTRIES = 256


def _open_probe_provider(role: ResolvedConnection) -> ModelProvider:
    if role.is_byok:
        return role.open_provider(timeout_seconds=_REQUEST_TIMEOUT, max_retries=0)
    provider = build_plan_provider(role)
    if isinstance(provider, OpenAICompatibleProvider):
        # A fresh, job-independent client shares the ordinary server provider settings.
        provider.client = provider.client.with_options(timeout=_REQUEST_TIMEOUT, max_retries=0)
    return provider


class JudgeCompatibilityProbe:
    """Keep conclusive results briefly and coalesce checks for the same model role."""

    def __init__(self) -> None:
        self._fingerprint_key = secrets.token_bytes(32)
        self._cache: OrderedDict[ProbeKey, tuple[float, ProbeOutcome]] = OrderedDict()
        self._inflight: dict[ProbeKey, asyncio.Task[ProbeOutcome]] = {}

    async def resolve(
        self, check: CheckRole, selected: ResolvedConnection, fallback: ResolvedConnection
    ) -> JudgeCompatibilityResult:
        outcome = await self._probe(check, selected)
        effective = selected
        status: Literal["compatible", "fallback", "unknown", "unavailable"]
        if outcome == "unsupported":
            if (
                self._key(check, selected) != self._key(check, fallback)
                and await self._probe(check, fallback) == "compatible"
            ):
                status = "fallback"
                effective = fallback
            else:
                status = "unavailable"
        else:
            status = outcome
        return JudgeCompatibilityResult(
            status=status, provider=effective.selection.provider, model=effective.model
        )

    def _key(self, check: CheckRole, role: ResolvedConnection) -> ProbeKey:
        provider = role.selection.provider
        if role.is_byok:
            key, endpoint = role.api_key, role.base_url
        else:
            key = os.getenv(f"{provider.upper()}_API_KEY")
            endpoint = os.getenv(f"{provider.upper()}_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        # Never retain a raw secret in a cache key, including user-supplied URL paths.
        fingerprint = hmac.digest(
            self._fingerprint_key,
            "\0".join(
                (
                    endpoint or "",
                    key or "",
                    os.getenv("OPENAI_ORG_ID", ""),
                    os.getenv("OPENAI_PROJECT_ID", ""),
                )
            ).encode(),
            "sha256",
        )
        return role.connection_id, provider, role.model, check, fingerprint

    async def _probe(self, check: CheckRole, role: ResolvedConnection) -> ProbeOutcome:
        key = self._key(check, role)
        cached = self._cache.pop(key, None)
        if cached is not None and time.monotonic() < cached[0]:
            self._cache[key] = cached
            return cached[1]
        task = self._inflight.get(key)
        if task is None:
            if len(self._inflight) >= _MAX_INFLIGHT:
                return "unknown"
            task = self._inflight[key] = asyncio.create_task(self._record(key, check, role))
        try:
            # A request timeout or cancellation must leave the shared task holding its slot.
            return await asyncio.wait_for(asyncio.shield(task), _PROBE_DEADLINE)
        except TimeoutError:
            return "unknown"

    async def _record(
        self, key: ProbeKey, check: CheckRole, role: ResolvedConnection
    ) -> ProbeOutcome:
        try:
            outcome = await asyncio.to_thread(_probe_decision, check, role)
            if outcome != "unknown":
                self._cache[key] = (time.monotonic() + _CACHE_TTL, outcome)
                if len(self._cache) > _MAX_CACHE_ENTRIES:
                    self._cache.popitem(last=False)
            return outcome
        finally:
            self._inflight.pop(key, None)


def _probe_decision(check: CheckRole, role: ResolvedConnection) -> ProbeOutcome:
    provider = None
    try:
        provider = _open_probe_provider(role)
        if not supports_decisions(provider):
            return "unsupported"
        labels = _CHECK_LABELS[check]
        decision = provider.decide(
            [
                {
                    "role": "system",
                    "content": (
                        f"Reply with exactly one label: {', '.join(labels)}. Do not explain."
                    ),
                },
                {"role": "user", "content": f"Reply with {labels[0]}."},
            ],
            labels,
        )
        # Revalidate the results of adapters that may construct their own decisions.
        if not isinstance(decision, ClassDecision):
            return "unsupported"
        validated = ClassDecision.model_validate(decision.model_dump())
        return "compatible" if set(validated.distribution) == set(labels) else "unsupported"
    except (DecisionCapabilityError, ValidationError):
        return "unsupported"
    except Exception:
        # A network, quota or configuration failure establishes no decision capability.
        return "unknown"
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            with suppress(Exception):
                close()
