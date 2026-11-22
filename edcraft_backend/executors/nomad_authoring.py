"""Nomad registration and allocation state for one durable authoring attempt."""

from __future__ import annotations

import os
from typing import Literal
from uuid import UUID

import httpx

from edcraft_backend.config.settings import NomadSettings

NomadAuthoringStatus = Literal["missing", "active", "stopped", "unreachable"]
_WORKER_TUNING = (
    "OPENAI_GENERATION_MAX_TOKENS",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_RESPONSE_FORMAT",
    "SOCLAAS_GENERATION_MAX_TOKENS",
    "SOCLAAS_REASONING_EFFORT",
    "SOCLAAS_RESPONSE_FORMAT",
    "EDCRAFT_JUDGE_REASONING_EFFORT",
    "EDCRAFT_RELEVANCE_THRESHOLD",
)


def nomad_authoring_job_id(job_id: UUID, attempt_id: UUID) -> str:
    return f"edcraft-authoring-{job_id}-{attempt_id}"


class NomadUnavailable(Exception):
    """The outcome of a Nomad API operation could not be confirmed."""


class NomadAuthoringExecutor:
    """Register only new job IDs, and preserve them when stopping an attempt."""

    def __init__(
        self,
        config: NomadSettings,
        *,
        image: str,
        callback_base_url: str,
        force_pull: bool = False,
        allow_private_provider_hosts: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._image = image
        self._callback_base_url = callback_base_url.rstrip("/")
        self._force_pull = force_pull
        self._allow_private_provider_hosts = allow_private_provider_hosts
        self._transport = transport

    async def register(
        self, job_id: UUID, attempt_id: UUID, library_revision: str, token: str
    ) -> None:
        """Create the batch registration under its already-persisted identity."""
        config = self._config
        job_name = nomad_authoring_job_id(job_id, attempt_id)
        callback = f"{self._callback_base_url}/ai-authoring/worker/{job_id}/{attempt_id}"
        env = {
            "EDCRAFT_JOB_ID": str(job_id),
            "EDCRAFT_LIBRARY_REVISION": library_revision,
            "EDCRAFT_WORKER_TOKEN": token,
            "EDCRAFT_CONTEXT_URL": f"{callback}/context",
            "EDCRAFT_PROGRESS_URL": f"{callback}/progress",
            "EDCRAFT_CALLBACK_URL": f"{callback}/result",
            "EDCRAFT_ERROR_URL": f"{callback}/error",
        }
        env.update({name: os.environ[name] for name in _WORKER_TUNING if name in os.environ})
        if self._allow_private_provider_hosts:
            env["PROVIDER_ALLOW_PRIVATE_HOSTS"] = "true"
        job = {
            "ID": job_name,
            "Name": job_name,
            "Type": "batch",
            "Datacenters": config.datacenters,
            "Namespace": config.namespace or "default",
            "TaskGroups": [
                {
                    "Name": "worker",
                    "Count": 1,
                    "RestartPolicy": {"Attempts": 0, "Mode": "fail"},
                    "ReschedulePolicy": {"Attempts": 0, "Unlimited": False},
                    "Tasks": [
                        {
                            "Name": "run",
                            "Driver": "docker",
                            "Config": {
                                "image": self._image,
                                "force_pull": self._force_pull,
                            },
                            "Env": env,
                            "Resources": {"CPU": config.cpu_mhz, "MemoryMB": config.memory_mb},
                        }
                    ],
                }
            ],
        }
        await self._request(
            "POST",
            "/jobs",
            json={"Job": job, "EnforceIndex": True, "JobModifyIndex": 0},
        )

    async def status(self, job_name: str) -> NomadAuthoringStatus:
        """Return stopped only when Nomad has no active allocation for the job."""
        job_path = f"/job/{job_name}"
        response = await self._request("GET", job_path, allow_missing=True)
        if response.status_code != 404:
            job = self._json(response, dict)
            if job.get("Status") != "dead":
                return "active"
        allocations = await self._request(
            "GET", f"{job_path}/allocations", params={"all": "true"}, allow_missing=True
        )
        unreachable = False
        if allocations.status_code != 404:
            for allocation in self._json(allocations, list):
                if not isinstance(allocation, dict):
                    raise NomadUnavailable()
                state = allocation.get("ClientStatus")
                if state in {"lost", "unknown"}:
                    unreachable = True
                elif state not in {"complete", "failed"}:
                    return "active"
        if unreachable:
            return "unreachable"
        return "missing" if response.status_code == 404 else "stopped"

    async def stop(self, job_name: str) -> None:
        """Deregister without purging, keeping create-only registration fenced."""
        await self._request(
            "DELETE", f"/job/{job_name}", params={"purge": "false"}, allow_missing=True
        )

    @staticmethod
    def _json[T](response: httpx.Response, container: type[T]) -> T:
        try:
            payload = response.json()
        except ValueError:
            raise NomadUnavailable() from None
        if not isinstance(payload, container):
            raise NomadUnavailable()
        return payload

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json: object | None = None,
        allow_missing: bool = False,
    ) -> httpx.Response:
        config = self._config
        try:
            async with httpx.AsyncClient(
                base_url=f"http://{config.host}:{config.port}/v1",
                timeout=10.0,
                transport=self._transport,
                trust_env=False,
                headers={"X-Nomad-Token": config.token} if config.token else {},
            ) as client:
                response = await client.request(
                    method,
                    path,
                    json=json,
                    params={"namespace": config.namespace or "default", **(params or {})},
                )
                if allow_missing and response.status_code == 404:
                    return response
                response.raise_for_status()
        except httpx.HTTPError:
            # Never emit a response body or a request containing the bearer.
            raise NomadUnavailable() from None
        return response
