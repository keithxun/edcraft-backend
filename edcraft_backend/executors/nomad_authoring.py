"""Nomad batch executor for flag-gated AI authoring worker jobs.

The worker image contains the pinned ``edcraft_validator`` code and starts with
``python -m edcraft_validator.worker``. The job spec deliberately carries only
non-secret job metadata: model credentials are fetched by the worker from the
token-authenticated context endpoint.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

AUTHORING_JOB_PREFIX = "edcraft-authoring-"
WORKER_COMMAND = "python"
WORKER_ARGS = ["-m", "edcraft_validator.worker"]


def build_authoring_job_spec(
    *,
    job_id: UUID,
    command: dict[str, Any],
    context_url: str,
    progress_url: str,
    callback_url: str,
    error_url: str,
    library_revision: str,
    datacenters: list[str],
    namespace: str | None,
    cpu_mhz: int,
    memory_mb: int,
    container_image: str,
    force_pull: bool,
) -> dict[str, Any]:
    """Build the exact Nomad batch job spec without credentials or source code."""
    nomad_job_id = f"{AUTHORING_JOB_PREFIX}{job_id}"
    params_b64 = base64.b64encode(
        json.dumps(command, separators=(",", ":"), default=str).encode()
    ).decode()

    spec: dict[str, Any] = {
        "ID": nomad_job_id,
        "Name": nomad_job_id,
        "Type": "batch",
        "Datacenters": list(datacenters),
        "TaskGroups": [
            {
                "Name": "worker",
                "Count": 1,
                "RestartPolicy": {"Attempts": 0, "Mode": "fail"},
                "Tasks": [
                    {
                        "Name": "run",
                        "Driver": "docker",
                        "Config": {
                            "image": container_image,
                            "force_pull": force_pull,
                            "command": WORKER_COMMAND,
                            "args": list(WORKER_ARGS),
                        },
                        "Env": {
                            "EDCRAFT_PARAMS_B64": params_b64,
                            "EDCRAFT_CONTEXT_URL": context_url,
                            "EDCRAFT_PROGRESS_URL": progress_url,
                            "EDCRAFT_CALLBACK_URL": callback_url,
                            "EDCRAFT_ERROR_URL": error_url,
                            "EDCRAFT_LIBRARY_REVISION": library_revision,
                            "EDCRAFT_JOB_ID": str(job_id),
                        },
                        "Resources": {
                            "CPU": cpu_mhz,
                            "MemoryMB": memory_mb,
                        },
                    }
                ],
            }
        ],
    }
    if namespace:
        spec["Namespace"] = namespace
    return spec


class NomadAuthoringExecutor:
    """Submit, stop and inspect one Nomad allocation per authoring job."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        library_revision: str,
        container_image: str,
        token: str | None = None,
        namespace: str | None = None,
        datacenters: list[str] | None = None,
        cpu_mhz: int = 500,
        memory_mb: int = 512,
        force_pull: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._token = token
        self._namespace = namespace
        self._datacenters = list(datacenters or ["dc1"])
        self._cpu_mhz = cpu_mhz
        self._memory_mb = memory_mb
        self._container_image = container_image
        self._force_pull = force_pull
        self._library_revision = library_revision
        self._transport = transport

    @property
    def _base_url(self) -> str:
        return f"http://{self._host}:{self._port}/v1"

    @property
    def _headers(self) -> dict[str, str]:
        if self._token:
            return {"X-Nomad-Token": self._token}
        return {}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=30.0, transport=self._transport)

    def job_spec(
        self,
        *,
        job_id: UUID,
        command: dict[str, Any],
        context_url: str,
        progress_url: str,
        callback_url: str,
        error_url: str,
    ) -> dict[str, Any]:
        """Expose the generated spec for tests and documentation samples."""
        return build_authoring_job_spec(
            job_id=job_id,
            command=command,
            context_url=context_url,
            progress_url=progress_url,
            callback_url=callback_url,
            error_url=error_url,
            library_revision=self._library_revision,
            datacenters=self._datacenters,
            namespace=self._namespace,
            cpu_mhz=self._cpu_mhz,
            memory_mb=self._memory_mb,
            container_image=self._container_image,
            force_pull=self._force_pull,
        )

    async def submit(
        self,
        job_id: UUID,
        command: dict[str, Any],
        callback_url: str,
        progress_url: str,
        context_url: str,
        error_url: str,
        timeout_seconds: float,
    ) -> str:
        """Submit one batch job and return its Nomad job ID."""
        spec = self.job_spec(
            job_id=job_id,
            command=command,
            context_url=context_url,
            progress_url=progress_url,
            callback_url=callback_url,
            error_url=error_url,
        )
        logger.info("Submitting Nomad authoring job", extra={"job_id": str(job_id)})
        async with self._client() as client:
            response = await client.post(
                f"{self._base_url}/jobs",
                json={"Job": spec},
                headers=self._headers,
            )
            response.raise_for_status()
        nomad_job_id = str(spec["ID"])
        logger.info(
            "Nomad authoring job submitted",
            extra={"job_id": str(job_id), "nomad_job_id": nomad_job_id},
        )
        return nomad_job_id

    async def stop(self, nomad_job_id: str) -> None:
        """Best-effort purge of a worker allocation; a missing job is success."""
        async with self._client() as client:
            response = await client.delete(
                f"{self._base_url}/job/{nomad_job_id}",
                params={"purge": "true"},
                headers=self._headers,
            )
        if response.status_code == 404:
            return
        response.raise_for_status()

    async def status(self, nomad_job_id: str) -> str:
        """Return a coarse ``running``/``completed``/``failed``/``missing`` status."""
        async with self._client() as client:
            response = await client.get(
                f"{self._base_url}/job/{nomad_job_id}/summary",
                headers=self._headers,
            )
        if response.status_code == 404:
            return "missing"
        response.raise_for_status()
        summary = response.json()
        groups = summary.get("Summary", {}) if isinstance(summary, dict) else {}
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            if group.get("Failed", 0) > 0:
                return "failed"
            if group.get("Complete", 0) > 0:
                return "completed"
        return "running"


__all__ = [
    "AUTHORING_JOB_PREFIX",
    "NomadAuthoringExecutor",
    "build_authoring_job_spec",
]
