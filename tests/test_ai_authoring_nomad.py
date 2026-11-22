"""The HTTP boundary preserves a single Nomad execution per authoring attempt."""

import json
from uuid import uuid4

import httpx
import pytest

from edcraft_backend.config.settings import NomadSettings
from edcraft_backend.executors.nomad_authoring import NomadAuthoringExecutor


async def test_registration_is_create_only_and_passes_no_job_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOCLAAS_REASONING_EFFORT", "low")
    monkeypatch.setenv("SOCLAAS_API_KEY", "a-synthetic-key")
    job_id = uuid4()
    attempt_id = uuid4()
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"EvalID": "example"})

    worker = NomadAuthoringExecutor(
        NomadSettings(namespace="authoring"),
        image="example/worker:source-sha",
        callback_base_url="http://api.test",
        transport=httpx.MockTransport(respond),
    )

    await worker.register(job_id, attempt_id, "source-sha", "worker-bearer")
    request = requests[0]
    assert request.url.params["namespace"] == "authoring"
    body = json.loads(request.content)
    assert body["EnforceIndex"] and body["JobModifyIndex"] == 0
    job = body["Job"]
    assert job["ID"] == f"edcraft-authoring-{job_id}-{attempt_id}"
    assert job["Type"] == "batch"
    group = job["TaskGroups"][0]
    assert group["RestartPolicy"]["Attempts"] == 0
    assert group["ReschedulePolicy"]["Attempts"] == 0
    task = group["Tasks"][0]
    # Use the image's entrypoint and default command together.
    assert task["Config"] == {"image": "example/worker:source-sha", "force_pull": False}
    root = f"http://api.test/ai-authoring/worker/{job_id}/{attempt_id}"
    expected = {
        "EDCRAFT_JOB_ID": str(job_id),
        "EDCRAFT_LIBRARY_REVISION": "source-sha",
        "EDCRAFT_WORKER_TOKEN": "worker-bearer",
        "EDCRAFT_CONTEXT_URL": f"{root}/context",
        "EDCRAFT_PROGRESS_URL": f"{root}/progress",
        "EDCRAFT_CALLBACK_URL": f"{root}/result",
        "EDCRAFT_ERROR_URL": f"{root}/error",
        "SOCLAAS_REASONING_EFFORT": "low",
    }
    assert all(task["Env"][key] == value for key, value in expected.items())
    assert "SOCLAAS_API_KEY" not in task["Env"]
    assert "EDCRAFT_PARAMS_B64" not in task["Env"]


async def test_admission_slot_waits_for_a_stopping_allocation() -> None:
    allocation_states = iter(["running", "complete"])

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/allocations"):
            return httpx.Response(200, json=[{"ClientStatus": next(allocation_states)}])
        return httpx.Response(200, json={"Status": "dead", "Stop": True})

    worker = NomadAuthoringExecutor(
        NomadSettings(),
        image="example/worker:source-sha",
        callback_base_url="http://api.test",
        transport=httpx.MockTransport(respond),
    )

    assert await worker.status("stopping") == "active"
    assert await worker.status("stopped") == "stopped"
