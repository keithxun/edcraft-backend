"""Focused cache and deadline checks for judge compatibility."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from uuid import uuid4

import pytest

from edcraft_backend.services import ai_authoring_compatibility as compatibility
from edcraft_backend.services.provider_endpoint import ResolvedConnection


async def test_probe_cache_is_scoped_by_connection_role_model_endpoint_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ResolvedConnection.for_connection(
        connection_id=uuid4(),
        provider="openai",
        model="demo",
        api_key="test-key",
        base_url="https://provider.example/v1",
    )
    fallback = ResolvedConnection.for_judge(provider="soclaas", model="x-test-2")
    calls = 0

    def compatible(*_args: object) -> str:
        nonlocal calls
        calls += 1
        return "compatible"

    monkeypatch.setattr(compatibility, "_probe_decision", compatible)
    probes = compatibility.JudgeCompatibilityProbe()
    candidates: list[tuple[compatibility.CheckRole, ResolvedConnection]] = [
        ("difficulty", connection),
        ("difficulty", connection),
        ("relevance", connection),
        ("difficulty", replace(connection, connection_id=uuid4())),
        ("difficulty", replace(connection, api_key="rotated-key")),
        ("difficulty", replace(connection, base_url="https://changed.example/v1")),
        (
            "difficulty",
            replace(
                connection, selection=connection.selection.model_copy(update={"model": "other"})
            ),
        ),
    ]
    for check, role in candidates:
        result = await probes.resolve(check, role, fallback)
        assert result.status == "compatible"
    assert calls == 6
    assert "test-key" not in repr(probes._cache)
    assert "rotated-key" not in repr(probes._cache)


async def test_deadline_keeps_capacity_for_inflight_work_and_coalesces_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compatibility, "_MAX_INFLIGHT", 1)
    monkeypatch.setattr(compatibility, "_PROBE_DEADLINE", 0.05)
    release = threading.Event()
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()
    calls = []

    def slow_probe(check: str, _role: ResolvedConnection) -> str:
        calls.append(check)
        release.wait(2)
        loop.call_soon_threadsafe(finished.set)
        return "compatible"

    monkeypatch.setattr(compatibility, "_probe_decision", slow_probe)
    probes = compatibility.JudgeCompatibilityProbe()
    model = ResolvedConnection.for_judge(provider="openai", model="demo")
    fallback = ResolvedConnection.for_judge(provider="soclaas", model="x-test-2")

    try:
        timed_out = await asyncio.gather(
            *(probes.resolve("difficulty", model, fallback) for _ in range(2))
        )
        assert [result.status for result in timed_out] == ["unknown", "unknown"]
        assert (await probes.resolve("relevance", model, fallback)).status == "unknown"
        assert calls == ["difficulty"]
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
