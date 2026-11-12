"""Legacy v1 differentiation records stay readable and regenerate identically.

The fixture was authored and approved by the frozen ``ef8bc08`` AI package
before the reusable-check catalogue. The backend must keep serving those
private records after the AI upgrade, without rewriting their content or
digests.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from edcraft_backend.config.settings import AuthoringSettings, CheckPilotSettings
from edcraft_backend.dependencies import (
    get_check_pilot_service,
    get_current_user,
    get_provider_connection_service,
)
from edcraft_backend.main import app
from edcraft_backend.services.check_pilot_service import CheckPilotService

FIXTURE = Path(__file__).parents[1] / "fixtures" / "check_pilot" / "v1_differentiation"
META: dict[str, Any] = json.loads((FIXTURE / "expected.json").read_text())


@dataclass
class _User:
    id: UUID


def _service(storage_dir: Path) -> CheckPilotService:
    return CheckPilotService(
        authoring=AuthoringSettings(storage_dir=storage_dir),
        settings=CheckPilotSettings(
            max_concurrent_requests=1,
            admission_timeout_seconds=1.0,
            request_timeout_seconds=120.0,
        ),
    )


def _install(service: CheckPilotService, owner_id: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: _User(UUID(owner_id))
    app.dependency_overrides[get_check_pilot_service] = lambda: service
    app.dependency_overrides[get_provider_connection_service] = lambda: object()


async def test_v1_differentiation_record_replays_through_http(tmp_path: Path) -> None:
    storage_dir = tmp_path / "authoring"
    shutil.copytree(FIXTURE / "storage", storage_dir / "check-pilot")
    service = _service(storage_dir)
    _install(service, META["owner_id"])
    base = "/ai-authoring/check-pilot"

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            templates = await client.get(f"{base}/templates")
            assert templates.status_code == 200, templates.text
            records = templates.json()
            assert [record["template_id"] for record in records] == [META["template_id"]]
            record = records[0]
            assert record["approval"] == "approved"
            assert record["status"] == "ready"
            assert record["reused_check"] is False
            assert record["template_sha256"] == META["template_sha256"]
            assert record["check"]["check_ref"]["check_id"] == META["check_id"]
            assert record["check"]["check_ref"]["sha256"] == META["check_sha256"]

            checks = await client.get(f"{base}/checks")
            assert checks.status_code == 200, checks.text
            assert [check["check_ref"]["check_id"] for check in checks.json()] == [META["check_id"]]

            generated = await client.post(
                f"{base}/templates/{META['template_id']}/generate",
                json={"seed": META["seed"], "quantity": META["quantity"]},
            )
            assert generated.status_code == 200, generated.text
            assert generated.json() == META["expected_batch"]

            # A stale digest still fails closed after the AI upgrade.
            stale = await client.post(
                f"{base}/templates/{META['template_id']}/approval",
                json={
                    "template_sha256": "0" * 64,
                    "check_sha256": META["check_sha256"],
                    "decision": "approved",
                },
            )
            assert stale.status_code == 409
    finally:
        app.dependency_overrides.clear()


async def test_v1_record_stays_private_to_its_owner(tmp_path: Path) -> None:
    storage_dir = tmp_path / "authoring"
    shutil.copytree(FIXTURE / "storage", storage_dir / "check-pilot")
    service = _service(storage_dir)
    _install(service, "99999999-8888-7777-6666-555555555555")
    base = "/ai-authoring/check-pilot"

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            assert (await client.get(f"{base}/templates")).json() == []
            assert (await client.get(f"{base}/checks")).json() == []
            assert (await client.get(f"{base}/templates/{META['template_id']}")).status_code == 404
            assert (await client.get(f"{base}/checks/{META['check_id']}")).status_code == 404
    finally:
        app.dependency_overrides.clear()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
