"""Bounded check-pilot configuration and storage-root derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from edcraft_backend.config.settings import (
    AuthoringSettings,
    CheckPilotSettings,
)


def test_check_pilot_defaults_are_small_and_bounded() -> None:
    settings = CheckPilotSettings()
    assert settings.max_concurrent_requests == 2
    assert settings.admission_timeout_seconds == 5.0
    assert settings.request_timeout_seconds == 180.0


def test_check_pilot_settings_reject_unbounded_values() -> None:
    with pytest.raises(ValueError):
        CheckPilotSettings(max_concurrent_requests=0)
    with pytest.raises(ValueError):
        CheckPilotSettings(admission_timeout_seconds=0)
    with pytest.raises(ValueError):
        CheckPilotSettings(request_timeout_seconds=0)


def test_check_pilot_root_is_derived_from_the_authoring_directory(tmp_path: Path) -> None:
    from edcraft_backend.services.check_pilot_service import CheckPilotService

    authoring = AuthoringSettings(storage_dir=tmp_path / "authoring")
    service = CheckPilotService(authoring=authoring, settings=CheckPilotSettings())

    assert service.root == tmp_path / "authoring" / "check-pilot"
    # Importing or constructing the service must not create storage.
    assert not service.root.exists()
