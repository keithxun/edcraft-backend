"""The allowlist decides which registered domains users may choose."""

import pytest

from edcraft_backend.schemas.ai_authoring import SubmitAuthoringJobRequest
from edcraft_backend.services import ai_domains


def test_default_enables_all_five_subjects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ai_domains.ENABLED_DOMAINS_ENV, raising=False)
    assert set(ai_domains.enabled_domains()) == {"code", "math", "biology", "chemistry", "physics"}


def test_allowlist_hides_a_registered_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ai_domains.ENABLED_DOMAINS_ENV, "math")
    assert ai_domains.enabled_domains() == ("math",)


def test_unregistered_names_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ai_domains.ENABLED_DOMAINS_ENV, "code, unknown ,")
    assert ai_domains.enabled_domains() == ("code",)


def test_submit_rejects_a_disabled_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ai_domains.ENABLED_DOMAINS_ENV, "code")
    with pytest.raises(ValueError, match="Unsupported domain 'math'"):
        SubmitAuthoringJobRequest.model_validate(
            {"domain": "math", "prompt": "Factorise x^2 - 1.", "difficulty": "easy"}
        )
