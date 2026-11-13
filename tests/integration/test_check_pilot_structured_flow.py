"""Exercise new AI-owned problem/answer shapes through unchanged HTTP routes.

Providers here are fixtures; live model evidence is collected separately.
"""

from pathlib import Path

import pytest

from tests.check_catalogue_fixtures import ProviderScripts
from tests.integration.test_check_pilot_catalogue_flow import (
    BASE,
    _approve,
    _author,
    _generate,
    _harness,
)
from tests.structured_check_fixtures import FAMILIES, script


@pytest.mark.parametrize("contract_id", list(FAMILIES))
async def test_structured_answer_author_approve_generate_restart_and_reuse(
    tmp_path: Path, contract_id: str
) -> None:
    scripts = ProviderScripts([script(contract_id)])
    async with _harness(tmp_path, scripts) as harness:
        response = await _author(harness.client, "Practice " + contract_id)
        assert response.status_code == 200, response.text
        record = response.json()
        assert record["status"] == "ready", record["findings"]
        assert record["selection"]["contract_id"] == contract_id
        assert record["check"]["scope"]["contract_id"] == contract_id
        assert record["check"]["definition"]["operation_set"] == "structured-check/1"
        assert record["template"]["schema_version"] == "pilot-template/3"
        assert record["check_qualification"]["status"] == "passed"
        assert record["qualification"]["status"] == "passed"

        unapproved = await _generate(harness.client, record["template_id"], seed=42, quantity=3)
        assert unapproved.status_code == 409
        approved = await _approve(harness.client, record)
        assert approved.status_code == 200, approved.text
        generated = await _generate(harness.client, record["template_id"], seed=42, quantity=3)
        assert generated.status_code == 200, generated.text
        batch = generated.json()
        assert batch["shortfall"] == 0
        assert len(batch["questions"]) == 3
        for question in batch["questions"]:
            assert question["choices"][question["correct_index"]] == question["answer"]
            assert question["check_ref"] == record["check"]["check_ref"]
            assert len(set(question["choices"])) == 4
        calls_after_generation = list(scripts.providers[0].calls)
        assert calls_after_generation == ["check_plan", "check_recipe", "polynomial_family"]

    # A fresh service reads the same immutable artifacts and approval.
    scripts.scripts.append(script(contract_id))
    async with _harness(tmp_path, scripts) as restarted:
        replay = await _generate(restarted.client, record["template_id"], seed=42, quantity=3)
        assert replay.status_code == 200, replay.text
        assert replay.json() == batch
        checks = await restarted.client.get(f"{BASE}/checks")
        assert checks.status_code == 200
        assert len(checks.json()) == 1
        reused_response = await _author(restarted.client, "More practice " + contract_id)
        assert reused_response.status_code == 200, reused_response.text
        reused = reused_response.json()
        assert reused["status"] == "ready", reused["findings"]
        assert reused["reused_check"] is True
        assert reused["check"]["check_ref"] == record["check"]["check_ref"]
        assert reused["approval"] == "unreviewed"
        assert scripts.providers[-1].calls == ["check_plan", "polynomial_family"]
