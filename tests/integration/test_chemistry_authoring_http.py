"""Chemistry deterministic authoring HTTP regression tests."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pytest
from edcraft_validator.domains.chemistry.chemistry_schemas import ChemistryQuestionInstance

from edcraft_backend.services.ai_domains import ENABLED_DOMAINS_ENV
from tests.chemistry_authoring_helpers import (
    CHEMISTRY_REACTION_IDS,
    chemistry_concept_proposal,
    chemistry_yield_proposal,
    run_scripted_chemistry_pipeline,
)
from tests.integration.test_ai_authoring_api import approve, complete_job, env_for, review


@pytest.mark.asyncio
async def test_real_scripted_chemistry_pipeline_round_trips_through_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,chemistry")
    async with env_for(tmp_path, run_scripted_chemistry_pipeline) as env:
        job_id = await complete_job(
            env,
            domain="chemistry",
            prompt="Create a balanced alkane combustion equation question",
            difficulty="medium",
        )
        envelope = await review(env, job_id)
        assert envelope["domain"] == "chemistry"
        assert envelope["request"]["domain"] == "chemistry"
        candidate = envelope["template"]["candidate"]
        assert candidate["item"]["family"] == "alkane_combustion"
        assert [
            reaction["id"] for reaction in candidate["catalogue"]["reactions"]
        ] == CHEMISTRY_REACTION_IDS
        assert len(candidate["item"]["distractors"]) == 3

        limits = envelope["validation_limits"]
        assert limits["validator_version"] == "mcp-chemistry-template-v1"
        assert limits["whole_domain"] is True

        assert [check["tool"] for check in envelope["checks"]] == [
            "chemistry_verify_balance",
            "chemistry_verify_options",
            "assess_template_difficulty",
            "assess_template_relevance",
        ]
        assert all(check["status"] == "passed" for check in envelope["checks"])
        assert [check["advisory"] for check in envelope["checks"]] == [
            False,
            False,
            True,
            True,
        ]

        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200
        preview = await env.client.post(f"/ai-authoring/jobs/{job_id}/preview", json={"seed": 3})
        assert preview.status_code == 200, preview.text
        instance = ChemistryQuestionInstance.model_validate(preview.json()["question"])
        assert instance.answer == instance.choices[instance.correct_index]
        assert instance.reaction_id in CHEMISTRY_REACTION_IDS

        batch = await env.client.post(
            f"/ai-authoring/jobs/{job_id}/generate",
            json={"artifact_sha256": digest, "seed": 3, "quantity": 2},
        )
        assert batch.status_code == 200, batch.text
        assert batch.json()["domain"] == "chemistry"
        assert [item["seed"] for item in batch.json()["questions"]] == [3, 4]
        for item in batch.json()["questions"]:
            generated = ChemistryQuestionInstance.model_validate(item)
            assert generated.answer == generated.choices[generated.correct_index]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("proposal_factory", "kind", "authority", "answers"),
    [
        (
            chemistry_concept_proposal,
            "concept",
            "chemistry_verify_content",
            {"12, 13, 12", "Proton"},
        ),
        (chemistry_yield_proposal, "theoretical_yield", "chemistry_verify_balance", {"0.67 mmol"}),
    ],
)
async def test_h2_chemistry_uses_normal_review_approval_and_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal_factory: Any,
    kind: str,
    authority: str,
    answers: set[str],
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,chemistry")
    pipeline = partial(run_scripted_chemistry_pipeline, proposal_factory=proposal_factory)
    async with env_for(tmp_path, pipeline) as env:
        job_id = await complete_job(env, domain="chemistry", prompt="Practise H2 chemistry")
        envelope = await review(env, job_id)
        item = envelope["template"]["candidate"]["item"]
        assert item["target"]["kind"] == kind
        assert len(item["distractors"]) == 3
        cases = envelope["template"]["case_previews"]
        if kind == "concept":
            assert {case["answer"] for case in cases} == answers
            assert [case["case_id"] for case in cases] == item["parameters"][0]["values"]
            assert all(case["question"] for case in cases)
            assert {case["case_id"]: case["key_evidence"] for case in cases} == {
                "skill-1-d-1-a": ["separate_reference"],
                "skill-1-a-1-a": None,
            }
        else:
            assert cases is None
        assert {row["tool"] for row in envelope["checks"] if not row["advisory"]} == {
            authority,
            "chemistry_verify_options",
        }
        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200
        generated = await env.client.post(
            f"/ai-authoring/jobs/{job_id}/generate",
            json={"artifact_sha256": digest, "seed": 5, "quantity": 8},
        )
        assert generated.status_code == 200, generated.text
        questions = [
            ChemistryQuestionInstance.model_validate(q) for q in generated.json()["questions"]
        ]
        assert {question.answer for question in questions} == answers
        assert all(len(set(question.choices)) == 4 for question in questions)
        if cases is not None:
            generated_choices = {q.reaction_id: set(q.choices) for q in questions}
            assert {case["case_id"] for case in cases} == set(generated_choices)
            for case in cases:
                assert set(case["choices"]) == generated_choices[case["case_id"]]
                assert case["choices"].count(case["answer"]) == 1
                assert case["key_source"] == "catalogue"


@pytest.mark.asyncio
async def test_chemistry_unsupported_format_preserves_a_typed_review_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,chemistry")
    async with env_for(tmp_path, run_scripted_chemistry_pipeline) as env:
        job_id = await complete_job(
            env,
            domain="chemistry",
            prompt="H2 skill 6.practical-data-9. Assess practical handling.",
        )
        status = (await env.client.get(f"/ai-authoring/jobs/{job_id}")).json()
        assert status["result"]["status"] == "needs_review"
        error = status["result"]["error"]
        assert error["code"] == "UNSUPPORTED_REQUEST"
        assert "needs assessment beyond this MCQ" in error["message"]
        envelope = await review(env, job_id)
        assert envelope["template"] is None
        assert envelope["attempts"] == []
        assert envelope["error"] == error
