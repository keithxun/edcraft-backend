"""Physics deterministic authoring HTTP regression tests."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from edcraft_validator.domains.physics.h2_authoring import H2CompactUnsupported
from edcraft_validator.domains.physics.physics_domain import PhysicsDomain
from edcraft_validator.domains.physics.physics_schemas import PhysicsQuestionInstance

from edcraft_backend.services.ai_domains import ENABLED_DOMAINS_ENV
from tests.integration.test_ai_authoring_api import (
    approve,
    complete_job,
    env_for,
    payload,
    review,
    submit,
)
from tests.physics_authoring_helpers import (
    run_scripted_h2_physics_pipeline,
    run_scripted_physics_pipeline,
)


@pytest.mark.asyncio
async def test_physics_can_be_disabled_and_round_trips_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math")
    async with env_for(tmp_path, run_scripted_physics_pipeline) as env:
        capabilities = (await env.client.get("/ai-authoring/capabilities")).json()
        assert "physics" not in capabilities["enabled_domains"]
        rejected = await env.client.post("/ai-authoring/jobs", json=payload(domain="physics"))
        assert rejected.status_code == 422

        monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,physics")
        job_id = await complete_job(
            env,
            domain="physics",
            prompt="Find the acceleration of a cart pushed against friction",
            difficulty="medium",
        )
        envelope = await review(env, job_id)
        assert envelope["request"]["domain"] == "physics"
        assert envelope["template"].get("case_previews") is None
        body = envelope["template"]["candidate"]["item"]["body"]
        assert body["target"] == "acceleration"
        assert len(body["distractors"]) == 3
        assert body["distractors"][0]["misconception"] == "added_opposing_forces"
        limits = envelope["validation_limits"]
        assert limits["validator_version"] == "mcp-physics-template-v1"
        assert limits["whole_domain"] is True
        assert limits["declared_domain_size"] == 12
        assert limits["examples_checked"] == 12
        assert [check["tool"] for check in envelope["checks"]] == [
            "physics_verify_item",
            "physics_validate_options",
            "physics_verify_text",
            "physics_verify_text_fidelity",
            "assess_template_difficulty",
            "assess_template_relevance",
        ]
        assert [check["status"] for check in envelope["checks"]] == ["passed"] * 6

        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200
        preview = await env.client.post(f"/ai-authoring/jobs/{job_id}/preview", json={"seed": 3})
        assert preview.status_code == 200, preview.text
        question = PhysicsQuestionInstance.model_validate(preview.json()["question"])
        assert len(question.choices) == 4
        assert question.answer == question.choices[question.correct_index]
        assert "Take right as the positive direction." in question.question
        batch = await env.client.post(
            f"/ai-authoring/jobs/{job_id}/generate",
            json={"artifact_sha256": digest, "seed": 3, "quantity": 2},
        )
        assert batch.status_code == 200, batch.text
        assert batch.json()["domain"] == "physics"
        assert [item["seed"] for item in batch.json()["questions"]] == [3, 4]


@pytest.mark.asyncio
async def test_h2_physics_review_approval_and_seeded_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,physics")
    async with env_for(tmp_path, run_scripted_h2_physics_pipeline) as env:
        job_id = await complete_job(
            env,
            domain="physics",
            prompt="H2 Physics 9478: find a wave's propagation speed.",
            difficulty="medium",
        )
        envelope = await review(env, job_id)
        item = envelope["template"]["candidate"]["item"]
        assert envelope["template"].get("case_previews") is None
        assert item["family_id"] == "wave_speed"
        assert item["rule"] == "f × w"
        assert len(item["body"]["distractors"]) == 3
        assert "frequency {f} Hz" in item["body"]["text"]
        assert {axis["name"] for axis in item["parameters"]} == {"f", "w"}
        assert envelope["approval"] is None
        preview_url = f"/ai-authoring/jobs/{job_id}/preview"
        previews = []
        for seed in (0, 3, 31):
            preview = await env.client.post(preview_url, json={"seed": seed})
            assert preview.status_code == 200, preview.text
            instance = PhysicsQuestionInstance.model_validate(preview.json()["question"])
            assert instance.seed == seed
            assert len(set(instance.choices) - {instance.answer}) == 3
            assert instance.choices[instance.correct_index] == instance.answer
            previews.append(instance)
        assert len({tuple(item.choices) for item in previews}) > 1
        assert (await approve(env, job_id, "0" * 64)).status_code == 409
        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200
        response = await env.client.post(
            f"/ai-authoring/jobs/{job_id}/generate",
            json={"artifact_sha256": digest, "seed": 2, "quantity": 2},
        )
        assert response.status_code == 200, response.text
        questions = response.json()["questions"]
        assert len(questions) == 2
        for question in questions:
            assert question["target"] == "h2_quantity"
            assert len(question["choices"]) == 4
            assert question["answer"] == question["choices"][question["correct_index"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "outcome", "stimulus"),
    [
        (
            "Write an H2 Physics MCQ about unequal-mass carts in a head-on elastic "
            "collision. Find both signed outgoing velocities using momentum and "
            "relative speed.",
            "6(d)",
            None,
        ),
        (
            "Set an H2 Physics MCQ with a displayed potential-position graph. "
            "Find the signed electric field and identify the zero-field interval.",
            "14(f)",
            "graph",
        ),
        (
            "Set an H2 Physics MCQ about unequal masses in a 1D elastic collision. "
            "Apply momentum and the approach/separation relation to find the "
            "individual outgoing velocities.",
            "6(d)",
            None,
        ),
        (
            "Use a plotted electric potential versus distance in an H2 Physics "
            "MCQ and ask students to infer the electric field from the gradient.",
            "14(f)",
            "graph",
        ),
        (
            "Set an H2 Physics MCQ from a velocity-time graph. Ask for signed "
            "displacement, not acceleration.",
            "3(c)",
            "graph",
        ),
        (
            "I need an H2 Physics warm-up giving a measured quantity in prefixed "
            "units and asking for the equivalent SI value.",
            "1(b)",
            None,
        ),
        (
            "Write a Singapore H2 Physics MCQ that uses a turntable to convert part "
            "of a revolution to radians and recognise the corresponding rim arc length.",
            "7(a)",
            None,
        ),
    ],
)
async def test_h2_physics_resolves_verified_task_after_author_decline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    outcome: str,
    stimulus: str | None,
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,physics")
    executor = partial(
        run_scripted_h2_physics_pipeline,
        unsupported=H2CompactUnsupported(unsupported="outside_catalogue"),
    )
    async with env_for(tmp_path, executor) as env:
        job_id = await complete_job(
            env,
            domain="physics",
            prompt=prompt,
            difficulty="easy" if outcome in {"1(b)", "7(a)"} else "hard",
        )
        envelope = await review(env, job_id)
        cases = envelope["template"]["case_previews"]
        assert len(cases) >= 4
        assert all(case["outcome"] == outcome for case in cases)
        assert all(
            (case["stimulus"]["kind"] if case.get("stimulus") else None) == stimulus
            for case in cases
        )
        assert all(len(case["choices"]) == len(set(case["choices"])) == 4 for case in cases)
        assert all(case["answer"] in case["choices"] for case in cases)
        if outcome == "3(c)":
            assert all("acceleration" not in case["question"] for case in cases)
            assert all(case["answer"].endswith(" m") for case in cases)
        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200

        preview = await env.client.post(f"/ai-authoring/jobs/{job_id}/preview", json={"seed": 31})
        generated = await env.client.post(
            f"/ai-authoring/jobs/{job_id}/generate",
            json={"artifact_sha256": digest, "seed": 31, "quantity": 2},
        )
        assert preview.status_code == generated.status_code == 200
        assert preview.json()["question"] == generated.json()["questions"][0]
        instance = PhysicsQuestionInstance.model_validate_json(
            json.dumps(preview.json()["question"])
        )
        assert instance.target == "h2_task"
        assert instance.choices[instance.correct_index] == instance.answer
        if outcome == "1(b)":
            assert all(
                choice.endswith(f" {instance.parameters['unit']}") for choice in instance.choices
            )
        if outcome == "7(a)":
            angle, arc = instance.answer.split(", ")
            alternatives = [
                choice.split(", ") for choice in instance.choices if choice != instance.answer
            ]
            assert angle.endswith("π rad") and arc.endswith("π m")
            assert any(first == angle and second != arc for first, second in alternatives)
            assert any(first != angle and second == arc for first, second in alternatives)


@pytest.mark.asyncio
async def test_h2_physics_does_not_substitute_graph_with_supplied_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,physics")
    async with env_for(tmp_path, run_scripted_h2_physics_pipeline) as env:
        job_id = await complete_job(
            env,
            domain="physics",
            prompt=(
                "H2 Physics outcome 3(c). Use the velocity–time graph through "
                "(0, 0), (2, 4) and (5, −2) to assess signed displacement."
            ),
        )
        envelope = await review(env, job_id)
        assert envelope["error"]["code"] == "UNSUPPORTED_REQUEST"
        assert "fixed points" in envelope["error"]["message"]
        assert envelope["artifact_id"] is None
        assert envelope["template"] is None
        assert (await approve(env, job_id, "0" * 64)).status_code == 409


@pytest.mark.asyncio
async def test_h2_physics_practical_request_stops_for_review_without_an_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,physics")
    executor = partial(
        run_scripted_h2_physics_pipeline,
        unsupported=H2CompactUnsupported(unsupported="practical_investigation"),
    )
    async with env_for(tmp_path, executor) as env:
        job_id = await complete_job(
            env,
            domain="physics",
            prompt="For H2 Physics, grade the student's own accelerometer readings.",
        )
        response = (await env.client.get(f"/ai-authoring/jobs/{job_id}")).json()
        assert response["result"]["status"] == "needs_review"
        envelope = await review(env, job_id)
        assert envelope["error"] == response["result"]["error"]
        assert envelope["error"]["code"] == "UNSUPPORTED_REQUEST"
        assert "original measurements" in envelope["error"]["message"]
        assert envelope["artifact_id"] is None
        assert envelope["template"] is None
        assert envelope["checks"] == []
        assert len(envelope["attempts"]) == 1
        assert (await approve(env, job_id, "0" * 64)).status_code == 409


@pytest.mark.asyncio
async def test_h2_physics_unexpected_construction_error_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,physics")

    def fail_to_build(self: PhysicsDomain, request: Any, proposal: Any) -> None:
        raise RuntimeError("internal failure")

    monkeypatch.setattr(PhysicsDomain, "build_candidate", fail_to_build)
    async with env_for(tmp_path, run_scripted_h2_physics_pipeline) as env:
        job_id = await submit(
            env, domain="physics", prompt="H2 Physics 9478: find a wave's propagation speed."
        )
        await env.service.wait_for_idle()
        response = (await env.client.get(f"/ai-authoring/jobs/{job_id}")).json()
        assert response["state"] == "failed"
        assert response["result"]["status"] == "error"
        assert response["error"]["code"] == "TEMPLATE_BUILDING_ERROR"
        assert "internal failure" not in response["error"]["message"]


@pytest.mark.asyncio
async def test_h2_physics_circuit_prompt_keeps_symbols_across_review_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,physics")
    async with env_for(tmp_path, run_scripted_h2_physics_pipeline) as env:
        job_id = await complete_job(
            env,
            domain="physics",
            prompt="H2 Physics 9478 outcome 16(a): identify visual circuit symbols.",
            difficulty="medium",
        )
        envelope = await review(env, job_id)
        item = envelope["template"]["candidate"]["item"]
        assert item["skill_id"] == "16a"
        assert item["body"]["target"] == "Statement set"
        case_previews = envelope["template"]["case_previews"]
        assert [(case["case_id"], case["outcome"], case["answer"]) for case in case_previews] == [
            ("16a-1", "16(a)", "A and B only"),
            ("16a-2", "16(a)", "A and B only"),
        ]
        assert "labelled A–F" in case_previews[0]["question"]
        assert "symbols G through L" in case_previews[1]["question"]
        assert [case["stimulus"]["components"]["0"] for case in case_previews] == [
            "resistor",
            "thermistor",
        ]
        for case in case_previews:
            assert set(case["choices"]) == {"A and B only", "A only", "B only", "A, B and C only"}
            assert len(case["choices"]) == 4
            assert case["choices"].count(case["answer"]) == 1
            alternate = case["incorrect_statements_example"]
            assert alternate["question"].endswith("Which listed statements are incorrect?")
            assert alternate["question"].split("\n\n")[1] != case["question"].split("\n\n")[1]
            assert alternate["answer"] == "B and C only"
            assert set(alternate["choices"]) == {
                "B and C only",
                "A, B and C only",
                "C only",
                "B only",
            }
            assert len(alternate["choices"]) == 4
        limits = envelope["validation_limits"]
        assert limits["declared_domain_size"] == 96
        assert limits["whole_domain"] is False
        assert 1 <= limits["examples_checked"] == limits["execution_calls"] <= 12
        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200
        preview = await env.client.post(f"/ai-authoring/jobs/{job_id}/preview", json={"seed": 31})
        assert preview.status_code == 200, preview.text
        response = await env.client.post(
            f"/ai-authoring/jobs/{job_id}/generate",
            json={"artifact_sha256": digest, "seed": 31, "quantity": 2},
        )
        assert response.status_code == 200, response.text
        questions = response.json()["questions"]
        assert preview.json()["question"] == questions[0]
        for question in questions:
            assert question["target"] == "h2_statement_set"
            assert question["stimulus"]["kind"] == "diagram"
            components = question["stimulus"]["components"]
            assert (components["0"], components["4"]) in {
                ("resistor", "switch_open"),
                ("thermistor", "battery"),
            }
            assert len(set(question["choices"])) == 4
            assert question["answer"] == question["choices"][question["correct_index"]]
