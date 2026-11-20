"""Biology deterministic authoring HTTP regression tests."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
from edcraft_validator.domains.biology import author_offer, h2_cases

from edcraft_backend.services.ai_domains import ENABLED_DOMAINS_ENV
from tests.biology_authoring_helpers import run_scripted_biology_pipeline
from tests.integration.test_ai_authoring_api import approve, complete_job, env_for, review


@pytest.mark.asyncio
async def test_real_scripted_biology_pipeline_round_trips_through_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,biology")
    async with env_for(tmp_path, run_scripted_biology_pipeline) as env:
        capabilities = (await env.client.get("/ai-authoring/capabilities")).json()
        assert "biology" in capabilities["enabled_domains"]
        job_id = await complete_job(
            env,
            domain="biology",
            prompt="Expected heterozygous offspring among purple offspring of a Pp cross",
            difficulty="hard",
        )
        envelope = await review(env, job_id)
        assert envelope["request"]["domain"] == "biology"
        assert envelope["template"]["case_previews"] is None
        candidate = envelope["template"]["candidate"]
        assert candidate["item"]["schema_version"] == "bio-mono/0.1"
        assert [recipe["rule"] for recipe in candidate["item"]["distractors"]] == [
            "unconditioned",
            "complement",
            "reverse_condition",
        ]
        limits = envelope["validation_limits"]
        assert limits["validator_version"] == "mcp-biology-template-v1"
        assert limits["whole_domain"] is True
        assert limits["declared_domain_size"] == 4
        checks = {check["tool"]: check for check in envelope["checks"]}
        for tool in (
            "biology_verify_cross",
            "biology_validate_options",
            "biology_verify_text_fidelity",
        ):
            assert checks[tool]["status"] == "passed"
            assert checks[tool]["advisory"] is False

        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200
        preview = await env.client.post(f"/ai-authoring/jobs/{job_id}/preview", json={"seed": 3})
        assert preview.status_code == 200, preview.text
        question = preview.json()["question"]
        assert question["choices"][question["correct_index"]] == question["answer"]
        expected = {"30": "20", "60": "40", "90": "60", "120": "80"}
        assert question["answer"] == expected[question["parameters"]["n"]]
        assert question["punnett"] == [["PP", "Pp"], ["Pp", "pp"]]
        batch = await env.client.post(
            f"/ai-authoring/jobs/{job_id}/generate",
            json={"artifact_sha256": digest, "seed": 3, "quantity": 2},
        )
        assert batch.status_code == 200, batch.text
        assert batch.json()["domain"] == "biology"
        assert [item["seed"] for item in batch.json()["questions"]] == [3, 4]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers",
    [
        {"dihybrid.self": "9/16", "dihybrid.test": "1/4"},
        {"dihybrid.self": "9/16"},
        {"skill.visual.punnett": "S"},
    ],
)
async def test_h2_catalogue_biology_approval_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: dict[str, str]
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,biology")
    run_pipeline = partial(
        run_scripted_biology_pipeline,
        item={
            "schema_version": "bio-h2/1",
            "cases": list(answers),
            "distractors": [0, 1, 2, 2, 3, 4],
        },
    )
    async with env_for(tmp_path, run_pipeline) as env:
        job_id = await complete_job(
            env,
            domain="biology",
            prompt=f"H2 Biology practice for {', '.join(answers)}",
            difficulty="medium",
        )
        envelope = await review(env, job_id)
        digest = envelope["artifact_sha256"]
        item = envelope["template"]["candidate"]["item"]
        case_previews = envelope["template"]["case_previews"]
        assert len(case_previews) == len(answers)
        assert {row["case_id"]: row["answer"] for row in case_previews} == answers
        if len(answers) > 1:
            assert case_previews[0]["outcome"] == "2(x)"
            assert "RrTt × rrtt" in case_previews[1]["question"]
        assert item["distractors"] == [1, 2, 3]
        assert len(item["catalogue_sha256"]) == 64
        assert envelope["validation_limits"]["declared_domain_size"] == len(answers)
        assert (await approve(env, job_id, "0" * 64)).status_code == 409
        assert (await approve(env, job_id, digest)).status_code == 200

        payload = {"artifact_sha256": digest, "seed": 3, "quantity": 5}
        url = f"/ai-authoring/jobs/{job_id}/generate"
        response = await env.client.post(url, json=payload)
        assert response.status_code == 200, response.text
        batch = response.json()["questions"]
        assert [question["seed"] for question in batch] == [3, 4, 5, 6, 7]
        for question in batch:
            assert question["answer"] == answers[question["case_id"]]
            assert question["answer"] == question["choices"][question["correct_index"]]
            assert len(set(question["choices"])) == 4
            review_case = next(
                row for row in case_previews if row["case_id"] == question["case_id"]
            )
            assert review_case["question"] == question["question"]
            assert set(review_case["choices"]) == set(question["choices"])
            assert review_case["superseded_by"] is None
            if question["case_id"] == "skill.visual.punnett":
                assert question["figure"]["data_url"].startswith("data:image/png;base64,")
                assert "Table headings RT" in question["figure"]["alt"]
                assert question["figure"]["credit"] == "Original EdCraft classroom schematic"
            else:
                assert "figure" not in question
        replay = await env.client.post(url, json=payload)
        assert replay.json()["questions"] == batch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("old_id", "new_id"),
    [
        ("skill.cell.osmosis", "skill.cell.osmosis_gradient"),
        ("skill.practical.axes", "skill.practical.axes_convention"),
        ("skill.energy.chemiosmosis_compare", "skill.energy.chemiosmosis_compartment_contrasts"),
        (
            "skill.energy.chemiosmosis_compartments",
            "skill.energy.chemiosmosis_compartment_contrasts",
        ),
        ("skill.energy.chemiosmosis_acidity", "skill.energy.chemiosmosis_acidity_contrasts"),
    ],
)
async def test_saved_biology_approval_flags_the_superseded_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_id: str, new_id: str
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,biology")
    earlier_catalogue = h2_cases.h2_catalogue(include_archived=True)
    pipeline = partial(
        run_scripted_biology_pipeline,
        item={
            "schema_version": "bio-h2/1",
            "cases": [old_id],
            "distractors": [1, 2, 3, 4, 5, 6],
        },
    )
    async with env_for(tmp_path, pipeline) as env:
        with monkeypatch.context() as earlier:
            earlier.setattr(author_offer, "h2_catalogue", lambda: earlier_catalogue)
            earlier.setattr(h2_cases, "h2_catalogue", lambda **_: earlier_catalogue)
            job_id = await complete_job(
                env, domain="biology", prompt=f"Make an H2 item for {old_id}."
            )
            checked = await review(env, job_id)
            digest = checked["artifact_sha256"]
            assert (await approve(env, job_id, digest)).status_code == 200
            url = f"/ai-authoring/jobs/{job_id}/generate"
            payload = {"artifact_sha256": digest, "seed": 31, "quantity": 1}
            baseline = await env.client.post(url, json=payload)
            assert baseline.status_code == 200, baseline.text
        approved = await review(env, job_id)
        assert approved["artifact_sha256"] == digest
        assert approved["template"]["candidate"] == checked["template"]["candidate"]
        assert approved["approval"]["decision"] == "approved"
        [case] = approved["template"]["case_previews"]
        assert case["case_id"] == old_id
        assert case["superseded_by"] == new_id
        generated = await env.client.post(url, json=payload)
        assert generated.status_code == 200, generated.text
        assert generated.json()["questions"] == baseline.json()["questions"]
        [item] = generated.json()["questions"]
        assert item["case_id"] == old_id
        assert item["question"] == case["question"]
        assert set(item["choices"]) == set(case["choices"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "reason"),
    [
        (
            "Write an essay assessment on protein structure for H2 Biology.",
            "The request requires an original written response.",
        ),
        (
            "Give observed counts of 24 and 16 and expected counts of 20 and 20. "
            "Ask for the chi-square statistic and the test decision.",
            "The verified chi-square cases cannot use those supplied data. "
            "They use prepared counts and critical values.",
        ),
        (
            "Give observed and expected counts. Ask students to calculate only "
            "the numeric chi-square statistic.",
            "The verified cases that ask for a statistic also ask for a test "
            "decision. A question asking for a statistic alone needs a different case.",
        ),
    ],
)
async def test_biology_unsupported_request_stops_for_review_without_an_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str, reason: str
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "code,math,biology")
    run_pipeline = partial(
        run_scripted_biology_pipeline,
        item={
            "schema_version": "bio-unsupported/1",
            "reason": "The catalogue does not cover the request.",
        },
    )
    async with env_for(tmp_path, run_pipeline) as env:
        job_id = await complete_job(env, domain="biology", prompt=prompt)
        status = (await env.client.get(f"/ai-authoring/jobs/{job_id}")).json()
        assert status["result"]["status"] == "needs_review"
        envelope = await review(env, job_id)
        assert envelope["error"] == {
            "code": "UNSUPPORTED_REQUEST",
            "message": reason,
        }
        assert envelope["artifact_id"] is None
        assert envelope["template"] is None
        assert envelope["checks"] == []
        assert (await approve(env, job_id, "0" * 64)).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "difficulty", "answers", "routine"),
    [
        (
            "What lets mitosis support growth, tissue repair and asexual reproduction "
            "while preserving chromosomes in daughter cells?",
            "easy",
            {
                "skill.genetic.mitosis_roles": "It can increase cell number while passing "
                "a copy of each chromosome to both daughter cells."
            },
            False,
        ),
        (
            "Warming advances the insect peak, but birds arrive as usual. "
            "Ask which change is likely in insect availability for breeding birds.",
            "medium",
            {
                "skill.climate.food_chains": "The birds may miss the insect peak, "
                "so less food is available for breeding."
            },
            False,
        ),
        (
            "Give observed and expected offspring counts and a 5% critical value. "
            "Ask students to report both the chi-square statistic and the test decision.",
            "hard",
            {
                "genetics.chi_statistic_below": "χ² = 5.90, so fail to reject the model.",
                "genetics.chi_statistic_above": "χ² = 6.08, so reject the model.",
            },
            True,
        ),
        (
            "Use chi-square to test whether offspring fit a two-locus inheritance model. "
            "Ask students to derive expected counts, determine degrees of freedom and conclude.",
            "hard",
            {
                "genetics.chi_model_petal": "Expected counts 90, 30, 40. df = 2. Reject the model.",
                "genetics.chi_model_leaf": "Expected counts 90, 60, 10. df = 2. "
                "Fail to reject the model.",
            },
            False,
        ),
    ],
)
async def test_biology_task_routing_preserves_requested_operation_through_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    difficulty: str,
    answers: dict[str, str],
    routine: bool,
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "biology")
    run_pipeline = partial(
        run_scripted_biology_pipeline,
        item={
            "schema_version": "bio-h2/1",
            "cases": list(answers),
            "distractors": [1, 2, 3, 4, 5, 6],
        },
    )
    async with env_for(tmp_path, run_pipeline) as env:
        job_id = await complete_job(env, domain="biology", prompt=prompt, difficulty=difficulty)
        envelope = await review(env, job_id)
        assert envelope["request"]["difficulty"] == difficulty
        assert envelope["request"]["prompt"] == prompt
        assert (
            any("available as a routine question" in warning for warning in envelope["warnings"])
            is routine
        )
        assert {
            row["case_id"]: row["answer"] for row in envelope["template"]["case_previews"]
        } == answers
        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200
        url = f"/ai-authoring/jobs/{job_id}/generate"
        args = {"artifact_sha256": digest, "seed": 3, "quantity": 5}
        first = await env.client.post(url, json=args)
        assert first.status_code == 200, first.text
        questions = first.json()["questions"]
        assert all(
            question["choices"][question["correct_index"]] == answers[question["case_id"]]
            for question in questions
        )
        assert (await env.client.post(url, json=args)).json()["questions"] == questions


@pytest.mark.asyncio
@pytest.mark.parametrize("num_distractors", [3, 4])
async def test_chemiosmosis_compartment_contrasts_replay_through_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_distractors: int
) -> None:
    monkeypatch.setenv(ENABLED_DOMAINS_ENV, "biology")
    cases = [
        "skill.energy.chemiosmosis_compartment_contrasts",
        "skill.energy.chemiosmosis_acidity_contrasts",
    ]
    pipeline = partial(
        run_scripted_biology_pipeline,
        item={
            "schema_version": "bio-h2/1",
            "cases": cases,
            "distractors": [6] * (num_distractors + 3),
        },
    )
    async with env_for(tmp_path, pipeline) as env:
        job_id = await complete_job(
            env,
            domain="biology",
            prompt="Compare chemiosmosis at the thylakoid and inner mitochondrial membranes. "
            "Which compartments accumulate protons and receive newly synthesised ATP?",
            difficulty="hard",
            num_distractors=num_distractors,
        )
        envelope = await review(env, job_id)
        assert any("available as a routine question" in text for text in envelope["warnings"])
        has_coverage_note = any("isolate 3 of four" in text for text in envelope["warnings"])
        assert has_coverage_note is (num_distractors == 3)
        assert envelope["template"]["candidate"]["item"]["distractors"] == list(
            range(1, num_distractors + 1)
        )
        assert [row["case_id"] for row in envelope["template"]["case_previews"]] == cases
        digest = envelope["artifact_sha256"]
        assert (await approve(env, job_id, digest)).status_code == 200
        url = f"/ai-authoring/jobs/{job_id}/generate"
        args = {"artifact_sha256": digest, "seed": 3, "quantity": 5}
        response = await env.client.post(url, json=args)
        assert response.status_code == 200, response.text
        questions = response.json()["questions"]
        assert (await env.client.post(url, json=args)).json()["questions"] == questions
        for question in questions:
            key = question["choices"][question["correct_index"]]
            expected = key.removesuffix(".").split(". ")
            assert len(expected) == 4
            assert "lumen" in expected[0] and "stroma" in expected[1]
            assert "intermembrane space" in expected[2] and "matrix" in expected[3]
            changed = set()
            has_different_sides = False
            for choice in question["choices"]:
                if choice == key:
                    continue
                cells = choice.removesuffix(".").split(". ")
                [contrast] = [
                    index
                    for index, (src, dst) in enumerate(zip(expected, cells, strict=True))
                    if src != dst
                ]
                changed.add(contrast)
                thylakoid_h, thylakoid_atp, mitochondrial_h, mitochondrial_atp = (
                    cell.split(" = ")[1] for cell in cells
                )
                has_different_sides |= (
                    thylakoid_h != thylakoid_atp and mitochondrial_h != mitochondrial_atp
                )
            assert len(changed) == num_distractors
            assert has_different_sides
