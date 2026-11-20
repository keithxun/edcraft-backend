from fractions import Fraction

import pytest

from edcraft_backend.services import material_practice

PHYSICS = (
    "physics",
    "physics-resistor-power",
    "For a resistor, electrical power dissipated is P = I²R.",
    "A resistor of resistance 10 Ω carries a steady current of 0.4 A. "
    "What power does it dissipate?",
    "Resistor power",
)
CHEMISTRY = (
    "chemistry",
    "chemistry-neutralisation-balance",
    "Acid-base neutralisation: an acid reacts with a base to form salt and water. "
    "Balance the equation by changing coefficients to conserve atoms.",
    "Balance the equation H2SO4 + NaOH → Na2SO4 + H2O.",
    "Acid-base neutralisation",
)
BIOLOGY = (
    "biology",
    "biology-monohybrid-probability",
    "For a monohybrid cross with complete dominance, use a Punnett square "
    "to predict the probability of offspring phenotypes.",
    "In pea plants, P is completely dominant over p. Crossing Pp × Pp, "
    "what is the probability of an offspring showing the recessive phenotype?",
    "Mendelian inheritance",
)
SPRING = (
    "physics",
    "physics-hooke-spring-force",
    "Hooke's law. Within the proportional limit, the magnitude of the spring force is F = kx.",
    "A spring of force constant 40 N/m is extended by 0.15 m within its proportional limit. "
    "What is the magnitude of the force holding it at that extension?",
    "Hooke's law",
)


@pytest.mark.parametrize(
    ("domain", "family", "lecture", "exam", "topic"),
    [
        PHYSICS,
        SPRING,
        CHEMISTRY,
        BIOLOGY,
    ],
)
def test_science_practice_is_checked_reproducible_and_scoped(
    domain: str, family: str, lecture: str, exam: str, topic: str
) -> None:
    assert material_practice.propose(lecture, exam, topic, domain)["family"] == family
    assert material_practice.propose(lecture, exam, topic)["family"] == family
    assert material_practice.propose(lecture, exam, topic, "code")["family"] is None
    assert material_practice.propose(lecture, exam, topic, enabled=("math",))["family"] is None
    with pytest.raises(ValueError, match="domain"):
        material_practice.generate(family, 42, lecture, exam, topic, "code")
    with pytest.raises(ValueError, match="domain"):
        material_practice.generate(family, 42, lecture, exam, topic, enabled=("math",))

    batch, version = material_practice.generate(family, 42, lecture, exam, topic, domain)
    assert (batch, version) == material_practice.generate(family, 42, lecture, exam, topic)
    assert (batch, version) != material_practice.generate(family, 43, lecture, exam, topic)
    assert len({question["stem"] for question in batch}) == 5
    for question in batch:
        assert len(question["options"]) == len(set(question["options"])) == 4
        assert 0 <= question["correct_index"] < 4
        assert question["explanation"]


@pytest.mark.parametrize(
    ("fixture", "unsafe_match"),
    [
        (PHYSICS, PHYSICS[3].replace("What power does it dissipate?", "Calculate the current.")),
        (PHYSICS, PHYSICS[3].replace("10 Ω", "0 Ω")),
        (PHYSICS, PHYSICS[3].replace("0.4 A", "0.4 V")),
        (PHYSICS, PHYSICS[3] + " Explain how the answer changes for two resistors."),
        (SPRING, SPRING[3].replace("within", "beyond")),
        (SPRING, SPRING[3].replace("0.15 m", "0.15 kg")),
        (SPRING, SPRING[3].split(". What")[0] + ". Calculate the elastic potential energy."),
        (CHEMISTRY, "Predict the products of H2SO4 + NaOH."),
        (CHEMISTRY, "Balance the equation H2SO4 + NaOH → K2SO4 + H2O."),
        (CHEMISTRY, "Balance the equation C2H6 + O2 → CO2 + H2O."),
        (BIOLOGY, BIOLOGY[3].replace("Pp × Pp", "Pp × Qq")),
        (BIOLOGY, BIOLOGY[3].replace("Pp × Pp", "pP × Pp")),
        (BIOLOGY, BIOLOGY[3].replace("completely dominant over", "codominant with")),
        (BIOLOGY, BIOLOGY[3].replace("offspring showing", "offspring with genotype PP showing")),
        (BIOLOGY, BIOLOGY[3] + " What is the expected count among 100 offspring?"),
    ],
)
def test_science_near_misses_do_not_offer_a_checked_operation(
    fixture: tuple[str, str, str, str, str], unsafe_match: str
) -> None:
    domain, family, lecture, _, topic = fixture
    assert material_practice.propose(lecture, unsafe_match, topic, domain)["family"] is None
    with pytest.raises(ValueError, match="excerpts"):
        material_practice.generate(family, 42, lecture, unsafe_match, topic, domain)


@pytest.mark.parametrize(
    ("domain", "family", "lecture", "exam", "topic"),
    [
        PHYSICS,
        SPRING,
        CHEMISTRY,
        BIOLOGY,
    ],
)
def test_science_requires_teaching_evidence_and_a_single_selected_exam_task(
    domain: str, family: str, lecture: str, exam: str, topic: str
) -> None:
    for passed in [(topic, exam, topic), (lecture, "", topic), (lecture, exam, "Matrices")]:
        assert material_practice.propose(*passed, domain)["family"] is None
        with pytest.raises(ValueError, match="excerpts"):
            material_practice.generate(family, 42, *passed, domain)


@pytest.mark.parametrize(
    ("fixture", "lecture"),
    [
        (PHYSICS, "Resistors. A common error is to write P = I²R."),
        (PHYSICS, PHYSICS[2] + " Do not use P = I²R."),
        (SPRING, "Hooke's law. Proportional limit. Do not use F = kx."),
        (CHEMISTRY, CHEMISTRY[2].replace("conserve atoms", "destroy atoms")),
        (CHEMISTRY, CHEMISTRY[2].replace("Balance the equation", "Do not balance the equation")),
        (
            BIOLOGY,
            "Unlike complete dominance, codominance is shown in this monohybrid cross. "
            "Use a Punnett square to predict the probability of offspring phenotypes.",
        ),
        (BIOLOGY, BIOLOGY[2].replace("with complete dominance", "without complete dominance")),
    ],
)
def test_science_rejects_examples_of_errors_and_contradictory_rules(
    fixture: tuple[str, str, str, str, str], lecture: str
) -> None:
    domain, family, _, exam, topic = fixture
    assert material_practice.propose(lecture, exam, topic, domain)["family"] is None
    with pytest.raises(ValueError, match="excerpts"):
        material_practice.generate(family, 42, lecture, exam, topic, domain)


@pytest.mark.parametrize(
    ("domain", "family", "lecture", "exam", "topic"),
    [
        PHYSICS,
        SPRING,
        CHEMISTRY,
        BIOLOGY,
    ],
)
def test_science_lecture_only_requires_one_labelled_example_with_a_complete_task(
    domain: str, family: str, lecture: str, exam: str, topic: str
) -> None:
    worked_example = f"{lecture}\nWorked example:\n{exam}\nSolution: Apply the rule above."
    proposal = material_practice.propose(worked_example, "", topic, domain)
    assert proposal["family"] == family
    assert "lecture" in proposal["reason"]
    assert material_practice.generate(family, 42, worked_example, "", topic, domain) == (
        material_practice.generate(family, 42, lecture, exam, topic, domain)
    )
    assert material_practice.propose(lecture + f"\n{exam}", "", topic, domain)["family"] is None
    duplicate = worked_example + f"\nWorked example 2: {exam}"
    assert material_practice.propose(duplicate, "", topic, domain)["family"] is None
    assert (
        material_practice.propose(worked_example, "Explain the rule.", topic, domain)["family"]
        is None
    )


def test_physics_rederives_wattage_for_each_new_resistor() -> None:
    _, family, lecture, exam, topic = PHYSICS
    batch, _ = material_practice.generate(family, 20, lecture, exam, topic)
    for question in batch:
        tokens = question["stem"].split()
        resistance = Fraction(tokens[tokens.index("Ω") - 1])
        current = Fraction(tokens[tokens.index("A.") - 1])
        assert (resistance, current) != (10, Fraction("0.4"))
        expected = current**2 * resistance
        values = [Fraction(option.removesuffix(" W")) for option in question["options"]]
        assert values.count(expected) == 1
        assert values[question["correct_index"]] == expected


def test_physics_accepts_a_labelled_question_given_current_before_resistance() -> None:
    domain, family, lecture, _, _ = PHYSICS
    exam = (
        "Q2(b) A steady current of 0.4 A flows through a resistor of resistance 10 Ω. "
        "Calculate the power dissipated in the resistor. [2]"
    )
    assert (
        material_practice.propose(lecture, exam, "Current electricity", domain)["family"] == family
    )
    assert material_practice.generate(family, 42, lecture, exam, "Current electricity", domain) == (
        material_practice.generate(family, 42, *PHYSICS[2:], domain)
    )


def test_physics_rederives_force_for_each_new_spring() -> None:
    _, family, lecture, exam, topic = SPRING
    batch, _ = material_practice.generate(family, 20, lecture, exam, topic)
    for question in batch:
        tokens = question["stem"].split()
        stiffness = Fraction(tokens[tokens.index("N/m") - 1])
        extension = Fraction(tokens[tokens.index("m") - 1])
        assert (stiffness, extension) != (40, Fraction("0.15"))
        expected = stiffness * extension
        values = [Fraction(option.removesuffix(" N")) for option in question["options"]]
        assert values.count(expected) == 1
        assert values[question["correct_index"]] == expected


@pytest.mark.parametrize(("fixture", "unit"), [(PHYSICS, " W"), (SPRING, " N")])
def test_physics_options_vary_without_a_half_and_double_shortcut(
    fixture: tuple[str, str, str, str, str], unit: str
) -> None:
    domain, family, lecture, exam, topic = fixture
    seen_stems: set[str] = set()
    seen_ranks: set[int] = set()
    for seed in range(24):
        batch, _ = material_practice.generate(family, seed, lecture, exam, topic, domain)
        for question in batch:
            values = [Fraction(option.removesuffix(unit)) for option in question["options"]]
            key = values[question["correct_index"]]
            assert len(set(values)) == 4
            assert not {key / 2, key * 2}.issubset(values)
            seen_stems.add(question["stem"])
            seen_ranks.add(sorted(values).index(key))
    assert len(seen_stems) == 8  # Every row of the 3 × 3 grid besides the selected example.
    assert seen_ranks == {0, 1, 2, 3}


def test_chemistry_balances_each_new_reaction_and_proves_wrong_options_unbalanced() -> None:
    from edcraft_validator.domains.chemistry.balance import first_unbalanced_element

    _, family, lecture, exam, topic = CHEMISTRY
    batch, _ = material_practice.generate(family, 20, lecture, exam, topic)
    for question in batch:
        skeleton = question["stem"].split("\n")[-1]
        assert skeleton != "H2SO4 + NaOH → Na2SO4 + H2O"
        left, right = (side.split(" + ") for side in skeleton.split(" → "))
        for index, option in enumerate(question["options"]):
            coefficients = tuple(int(value) for value in option.split(", "))
            imbalance = first_unbalanced_element(left, right, coefficients)
            assert (imbalance is None) == (index == question["correct_index"])


def test_chemistry_foils_keep_the_key_from_being_the_unique_smallest_total() -> None:
    domain, family, lecture, exam, topic = CHEMISTRY
    saw_lower_sum = saw_higher_sum = False
    seen_stems: set[str] = set()
    for seed in range(16):
        batch, _ = material_practice.generate(family, seed, lecture, exam, topic, domain)
        for question in batch:
            seen_stems.add(question["stem"])
            totals = [sum(map(int, option.split(", "))) for option in question["options"]]
            key_total = totals[question["correct_index"]]
            assert totals.count(key_total) >= 2
            saw_lower_sum |= min(totals) < key_total
            saw_higher_sum |= max(totals) > key_total
    assert len(seen_stems) == 8  # Ten catalogue reactions minus the exam and all-unit equation.
    assert saw_lower_sum and saw_higher_sum


def test_biology_keeps_the_exam_allele_and_requested_phenotype() -> None:
    _, family, lecture, exam, topic = BIOLOGY
    exam = exam.replace("P", "T").replace(" p.", " t.").replace("Tp", "Tt")
    exam = exam.replace("recessive phenotype", "dominant phenotype")
    batch, _ = material_practice.generate(family, 20, lecture, exam, topic)
    for question in batch:
        assert "dominant phenotype" in question["stem"]
        cross = question["stem"].split("Crossing ")[1].split(",")[0]
        parent1, parent2 = cross.split(" × ")
        assert set(parent1 + parent2) <= {"T", "t"}
        assert (parent1, parent2) != ("Tt", "Tt")
        expected = Fraction(sum("T" in a + b for a in parent1 for b in parent2), 4)
        values = [Fraction(value) for value in question["options"]]
        assert values.count(expected) == 1
        assert values[question["correct_index"]] == expected


def test_chemistry_choices_hide_the_key_from_a_position_majority() -> None:
    domain, family, lecture, exam, topic = CHEMISTRY
    for seed in range(16):
        batch, _ = material_practice.generate(family, seed, lecture, exam, topic, domain)
        for question in batch:
            options = [option.split(", ") for option in question["options"]]
            changing_columns = [
                column for column in zip(*options, strict=True) if len(set(column)) > 1
            ]
            assert len(changing_columns) >= 2
            assert all(
                max(column.count(value) for value in column) == 2 for column in changing_columns
            )


@pytest.mark.parametrize("caution", ["Don't use", "Don’t use", "Avoid using", "A misconception is"])
def test_science_declines_a_cautioned_formula(caution: str) -> None:
    domain, _, _, exam, topic = SPRING
    lecture = f"Hooke's law. Within the proportional limit. {caution} F = kx."
    assert material_practice.propose(lecture, exam, topic, domain)["family"] is None
