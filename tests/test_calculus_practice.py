import pytest

from edcraft_backend.services.calculus_practice import generate, propose

RATIO = "calculus-factorial-ratio"
COMPARISON = "calculus-power-comparison"
ASCENT = "calculus-steepest-ascent"
CRITICAL = "calculus-quartic-critical-points"

RATIO_LECTURE = (
    "The Ratio Test. If 0 ≤ L < 1, then the series is absolutely convergent. "
    "If L > 1, then the series is divergent."
)
COMPARISON_LECTURE = (
    "The Limit Comparison Test for positive terms. If the limit is a positive constant, "
    "both series converge or both diverge."
)
ASCENT_LECTURE = (
    "The maximum directional derivative is attained in the direction of the gradient vector. "
    "Use a unit vector."
)
CRITICAL_LECTURE = (
    "Second Derivative Test. A critical point can be a local maximum, local minimum "
    "or saddle point of a function of two variables."
)

RATIO_EXAM = """Question 2. [10 marks]
For each of the following series, determine whether it converges or diverges. Justify
your answers.
     ∞
    X   n!(2n)!
(a)             ,
    n=1
         (3n)!"""

COMPARISON_EXAM = """2. (10 marks)
                                 ∞            1
                                 X      n2
   (a) Determine if the series            3
                                                   is convergent or divergent. Justify
                                 n=1 5n
                                          2   −1
       your answer."""

ASCENT_EXAM = """                                         √
4. (10 marks) Let f (x, y) = ln(x2 y) − x y.
   (a) Find ∇f (x, y).
   (b) Find the unit vector v such that the directional derivative Dv f (2, 14 ) at
       the point (2, 14 ) attains the maximum value."""

CRITICAL_EXAM = """5. (10 marks) Determine the critical points of the function
                         f (x, y) = 2x4 + y 2 − x2 − 2y
  and classify each one as a local maximum point, a local minimum point or
  a saddle point for f ."""


@pytest.mark.parametrize(
    ("family", "lecture", "exam", "topic"),
    [
        (RATIO, RATIO_LECTURE, RATIO_EXAM, "Ratio test for a factorial series"),
        (COMPARISON, COMPARISON_LECTURE, COMPARISON_EXAM, "Limit comparison with a p-series"),
        (ASCENT, ASCENT_LECTURE, ASCENT_EXAM, "Steepest ascent"),
        (CRITICAL, CRITICAL_LECTURE, CRITICAL_EXAM, "Classify critical points using the Hessian"),
    ],
)
def test_generates_distinct_seeded_items_from_supported_material(
    family: str, lecture: str, exam: str, topic: str
) -> None:
    assert propose(lecture, exam, topic=topic)["family"] == family
    first = generate(family, 42, lecture, exam, topic=topic)
    assert first == generate(family, 42, lecture, exam, topic=topic)
    assert first != generate(family, 43, lecture, exam, topic=topic)
    assert len(first) == 5
    assert len({question["stem"] for question in first}) == 5
    for question in first:
        assert len(set(question["options"])) == 4
        assert 0 <= question["correct_index"] < 4
        assert question["explanation"]


def test_topic_can_disambiguate_multiple_supported_passages() -> None:
    lecture = RATIO_LECTURE + "\n" + COMPARISON_LECTURE
    assert propose(lecture)["family"] is None
    assert propose(RATIO_LECTURE)["family"] == RATIO
    assert propose(lecture, topic="factorial series")["family"] == RATIO
    assert len(generate(RATIO, 42, lecture, topic="factorial series")) == 5
    assert propose(lecture, RATIO_EXAM, topic="limit comparison")["family"] is None
    assert propose(lecture, RATIO_EXAM, topic="Bernoulli equation")["family"] is None


def test_rejects_unsupported_exam_even_with_a_supported_lecture() -> None:
    lecture = RATIO_LECTURE
    for exam in (
        "Find y(2) by solving the Bernoulli equation y' + y = x*y^3.",
        "Show this ratio test has a limit of 1.",
        RATIO_EXAM.replace("(3n)!", "(4n)!"),
        RATIO_EXAM + "\n(b) Evaluate a different series.",
    ):
        assert propose(lecture, exam)["family"] is None
        with pytest.raises(ValueError, match="exam"):
            generate(RATIO, 1, lecture, exam)


def test_rejects_a_heading_as_lecture_evidence_or_a_changed_task() -> None:
    assert (
        propose("Chapter 6.7 The Ratio Test and Root Test ..... 86", RATIO_EXAM)["family"] is None
    )
    assert propose(ASCENT_LECTURE, RATIO_EXAM)["family"] is None
    with pytest.raises(ValueError, match="confirmed task"):
        generate(COMPARISON, 1, RATIO_LECTURE, RATIO_EXAM)


@pytest.mark.parametrize(
    ("family", "lecture", "exam", "near_miss"),
    [
        (
            RATIO,
            RATIO_LECTURE,
            "Q7(b) Use the ratio test to determine convergence of the series "
            "sum_(n=1)^infinity 5^n n!(4n)!/(5n)!. Justify your answer.",
            "Q7(b) Use the ratio test to determine convergence of the series "
            "sum_(n=1)^infinity 5^n n!(4n)!/(6n)!. Justify your answer.",
        ),
        (
            COMPARISON,
            COMPARISON_LECTURE,
            "(d) Decide whether the series sum_(n=1)^∞ n^(2/3)/(4n^(5/3)-3) converges or diverges.",
            "(d) Decide whether the series sum_(n=1)^∞ n^(2/3)/(4n^(5/3)-4) converges or diverges.",
        ),
        (
            ASCENT,
            ASCENT_LECTURE,
            "Question 8. Let f(x,y) = ln(x^2*y) - 3x sqrt(y). "
            "At (4, 1/9), find the unit vector in the direction of steepest ascent.",
            "Question 8. Let f(x,y) = ln(x^2*y) - 3x sqrt(y). "
            "At (4, 1/9), find the maximum rate of change.",
        ),
        (
            CRITICAL,
            CRITICAL_LECTURE,
            "Q13(a) Let f(x,y) = 3x^4 - 5x^2 + 4y^2 - 8y. "
            "Locate and classify all stationary points of f.",
            "Q13(a) Let f(x,y) = 3x^4 + 5x^2 + 4y^2 - 8y. "
            "Locate and classify all stationary points of f.",
        ),
    ],
)
def test_recognizes_equivalent_typed_exam_forms_with_varied_parameters(
    family: str, lecture: str, exam: str, near_miss: str
) -> None:
    assert propose(lecture, exam)["family"] == family
    assert propose(lecture, near_miss)["family"] is None
    assert propose(lecture, exam + "\n(c) Find the value at the point.")["family"] is None
