"""Match calculus material to bounded problem forms with exact answer keys."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from random import Random
from typing import TypedDict
from unicodedata import normalize

from edcraft_backend.services import calculus_exam

VERSION = 1

RATIO = "calculus-factorial-ratio"
COMPARISON = "calculus-power-comparison"
ASCENT = "calculus-steepest-ascent"
CRITICAL = "calculus-quartic-critical-points"


class Proposal(TypedDict):
    family: str | None
    task: str
    reason: str


class Question(TypedDict):
    stem: str
    options: list[str]
    correct_index: int
    explanation: str


@dataclass(frozen=True)
class Task:
    description: str
    topic: str
    lecture_checks: tuple[str, ...]
    exam_matches: Callable[[str], bool]


TASKS = {
    RATIO: Task(
        "Apply the ratio test to a factorial series. Choose the limiting ratio and "
        "the convergence conclusion. The worked solution shows the justification.",
        r"\b(?:ratio test|factorial(?: series| ratio|s))\b",
        (
            r"ratio test",
            r"l\s*<\s*1.{0,250}convergent",
            r"l\s*>\s*1.{0,150}divergent",
        ),
        calculus_exam.factorial,
    ),
    COMPARISON: Task(
        "Compare a positive series with a p-series. Choose the comparison exponent "
        "and the convergence conclusion.",
        r"\b(?:limit comparison|comparison test|p[ -]series)\b",
        (
            r"comparison test",
            r"(?:positive|nonnegative|non-negative) terms",
            r"both diverge|p-series.{0,150}(?:convergen|divergen)",
        ),
        calculus_exam.power_comparison,
    ),
    ASCENT: Task(
        "Find the unit direction of greatest increase by computing and normalizing "
        "the gradient of a log and square-root expression.",
        r"\b(?:steepest ascent|directional derivative|gradients?|maximum rate of change)\b",
        (
            r"directional derivative",
            r"unit (?:vector|direction)",
            r"maximum (?:directional derivative|(?:rate|value)).{0,350}"
            r"direction of (?:the gradient|of\b|∇f)",
        ),
        calculus_exam.steepest_ascent,
    ),
    CRITICAL: Task(
        "Find and classify every critical point of a separable two-variable quartic "
        "using second derivatives.",
        r"\b(?:critical (?:points?|values?)|hessian|second derivative test|saddle points?)\b",
        (
            r"second derivative test",
            r"critical point",
            r"local maximum",
            r"local minimum",
            r"saddle point",
        ),
        calculus_exam.critical_points,
    ),
}


def _plain(text: str) -> str:
    return " ".join(normalize("NFKC", text).lower().replace("−", "-").split())


def _match(lecture: str, exam: str, topic: str) -> str:
    lesson = _plain(lecture)
    selected_exam = _plain(exam)
    prompt = _plain(topic)
    candidates = [
        family for family, task in TASKS.items() if not prompt or re.search(task.topic, prompt)
    ]
    if not candidates:
        raise ValueError("This topic has no checked calculus question family.")
    if selected_exam:
        candidates = [family for family in candidates if TASKS[family].exam_matches(selected_exam)]
        if not candidates:
            raise ValueError(
                "The selected exam subquestion does not match a checked calculus task "
                "for this topic. Include its complete statement and shared mathematical setup."
            )
    candidates = [
        family
        for family in candidates
        if all(re.search(pattern, lesson) for pattern in TASKS[family].lecture_checks)
    ]
    if not candidates:
        raise ValueError(
            "Select lecture evidence that shows how to solve the chosen calculus task."
        )
    if len(candidates) > 1:
        raise ValueError("Name one calculus topic or select a complete exam subquestion.")
    return candidates[0]


def propose(lecture: str, exam: str = "", *, topic: str = "") -> Proposal:
    try:
        family = _match(lecture, exam, topic)
    except ValueError as exc:
        return {"family": None, "task": "", "reason": str(exc)}
    reason = (
        "Uses the method in the selected lecture and the problem form in the exam subquestion."
        if exam.strip()
        else "Uses the method demonstrated in the selected lecture passage."
    )
    return {"family": family, "task": TASKS[family].description, "reason": reason}


@dataclass(frozen=True)
class Item:
    stem: str
    key: str
    distractors: tuple[str, ...]
    explanation: str


def _sample_items(draw: Callable[[], Item | None]) -> list[Item]:
    items: dict[str, Item] = {}
    for _ in range(100):
        item = draw()
        if item is not None:
            items.setdefault(item.stem, item)
        if len(items) == 5:
            return list(items.values())
    raise ValueError("The seed produced too few distinct checked variants.")


def _series_result(value: Fraction, converges: bool) -> str:
    conclusion = "converges" if converges else "diverges"
    return f"{value} · Series {conclusion}"


def _ratio_item(k: int, base: int) -> Item:
    unweighted = Fraction(k**k, (k + 1) ** (k + 1))
    limit = base * unweighted
    reciprocal = 1 / limit
    conclusion = "converges because L < 1" if limit < 1 else "diverges because L > 1"
    return Item(
        stem=(
            f"Let a_n = {base}^n n! ({k}n)! / ({k + 1}n)!. "
            "The sum runs from n = 1 to infinity. Find lim (a_(n+1) / a_n) as n tends "
            "to infinity and decide whether the series converges."
        ),
        key=_series_result(limit, limit < 1),
        distractors=(
            _series_result(limit, limit > 1),
            _series_result(reciprocal, reciprocal < 1),
            _series_result(unweighted, True),
        ),
        explanation=(
            f"In a_(n+1) / a_n, n! contributes (n+1), ({k}n)! contributes {k} factors "
            f"with leading term {k}n, and ({k + 1}n)! contributes {k + 1} denominator "
            f"factors with leading term ({k + 1})n. Cancel n^({k + 1}) from their "
            f"leading terms to get L = {base} · {k}^{k} / {k + 1}^{k + 1} = {limit}. "
            f"The positive-term series {conclusion}."
        ),
    )


def _ratio_batch(rng: Random) -> list[Item]:
    def draw() -> Item:
        k = rng.randint(2, 8)
        threshold = (k + 1) ** (k + 1) // k**k
        return _ratio_item(k, rng.randint(threshold - 3, threshold + 4))

    return _sample_items(draw)


def _comparison_item(q: int, p: int, gap: int, a: int, b: int) -> Item:
    numerator = Fraction(p, q)
    denominator = Fraction(p + gap, q)
    exponent = denominator - numerator
    summation = numerator + denominator
    return Item(
        stem=(
            f"Let a_n = n^({numerator}) / ({a}n^({denominator}) - {b}). "
            "The sum runs from n = 1 to infinity. Which p and conclusion are valid if "
            "you compare a_n to 1/n^p with a finite, positive limit for the term ratio?"
        ),
        key=_series_result(exponent, exponent > 1),
        distractors=(
            _series_result(exponent, exponent <= 1),
            _series_result(denominator, denominator > 1),
            _series_result(summation, summation > 1),
        ),
        explanation=(
            f"The denominator is positive for n ≥ 1. Its leading power is "
            f"n^({denominator}). Subtract the numerator's exponent {numerator} to get "
            f"p = {exponent}. The ratio a_n / (1/n^p) tends to 1/{a}, so both series "
            f"have the same convergence behavior. The p-series "
            f"{'converges because p > 1' if exponent > 1 else 'diverges because p ≤ 1'}."
        ),
    )


def _comparison_batch(rng: Random) -> list[Item]:
    def draw() -> Item | None:
        q = rng.randint(2, 6)
        p = rng.randint(1, 8)
        if gcd(p, q) != 1:
            return None
        a = rng.randint(2, 7)
        return _comparison_item(q, p, rng.randint(q - 1, q + 2), a, rng.randrange(1, a))

    return _sample_items(draw)


def _ascent_item(a: int, x: int, q: int) -> Item:
    gx = Fraction(2, x) - Fraction(a, q)
    gy = q**2 - Fraction(a * x * q, 2)
    # Clear denominators using the exact proportionality of the partial derivatives.
    dx, dy = 2, x * q**2
    divisor = gcd(dx, dy)
    sign = 1 if gx > 0 else -1
    dx, dy = sign * dx // divisor, sign * dy // divisor
    norm_squared = dx**2 + dy**2

    def direction(first: int, second: int) -> str:
        return f"({first}, {second}) / sqrt({norm_squared})"

    return Item(
        stem=(
            f"Let f(x, y) = ln(x^2 y) - {a}x sqrt(y). "
            f"Which unit vector maximizes the directional derivative at ({x}, 1/{q**2})?"
        ),
        key=direction(dx, dy),
        distractors=(direction(-dx, -dy), direction(-dy, dx), direction(dy, dx)),
        explanation=(
            f"The partial derivatives are f_x = 2/x - {a} sqrt(y) and "
            f"f_y = 1/y - {a}x/(2 sqrt(y)). At ({x}, 1/{q**2}) the gradient is "
            f"({gx}, {gy}). It points along ({dx}, {dy}). Dividing by sqrt({norm_squared}) "
            "gives the unit vector in the direction of greatest increase."
        ),
    )


def _ascent_batch(rng: Random) -> list[Item]:
    def draw() -> Item | None:
        a, x, q = rng.randint(1, 8), rng.randint(2, 12), rng.randint(2, 10)
        if 2 * q == a * x or (a, x, q) == (1, 2, 2):
            return None
        return _ascent_item(a, x, q)

    return _sample_items(draw)


def _critical_item(s: int, t: int, h: int, k: int, a: int) -> Item:
    u = f"(x {'-' if h > 0 else '+'} {abs(h)})"
    v = f"(y {'-' if k > 0 else '+'} {abs(k)})"
    y_term = f"{'+' if t > 0 else '-'} {abs(t)}{v}^2"
    left, center, right = (h - a, k), (h, k), (h + a, k)
    outer = f"{left}, {right}"
    choices = (
        f"Minima: {outer} | Maxima: none | Saddles: {center}",
        f"Minima: {center} | Maxima: none | Saddles: {outer}",
        f"Minima: none | Maxima: {outer} | Saddles: {center}",
        f"Minima: none | Maxima: {center} | Saddles: {outer}",
    )
    key = (s * t < 0) + 2 * (t < 0)
    return Item(
        stem=(
            f"Let f(x, y) = {s}[{u}^4 - {2 * a**2}{u}^2] {y_term}. "
            "Which option lists and classifies all the critical points?"
        ),
        key=choices[key],
        distractors=tuple(choice for i, choice in enumerate(choices) if i != key),
        explanation=(
            f"Set u = {u}. The first partials are {4 * s}u(u^2 - {a**2}) "
            f"and {2 * t}{v}, yielding the three points {left}, {center}, and {right}. "
            f"The Hessian is diagonal. Its y entry is {2 * t}. Its x entry is "
            f"{-4 * s * a**2} at the center and {8 * s * a**2} at the two outer points. "
            "Matching signs give a local minimum for positive entries or a local maximum "
            "for negative entries. Opposite signs give a saddle point."
        ),
    )


def _critical_batch(rng: Random) -> list[Item]:
    def signed(maximum: int) -> int:
        return rng.choice((-1, 1)) * rng.randint(1, maximum)

    def draw() -> Item:
        return _critical_item(signed(5), signed(5), signed(12), signed(12), rng.randint(1, 8))

    return _sample_items(draw)


_RENDERERS: dict[str, Callable[[Random], list[Item]]] = {
    RATIO: _ratio_batch,
    COMPARISON: _comparison_batch,
    ASCENT: _ascent_batch,
    CRITICAL: _critical_batch,
}


def generate(
    family: str, seed: int, lecture: str, exam: str = "", *, topic: str = ""
) -> list[Question]:
    if _match(lecture, exam, topic) != family:
        raise ValueError("The selected sources no longer support the confirmed task.")
    rng = Random(f"calculus-{VERSION}/{family}/{seed}")  # noqa: S311
    positions = [0, 1, 2, 3, rng.randrange(4)]
    rng.shuffle(positions)
    questions = []
    for item, position in zip(_RENDERERS[family](rng), positions, strict=True):
        options = list(item.distractors)
        rng.shuffle(options)
        options.insert(position, item.key)
        questions.append(
            Question(
                stem=item.stem,
                options=options,
                correct_index=position,
                explanation=item.explanation,
            )
        )
    return questions
