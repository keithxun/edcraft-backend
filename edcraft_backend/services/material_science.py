"""Bind reviewed science excerpts to a small set of deterministic operations."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, product
from random import Random
from typing import TypedDict
from unicodedata import normalize

from edcraft_validator.domains.biology.cross import (
    BiologyCheckError,
    genotype_distribution,
    parse_genotype,
    phenotype_of,
    punnett,
)
from edcraft_validator.domains.chemistry.balance import (
    balance_equation,
    element_totals,
    first_unbalanced_element,
)
from edcraft_validator.domains.chemistry.catalogue import ReactionEntry, load_catalogue
from edcraft_validator.domains.physics.h2 import checked_options, render_h2_stem
from edcraft_validator.domains.physics.h2_catalog import FAMILIES as PHYSICS_FAMILIES
from edcraft_validator.domains.physics.physics_schemas import (
    H2Body,
    H2TemplateSpec,
    PhysicsDistractor,
    RecipeNode,
)

_EXAMPLE_LABEL = re.compile(
    r"^[ \t]*(?:worked[ \t]+)?example(?:[ \t]+\d+)?[ \t]*(?::|$)", re.I | re.M
)
_EXAMPLE_END = re.compile(r"\n[ \t]*(?:\n|(?:answer|solution)[ \t]*[:=])", re.I)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n[ \t]*\n")
_QUALIFIED_RULE = (
    r"\b(?:errors?|incorrect(?:ly)?|false|mistak\w*|wrong(?:ly)?|no|not|never|cannot|"
    r"without|unlike|except|instead|rather than)\b"
)


class Question(TypedDict):
    stem: str
    options: list[str]
    correct_index: int
    explanation: str


@dataclass(frozen=True)
class Operation:
    domain: str
    task: str
    reason: str
    topic: str
    lecture_checks: tuple[str, ...]
    read_exam: Callable[[str], tuple[str, ...] | None]
    questions: Callable[[tuple[str, ...], Random], list[Question]]
    lecture_conflicts: str | None = None
    # Bump when this operation's visible output or its pinned catalogue changes.
    version: int = 1

    def _teaches(self, lecture: str) -> bool:
        sentences = _SENTENCE_BREAK.split(lecture)
        conflict = _QUALIFIED_RULE
        if self.lecture_conflicts:
            conflict += "|" + self.lecture_conflicts
        for rule in self.lecture_checks:
            passages = [
                sentence for sentence in sentences if re.search(rule, sentence, re.I | re.M)
            ]
            if not passages or any(re.search(conflict, passage, re.I) for passage in passages):
                return False
        return True

    def select(self, lecture: str, exam: str, topic: str) -> tuple[str, ...] | None:
        if topic.strip() and not re.search(self.topic, normalize("NFKC", topic), re.I):
            return None
        lesson = normalize("NFKC", lecture.replace("²", "^2"))
        if not self._teaches(lesson):
            return None
        if exam.strip():
            return self.read_exam(_exam_text(exam))
        examples = _EXAMPLE_LABEL.split(lesson)[1:]
        matches = [
            evidence
            for example in examples
            if (evidence := self.read_exam(_exam_text(_EXAMPLE_END.split(example.strip(), 1)[0])))
        ]
        return matches[0] if len(matches) == 1 else None


def _exam_text(exam: str) -> str:
    text = " ".join(normalize("NFKC", exam).split())
    text = re.sub(
        r"^(?:(?:question\s+|q\s*)?\d+(?:[.):]|(?=\s*\([a-z]\))))?\s*(?:\([a-z]\)\s*)?",
        "",
        text,
        flags=re.I,
    )
    return re.sub(r"\s*(?:\[\d{1,2}(?:\s+marks?)?\]|\(\d{1,2}\s+marks?\))$", "", text, flags=re.I)


def _question(stem: str, options: list[str], explanation: str, rng: Random) -> Question:
    key = options[0]
    rng.shuffle(options)
    return Question(
        stem=stem, options=options, correct_index=options.index(key), explanation=explanation
    )


@dataclass(frozen=True)
class PhysicsPractice:
    """Fixed exercises that use the H2 catalogue's numeric and option checks."""

    family_id: str
    exam_axes: tuple[str, str]
    answer_recipe: RecipeNode
    distractors: tuple[tuple[RecipeNode, str], ...]
    explanation: str

    def questions(self, exam: tuple[str, ...], rng: Random) -> list[Question]:
        axes = PHYSICS_FAMILIES[self.family_id].default_axes
        selected = dict(zip(self.exam_axes, map(Fraction, exam), strict=True))
        variants = [
            dict(zip(axes, values, strict=True))
            for values in product(*axes.values())
            if dict(zip(axes, map(Fraction, values), strict=True)) != selected
        ]
        questions = []
        for inputs in rng.sample(variants, 5):
            spec = H2TemplateSpec(
                schema_version="physics-h2/1.0",
                family_id=self.family_id,
                body=H2Body(
                    answer_recipe=self.answer_recipe,
                    distractors=[
                        PhysicsDistractor(recipe=recipe, misconception="other", explanation=message)
                        for recipe, message in rng.sample(self.distractors, 3)
                    ],
                ),
            )
            options = checked_options(spec, inputs)
            explanation = self.explanation.format(**inputs, answer=options[0])
            questions.append(_question(render_h2_stem(spec, inputs), options, explanation, rng))
        return questions


_NUMBER = r"(?:\d{1,5}(?:\.\d{1,4})?|\.\d{1,4})"
_POWER_TARGET = (
    r"(?:what power does it dissipate\?|"
    r"(?:calculate|determine|find) (?:the )?power dissipated (?:by|in) (?:the|this) resistor[.?]?)"
)
_RESISTOR_EXAM = re.compile(
    rf"a resistor (?:of|with) resistance ({_NUMBER})\s*(?:Ω|ohms?) "
    rf"carries (?:a )?(?:(?:steady|direct) )?current of ({_NUMBER})\s*A[.;] {_POWER_TARGET}",
    re.I,
)
_CURRENT_FIRST_EXAM = re.compile(
    rf"a (?:(?:steady|direct) )?current of ({_NUMBER})\s*A flows through a resistor "
    rf"(?:of|with) resistance ({_NUMBER})\s*(?:Ω|ohms?)[.;] {_POWER_TARGET}",
    re.I,
)
_SPRING_EXAM = re.compile(
    rf"a spring (?:of|with) (?:force constant|spring constant) ({_NUMBER})\s*"
    rf"(?:N\s*/\s*m|N\s+m[−-]1) is extended by ({_NUMBER})\s*m "
    r"within (?:its proportional limit|the limit of proportionality)\. "
    r"(?:what is the magnitude of the force holding it at that extension\?|"
    r"(?:calculate|determine|find) (?:the )?(?:magnitude of the )?force required to hold "
    r"(?:it|the spring) at (?:this|that) extension[.?]?)",
    re.I,
)


def _read_positive_values(pattern: re.Pattern[str], exam: str) -> tuple[str, ...] | None:
    match = pattern.fullmatch(exam)
    if match and all(Fraction(value) > 0 for value in match.groups()):
        return match.groups()
    return None


def _read_resistor_exam(exam: str) -> tuple[str, ...] | None:
    values = _read_positive_values(_RESISTOR_EXAM, exam)
    # Return current and resistance in the operation's exam_axes order.
    return values[::-1] if values else _read_positive_values(_CURRENT_FIRST_EXAM, exam)


def _read_spring_exam(exam: str) -> tuple[str, ...] | None:
    return _read_positive_values(_SPRING_EXAM, exam)


_TWO = RecipeNode(const="2")
_CURRENT = RecipeNode(param="I")
_RESISTANCE = RecipeNode(param="R")
_SQUARED_CURRENT = RecipeNode(op="mul", args=[_CURRENT, _CURRENT])
_VOLTAGE = RecipeNode(op="mul", args=[_CURRENT, _RESISTANCE])
_POWER = RecipeNode(op="mul", args=[_SQUARED_CURRENT, _RESISTANCE])
_RESISTOR = PhysicsPractice(
    family_id="resistor_power",
    exam_axes=("I", "R"),
    answer_recipe=_POWER,
    distractors=(
        (_VOLTAGE, "Find the voltage drop instead of the power."),
        (_SQUARED_CURRENT, "Square the current and leave out the resistance."),
        (RecipeNode(op="div", args=[_POWER, _TWO]), "Apply the AC peak-to-RMS rule to DC."),
        (RecipeNode(op="mul", args=[_POWER, _CURRENT]), "Cube the current instead of squaring it."),
        (
            RecipeNode(op="mul", args=[_POWER, _RESISTANCE]),
            "Square the voltage and omit division by resistance.",
        ),
        (
            RecipeNode(op="add", args=[_SQUARED_CURRENT, _RESISTANCE]),
            "Add the terms that should be multiplied.",
        ),
    ),
    explanation="P = I²R = ({I} A)² × {R} Ω = {answer}.",
)

_STIFFNESS = RecipeNode(param="k")
_EXTENSION = RecipeNode(param="x")
_SPRING_FORCE = RecipeNode(op="mul", args=[_STIFFNESS, _EXTENSION])
_SQUARED_EXTENSION_FORCE = RecipeNode(op="mul", args=[_SPRING_FORCE, _EXTENSION])
_SPRING = PhysicsPractice(
    family_id="spring_force",
    exam_axes=("k", "x"),
    answer_recipe=_SPRING_FORCE,
    distractors=(
        (_STIFFNESS, "Report the force constant as the force."),
        (_SQUARED_EXTENSION_FORCE, "Square the extension instead of using the linear force law."),
        (
            RecipeNode(op="div", args=[_SQUARED_EXTENSION_FORCE, _TWO]),
            "Use the elastic-energy formula.",
        ),
        (
            RecipeNode(op="mul", args=[_SPRING_FORCE, RecipeNode(const="0.01")]),
            "Treat the stated metres as centimetres.",
        ),
        (RecipeNode(op="mul", args=[_SPRING_FORCE, _TWO]), "Use twice the given extension."),
        (
            RecipeNode(op="add", args=[_STIFFNESS, _EXTENSION]),
            "Add the inputs instead of multiplying them.",
        ),
    ),
    explanation="Within the proportional limit, F = kx = ({k} N/m) × ({x} m) = {answer}.",
)


# The equation must name a neutralisation reaction that the curated catalogue
# can identify in the same direction. The authored question asks for balancing.
_FORMULA = r"[A-Z][A-Za-z0-9()]{0,31}"
_PAIR = rf"{_FORMULA}\s*\+\s*{_FORMULA}"
_NEUTRALISATION_EXAM = re.compile(
    rf"(?i:balance (?:the )?(?:following )?(?:chemical )?equation)\s*:?\s*"
    rf"({_PAIR})\s*(?:→|⟶|->)\s*({_PAIR})[.?]?"
)


def _neutralisations() -> list[ReactionEntry]:
    return [
        reaction for reaction in load_catalogue().reactions if reaction.family == "neutralisation"
    ]


def _read_neutralisation_exam(exam: str) -> tuple[str, ...] | None:
    match = _NEUTRALISATION_EXAM.fullmatch(exam)
    if match is None:
        return None
    reactants, products = (
        sorted(part.strip() for part in side.split("+")) for side in match.groups()
    )
    for reaction in _neutralisations():
        if reactants == sorted(reaction.reactants) and products == sorted(reaction.products):
            return (reaction.id,)
    return None


def _coefficient_foils(
    reaction: ReactionEntry, key: tuple[int, ...], rng: Random
) -> list[tuple[int, ...]]:
    def unbalanced(coefficients: tuple[int, ...]) -> bool:
        return (
            first_unbalanced_element(reaction.reactants, reaction.products, coefficients)
            is not None
        )

    # A swap preserves the sum, so the sum alone cannot identify the key.
    swaps = {
        (*key[:i], key[j], *key[i + 1 : j], key[i], *key[j + 1 :])
        for i, j in combinations(range(len(key)), 2)
        if key[i] != key[j]
    }
    shifted = [
        key[:i] + (changed,) + key[i + 1 :]
        for i, coefficient in enumerate(key)
        for changed in (coefficient - 1, coefficient + 1)
        if changed > 0
    ]
    return [
        rng.choice([choice for choice in sorted(swaps) if unbalanced(choice)]),
        *rng.sample([choice for choice in shifted if unbalanced(choice)], 2),
    ]


def _neutralisation_questions(exam: tuple[str, ...], rng: Random) -> list[Question]:
    (exam_id,) = exam
    variants = []
    for reaction in _neutralisations():
        if reaction.id == exam_id:
            continue
        key = balance_equation(reaction.reactants, reaction.products).coefficients
        if len(set(key)) > 1:
            variants.append((reaction, key))
    questions = []
    for reaction, key in rng.sample(variants, 5):
        reactants, products = reaction.reactants, reaction.products
        options = [
            ", ".join(map(str, choice)) for choice in (key, *_coefficient_foils(reaction, key, rng))
        ]
        skeleton = f"{' + '.join(reactants)} → {' + '.join(products)}"
        stem = (
            "Which smallest whole-number coefficients balance the equation, in the order "
            f"shown? Include every coefficient, even 1.\n{skeleton}"
        )
        totals = element_totals(reactants, key[: len(reactants)])
        atom_counts = ", ".join(f"{element}: {total}" for element, total in totals.items())
        explanation = f"Using {options[0]}, each side has the same atom counts ({atom_counts})."
        questions.append(_question(stem, options, explanation, rng))
    return questions


# Keep the exam's complete-dominance, pea-plant context, allele symbol and
# requested phenotype. Only the canonical genotypes of the two parents vary.
_MONOHYBRID_EXAM = re.compile(
    r"(?i:in pea plants, (?:allele )?)([A-Z]) "
    r"(?i:is completely dominant over (?:allele )?)([a-z])\. "
    r"(?i:crossing )([A-Za-z]{2}) (?:×|[xX]) ([A-Za-z]{2}), "
    r"(?i:what is the probability (?:of an offspring showing|that an offspring shows) the "
    r"(dominant|recessive) phenotype)\?"
)


def _read_monohybrid_exam(exam: str) -> tuple[str, ...] | None:
    match = _MONOHYBRID_EXAM.fullmatch(exam)
    if match is None:
        return None
    upper, lower, parent1, parent2, phenotype = match.groups()
    if lower != upper.lower():
        return None
    try:
        parse_genotype(parent1, upper)
        parse_genotype(parent2, upper)
    except BiologyCheckError:
        return None
    return upper, parent1, parent2, phenotype.lower()


def _monohybrid_questions(exam: tuple[str, ...], rng: Random) -> list[Question]:
    upper, given1, given2, phenotype = exam
    lower = upper.lower()
    genotypes = [upper * 2, upper + lower, lower * 2]
    given = frozenset((given1, given2))
    variants = [
        (parent1, parent2)
        for parent1, parent2 in combinations_with_replacement(genotypes, 2)
        if frozenset((parent1, parent2)) != given
    ]
    probabilities = [Fraction(value, 4) for value in range(5)]
    questions = []
    for parent1, parent2 in rng.sample(variants, 5):
        dist = genotype_distribution(parent1, parent2)
        key = sum(
            (value for genotype, value in dist.items() if phenotype_of(genotype) == phenotype),
            Fraction(0),
        )
        wrong = rng.sample([value for value in probabilities if value != key], 3)
        stem = (
            f"In pea plants, {upper} is completely dominant over {lower}. Crossing {parent1} × "
            f"{parent2}, what is the probability of an offspring showing the {phenotype} phenotype?"
        )
        cells = ", ".join(cell for row in punnett(parent1, parent2) for cell in row)
        explanation = (
            f"The four equally likely Punnett square cells are {cells}. "
            f"{key * 4} of the four show the {phenotype} phenotype, giving {key}."
        )
        questions.append(_question(stem, [str(value) for value in (key, *wrong)], explanation, rng))
    return questions


OPERATIONS = {
    "physics-resistor-power": Operation(
        domain="physics",
        task="Calculate the power dissipated by a resistor carrying a steady current.",
        reason="The lecture supplies P = I²R, and the exam gives resistance and steady current.",
        topic=(
            r"\b(?:resistor power|(?:electrical )?power dissipation|power in (?:a )?resistors?|"
            r"electric(?:al)? circuits?|dc circuits?|current electricity)\b"
        ),
        lecture_checks=(
            r"\bresistors?\b",
            r"\bP\s*=\s*I\s*\^\s*2\s*[×*·]?\s*R\b\s*(?=[.,!;:]|$)",
        ),
        read_exam=_read_resistor_exam,
        questions=_RESISTOR.questions,
        version=2,
    ),
    "physics-hooke-spring-force": Operation(
        domain="physics",
        task="Calculate the force needed to hold a spring at a given extension.",
        reason="The lecture supplies F = kx, and the exam gives a stiffness and an extension.",
        topic=(
            r"\b(?:hooke['’]s law|springs?|elasticity|elastic deformation|"
            r"deformation of solids)\b"
        ),
        lecture_checks=(
            r"\b(?:hooke['’]s law|spring force)\b",
            r"\b(?:proportional limit|limit of proportionality)\b",
            r"\bF\s*=\s*k\s*[×*·]?\s*x\b\s*(?=[.,!;:]|$)",
        ),
        read_exam=_read_spring_exam,
        questions=_SPRING.questions,
        version=2,
    ),
    "chemistry-neutralisation-balance": Operation(
        domain="chemistry",
        task="Balance an acid-base neutralisation equation using whole-number coefficients.",
        reason=(
            "The lecture teaches atom conservation for neutralisation, "
            "and the exam asks for a balance."
        ),
        topic=(
            r"\b(?:neutrali[sz]ation|stoichiometry|balancing (?:chemical )?equations?|"
            r"acids and bases)\b"
        ),
        lecture_checks=(
            r"\bneutrali[sz]ation\b",
            r"\bacid\b[^.!?]{0,100}\bbase\b[^.!?]{0,100}\b(?:forms?|gives?|produces?|yields?)\b"
            r"[^.!?]{0,50}\bsalt\b[^.!?]{0,50}\bwater\b",
            r"\bbalanc\w*\b[^.!?]{0,150}\bcoefficients?\b[^.!?]{0,150}"
            r"\b(?:conserv\w*|preserv\w*|the same (?:number|count))\b[^.!?]{0,50}\batoms?\b",
        ),
        read_exam=_read_neutralisation_exam,
        questions=_neutralisation_questions,
        version=2,
    ),
    "biology-monohybrid-probability": Operation(
        domain="biology",
        task="Use a Punnett square to find the probability of the requested offspring phenotype.",
        reason="The lecture teaches complete dominance, and the exam supplies a monohybrid cross.",
        topic=(
            r"\b(?:mendelian|monohybrid|punnett square|complete dominance|inheritance|genetics)\b"
        ),
        lecture_checks=(
            r"\bmonohybrid (?:cross|inheritance)\b",
            r"\bcomplete dominance\b",
            r"\bpunnett square\b[^.!?]{0,150}\b(?:probabilit\w+|offspring|phenotypes?)\b",
        ),
        read_exam=_read_monohybrid_exam,
        questions=_monohybrid_questions,
        lecture_conflicts=r"\b(?:co[ -]?dominan\w*|incomplete dominance)\b",
    ),
}


def generate(
    family: str, seed: int, lecture: str, exam: str, topic: str
) -> tuple[list[Question], int]:
    operation = OPERATIONS[family]
    evidence = operation.select(lecture, exam, topic)
    if evidence is None:
        raise ValueError("The current excerpts do not support the confirmed science operation.")
    rng = Random(f"material-science-{operation.version}/{family}/{seed}")  # noqa: S311
    return operation.questions(evidence, rng), operation.version
