"""Match a lecture and a selected exam task to trusted CS1010E question families."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from random import Random
from typing import NotRequired, TypedDict

from pycparser import CParser
from pycparser import c_ast as c
from pycparser.c_parser import ParseError

from edcraft_backend.services.cs1010e.functions import (
    matches,
    render_max,
    render_pointer,
    render_square_sum,
    teaches_square_sum,
)
from edcraft_backend.services.cs1010e.loops import (
    CountingLoop,
    NestedLoops,
    UnsupportedLoop,
    read_loops,
)

VERSION = 3
SIZE_LIMIT = 24_000
TASKS = {
    "pointer-swap": (
        "Trace a swap of local pointer targets, a swap of pointed-to integers, "
        "and the weighted writes in both calls. Choose the printed sum."
    ),
    "recursive-max": (
        "Follow a C function that splits an inclusive array range at its midpoint, "
        "combines both child results, and prints the final integer."
    ),
}


class Task(TypedDict):
    family: str | None
    task: str
    reason: str


class Proposal(Task):
    alternatives: NotRequired[list[Task]]


class Question(TypedDict):
    stem: str
    options: list[str]
    correct_index: int
    explanation: str


@dataclass(frozen=True)
class Match:
    family: str
    task: str
    loop: NestedLoops | None = None


def _source(exam: str) -> str:
    # The shared section directive can come from another page. Only an explicitly
    # selected output-tracing task is eligible for an integer-answer family.
    text = exam.replace("\u00a0", " ").strip()
    if len(text) > SIZE_LIMIT:
        raise ValueError("Select a shorter exam passage containing one subquestion.")
    # C comments are discarded by a preprocessor in normal C compilation.
    # Keep string literals intact and preserve newlines for passage boundaries.
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
        lambda m: m.group() if m.group().startswith('"') else " ",
        text,
    )
    after = re.search(
        r"(?im)^[ \t]*(?:what|which) [^\n]{0,100}\b(?:output|printed|print)[^\n]{0,80}\?\s*\Z", text
    )
    suffix = after.group() if after else ""
    if after:
        text = text[: after.start()]
    start = re.search(
        r"(?m)^[ \t]*(?:(?:Q?\d{1,2}[.)])\s*)?(?P<code>#(?:include|define)\b|"
        r"(?:int|void|main)\b\s*(?=[A-Za-z_(]))",
        text,
    )
    if not start:
        raise ValueError("Select the C code for one exam subquestion, including its setup.")
    directive = (text[: start.start("code")] + " " + suffix).lower()
    if not re.search(r"\b(output|printed|print out|prints|displayed)\b", directive) or re.search(
        r"\b(complexity|prove|implement|rewrite|draw|explain|justify|"
        r"how many times|write (?:a|the|an) (?:function|program))\b",
        directive,
    ):
        raise ValueError("Include the exam instruction asking for the printed output.")
    code = text[start.start("code") :]
    # Options and printed page footers are outside the program. Never truncate
    # another numbered question implicitly: the user must choose a single subpart.
    if re.search(r"(?m)^[ \t]*(?:Q\d+\.|\d+\.\s*(?:\(|What|Study|Given|Let))", code):
        raise ValueError("Select a single exam subquestion and its shared stem.")
    code = re.split(r"(?m)^[ \t]*(?:[A-E]\.\s+|\([a-e]\)\s+|Page \d+\s*$)", code)[0]
    code = re.sub(r"(?m)^[ \t]*#\s*include\s*<(?:stdio|stdlib)\.h>\s*$", "", code)
    defines = re.findall(r"(?m)^[ \t]*#define\s+([A-Za-z_]\w*)\s+([1-9]\d{0,2})\s*$", code)
    if len(defines) == 1:
        name, size = defines[0]
        code = re.sub(r"(?m)^[ \t]*#define\s+\w+\s+\d+\s*$", "", code)
        code = re.sub(r"\b" + re.escape(name) + r"\b", size, code)
    # Bound parser work before constructing an AST. Uploaded code is never executed.
    if "#" in code or len(code) > 6000 or len(re.findall(r"\w+|[^\w\s]", code)) > 600:
        raise ValueError(
            "This source exceeds the supported C fragment or uses unsupported directives."
        )
    return code


def _recognise(exam: str) -> Match:
    code = _source(exam)
    parser = CParser()
    try:
        if re.search(r"\bmain\s*\(", code):
            tree = parser.parse(code)
            for family in TASKS:
                if matches(tree, family):
                    return Match(family, TASKS[family])
        else:
            tree = parser.parse("void selected(void) {\n" + code + "\n}")
            if len(tree.ext) == 1 and isinstance(tree.ext[0], c.FuncDef):
                spec = read_loops(tree.ext[0].body)
                return Match("nested-loops", spec.task, spec)
    except (ParseError, UnsupportedLoop, RecursionError, AssertionError):
        pass
    raise ValueError(
        "This exam fragment falls outside the checked loop, pointer-swap, "
        "and midpoint-recursion structures. Check the selection and its code layout."
    )


def _check_lecture(result: Match, lecture: str) -> Match:
    lesson = " ".join(lecture.lower().split())
    if result.family == "nested-loops":
        evidence = re.search(r"nested loops?|nesting (?:a )?loop", lesson) and (
            re.search(r"\binner\b", lesson)
            and re.search(r"\bouter\b", lesson)
            or lesson.count("for (") >= 2
        )
        if result.loop and result.loop.guard and result.loop.guard.action in {"break", "continue"}:
            required = result.loop.guard.action
            if not re.search(r"\b" + required + r"\b", lesson):
                raise ValueError(f"The selected lecture needs to cover the {required} statement.")
        needed = "how the inner loop runs within the outer loop"
    elif result.family == "pointer-swap":
        evidence = bool(
            "pointer" in lesson
            and "swap" in lesson
            and re.search(r"\*\s*\w+\s*=\s*\*\s*\w+", lesson)
            and re.search(r"\b\w+\s*=\s*&\s*\w+", lesson)
        )
        needed = "retargeting pointers and swapping values through pointer parameters"
    else:
        evidence = bool(
            re.search(r"\b(recurs\w*|trace tree)\b", lesson)
            and re.search(r"\(\s*\w+\s*\+\s*\w+\s*\)\s*/\s*2", lesson)
            and re.search(r"\(\s*\w+\s*\+\s*1\s*,", lesson)
        )
        needed = "the recursive midpoint split into two adjacent ranges"
    if not evidence:
        raise ValueError(f"The selected lecture needs a passage showing {needed}.")
    return result


# Topic words describe a checked operation. A match still requires independent lecture evidence.
_TOPIC = {
    "nested-loops": r"\b(loop\w*|iteration\w*|repetition\w*)\b",
    "pointer-swap": r"\b(pointer\w*|address\w*|pointee\w*|swap\w*)\b",
    "recursive-max": r"\b(recurs\w*|midpoint|divide and conquer)\b",
    "recursive-square-sum": r"\b(recurs\w*|midpoint|sum of squares|squared integers)\b",
}


FAMILIES = frozenset(_TOPIC)


def _topic_matches(family: str, topic: str) -> bool:
    return not topic.strip() or bool(re.search(_TOPIC[family], topic, re.IGNORECASE))


def _lecture_tasks(lecture: str, topic: str) -> list[Match]:
    candidates: list[Match] = []
    if teaches_square_sum(lecture):
        candidates.append(
            Match(
                "recursive-square-sum",
                "Trace the lecture’s midpoint sum of squares and choose the printed integer.",
            )
        )
    if re.search(r"\bprintf\b", lecture) and re.search(r"\b(for|while)\s*\(", lecture):
        loop = NestedLoops(
            CountingLoop(0, "<", 8, "+=", 1),
            CountingLoop(0, "<", 8, "+=", 1),
            0,
            0,
            1,
            "total",
            None,
        )
        candidates.append(Match("nested-loops", loop.task, loop))
    candidates.append(Match("pointer-swap", TASKS["pointer-swap"]))
    supported = []
    for candidate in candidates:
        if _topic_matches(candidate.family, topic):
            try:
                supported.append(_check_lecture(candidate, lecture))
            except ValueError:
                continue
    return supported


def _match(lecture: str, exam: str, topic: str, family: str | None = None) -> Match:
    if not exam.strip():
        for match in _lecture_tasks(lecture, topic):
            if family is None or family == match.family:
                return match
        raise ValueError("No checked C practice task matches that topic and lecture evidence.")
    match = _check_lecture(_recognise(exam), lecture)
    if not _topic_matches(match.family, topic):
        raise ValueError("The selected topic and the checked C exam operation differ.")
    return match


def _alternatives(match: Match, lecture: str) -> list[Task]:
    if match.family != "recursive-max" or not teaches_square_sum(lecture):
        return []
    return [
        {
            "family": "recursive-square-sum",
            "task": "Trace the lecture’s midpoint sum of squares and choose the printed integer.",
            "reason": "Uses the lecture’s singleton and addition rules with the exam’s "
            "two-branch reduction and final-output question.",
        }
    ]


def propose(lecture: str, exam: str, topic: str = "") -> Proposal:
    try:
        match = _match(lecture, exam, topic)
    except ValueError as exc:
        return {"family": None, "task": "", "reason": str(exc)}
    return {
        "family": match.family,
        "task": match.task,
        "reason": "",
        "alternatives": (
            _alternatives(match, lecture)
            if exam.strip()
            else [
                {
                    "family": m.family,
                    "task": m.task,
                    "reason": "Uses the selected lecture evidence.",
                }
                for m in _lecture_tasks(lecture, topic)[1:]
            ]
        ),
    }


def _choices(key: int, foils: list[int], rank: int, rng: Random, required: int | None) -> list[int]:
    """Use the small reserve pool without teaching a fixed numeric answer pattern."""
    distinct = list(dict.fromkeys(v for v in foils if v != key))
    rng.shuffle(distinct)
    groups = [g for g in combinations(distinct, 3) if required is None or required in g]
    if not groups:
        return []

    def preference(group: tuple[int, ...]) -> tuple[bool, int]:
        midpoint = any(a + b == 2 * key for a, b in combinations(group, 2))
        return midpoint, abs(sum(v < key for v in group) - rank)

    return [key, *min(groups, key=preference)]


def generate(family: str, seed: int, lecture: str, exam: str, topic: str = "") -> list[Question]:
    match = _match(lecture, exam, topic, family)
    if match.family != family and family not in {
        t["family"] for t in _alternatives(match, lecture)
    }:
        raise ValueError("The selected sources no longer support the confirmed task.")
    shape = json.dumps(asdict(match.loop), sort_keys=True) if match.loop else family
    rng = Random(f"cs1010e-{VERSION}/{shape}/{seed}")  # noqa: S311
    ranks = [0, 1, 2, 3, rng.randrange(4)]
    rng.shuffle(ranks)
    batch: list[Question] = []
    used_keys: set[int] = set()
    # A shuffled bounded size pool avoids repeats without precomputing instances.
    geometric = match.loop and any(
        h.step_op in {"/=", "*="} for h in (match.loop.outer, match.loop.inner)
    )
    sizes = list(range(4, 129 if geometric else 33))
    rng.shuffle(sizes)
    for size in sizes:
        if match.loop:
            if size == match.loop.size:
                continue
            try:
                code, key, foils, explanation = match.loop.render(size)
            except UnsupportedLoop:
                continue
        elif family == "recursive-max":
            code, key, foils, explanation = render_max(rng)
        elif family == "recursive-square-sum":
            code, key, foils, explanation = render_square_sum(rng)
        else:
            code, key, foils, explanation = render_pointer(rng)
        required = key + 1 if match.loop and match.loop.printed == "total++" else None
        choices = _choices(key, foils, ranks[len(batch)], rng, required)
        if key in used_keys or not choices:
            continue
        used_keys.add(key)
        rng.shuffle(choices)
        batch.append(
            {
                "stem": ("Assume standard C headers. Which integer does this code print?\n\n")
                + code,
                "options": [str(v) for v in choices],
                "correct_index": choices.index(key),
                "explanation": explanation,
            }
        )
        if len(batch) == 5:
            return batch
    raise ValueError("These parameters produced too few distinct checked variants.")
