"""Literal term identification from a labelled glossary supplied by the user."""

import random
import re
from dataclasses import dataclass

FAMILY = "glossary-term"
VERSION = 1
_HEADING = re.compile(
    r"^(?:[\w /&–-]{1,64} )?(?:glossary|definitions)(?:\s*[:–-]\s*[\w /&–-]{1,64})?$", re.I
)
_ENTRY = re.compile(r"^([^:\n]{1,70}):\s+(\S.{7,299})$")
_RECALL = re.compile(
    r"(?:Q?\d+[.)]\s*)?(?:which|what) (?:term|word|concept) "
    r"(?:do the notes define as|is defined as|means|refers to) [^?\n]{1,300}\?",
    re.I,
)


@dataclass(frozen=True)
class Glossary:
    title: str
    entries: dict[str, str]


def _entries(lines: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    seen_terms: set[str] = set()
    for entry in lines:
        if not entries and not entry.strip():
            continue
        match = _ENTRY.fullmatch(entry.strip())
        if not entry.strip():
            break
        if not match or entry[:1].isspace():
            return {}
        term, definition = (s.strip() for s in match.groups())
        if (
            term.casefold() in {"e.g.", "i.e.", "for example", "example", "note"}
            or term.casefold() in seen_terms
        ):
            return {}
        seen_terms.add(term.casefold())
        entries[term] = definition
    definitions = {" ".join(value.casefold().split()) for value in entries.values()}
    return entries if len(entries) >= 5 and len(definitions) == len(entries) else {}


def select(lecture: str, exam: str, topic: str) -> Glossary | None:
    if exam.strip() and not _RECALL.fullmatch(exam.strip()):
        return None
    matches = []
    lines = lecture.splitlines()
    for idx, line in enumerate(lines):
        heading = line.strip()
        if not _HEADING.fullmatch(heading):
            continue
        entries = _entries(lines[idx + 1 :])
        if not entries:
            continue
        # Use a source heading or a listed term to associate the block with the topic.
        grounding = [heading.removesuffix(":"), *entries]
        words = {w.casefold() for w in re.findall(r"\w+", heading)} - {
            "of",
            "and",
            "the",
            "for",
            "in",
            "glossary",
            "definitions",
        }
        if topic and not (
            words.intersection(re.findall(r"\w+", topic.casefold()))
            or any(re.search(r"\b" + re.escape(label) + r"\b", topic, re.I) for label in grounding)
        ):
            continue
        matches.append(Glossary(heading, entries))
    return matches[0] if len(matches) == 1 else None


def questions(block: Glossary, seed: int) -> list[dict]:
    rng = random.Random(f"glossary-{VERSION}/{seed}")  # noqa: S311
    terms = list(block.entries)
    batch = []
    for term in rng.sample(terms, 5):
        options = [term, *rng.sample([other for other in terms if other != term], 3)]
        rng.shuffle(options)
        definition = block.entries[term]
        batch.append(
            {
                "stem": f"Which exact term does the lecture glossary use for “{definition}”?",
                "options": options,
                "correct_index": options.index(term),
                "explanation": f"The entry under {term} reads “{definition}”.",
            }
        )
    return batch
