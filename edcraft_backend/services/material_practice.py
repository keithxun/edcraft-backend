"""Dispatch by an evidenced operation. No module names take part in matching."""

from edcraft_backend.services import (
    calculus_practice,
    cs1010e_practice,
    material_glossary,
    material_tables,
)


def propose(lecture: str, exam: str, topic: str) -> dict:
    programming = cs1010e_practice.propose(lecture, exam, topic)
    calculus = calculus_practice.propose(lecture, exam, topic=topic)
    candidates = []
    if programming["family"]:
        candidates.append(dict(programming))
        candidates.extend(dict(option) for option in programming.get("alternatives", []))
    if calculus["family"]:
        candidates.append(dict(calculus))
    glossary = material_glossary.select(lecture, exam, topic)
    if glossary:
        candidates.append(
            {
                "family": material_glossary.FAMILY,
                "task": f"Identify the term for an exact definition from {glossary.title}.",
                "reason": "The labels and definitions come from the reviewed glossary.",
            }
        )
    table = material_tables.select(lecture, exam, topic)
    if table:
        candidates.append(
            {
                "family": material_tables.FAMILY,
                "task": f"Look up the {table.value_label} for a given {table.key_label}.",
                "reason": "Each question uses a different row of the selected table.",
            }
        )
    if candidates:
        first, *remaining = candidates
        return {**first, "alternatives": remaining}
    reason = (
        "No checked practice question matches this topic and evidence. "
        "Choose a more specific passage or a different exam subquestion."
    )
    if exam.strip() and "printf(" in exam:
        reason = programming["reason"]
    return {"family": None, "task": "", "reason": reason, "alternatives": []}


def generate(family: str, seed: int, lecture: str, exam: str, topic: str) -> tuple[list, int]:
    if family == material_glossary.FAMILY:
        glossary = material_glossary.select(lecture, exam, topic)
        if not glossary:
            raise ValueError("The current excerpts do not identify one unambiguous glossary.")
        return material_glossary.questions(glossary, seed), material_glossary.VERSION
    if family == material_tables.FAMILY:
        table = material_tables.select(lecture, exam, topic)
        if not table:
            raise ValueError("The current excerpts do not support a unique lookup table.")
        return material_tables.questions(table, seed), material_tables.VERSION
    if family.startswith("calculus-"):
        return calculus_practice.generate(
            family, seed, lecture, exam, topic=topic
        ), calculus_practice.VERSION
    if family in cs1010e_practice.FAMILIES:
        return cs1010e_practice.generate(
            family, seed, lecture, exam, topic
        ), cs1010e_practice.VERSION
    raise ValueError("Unknown checked operation.")
