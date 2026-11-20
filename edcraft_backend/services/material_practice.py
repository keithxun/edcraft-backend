"""Dispatch by an evidenced operation. No module names take part in matching."""

from edcraft_backend.services import (
    calculus_practice,
    cs1010e_practice,
    material_glossary,
    material_science,
    material_tables,
)


def _allows(operation_domain: str, domain: str | None, enabled: tuple[str, ...] | None) -> bool:
    return (domain is None or domain == operation_domain) and (
        enabled is None or operation_domain in enabled
    )


def propose(
    lecture: str,
    exam: str,
    topic: str,
    domain: str | None = None,
    *,
    enabled: tuple[str, ...] | None = None,
) -> dict:
    programming = (
        cs1010e_practice.propose(lecture, exam, topic) if _allows("code", domain, enabled) else None
    )
    calculus = (
        calculus_practice.propose(lecture, exam, topic=topic)
        if _allows("math", domain, enabled)
        else None
    )
    candidates = []
    if programming and programming["family"]:
        candidates.append(dict(programming))
        candidates.extend(dict(option) for option in programming.get("alternatives", []))
    if calculus and calculus["family"]:
        candidates.append(dict(calculus))
    candidates.extend(
        {
            "family": family,
            "task": operation.task,
            "reason": (
                operation.reason
                if exam.strip()
                else "The selected lecture teaches this operation and contains a complete example."
            ),
        }
        for family, operation in material_science.OPERATIONS.items()
        if _allows(operation.domain, domain, enabled) and operation.select(lecture, exam, topic)
    )
    candidates.extend(_source_lookup(lecture, exam, topic))
    if candidates:
        first, *remaining = candidates
        return {**first, "alternatives": remaining}
    reason = (
        "No checked practice question matches this topic and evidence. "
        "Choose a more specific passage or a different exam subquestion."
    )
    if programming and exam.strip() and "printf(" in exam:
        reason = programming["reason"]
    return {"family": None, "task": "", "reason": reason, "alternatives": []}


def _source_lookup(lecture: str, exam: str, topic: str) -> list[dict[str, object]]:
    """Offer literal recall and lookup independently of enabled specialist subjects."""
    candidates: list[dict[str, object]] = []
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
    return candidates


def generate(
    family: str,
    seed: int,
    lecture: str,
    exam: str,
    topic: str,
    domain: str | None = None,
    *,
    enabled: tuple[str, ...] | None = None,
) -> tuple[list, int]:
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
    if family in material_science.OPERATIONS:
        operation = material_science.OPERATIONS[family]
        _require_domain(operation.domain, domain, enabled)
        return material_science.generate(family, seed, lecture, exam, topic)
    if family in calculus_practice.TASKS:
        _require_domain("math", domain, enabled)
        return calculus_practice.generate(
            family, seed, lecture, exam, topic=topic
        ), calculus_practice.VERSION
    if family in cs1010e_practice.FAMILIES:
        _require_domain("code", domain, enabled)
        return cs1010e_practice.generate(
            family, seed, lecture, exam, topic
        ), cs1010e_practice.VERSION
    raise ValueError("Unknown checked operation.")


def _require_domain(
    operation_domain: str, domain: str | None, enabled: tuple[str, ...] | None
) -> None:
    if not _allows(operation_domain, domain, enabled):
        raise ValueError("The checked operation does not belong to an enabled, selected domain.")
