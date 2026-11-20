import pytest

from edcraft_backend.services import material_practice

NOTES = """Thermodynamics glossary
Isobaric: A process that takes place at constant pressure.
Isochoric: A process that takes place at constant volume.
Isothermal: A process that takes place at constant temperature.
Adiabatic: A process in which no heat is exchanged.
Reversible: An idealized process that can be reversed without net change."""


@pytest.mark.parametrize(
    "exam", ["", "Which term do the notes define as a transfer of thermal energy?"]
)
def test_grounded_recall_reuses_only_explicit_lecture_definitions(exam):
    proposal = material_practice.propose(NOTES, exam, "Thermodynamics")
    assert proposal["family"] == "glossary-term"
    batch, _ = material_practice.generate("glossary-term", 42, NOTES, exam, "Thermodynamics")
    definitions = dict(line.split(": ", 1) for line in NOTES.splitlines()[1:])
    assert len({q["stem"] for q in batch}) == 5
    for q in batch:
        term = q["options"][q["correct_index"]]
        assert definitions[term] in q["stem"]
        assert len(set(q["options"])) == 4


@pytest.mark.parametrize(
    "topic,exam,notes",
    [
        ("Thermodynamics", "Justify why this process is reversible.", NOTES),
        ("Recursion", "", NOTES),
        ("Thermodynamics", "", NOTES.replace("constant volume", "constant pressure")),
        ("Thermodynamics", "", NOTES.replace("Isothermal:", "Isobaric:")),
    ],
)
def test_refuses_ambiguous_definitions_and_incompatible_requests(topic, exam, notes):
    assert material_practice.propose(notes, exam, topic)["family"] is None
    with pytest.raises(ValueError):
        material_practice.generate("glossary-term", 7, notes, exam, topic)


def test_wrapped_or_malformed_glossary_cannot_silently_be_truncated():
    for following in [
        "followed by a qualifying condition.",
        "  e.g.: a special case",
        "For example: the previous definition only holds in a fluid.",
    ]:
        assert (
            material_practice.propose(NOTES + "\n" + following, "", "Thermodynamics")["family"]
            is None
        )
