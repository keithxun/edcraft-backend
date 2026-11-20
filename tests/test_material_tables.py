from edcraft_backend.services import material_tables

TABLE = """Table 1. Orbital observations
Moon              Planet
----              ------
Io                Jupiter
Titan             Saturn
Triton            Neptune
Phobos            Mars
Miranda           Uranus"""


def test_lookup_reads_aligned_pdf_columns_and_cites_the_exact_cell():
    table = material_tables.select(TABLE, "", "Mapping moon to planet")
    assert table
    source = dict(table.rows)
    for q in material_tables.questions(table, 611):
        key = q["stem"].split("“")[1].split("”")[0]
        assert q["options"][q["correct_index"]] == source[key]
        assert len(q["options"]) == len(set(q["options"])) == 4


def test_rejects_ambiguous_or_unrelated_layout():
    for text in [
        TABLE.replace("Titan ", "Io    "),
        TABLE.replace("Saturn", "\nSaturn"),
        TABLE.replace("Table 1. Orbital observations", "Notes"),
        TABLE.replace("----              ------\n", ""),
        TABLE + "\n\n" + TABLE,
    ]:
        assert material_tables.select(text, "", "Moon") is None
    assert material_tables.select(TABLE, "Explain the composition.", "Moon") is None
    assert material_tables.select(TABLE, "", "Banking") is None
