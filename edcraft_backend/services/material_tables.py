"""Create literal lookup exercises from an unambiguous, user-selected two-column table."""

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass

VERSION = 1
FAMILY = "table-lookup"


@dataclass(frozen=True)
class Table:
    key_label: str
    value_label: str
    rows: tuple[tuple[str, str], ...]


def _cell_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _table(cells: list[tuple[str, str]]) -> Table | None:
    if len(cells) < 6:
        return None
    headers, rows = cells[0], cells[1:]
    if not all(0 < len(v) <= 100 for row in [headers, *rows] for v in row):
        return None
    if _cell_key(headers[0]) == _cell_key(headers[1]):
        return None
    if len({_cell_key(row[0]) for row in rows}) != len(rows):
        return None
    if len({_cell_key(row[1]) for row in rows}) < 4:
        return None
    return Table(headers[0], headers[1], tuple((row[0], row[1]) for row in rows))


def _pipe_tables(lecture: str) -> Iterator[Table | None]:
    for group in re.finditer(r"(?m)^(?:[^\n]*\|[^\n]*(?:\n|$))+", lecture):
        rows = [line.strip().strip("|").split("|") for line in group.group().splitlines()]
        if len(rows) < 7 or any(len(row) != 2 for row in rows):
            continue
        if all(re.fullmatch(r"\s*:?-{3,}:?\s*", v) for v in rows[1]):
            yield _table([(a.strip(), b.strip()) for a, b in [rows[0], *rows[2:]]])


def _aligned_tables(lecture: str) -> Iterator[Table | None]:
    # A caption and a fixed second-column position distinguish tables from slide prose.
    for block in re.split(r"\n[ \t]*\n", lecture):
        lines = block.splitlines()
        caption = next(
            (
                i
                for i, line in enumerate(lines)
                if re.match(r"^\s*table\s+(?:\d+|[IVX]+)\b", line, re.I)
            ),
            None,
        )
        if caption is None:
            continue
        cells: list[tuple[str, str]] = []
        header_rule = False
        second_column = None
        for line in lines[caption + 1 :]:
            row = re.fullmatch(r"[ \t]*(\S.*?)[ \t]{2,}(\S(?:.*\S)?)[ \t]*", line)
            if len(cells) == 1 and re.fullmatch(r"\s*[-=]{2,}(?:[ \t]+[-=]{2,})?\s*", line):
                header_rule = True
                continue
            if not row or re.search(r"\s{2,}", row[2]):
                break
            if second_column is not None and second_column != row.start(2):
                break
            second_column = row.start(2)
            cells.append((row[1], row[2]))
        else:
            if header_rule:
                yield _table(cells)


def select(lecture: str, exam: str, topic: str) -> Table | None:
    if exam.strip():
        return None
    found = [
        table
        for table in [*_pipe_tables(lecture), *_aligned_tables(lecture)]
        if table
        and (
            not topic
            or any(
                re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", topic, re.I)
                for label in (table.key_label, table.value_label)
            )
        )
    ]
    return found[0] if len(found) == 1 else None


def questions(table: Table, seed: int) -> list[dict]:
    rng = random.Random(f"table-{VERSION}/{seed}")  # noqa: S311
    values = {_cell_key(v): v for _, v in table.rows}
    result = []
    for key, correct in rng.sample(table.rows, 5):
        foils = [v for k, v in values.items() if k != _cell_key(correct)]
        options = [correct, *rng.sample(foils, 3)]
        rng.shuffle(options)
        result.append(
            {
                "stem": f"In the selected table, which exact {table.value_label} entry "
                f"is listed for {table.key_label} “{key}”?",
                "options": options,
                "correct_index": options.index(correct),
                "explanation": f"The row for {key} lists {correct} under {table.value_label}.",
            }
        )
    return result
