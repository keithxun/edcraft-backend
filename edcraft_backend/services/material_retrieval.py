"""Rank course-material pages while preserving their source references and text."""

import re
from collections import Counter, defaultdict
from itertools import zip_longest
from math import log1p
from typing import TypedDict

from edcraft_backend.services.material_embeddings import (
    SemanticRetrievalUnavailable as SemanticRetrievalUnavailable,
)
from edcraft_backend.services.material_embeddings import semantic_scores

TOKEN = re.compile(r"[^\W_]+(?:_[^\W_]+)*", re.UNICODE)
K1 = 1.2
B = 0.75

_HEADING = re.compile(r"[\w][\w\s&,/():+'’–-]{1,79}")
_CONTENTS = re.compile(r"(?:table of )?contents(?:\s*\(continued\))?", re.I)
_DOTTED_PAGE = re.compile(r"(?:(?:[.·]\s*){3,}|…+\s*)\d+$")
_ALIGNED_PAGE = re.compile(r"[ \t]{3,}\d+$")
_SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+")
_NAMED_SECTION = r"(?:chapter|section|part|unit)\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.:)]?"
_NUMBERED_HEADING = re.compile(
    rf"^(?:{_NAMED_SECTION}|(?P<number>\d+(?:\.\d+)+[.)]?))(?P<gap>[ \t]+)", re.I
)
_SECTION_MARKER = re.compile(_NAMED_SECTION, re.I)
_SLIDE_CONTINUATION = re.compile(r"\s*\((?:[0-9]+\s*/\s*[0-9]+|continued)\)$", re.I)
_SLIDE_LABEL = re.compile(r".*\b(?:lecture|lesson|page|slide|week)\s*#?\d+[:.]?", re.I)
_FRONT_MATTER = re.compile(
    r"(?:table of )?contents|(?:learning )?(?:objectives|outcomes)|"
    r"(?:quiz|exercise|example)(?: \d+)?|references|bibliography|"
    r"previous|next|announcements?|common (?:mistakes?|errors?)|(?:today[’']s )?summary|"
    r"by .+|(?:authors?|instructors?|lecturers?)[: ].+",
    re.I,
)
_BULLET = re.compile(r"^(?:[•●▪◦◆○\ue000-\uf8ff]|[-*]\s)")


class Page(TypedDict):
    document: str
    page: int
    text: str


class RankedPage(Page):
    score: float | None


def _ranked(page: Page, score: float | None) -> RankedPage:
    return {**page, "score": score}


def _bm25(texts: list[str], topic: str) -> list[float]:
    query = set(TOKEN.findall(topic.casefold()))
    counts = [Counter(TOKEN.findall(text.casefold())) for text in texts]
    lengths = [sum(count.values()) for count in counts]
    avg_length = sum(lengths) / len(texts)
    if not query or not avg_length:
        return [0.0] * len(texts)

    document_frequency = Counter(term for count in counts for term in query & count.keys())
    idf = {
        term: log1p((len(counts) - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in sorted(document_frequency.items())
    }
    scores = []
    for count, length in zip(counts, lengths, strict=True):
        length_penalty = K1 * (1 - B + B * length / avg_length)
        scores.append(
            sum(
                weight * count[term] * (K1 + 1) / (count[term] + length_penalty)
                for term, weight in idf.items()
            )
        )
    return scores


def retrieve(
    pages: list[Page], topic: str, strategy: str, selected: list[tuple[str, int]], limit: int
) -> list[RankedPage]:
    """Return explicit selections in order or the highest-scoring pages up to ``limit``."""
    if strategy == "selected":
        by_ref = {(page["document"], page["page"]): page for page in pages}
        if any(ref not in by_ref for ref in selected):
            raise ValueError("Unknown selected page. Select pages from the supplied materials.")
        return [_ranked(by_ref[ref], None) for ref in selected]
    if strategy not in {"keyword", "semantic"}:
        raise ValueError("Unknown retrieval strategy.")

    candidates = [page for page in pages if page["text"].strip()]
    if not candidates or not topic.strip() or limit <= 0:
        return []
    texts = [page["text"] for page in candidates]
    scores = _bm25(texts, topic) if strategy == "keyword" else semantic_scores(texts, topic)
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda result: -result[1])
    return [
        _ranked(page, score)
        for page, score in ranked[:limit]
        if strategy == "semantic" or score > 0
    ]


def _title(line: str) -> str | None:
    title = _SLIDE_CONTINUATION.sub("", " ".join(line.split()).rstrip(","))
    if (
        _HEADING.fullmatch(title)
        and any(char.isalpha() for char in title)
        and len(title.split()) <= 12
        and not _FRONT_MATTER.fullmatch(title)
        and not _SLIDE_LABEL.fullmatch(title)
        and not _SECTION_MARKER.fullmatch(title)
    ):
        return title
    return None


def _contents_titles(lines: list[str]) -> list[str]:
    lines = [line for line in lines if len(line) <= 240]
    dotted = [_DOTTED_PAGE.sub("", line) for line in lines]
    is_contents = (
        any(_CONTENTS.fullmatch(line) for line in lines[:4])
        or sum(source != stripped for source, stripped in zip(lines, dotted, strict=True)) >= 2
    )
    titles = []
    for source, stripped in zip(lines, dotted, strict=True):
        if source == stripped:
            if not is_contents:
                continue
            stripped = _ALIGNED_PAGE.sub("", source)
            if stripped == source and not _SECTION_NUMBER.match(source):
                continue
        if title := _title(_SECTION_NUMBER.sub("", _NUMBERED_HEADING.sub("", stripped))):
            titles.append(title)
    return titles


def _numbered_titles(lines: list[str]) -> list[str]:
    titles = []
    for index, line in enumerate(lines):
        if _SECTION_MARKER.fullmatch(line):
            heading = lines[index + 1] if index + 1 < len(lines) else ""
        elif len(line) <= 240 and (prefix := _NUMBERED_HEADING.match(line)):
            heading = _ALIGNED_PAGE.sub("", line[prefix.end() :])
            # A single space and a lowercase instruction often indicate a numbered list.
            if prefix["number"] and len(prefix["gap"]) == 1 and heading[:1].islower():
                continue
        else:
            continue
        if title := _title(heading):
            titles.append(title)
    return titles


def _opening_titles(lines: list[str]) -> list[str]:
    titles = []
    for line in lines[:4]:
        if _BULLET.match(line):
            break
        # A layout-extracted right column often holds a running label or a page number.
        left_column = re.split(r"[ \t]{8,}", line, maxsplit=1)[0]
        if title := _title(left_column):
            titles.append(title)
    return titles


def _document_topics(pages: list[list[str]]) -> list[str]:
    contents = [title for lines in pages for title in _contents_titles(lines)]
    if contents:
        return contents
    sections = [title for lines in pages for title in _numbered_titles(lines)]
    if len({title.casefold() for title in sections}) >= 2:
        return sections

    openings = [titles for lines in pages if (titles := _opening_titles(lines))]
    firsts = Counter(titles[0].casefold() for titles in openings)
    for titles in openings:
        frequency = firsts[titles[0].casefold()]
        running_header = frequency >= 2 and frequency * 2 >= len(openings)
        sections.append(titles[1] if running_header and len(titles) > 1 else titles[0])
    return sections


def topics(pages: list[Page]) -> list[str]:
    """Suggest distinct contents entries, numbered sections or slide titles from the collection."""
    documents: dict[str, list[list[str]]] = defaultdict(list)
    for page in pages:
        documents[page["document"]].append(
            [line.strip() for line in page["text"].splitlines() if line.strip()]
        )
    suggestions: dict[str, str] = {}
    document_suggestions = [_document_topics(lines) for lines in documents.values()]
    # Surface material from each document before showing deeper headings from any of them.
    for entries in zip_longest(*document_suggestions):
        for title in entries:
            if title is not None:
                suggestions.setdefault(title.casefold(), title)
    return list(suggestions.values())
