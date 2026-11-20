"""The source pair must support the student action and the displayed C program."""

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from edcraft_backend.services.cs1010e_practice import generate, propose

LECTURE = "Nested loops. In each round of the outer loop all rounds of the inner loop run."
DIRECTIVE = "What integer is printed by this code fragment?\n"
LOOP = """
int i, j, n = 0;
for (i=0; i<33; i+=2) {
    n++;
    for (j=1; j<33; j+=2) n--;
}
printf("%d", n);
"""


def test_both_selections_must_support_the_task():
    assert propose(LECTURE, DIRECTIVE + LOOP)["family"] == "nested-loops"
    assert propose("How to declare a pointer", DIRECTIVE + LOOP)["family"] is None
    assert propose(LECTURE, "Give its time complexity.\n" + LOOP)["family"] is None
    assert propose(LECTURE, DIRECTIVE + LOOP + " k++;")["family"] is None
    assert propose(LECTURE, DIRECTIVE + LOOP.replace("j+=2", "j-=2"))["family"] is None


@given(st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=None)
def test_rendered_loop_batch_preserves_strides_and_has_one_right_answer(seed):
    batch = generate("nested-loops", seed, LECTURE, DIRECTIVE + LOOP)
    assert len({q["stem"] for q in batch}) == 5
    for q in batch:
        limits = [int(x) for x in re.findall(r"< (\d+);", q["stem"])]
        assert len(limits) == 2
        outer = len(range(0, limits[0], 2))
        inner = len(range(1, limits[1], 2))
        assert len(set(q["options"])) == 4
        assert q["options"][q["correct_index"]] == str(outer * (1 - inner))


def test_generation_rechecks_compatibility_and_replays():
    with pytest.raises(ValueError):
        generate("pointer-swap", 13, LECTURE, DIRECTIVE + LOOP)
    assert generate("nested-loops", 42, LECTURE, DIRECTIVE + LOOP) == generate(
        "nested-loops", 42, LECTURE, DIRECTIVE + LOOP
    )


def test_halving_practice_has_varied_outcomes():
    exam = (
        DIRECTIVE
        + """
int row, col, count = 0;
for(row=32; row>0; row/=2)
    for(col=row; col>0; col/=2) count++;
printf("%d", count);
"""
    )
    batch = generate("nested-loops", 1010, LECTURE, exam)
    assert len({q["options"][q["correct_index"]] for q in batch}) == 5


def test_lecture_reduction_can_be_practised_in_the_exam_output_format():
    from edcraft_backend.services.cs1010e.functions import MAX_OUTLINE

    lecture = """Draw the trace tree for this recursion:
int study(int x, int y) {
    int mid;
    if (x == y) return x*x;
    else { mid=(x+y)/2; return study(x,mid) + study(mid+1,y); }
}"""
    exam = DIRECTIVE + MAX_OUTLINE
    result = propose(lecture, exam)
    assert result["family"] == "recursive-max"
    assert result["alternatives"][0]["family"] == "recursive-square-sum"
    for q in generate("recursive-square-sum", 11, lecture, exam):
        lo, hi = map(int, re.search(r'printf\("%d", reduce\((\d+), (\d+)\)', q["stem"]).groups())
        assert q["options"][q["correct_index"]] == str(sum(i * i for i in range(lo, hi + 1)))
    assert not propose(lecture.replace("x*x", "x+x"), exam).get("alternatives")
    with pytest.raises(ValueError):
        generate("recursive-square-sum", 11, lecture.replace("x*x", "x+x"), exam)


def test_rank_of_recursive_maximum_varies_within_each_batch():
    from edcraft_backend.services.cs1010e.functions import MAX_OUTLINE

    lecture = "Trace tree: mid=(x+y)/2, second call f(mid+1,y)."
    ranks = []
    for q in generate("recursive-max", 11, lecture, DIRECTIVE + MAX_OUTLINE):
        values = list(map(int, q["options"]))
        ranks.append(sorted(values).index(values[q["correct_index"]]))
    assert set(ranks) == {0, 1, 2, 3}


@pytest.mark.parametrize(
    "inner", ["if (col > row) break; count++;", "if (abs(row-col)==1) count++;"]
)
def test_guarded_updates_remain_in_display_and_change_counts(inner):
    from math import comb

    lecture = LECTURE + " Stop an inner loop early using break. Use if to test a condition."
    exam = (
        DIRECTIVE
        + f"""
int row, col, count=0;
for(row=1;row<=5;row++)
    for(col=1;col<=5;col++) {{ {inner} }}
printf("%d",count);
"""
    )
    for q in generate("nested-loops", 6, lecture, exam):
        n = int(re.search(r"i <= (\d+)", q["stem"])[1])
        expected = comb(n + 1, 2) if "break" in inner else 2 * (n - 1)
        assert "if (" in q["stem"]
        assert q["options"][q["correct_index"]] == str(expected)
    if "break" in inner:
        assert propose(LECTURE, exam)["family"] is None


def test_geometric_outer_loop_preserves_a_single_inner_iteration():
    exam = (
        DIRECTIVE
        + """
int row, col, count=0;
for(row=1;row<=64;row*=2)
    for(col=row;col<=row;col++) count++;
printf("%d",count);
"""
    )
    for q in generate("nested-loops", 99, LECTURE, exam):
        bound = int(re.search(r"i <= (\d+)", q["stem"])[1])
        assert q["options"][q["correct_index"]] == str(bound.bit_length())
    assert propose(LECTURE, exam.replace("row=1", "row=0"))["family"] is None


def test_lecture_topic_heading_alone_is_insufficient():
    assert propose("Next lecture: Nested loops", DIRECTIVE + LOOP)["family"] is None


@given(st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=32, deadline=None)
def test_rendered_pointer_and_reduction_keys_across_seeds(seed):
    from edcraft_backend.services.cs1010e.functions import MAX_OUTLINE, POINTER_OUTLINE

    pointer_notes = "Retarget a pointer with p = &a; swap values using *p = *q;"
    for q in generate("pointer-swap", seed, pointer_notes, DIRECTIVE + POINTER_OUTLINE):
        code = q["stem"]
        a, b = map(int, re.search(r"int a = (\d+), b = (\d+);", code).groups())
        end_p, end_q, first_p, first_q = map(int, re.findall(r"\*= (\d+);", code))
        # Model addresses as list indices and perform writes in C call order.
        memory = [a, b]
        p, r = 1, 0
        memory[p] *= first_p
        memory[r] *= first_q
        memory[p], memory[r] = memory[r], memory[p]
        memory[p] *= end_p
        memory[r] *= end_q
        assert q["options"].count(str(sum(memory))) == 1
        assert q["options"][q["correct_index"]] == str(sum(memory))
    notes = "Trace tree: mid=(x+y)/2; the next call starts at f(mid+1,y)."
    for q in generate("recursive-max", seed, notes, DIRECTIVE + MAX_OUTLINE):
        array = re.search(r"= \{([\d, ]+)\};", q["stem"])[1]
        key = max(map(int, array.split(",")))
        assert q["options"].count(str(key)) == 1
        assert q["options"][q["correct_index"]] == str(key)


def test_loop_key_is_not_systematically_the_midpoint_of_two_foils():
    from itertools import combinations

    hits = 0
    for q in generate("nested-loops", 1010, LECTURE, DIRECTIVE + LOOP):
        key = int(q["options"][q["correct_index"]])
        foils = [int(v) for v in q["options"] if int(v) != key]
        hits += any(a + b == 2 * key for a, b in combinations(foils, 2))
    assert hits <= 1


def test_c_source_with_comments_headers_or_an_instruction_after_the_code():
    exam = "#include <stdlib.h>\n" + LOOP.replace("n++;", "n++; /* one outer round */")
    assert propose(LECTURE, exam + "\nWhat integer is printed?")["family"] == "nested-loops"
    assert (
        propose("A beginner's guide to routers and nested loops.", DIRECTIVE + LOOP)["family"]
        is None
    )
