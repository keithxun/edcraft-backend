"""Recognise two narrowly checked C function structures and render fresh instances."""

from __future__ import annotations

import copy
import re
from functools import cache
from random import Random

from pycparser import CParser, c_generator
from pycparser import c_ast as c

# These outlines define the exact dependencies the corresponding solvers support.
# Source code is parsed for comparison only. Only our own templates are rendered.
POINTER_OUTLINE = """
void route(int *u, int *v);
void exchange(int *u, int *v);
int main(void) {
    int u = 2, v = 3;
    route(&u, &v);
    printf("%d", u + v);
    return 0;
}
void route(int *u, int *v) {
    int *hold;
    hold = u; u = v; v = hold;
    *u *= 2; *v *= 3;
    exchange(u, v);
    return;
}
void exchange(int *u, int *v) {
    int hold;
    hold = *u; *u = *v; *v = hold;
    *u *= 4; *v *= 5;
    return;
}
"""

MAX_OUTLINE = """
int select_value(int, int);
int items[4] = {4, 1, 8, 2};
int main() {
    printf("%d", select_value(0, 4 - 1));
}
int select_value(int lo, int hi) {
    int left, right, middle;
    if (hi > lo) {
        middle = (lo + hi) / 2;
        return ((left = select_value(lo, middle)) > (right = select_value(middle+1, hi))
                ? left : right);
    }
    return items[lo];
}
"""


def _is_positive(node: c.Node) -> bool:
    return (
        isinstance(node, c.Constant)
        and node.type == "int"
        and node.value.isdecimal()
        and str(int(node.value)) == node.value
        and 1 <= int(node.value) <= 100
    )


def _erase_parameters(tree: c.FileAST, family: str) -> bool:
    """Erase the data values only, preserving argument order, operations and literals."""
    funcs = [n for n in tree.ext if isinstance(n, c.FuncDef)]
    main = next((f for f in funcs if f.decl.name == "main"), None)
    if main is None:
        return False
    if family == "pointer-swap":
        return _pointer_parameters(funcs, main)
    return _array_parameters(tree, funcs, main)


def _pointer_parameters(funcs: list[c.FuncDef], main: c.FuncDef) -> bool:
    if len(funcs) != 3:
        return False
    for d in main.body.block_items or []:
        if isinstance(d, c.Decl) and _is_positive(d.init):
            d.init.value = "1"
    for func in funcs:
        for stmt in func.body.block_items or []:
            if isinstance(stmt, c.Assignment) and stmt.op == "*=" and _is_positive(stmt.rvalue):
                stmt.rvalue.value = "1"
    return True


def _array_parameters(tree: c.FileAST, funcs: list[c.FuncDef], main: c.FuncDef) -> bool:
    arrays = [d for d in tree.ext if isinstance(d, c.Decl) and isinstance(d.type, c.ArrayDecl)]
    if len(funcs) != 2 or len(arrays) != 1:
        return False
    array = arrays[0]
    if not (
        isinstance(array.init, c.InitList)
        and _is_positive(array.type.dim)
        and int(array.type.dim.value) >= 2
        and len(array.init.exprs) == int(array.type.dim.value)
        and all(
            isinstance(v, c.Constant)
            and v.type == "int"
            and v.value.isdecimal()
            and str(int(v.value)) == v.value
            and int(v.value) < 32768
            for v in array.init.exprs
        )
    ):
        return False
    # Only the declared length and the matching upper limit may vary.
    body = main.body.block_items or []
    if (
        len(body) == 2
        and isinstance(body[-1], c.Return)
        and isinstance(body[-1].expr, c.Constant)
        and body[-1].expr.value == "0"
    ):
        main.body.block_items = body[:1]
    if len(main.body.block_items or []) != 1:
        return False
    output = main.body.block_items[0]
    if not (isinstance(output, c.FuncCall) and output.args and len(output.args.exprs) == 2):
        return False
    call = output.args.exprs[1]
    if not (isinstance(call, c.FuncCall) and call.args and len(call.args.exprs) == 2):
        return False
    upper = call.args.exprs[1]
    if not (
        isinstance(upper, c.BinaryOp)
        and isinstance(upper.left, c.Constant)
        and upper.left.value == array.type.dim.value
    ):
        return False
    upper.left.value = array.type.dim.value = "4"
    array.init.exprs = [c.Constant("int", "1") for _ in range(4)]
    return True


def _shape(tree: c.FileAST, family: str) -> str | None:
    tree = copy.deepcopy(tree)
    if not _erase_parameters(tree, family):
        return None
    # CGenerator discards layout and redundant grouping but preserves the expression tree.
    code = c_generator.CGenerator().visit(tree)
    code = re.sub(r'"%[id](?:\\n| )?"', '"%d"', code)
    reserved = {"int", "void", "return", "if", "main", "printf", "d"}
    names: dict[str, str] = {}

    def rename(match: re.Match[str]) -> str:
        name = match.group()
        return name if name in reserved else names.setdefault(name, f"v{len(names)}")

    return re.sub(r"[a-zA-Z_]\w*", rename, code)


@cache
def outline_shape(family: str) -> str | None:
    outline = POINTER_OUTLINE if family == "pointer-swap" else MAX_OUTLINE
    return _shape(CParser().parse(outline), family)


def matches(tree: c.FileAST, family: str) -> bool:
    shape = _shape(tree, family)
    return shape is not None and shape == outline_shape(family)


def render_pointer(rng: Random) -> tuple[str, int, list[int], str]:
    a, b = rng.sample(range(2, 16), 2)
    p, q, r, s = rng.sample(range(2, 10), 4)
    code = f"""void finish(int *p, int *q) {{
    int held = *p;
    *p = *q;
    *q = held;
    *p *= {r};
    *q *= {s};
}}

void update(int *p, int *q) {{
    int *held = p;
    p = q;
    q = held;
    *p *= {p};
    *q *= {q};
    finish(p, q);
}}

int main(void) {{
    int a = {a}, b = {b};
    update(&a, &b);
    printf("%d", a + b);
    return 0;
}}"""
    # After local retargeting p refers to b. After the value swap, b holds a*q.
    final_a, final_b = b * p * s, a * q * r
    key = final_a + final_b
    candidates = [
        a * p * s + b * q * r,
        a * q * s + b * p * r,
        a * q + b * p,
        a + b,
        final_b,
        final_a,
    ]
    explanation = (
        f"After update retargets the pointers, a is {a * q} and b is {b * p} "
        f"after the first multiplications. finish swaps the stored values, then "
        f"multiplies b by {r} and a by {s}. The sum is {final_a} + {final_b} = {key}."
    )
    return code, key, candidates, explanation


def render_max(rng: Random) -> tuple[str, int, list[int], str]:
    items = rng.sample(range(2, 75), rng.randint(6, 10))
    size = len(items)
    code = f"""int values[{size}] = {{{", ".join(map(str, items))}}};

int reduce(int lo, int hi) {{
    if (lo == hi) return values[lo];
    int mid = (lo + hi) / 2;
    int first = reduce(lo, mid);
    int second = reduce(mid + 1, hi);
    return first > second ? first : second;
}}

int main(void) {{
    printf("%d", reduce(0, {size - 1}));
    return 0;
}}"""
    middle = (size - 1) // 2
    left, right = max(items[: middle + 1]), max(items[middle + 1 :])
    key = max(left, right)
    # Lower foils use an incorrect child or order statistic. Higher foils
    # double-count a child, the split index, or the comparisons along a path.
    higher = [left + right, key + middle, key + size - 1]
    lower = [min(left, right), min(items), sorted(items)[-2]]
    candidates = [*lower, *higher]
    explanation = (
        f"Singleton ranges return their array entries. Each call keeps the larger "
        f"child result. The final children return {left} and {right}, so the "
        f"printed value is {key}."
    )
    return code, key, candidates, explanation


def teaches_square_sum(lecture: str) -> bool:
    """Bind the worked example's base case, split, and reducer to the same variables."""
    lesson = " ".join(re.sub(r"//[^\n]*", "", lecture).split())
    outline = (
        r"int\s+(?P<fn>\w+)\s*\(\s*int\s+(?P<lo>\w+)\s*,\s*int\s+(?P<hi>\w+)\s*\)"
        r"\s*\{\s*int\s+(?P<mid>\w+)\s*;\s*if\s*\(\s*(?P=lo)\s*==\s*(?P=hi)\s*\)"
        r"\s*return\s+(?P=lo)\s*\*\s*(?P=lo)\s*;\s*else\s*\{\s*(?P=mid)\s*=\s*"
        r"\(\s*(?P=lo)\s*\+\s*(?P=hi)\s*\)\s*/\s*2\s*;\s*return\s+"
        r"(?P=fn)\s*\(\s*(?P=lo)\s*,\s*(?P=mid)\s*\)\s*\+\s*"
        r"(?P=fn)\s*\(\s*(?P=mid)\s*\+\s*1\s*,\s*(?P=hi)\s*\)\s*;\s*\}\s*\}"
    )
    return bool(re.search(outline, lesson))


def render_square_sum(rng: Random) -> tuple[str, int, list[int], str]:
    lo = rng.randint(1, 15)
    hi = lo + rng.randint(3, 8)
    code = f"""int reduce(int lo, int hi) {{
    if (lo == hi) return lo * lo;
    int mid = (lo + hi) / 2;
    return reduce(lo, mid) + reduce(mid + 1, hi);
}}

int main(void) {{
    printf("%d", reduce({lo}, {hi}));
    return 0;
}}"""
    middle = (lo + hi) // 2
    left = sum(i * i for i in range(lo, middle + 1))
    right = sum(i * i for i in range(middle + 1, hi + 1))
    key = left + right
    candidates = [
        key - hi * hi,
        sum(range(lo, hi + 1)),
        key - lo * lo,
        key + middle * middle,
        key + (hi + 1) ** 2,
        key + left,
    ]
    why = (
        f"Each singleton returns its square. The inclusive halves {lo}–{middle} "
        f"and {middle + 1}–{hi} contribute {left} and {right}. Their sum is {key}."
    )
    return code, key, candidates, why
