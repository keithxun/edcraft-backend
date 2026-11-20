"""A bounded language for two nested counting loops and a printed integer."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pycparser import c_ast as c

from .conditions import COMPARE, Action, Guard, operand


class UnsupportedLoop(ValueError):
    pass


def small_integer(node: c.Node) -> int:
    if isinstance(node, c.Constant) and node.type == "int" and node.value.isdecimal():
        value = int(node.value)
        if 0 <= value <= 4096 and str(value) == node.value:
            return value
    raise UnsupportedLoop


def is_name(node: c.Node, name: str) -> bool:
    return isinstance(node, c.ID) and node.name == name


def bound(node: c.Node, outer: str | None) -> int | str:
    if outer and is_name(node, outer):
        return "i"
    return small_integer(node)


@dataclass(frozen=True)
class CountingLoop:
    start: int | str
    op: str
    stop: int | str
    step_op: str
    step: int

    def sizes(self) -> tuple[int, ...]:
        return tuple(v for v in (self.start, self.stop) if isinstance(v, int))

    def resized(self, old: int, new: int, scale_ranges: bool) -> CountingLoop:
        def resize(value: int | str) -> int | str:
            if value == old:
                return new
            if scale_ranges and isinstance(value, int) and value > 1:
                return max(2, round(value * new / old))
            return value

        return replace(
            self,
            start=resize(self.start),
            stop=resize(self.stop),
        )

    def values(self, outer: int = 0) -> list[int]:
        start = outer if isinstance(self.start, str) else self.start
        stop = outer if isinstance(self.stop, str) else self.stop
        if self.step_op == "*=":
            values = []
            if start <= 0:
                raise UnsupportedLoop
            while COMPARE[self.op](start, stop):
                values.append(start)
                start *= self.step
            return values
        if self.step_op == "/=":
            values = []
            while start > stop:
                values.append(start)
                start //= self.step
            return values
        delta = self.step if self.step_op == "+=" else -self.step
        end = stop + int(self.op == "<=") - int(self.op == ">=")
        return list(range(start, end, delta))

    def header(self, name: str) -> str:
        return (
            f"for (int {name} = {self.start}; {name} {self.op} {self.stop}; "
            f"{name} {self.step_op} {self.step})"
        )


def read_header(loop: c.For, outer: str | None) -> tuple[str, CountingLoop]:
    init, cond, update = loop.init, loop.cond, loop.next
    if not (
        isinstance(init, c.Assignment)
        and init.op == "="
        and isinstance(init.lvalue, c.ID)
        and isinstance(cond, c.BinaryOp)
        and is_name(cond.left, init.lvalue.name)
    ):
        raise UnsupportedLoop
    name = init.lvalue.name
    if isinstance(update, c.UnaryOp) and is_name(update.expr, name):
        step_op = {"++": "+=", "p++": "+=", "--": "-=", "p--": "-="}.get(update.op, "")
        step = 1
    elif isinstance(update, c.Assignment) and is_name(update.lvalue, name):
        step_op, step = update.op, small_integer(update.rvalue)
    else:
        raise UnsupportedLoop
    allowed = {"+=": ("<", "<="), "-=": (">", ">="), "/=": (">",), "*=": ("<", "<=")}
    if cond.op not in allowed.get(step_op, ()) or not 1 <= step <= 8:
        raise UnsupportedLoop
    if step_op in {"/=", "*="} and step < 2:
        raise UnsupportedLoop
    if step_op == "*=" and (
        not isinstance(init.rvalue, c.Constant) or small_integer(init.rvalue) == 0
    ):
        raise UnsupportedLoop
    return name, CountingLoop(
        bound(init.rvalue, outer), cond.op, bound(cond.right, outer), step_op, step
    )


def statements(node: c.Node) -> list[c.Node]:
    return (node.block_items or []) if isinstance(node, c.Compound) else [node]


def change(node: c.Node, name: str) -> int:
    if isinstance(node, c.UnaryOp) and is_name(node.expr, name):
        if node.op in {"p++", "++", "p--", "--"}:
            return 1 if "+" in node.op else -1
    if isinstance(node, c.Assignment) and is_name(node.lvalue, name) and node.op in {"+=", "-="}:
        return small_integer(node.rvalue) * (1 if node.op == "+=" else -1)
    raise UnsupportedLoop


@dataclass(frozen=True)
class NestedLoops:
    outer: CountingLoop
    inner: CountingLoop
    start: int
    outer_change: int
    inner_change: int
    printed: str
    guard: Guard | None

    @property
    def size(self) -> int:
        return max((*self.outer.sizes(), *self.inner.sizes()), default=0)

    @property
    def task(self) -> str:
        dependency = (
            "The inner loop depends on the current outer counter."
            if "i" in (self.inner.start, self.inner.stop)
            else "The inner loop restarts from the same value each time."
        )
        if self.guard:
            step = {
                "break": "apply the inner-loop break condition",
                "continue": "apply the inner-loop continue condition",
                "count": "test the condition before each accumulator update",
            }
            dependency += f" The student must {step[self.guard.action]}."
        return f"Trace two nested C loops and choose the printed integer. {dependency}"

    def render(self, size: int) -> tuple[str, int, list[int], str]:
        outer, inner = (
            self.outer.resized(self.size, size, bool(self.guard)),
            self.inner.resized(self.size, size, bool(self.guard)),
        )
        rounds = outer.values()
        available = [inner.values(i) for i in rounds]
        counts = [
            self.guard.counted_iterations(i, values) if self.guard else len(values)
            for i, values in zip(rounds, available, strict=True)
        ]
        if self.guard and sum(counts) == sum(map(len, available)):
            raise UnsupportedLoop
        if len(rounds) < 2 or sum(n > 0 for n in counts) < 2:
            raise UnsupportedLoop
        outer_total = self.outer_change * len(rounds)
        inner_total = self.inner_change * sum(counts)
        key = self.start + outer_total + inner_total
        lines = [f"int total = {self.start};", f"{outer.header('i')} {{"]
        if self.outer_change:
            lines.append(f"    total += {self.outer_change};")
        update = f"total += {self.inner_change};"
        inner_lines = self.guard.render(update) if self.guard else [update]
        lines.extend(
            [
                f"    {inner.header('j')} {{",
                *[f"        {line}" for line in inner_lines],
                "    }",
                "}",
                f'printf("%d", {self.printed});',
            ]
        )
        explanation = (
            f"The outer loop runs {len(rounds)} times. The inner update runs "
            f"{sum(counts)} times in total. The printed value is "
            f"{self.start} + ({outer_total}) + ({inner_total}) = {key}."
        )
        distractors = [
            key - self.inner_change * len(rounds),
            key + self.outer_change + self.inner_change * counts[-1],
            self.start + outer_total,
            self.start + self.inner_change * counts[0],
            key - 1,
            key + 1,
        ]
        if self.guard:
            ungated = self.start + outer_total + self.inner_change * sum(map(len, available))
            distractors = [ungated, *distractors[:5]]
        if self.printed == "total++":
            explanation += " Post-increment passes the old total to printf."
            distractors = [key + 1, *distractors[:5]]
        return "\n".join(lines), key, distractors, explanation


def read_updates(node: c.Node, accumulator: str, names: dict[str, str]) -> tuple[int, Guard | None]:
    body = statements(node)
    if not body or len(body) > 2:
        raise UnsupportedLoop
    cond = body[0]
    if not isinstance(cond, c.If):
        if len(body) != 1:
            raise UnsupportedLoop
        return change(cond, accumulator), None
    if cond.iffalse or not isinstance(cond.cond, c.BinaryOp) or cond.cond.op not in COMPARE:
        raise UnsupportedLoop
    yes = statements(cond.iftrue)
    if len(yes) != 1:
        raise UnsupportedLoop
    action: Action
    if len(body) == 2 and isinstance(yes[0], (c.Break, c.Continue)):
        action = "break" if isinstance(yes[0], c.Break) else "continue"
        update = body[1]
    elif len(body) == 1:
        action, update = "count", yes[0]
    else:
        raise UnsupportedLoop
    guard = Guard(
        operand(cond.cond.left, names), cond.cond.op, operand(cond.cond.right, names), action
    )
    return change(update, accumulator), guard


def read_loops(body: c.Compound) -> NestedLoops:
    nodes = statements(body)
    if len(nodes) != 5:
        raise UnsupportedLoop
    *decls, outer, output = nodes
    if not all(
        isinstance(d, c.Decl)
        and isinstance(d.type, c.TypeDecl)
        and isinstance(d.type.type, c.IdentifierType)
        and d.type.type.names == ["int"]
        and not d.quals
        and not d.storage
        for d in decls
    ):
        raise UnsupportedLoop
    accumulators = [d for d in decls if d.init is not None]
    if len(accumulators) != 1 or not isinstance(outer, c.For):
        raise UnsupportedLoop
    accumulator = accumulators[0]
    outer_name, outer_spec = read_header(outer, None)
    body_nodes = statements(outer.stmt)
    if len(body_nodes) not in (1, 2) or not isinstance(body_nodes[-1], c.For):
        raise UnsupportedLoop
    inner = body_nodes[-1]
    inner_name, inner_spec = read_header(inner, outer_name)
    if {outer_name, inner_name, accumulator.name} != {d.name for d in decls}:
        raise UnsupportedLoop
    if len({d.name for d in decls}) != 3:
        raise UnsupportedLoop
    with_outer = change(body_nodes[0], accumulator.name) if len(body_nodes) == 2 else 0
    with_inner, guard = read_updates(
        inner.stmt, accumulator.name, {outer_name: "i", inner_name: "j"}
    )
    if not with_inner:
        raise UnsupportedLoop
    printed = printed_value(output, accumulator.name)
    spec = NestedLoops(
        outer_spec,
        inner_spec,
        small_integer(accumulator.init),
        with_outer,
        with_inner,
        printed,
        guard,
    )
    if spec.size <= max(spec.outer.step, spec.inner.step, 2):
        raise UnsupportedLoop
    # Scaling replaces just the largest endpoint. Other literal bounds must stay small.
    if any(x > 16 and x != spec.size for x in (*outer_spec.sizes(), *inner_spec.sizes())):
        raise UnsupportedLoop
    return spec


def printed_value(output: c.Node, accumulator: str) -> str:
    if not (
        isinstance(output, c.FuncCall)
        and is_name(output.name, "printf")
        and output.args
        and len(output.args.exprs) == 2
    ):
        raise UnsupportedLoop
    fmt, value = output.args.exprs
    if not (
        isinstance(fmt, c.Constant)
        and fmt.value in {'"%d"', '"%i"', '"%d "', '"%i "', '"%d\\n"', '"%i\\n"'}
    ):
        raise UnsupportedLoop
    printed = "total"
    if isinstance(value, c.UnaryOp) and value.op == "p++":
        value, printed = value.expr, "total++"
    if not is_name(value, accumulator):
        raise UnsupportedLoop
    return printed
