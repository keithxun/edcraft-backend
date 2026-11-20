"""Predicates accepted inside a counted inner loop."""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pycparser import c_ast as c

Action = Literal["break", "continue", "count"]
COMPARE: dict[str, Callable[[int, int], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class Operand:
    expression: str
    literal: int = 0

    def value(self, i: int, j: int) -> int:
        if self.expression == "i":
            return i
        if self.expression == "j":
            return j
        if self.expression == "abs(i - j)":
            return abs(i - j)
        return self.literal


def operand(node: c.Node, names: dict[str, str]) -> Operand:
    if isinstance(node, c.ID) and node.name in names:
        return Operand(names[node.name])
    if isinstance(node, c.Constant) and node.type == "int" and node.value in ("0", "1", "2"):
        return Operand(node.value, int(node.value))
    if (
        isinstance(node, c.FuncCall)
        and isinstance(node.name, c.ID)
        and node.name.name == "abs"
        and node.args
        and len(node.args.exprs) == 1
    ):
        diff = node.args.exprs[0]
        if (
            isinstance(diff, c.BinaryOp)
            and diff.op == "-"
            and isinstance(diff.left, c.ID)
            and isinstance(diff.right, c.ID)
            and {diff.left.name, diff.right.name} == set(names)
        ):
            return Operand("abs(i - j)")
    raise ValueError("Unsupported inner-loop comparison operand.")


@dataclass(frozen=True)
class Guard:
    left: Operand
    op: str
    right: Operand
    action: Action

    def allows(self, i: int, j: int) -> bool:
        return COMPARE[self.op](self.left.value(i, j), self.right.value(i, j))

    def render(self, update: str) -> list[str]:
        condition = f"{self.left.expression} {self.op} {self.right.expression}"
        if self.action == "count":
            return [f"if ({condition}) {update}"]
        return [f"if ({condition}) {self.action};", update]

    def counted_iterations(self, i: int, values: list[int]) -> int:
        count = 0
        for j in values:
            matched = self.allows(i, j)
            if matched and self.action == "break":
                break
            if matched == (self.action == "count"):
                count += 1
        return count
