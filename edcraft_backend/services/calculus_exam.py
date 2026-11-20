"""Read the directive and mathematical shape of a single calculus exam selection.

These recognizers accept a small grammar and known Poppler fraction layouts.
They never evaluate the user's expression or parse an arbitrary math language.
"""

import re
from fractions import Fraction

_TITLE = r"(?:(?:(?:question|q)?\d+(?:[.):]|(?=\())(?:[\[(]\d+marks[\])])?)?(?:\([a-z]\))?)"
_JUSTIFY = r"[.?!]?(?:(?:justify|explain)(?:youranswer|yourreasoning)[.?!]?)?"
_SUM = r"(?:(?:sum_?\(n=1\)\^?(?:∞|infinity)|[∑Σσ]_?\(n=1\)\^?∞))?"
_CONVERGES = r"(?:convergesordiverges|isconvergentordivergent)"
_CONVERGENCE = r"(?:determine|find|test|decide)(?:the)?convergenceof"
_INT = r"[1-9]\d{0,2}"


def _question(exam: str) -> str:
    return re.sub("^" + _TITLE, "", exam.replace(" ", ""), count=1)


def _convergence_task(text: str, term: str, method: str) -> re.Match[str] | None:
    using_method = rf"(?:usethe{method}to)?"
    statement = rf"(?:theseries)?{_SUM}{term}"
    return re.fullmatch(
        rf"{using_method}{_CONVERGENCE}{statement}{_JUSTIFY}", text
    ) or re.fullmatch(
        rf"{using_method}(?:determine|decide|state|test)(?:if|whether)"
        rf"{statement}{_CONVERGES}{_JUSTIFY}",
        text,
    )


def factorial(exam: str) -> bool:
    selection = _question(exam)
    match = _convergence_task(
        selection,
        r"(?:[1-9]\d?\^n\*?)?n!\*?\(([2-8])n\)!/\(([3-9])n\)!",
        "ratiotest",
    ) or re.fullmatch(
        r"foreachofthefollowingseries,(?:determine|decide)(?:whether|if)"
        r"it(?:convergesordiverges|isconvergentordivergent)\."
        r"(?:justify|explain)youranswers\.∞xn!\(([2-8])n\)!\([a-z]\),?n=1\(([3-9])n\)!",
        selection,
    )
    return bool(match and int(match[2]) == int(match[1]) + 1)


def power_comparison(exam: str) -> bool:
    selection = _question(exam)
    # Fraction exponents need explicit parentheses to distinguish n^(2/3) from n^2/3.
    exponent = r"(?:\(\d{1,2}/[1-9]\d?\)|[1-9]\d?)"
    match = _convergence_task(
        selection,
        rf"n\^({exponent})/\(({_INT})\*?n\^({exponent})-({_INT})\)",
        r"(?:limit)?comparisontest",
    )
    if match:
        numerator, coefficient, denominator, offset = match.groups()
        return int(coefficient) > int(offset) and 0 < Fraction(numerator.strip("()")) < Fraction(
            denominator.strip("()")
        )
    # Poppler's layout places the top and bottom of the superscript fractions
    # on opposite sides of the directive. Keep a dedicated, bounded reader.
    match = re.fullmatch(
        rf"∞([1-9])xn([2-9])\([a-z]\)(?:determine|decide)(?:if|whether)"
        rf"theseries([1-9]){_CONVERGES}\.justifyn=1({_INT})n([2-9])-({_INT})youranswer\.",
        selection,
    )
    if not match:
        return False
    pn, qn, pd, a, qd, b = map(int, match.groups())
    return a > b and Fraction(pn, qn) < Fraction(pd, qd)


_NUMBER = r"-?\d{1,3}(?:/[1-9]\d{0,2}|\.\d{1,3})?"
_POINT = rf"\(({_NUMBER}),({_NUMBER})\)"
_DIRECTION = (
    r"(?:find|determine)(?:the)?unit(?:vector|direction)"
    r"(?:thatmaximizesthedirectionalderivative"
    r"|forwhichthedirectionalderivativeis(?:maximal|largest)"
    r"|inthedirectionof(?:greatestincrease|steepestascent))"
    r"|(?:find|determine)(?:the)?directionof(?:steepestascent|greatestincrease)"
)


def _nonzero_gradient(coefficient: str, x: str, y: str) -> bool:
    a = int(coefficient or "1")
    xv, yv = Fraction(x), Fraction(y)
    return xv != 0 and yv > 0 and not (a * xv > 0 and (a * xv) ** 2 * yv == 4)


def steepest_ascent(exam: str) -> bool:
    if exam.startswith("√"):
        selection = _question(exam.removeprefix("√"))
        match = re.fullmatch(
            rf"letf\(x,y\)=ln\(x2y\)-({_INT})?xy\."
            rf"(?:\([a-z]\)find∇f\(x,y\)\.)?\([a-z]\)"
            rf"(?:find|determine)theunitvectorvsuchthatthedirectionalderivativedvf"
            rf"{_POINT}atthepoint{_POINT}attains(?:its|the)maximum(?:value)?[.!?]?",
            selection,
        )
        if not match:
            return False
        coefficient, x, y, repeated_x, repeated_y = match.groups()
        if (x, y) != (repeated_x, repeated_y):
            return False
        # The inspected document flattens 1/4 as "14". Do not guess at other
        # flattened, multi-digit numbers where the two-dimensional layout was lost.
        if (coefficient, x, y) == (None, "2", "14"):
            y = "1/4"
        elif y.isdigit() and len(y) > 1:
            return False
        return _nonzero_gradient(coefficient, x, y)
    selection = _question(exam)
    setup = re.match(
        rf"(?:let|given)?f\(x,y\)=ln\(x\^?2\*?y\)-({_INT})?\*?x\*?"
        r"(?:sqrt\(y\)|√\(?y\)?)\.",
        selection,
    )
    if not setup:
        return False
    directive = selection[setup.end() :]
    match = re.fullmatch(rf"(?:{_DIRECTION})at{_POINT}{_JUSTIFY}", directive) or re.fullmatch(
        rf"at{_POINT},?(?:{_DIRECTION}){_JUSTIFY}", directive
    )
    return bool(match and _nonzero_gradient(setup[1], match[1], match[2]))


_STATIONARY = r"(?:critical|stationary)points"
_MONOMIAL = rf"([+-])({_INT})?\*?([xy](?:\^?[24])?)"
_POLYNOMIAL = rf"([+-]?(?:{_INT})?\*?x\^?4(?:[+-](?:{_INT})?\*?[xy](?:\^?[24])?){{3}})"


def critical_points(exam: str) -> bool:
    selection = _question(exam)
    match = re.fullmatch(
        rf"(?:let|given)?f\(x,y\)={_POLYNOMIAL}\."
        rf"(?:find|locate|determine)andclassify(?:all)?(?:the)?{_STATIONARY}off{_JUSTIFY}",
        selection,
    ) or re.fullmatch(
        rf"(?:find|locate|determine)(?:all)?the{_STATIONARY}ofthefunction"
        rf"f\(x,y\)={_POLYNOMIAL}(?:\.|and)"
        r"classifyeachoneasalocalmaximumpoint,alocalminimumpoint(?:,)?orasaddlepointforf"
        rf"{_JUSTIFY}",
        selection,
    )
    if not match:
        return False
    polynomial = match[1].replace("^", "")
    monomials = re.findall(_MONOMIAL, polynomial if polynomial[0] in "+-" else "+" + polynomial)
    signs = {variable: sign for sign, _, variable in monomials}
    return signs.keys() == {"x4", "x2", "y2", "y"} and signs["x4"] != signs["x2"]
