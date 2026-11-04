"""P0.T9: chấm điểm (test dạng bảng)."""

from __future__ import annotations

import pytest

from ttalloc.data.grading import canonical_key, classify_answer, equivalent, extract_boxed, extract_choice, grade


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so \\boxed{\\frac{1}{2}} done", "\\frac{1}{2}"),  # ngoặc lồng nhau
        ("\\boxed{a_{1}^{2}}", "a_{1}^{2}"),
        ("first \\boxed{1} then \\boxed{2}", "2"),  # nhiều khung: lấy khung cuối
        ("no box here", None),
        ("\\boxed{\\frac{1}{2}", None),  # ngoặc không cân
        ("answer \\boxed 5", "5"),
        ("answer $\\boxed 5$.", "5"),
        ("\\fbox{42}", "42"),
        ("\\boxed {7}", "7"),
        ("", None),
    ],
)
def test_extract_boxed(text: str, expected: str | None) -> None:
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("0.5", "\\frac12", True),
        ("1/2", "\\frac12", True),
        ("\\frac{1}{2}", "0.5", True),
        ("\\dfrac{2}{3}", "\\frac{2}{3}", True),
        ("2", "3", False),
        ("70", "70", True),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})", True),
        ("x^2+1", "1+x^2", True),
        (None, "1", False),
    ],
)
def test_grade_math(pred: str | None, gold: str, ok: bool) -> None:
    assert grade(pred, gold, "math") is ok


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        ("\\boxed{B}", "B"),
        ("so the answer is \\boxed{(C)}", "C"),
        ("Answer: (C)", "C"),
        ("\\boxed{\\text{D}}", "D"),
        ("I think the answer is A.", "A"),
        ("nothing", None),
    ],
)
def test_extract_choice(text: str, letter: str | None) -> None:
    assert extract_choice(text) == letter


def test_grade_choice() -> None:
    assert grade("B", "B", "choice")
    assert grade("(B)", "B", "choice")
    assert not grade("C", "B", "choice")


def test_canonical_key_and_equivalent() -> None:
    assert canonical_key("0.5") == canonical_key("\\frac12") == canonical_key("1/2")
    assert canonical_key("2") != canonical_key("3")
    assert equivalent("0.5", "1/2")
    assert not equivalent("2", "3")


@pytest.mark.parametrize(
    ("ans", "cls"),
    [
        ("42", "integer"),
        ("-7", "integer"),
        ("1,000", "integer"),
        ("\\frac{1}{2}", "fraction"),
        ("3/4", "fraction"),
        ("0.25", "fraction"),
        ("x^2+1", "expression"),
        ("\\sqrt{2}", "expression"),
        ("(3, \\frac{\\pi}{2})", "expression"),
        ("B", "choice"),
        ("(C)", "choice"),
        ("", "other"),
        (None, "other"),
        ("Tuesday", "other"),
    ],
)
def test_classify_answer(ans: str | None, cls: str) -> None:
    assert classify_answer(ans) == cls
