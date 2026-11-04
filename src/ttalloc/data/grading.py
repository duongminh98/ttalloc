"""Trích đáp án và chấm điểm (PLAN P0.T9)."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Literal

import sympy

# math_verify dùng signal.alarm để giới hạn thời gian; Windows không có signal.alarm và cách thay thế
# bằng multiprocessing rất chậm, nên trên Windows (máy phát triển) bỏ giới hạn thời gian.
_TIMEOUT: int | None = None if os.name == "nt" else 5

AnswerClass = Literal["integer", "fraction", "expression", "choice", "other"]

_BOX_RE = re.compile(r"\\(?:boxed|fbox)")
_CHOICE_LETTERS = "ABCDEFGHIJ"


def extract_boxed(text: str | None) -> str | None:
    """Nội dung của khung `\\boxed{...}` (hoặc `\\fbox{...}`) cuối cùng; xử lý ngoặc lồng nhau và `\\boxed 5`.

    Trả về None nếu không có khung hoặc khung cuối không đóng ngoặc.
    """
    if not text:
        return None
    matches = list(_BOX_RE.finditer(text))
    if not matches:
        return None
    i = matches[-1].end()
    n = len(text)
    while i < n and text[i] == " ":
        i += 1
    if i >= n:
        return None
    if text[i] != "{":
        # Dạng `\boxed 5`: lấy tới khoảng trắng hoặc dấu `$` kế tiếp.
        m = re.match(r"[^\s$]+", text[i:])
        return m.group(0).rstrip(".") if m else None
    depth = 0
    for j in range(i, n):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


_CHOICE_PATTERNS = [
    re.compile(r"^\s*\(?\s*([A-J])\s*\)?\s*[.:)]?\s*$"),
    re.compile(r"^\s*\\text\{\s*\(?\s*([A-J])\s*\)?\s*\}\s*$"),
    re.compile(r"^\s*\\textbf\{\s*\(?\s*([A-J])\s*\)?\s*\}\s*$"),
    re.compile(r"^\s*\(?\s*([A-J])\s*\)\s+\S"),
]
_CHOICE_IN_TEXT = re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*:?\s*\(?\s*([A-J])\b\s*\)?", re.IGNORECASE)


def extract_choice(text: str | None) -> str | None:
    """Chữ cái của lựa chọn: ưu tiên nội dung khung cuối, sau đó các mẫu như `Answer: (C)`."""
    if not text:
        return None
    boxed = extract_boxed(text)
    for cand in [boxed] if boxed is not None else []:
        for pat in _CHOICE_PATTERNS:
            m = pat.match(cand)
            if m:
                return m.group(1).upper()
    for pat in _CHOICE_PATTERNS[:3]:
        m = pat.match(text)
        if m:
            return m.group(1).upper()
    found = _CHOICE_IN_TEXT.findall(text)
    return found[-1].upper() if found else None


@lru_cache(maxsize=65536)
def _parse(ans: str) -> tuple[Any, ...]:
    from math_verify import parse

    s = ans.strip()
    if not (s.startswith("$") and s.endswith("$")):
        s = f"${s}$"
    return tuple(parse(s, parsing_timeout=_TIMEOUT))


def _verify(gold: tuple[Any, ...], pred: tuple[Any, ...]) -> bool:
    from math_verify import verify

    if not gold or not pred:
        return False
    return bool(verify(list(gold), list(pred), timeout_seconds=_TIMEOUT))


_NORM_DROP = re.compile(r"\\left|\\right|\\!|\\,|\\;|\\:|\\displaystyle|\$|\s")


def _normalize_text(ans: str) -> str:
    s = _NORM_DROP.sub("", ans)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.rstrip(".")


def grade(pred: str | None, gold: str, answer_type: Literal["math", "choice"]) -> bool:
    """Chấm một đáp án đã trích (nội dung khung) so với đáp án chuẩn."""
    if pred is None:
        return False
    if answer_type == "choice":
        letter = extract_choice(pred) if len(pred.strip()) > 1 else pred.strip().upper()
        return letter is not None and letter == gold.strip().upper()
    if _normalize_text(pred) == _normalize_text(gold):
        return True
    return _verify(_parse(gold), _parse(pred))


@lru_cache(maxsize=65536)
def canonical_key(ans: str) -> str:
    """Khóa chuẩn hóa để nhóm các đáp án tương đương (ví dụ 0.5, \\frac12, 1/2 cùng một khóa)."""
    letter = extract_choice(ans) if classify_answer(ans) == "choice" else None
    if letter is not None:
        return f"choice:{letter}"
    parsed = _parse(ans)
    if parsed:
        expr = parsed[0]
        if isinstance(expr, sympy.Basic):
            if expr.is_Number:
                expr = sympy.nsimplify(expr, rational=True)
            return f"expr:{sympy.srepr(expr)}"
    return f"str:{_normalize_text(ans)}"


@lru_cache(maxsize=65536)
def equivalent(a: str, b: str) -> bool:
    if canonical_key(a) == canonical_key(b):
        return True
    return _verify(_parse(a), _parse(b))


_INT_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+$|^[+-]?\d+$")
_FRAC_RE = re.compile(r"^[+-]?(?:\\[dt]?frac\{\s*[+-]?\d+\s*\}\{\s*\d+\s*\}|\\[dt]?frac\d\d|\d+\s*/\s*\d+|\d*\.\d+)$")
_CHOICE_ONLY_RE = re.compile(r"^\(?[A-J]\)?$|^\\text\{\(?[A-J]\)?\}$")
_MATHY_RE = re.compile(r"[\\^_=+\-*/()\[\]{}<>]|\d|^[a-z]$")


def classify_answer(ans: str | None) -> AnswerClass:
    if ans is None:
        return "other"
    s = _normalize_text(ans)
    if not s:
        return "other"
    if _CHOICE_ONLY_RE.match(s):
        return "choice"
    if _INT_RE.match(s):
        return "integer"
    if _FRAC_RE.match(s):
        return "fraction"
    if re.fullmatch(r"[A-Za-z]{2,}", s):
        return "other"
    if _MATHY_RE.search(s):
        return "expression"
    return "other"
