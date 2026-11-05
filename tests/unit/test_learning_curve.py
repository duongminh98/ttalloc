"""ADR-015: luật quyết định cỡ train theo learning curve."""

from __future__ import annotations

import pytest

from ttalloc.estimator.learning_curve import decide, nested_subsets


def _row(n: int, auroc: float, ll: float) -> dict[str, float]:
    return {"n_questions": n, "auroc_q0": auroc, "mean_log_loss": ll}


def test_nested_subsets_are_prefixes() -> None:
    order = [f"q{i}" for i in range(10)]
    s = nested_subsets(order, [5, 2])
    assert s[2] == {"q0", "q1"} and s[2] < s[5]
    with pytest.raises(ValueError):
        nested_subsets(order, [11])


def test_decide_extends_only_on_clear_gain() -> None:
    flat = decide([_row(250, 0.800, 0.400), _row(500, 0.805, 0.398)], logged=500)
    assert not flat["extend"] and flat["train_limit"] == 500
    by_auroc = decide([_row(250, 0.80, 0.40), _row(500, 0.82, 0.40)], logged=500)
    assert by_auroc["extend"] and by_auroc["train_limit"] == 1000
    by_loss = decide([_row(500, 0.80, 0.40), _row(1000, 0.80, 0.39)], logged=1000)
    assert by_loss["extend"] and by_loss["train_limit"] == 1500  # không quá cỡ tập train
    with pytest.raises(ValueError):
        decide([_row(250, 0.8, 0.4)], logged=250)
