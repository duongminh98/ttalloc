"""Regrade saved raw P1 responses using the pinned DEER parser and grader."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEER_COMMIT = "c9dd19fbffa27f841cfe47502d015b63811b4d1b"
SCORER_PACKAGES = ROOT / "third_party" / "deer-scorer"
DEER_REPO = ROOT / "third_party" / "DEER"


def load_scorer(repo: Path, packages: Path) -> tuple[Any, Any]:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != DEER_COMMIT:
        raise ValueError(f"DEER scoring requires {DEER_COMMIT}, found {head}")
    # latex2sympy2 generated its parser with 4.7.2. Load it first in this isolated
    # process, then let SymPy load its own runtime as required by the DEER repo.
    sys.path[:0] = [str(packages / "antlr47"), str(packages), str(repo)]
    legacy = importlib.import_module("latex2sympy2")
    del sys.path[0]
    for name in tuple(sys.modules):
        if name == "antlr4" or name.startswith("antlr4."):
            del sys.modules[name]
    sympy = importlib.import_module("sympy")
    sympy_latex = importlib.import_module("sympy.parsing.latex")
    half = sympy.Rational(1, 2)
    fraction = r"\frac{1}{2}"
    parsed = (legacy.latex2sympy(fraction), sympy_latex.parse_latex(fraction))
    if any(sympy.simplify(value) != half for value in parsed):
        raise RuntimeError("DEER scoring parser smoke check failed")
    return importlib.import_module("utils.parser"), importlib.import_module("utils.grader")


def comparison(rows: list[dict[str, Any]], parser: Any, grader: Any) -> dict[str, Any]:
    labels = []
    for row in rows:
        if not isinstance(row.get("raw_text"), str) or not isinstance(row.get("raw_token_ids"), list):
            raise ValueError(f"missing raw output for {row['qid']} seed {row['seed']}")
        _, gold = parser.parse_ground_truth({"answer": row["gold"]}, row["dataset"])
        answer = parser.extract_answer(row["raw_text"], row["dataset"])
        labels.append({
            "qid": row["qid"],
            "seed": row["seed"],
            "dataset": row["dataset"],
            "grade_correct": row["correct"],
            "deer_extracted": answer,
            "deer_correct": bool(grader.check_is_correct(answer, gold)),
        })
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for label in labels:
        by_seed.setdefault(label["seed"], []).append(label)
    return {
        "deer_commit": DEER_COMMIT,
        "response_source": "raw_text",
        "token_sources": sorted({row["raw_token_source"] for row in rows}),
        "overall": counts(labels),
        "by_seed": {str(seed): counts(part) for seed, part in by_seed.items()},
        "rows": labels,
    }


def counts(labels: list[dict[str, Any]]) -> dict[str, int | float]:
    n = len(labels)
    grade = sum(row["grade_correct"] for row in labels)
    deer = sum(row["deer_correct"] for row in labels)
    return {"n": n, "grade_correct": grade, "grade_accuracy": grade / n,
            "deer_correct": deer, "deer_accuracy": deer / n}


def main() -> None:
    from ttalloc.baseline.checkpoint import read_payload

    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--repo", type=Path, default=DEER_REPO)
    args.add_argument("--packages", type=Path, default=SCORER_PACKAGES)
    args.add_argument("--smoke", action="store_true")
    args.add_argument("checkpoint", type=Path, nargs="?")
    args.add_argument("out", type=Path, nargs="?")
    config = args.parse_args()
    parser, grader = load_scorer(config.repo.resolve(), config.packages.resolve())
    if config.smoke:
        if not grader.check_is_correct(parser.extract_answer(r"\boxed{1/2}"), r"\frac{1}{2}"):
            raise RuntimeError("DEER scoring smoke check failed")
        print(f"DEER_SCORER_OK {DEER_COMMIT}", flush=True)
        return
    if config.checkpoint is None or config.out is None:
        args.error("checkpoint and output paths are required")
    report = comparison(read_payload(config.checkpoint)["rows"], parser, grader)
    tmp = config.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(config.out)
    print(f"P1_BASELINE_SCORES {json.dumps(report['overall'])}", flush=True)


if __name__ == "__main__":
    main()
