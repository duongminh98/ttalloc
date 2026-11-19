"""Combine the two approved equivalence runs and apply the original P1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttalloc.baseline.checkpoint import read_payload
from ttalloc.baseline.equivalence import equivalence_report
from ttalloc.config import config_hash, load_config

CONFIG = Path("configs/experiments/p1_safe.yaml")


def load_rows(
    run: Path, filename: str, role: str, cfg_hash: str, seeds: tuple[int, int],
) -> tuple[list[dict[str, Any]], set[str]]:
    payload = read_payload(run / filename)
    identity = payload["identity"]
    rows = payload["rows"]
    qids = {row["qid"] for row in rows}
    expected = {(qid, seed) for qid in qids for seed in seeds}
    actual = {(row["qid"], row["seed"]) for row in rows}
    declared = [tuple(key) for key in identity["keys"]]
    if (
        identity["role"] != role or identity["config_hash"] != cfg_hash or len(qids) != 400
        or actual != expected or len(rows) != len(expected)
        or set(declared) != expected or len(declared) != len(expected)
    ):
        raise ValueError(f"invalid equivalence checkpoint identity or rows: {run / filename}")
    return rows, qids


def load_run(
    run: Path, cfg_hash: str, seeds: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    full, full_qids = load_rows(run, "full_checkpoint.json", "equivalence:full", cfg_hash, seeds)
    segmented, segmented_qids = load_rows(run, "seg_checkpoint.json", "equivalence:seg", cfg_hash, seeds)
    if full_qids != segmented_qids:
        raise ValueError("full and segmented checkpoints must cover the same questions")
    return full, segmented, full_qids


def combined_report(first: Path, second: Path) -> dict[str, Any]:
    cfg = load_config(CONFIG)
    cfg_hash = config_hash(cfg)
    full_a, segmented_a, qids_a = load_run(first, cfg_hash, (0, 1))
    full_b, segmented_b, qids_b = load_run(second, cfg_hash, (2, 3))
    if qids_a != qids_b:
        raise ValueError("equivalence runs must cover the same 400 questions")
    report = equivalence_report(
        pd.DataFrame(full_a + full_b), pd.DataFrame(segmented_a + segmented_b),
        np.random.default_rng(cfg.sim.seed), cfg.sim.n_boot,
    )
    return {**report, "seeds": [0, 1, 2, 3], "config_hash": cfg_hash, "source_runs": [first.name, second.name]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path, nargs="?", help="omit to verify the original two-seed run")
    args = parser.parse_args()
    if args.second is None:
        load_run(args.first, config_hash(load_config(CONFIG)), (0, 1))
        print("Verified original equivalence checkpoints for seeds 0–1", flush=True)
        return 0
    report = combined_report(args.first, args.second)
    with (args.second / "equivalence_4seed.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
