"""Compare short and natural answers on the same segmented val reasoning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from merge_equivalence import CONFIG, load_run
from p1_run import save

from ttalloc.cli import _bounds_extra, _live_source, _questions, _run_dir, _verify_splits, _vllm
from ttalloc.config import ExperimentCfg, config_hash, load_config
from ttalloc.context import model_context, prompt_ids_for
from ttalloc.data.grading import grade
from ttalloc.engine.live_source import LiveSource, extract_answer
from ttalloc.logging.studies import walk_chains
from ttalloc.types import GenRequest, SegmentResult, TraceKey


def select_cases(first: Path, second: Path, cfg: ExperimentCfg) -> dict[str, str]:
    cfg_hash = config_hash(cfg)
    full_a, seg_a, qids_a = load_run(first, cfg_hash, (0, 1))
    full_b, seg_b, qids_b = load_run(second, cfg_hash, (2, 3))
    if qids_a != qids_b:
        raise ValueError("diagnostic source runs must cover the same questions")
    full = pd.DataFrame(full_a + full_b).groupby("qid").agg(
        correct=("correct", "sum"), length=("think_tokens", "mean"),
    )
    segmented = pd.DataFrame(seg_a + seg_b).groupby("qid")["correct"].sum()
    focus = [str(qid) for qid in full.index if full.at[qid, "correct"] == 4 and segmented.at[qid] == 0]
    if len(focus) != 6 or any(not qid.startswith("math_train/") for qid in focus):
        raise ValueError("expected six all-seed MATH disagreements")
    available = {str(qid) for qid in full.index if str(qid).startswith("math_train/")
                 and full.at[qid, "correct"] == segmented.at[qid] == 4}
    controls = []
    for qid in focus:
        match = min(available, key=lambda other: (abs(full.at[other, "length"] - full.at[qid, "length"]), other))
        controls.append(match)
        available.remove(match)
    return dict.fromkeys(focus, "discordant") | dict.fromkeys(controls, "control")


def diagnose(source: LiveSource, cases: dict[str, str], cfg: ExperimentCfg, output: Path) -> dict[str, Any]:
    keys = [TraceKey(qid, seed) for seed in range(4) for qid in cases]
    rows: list[dict[str, Any]] = []

    def terminal(segments: list[SegmentResult]) -> None:
        ended = [segment.key for segment in segments]
        short = source.answer(ended, finalize=False, mode="short")
        natural = []
        requests = []
        for segment in segments:
            if segment.kind == "natural_end":
                natural.append(segment.key)
                chain = source.chains[segment.key]
                requests.append(GenRequest(prompt_ids=tuple(chain.prompt_ids + chain.chain[:chain.pos]),
                                           max_tokens=source.r_ans("full"), temperature=0.0))
        continuations = dict(zip(natural, source.gen.generate(requests) if requests else [], strict=True))
        for segment, answer in zip(segments, short, strict=True):
            k = segment.key
            q = source.questions[k.qid]
            result = continuations.get(k)
            text = source.codec.decode(list(result.token_ids)) if result is not None else None
            extracted = extract_answer(text, q.answer_type) if text is not None else None
            chain = source.chains[k].chain[:segment.pos]
            prefix_hash = hashlib.sha256(json.dumps(chain).encode()).hexdigest()
            rows.append({
                "qid": k.qid, "seed": k.seed, "group": cases[k.qid], "end_kind": segment.kind,
                "thinking_tokens": segment.gen_end, "chain_sha256": prefix_hash,
                "gold": q.gold, "short_prompt": source.prompts.answer("short"), "short_raw": answer.text,
                "short_extracted": answer.extracted, "short_correct": answer.correct,
                "natural_raw": text, "natural_extracted": extracted,
                "natural_correct": grade(extracted, q.gold, q.answer_type) if result is not None else None,
                "natural_finish_reason": result.finish_reason if result is not None else None,
            })
        source.drop(ended)
        save(output, {"expected": len(keys), "rows": rows})

    for seed in range(4):
        walk_chains(source, [key for key in keys if key.seed == seed], cfg,
                    lambda segments: source.probe([segment.key for segment in segments]), terminal)
    completed = {(r["qid"], r["seed"]) for r in rows}
    missing = [(key.qid, key.seed) for key in keys if (key.qid, key.seed) not in completed]
    groups = {}
    for group in ("discordant", "control"):
        group_rows = [row for row in rows if row["group"] == group]
        groups[group] = {
            "n": len(group_rows),
            "natural_available": sum(r["natural_raw"] is not None for r in group_rows),
            "short_unextracted": sum(r["short_extracted"] is None for r in group_rows),
            "short_wrong_natural_correct": sum(
                not r["short_correct"] and r["natural_correct"] is True for r in group_rows
            ),
            "short_correct_natural_wrong": sum(
                bool(r["short_correct"]) and r["natural_correct"] is False for r in group_rows
            ),
        }
    return {"completed": len(rows), "missing": missing, "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--check-selection", action="store_true")
    args = parser.parse_args()
    cfg = load_config(CONFIG)
    cases = select_cases(args.first, args.second, cfg)
    if args.check_selection:
        print(f"Verified diagnostic selection: {len(cases)} val cases × 4 seeds", flush=True)
        return
    _verify_splits()
    questions = [q for q in _questions("val", ["math_train"]) if q.qid in cases]
    if {q.qid for q in questions} != cases.keys():
        raise ValueError("diagnostic questions must belong to the locked val split")
    ctx = model_context(cfg)
    prompts = prompt_ids_for(questions, ctx)
    run = _run_dir("equivalence-diagnostic", cfg)
    gen = _vllm(ctx, max(map(len, prompts.values())) + cfg.log.l_log + _bounds_extra(ctx))
    source = _live_source(ctx, gen, questions, prompts, "short")
    report = diagnose(source, cases, cfg, run / "diagnostic_checkpoint.json")
    save(run / "diagnostic_summary.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
