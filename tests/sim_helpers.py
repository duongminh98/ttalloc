"""Dựng môi trường mô phỏng trên dữ liệu tổng hợp cho test (không cần GPU, mạng hay tokenizer)."""

from __future__ import annotations

from typing import Any

import numpy as np

from synth import make_logs
from ttalloc.config import ExperimentCfg
from ttalloc.features.events import LoggedChain, chains_from_tables
from ttalloc.sim.env import SimEnv
from ttalloc.types import Question, TraceKey

C_ANS_MAX = 3 + 40  # synth: answer_prompt_tokens = 3, answer_output_tokens < 40
C_PROBE_MAX = 6 + 12  # synth: probe_prompt_tokens = 6, probe_output_tokens < 12


def synth_cfg(**over: Any) -> ExperimentCfg:
    base: dict[str, Any] = {
        "name": "synth",
        "log": {"l_log": 4096},
        "policy": {"mu_min": 0.0, "horizons": [512, 1024, 2048]},
        "sim": {"draws": 4, "calib_draws": 4},
        "estimator": {"h_train": [256, 512, 1024, 2048], "n_folds": 3, "max_rounds": 60, "early_stopping": 10},
    }
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return ExperimentCfg.model_validate(base)


def synth_chains(n: int, seeds: list[int], seed: int = 0, **knobs: Any) -> dict[TraceKey, LoggedChain]:
    tr, pt = make_logs(n, seeds, np.random.default_rng(seed), **knobs)
    return chains_from_tables(tr, pt)


def synth_questions(n: int) -> dict[str, Question]:
    return {f"synth/{i}": Question(f"synth/{i}", "synth", f"p{i}", None, "gold", "math", 1 + i % 5) for i in range(n)}


def synth_env(
    n: int = 12, seeds: list[int] | None = None, cfg: ExperimentCfg | None = None, seed: int = 0, **kw: Any
) -> SimEnv:
    seeds = seeds if seeds is not None else [0, 1, 2]
    chains = synth_chains(n, seeds, seed)
    lens = [c.chain_len for c in chains.values()]
    return SimEnv(
        cfg=cfg or synth_cfg(),
        split="val",
        chains=chains,
        questions=synth_questions(n),
        c_ans_max=C_ANS_MAX,
        c_probe_max=C_PROBE_MAX,
        ref_median=float(np.median(lens)),
        **kw,
    )
