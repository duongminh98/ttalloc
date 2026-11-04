"""Cổng quyết định G1, G2, G3 (PLAN mục 7.4, P2.T19, P4.T4). Lệnh `gate` chỉ báo đạt hay không và số liệu."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ttalloc.eval.stats import acceptance_band, bonferroni_level, compare_methods

ALPHA_MAX = 0.5


def _crit(name: str, ok: bool, **values: Any) -> dict[str, Any]:
    return {"criterion": name, "pass": bool(ok), **values}


def gate_g1(
    est_report: dict[str, Any],
    pq: pd.DataFrame,
    validate_report: dict[str, Any],
    rng: np.random.Generator,
    n_boot: int = 2000,
    k: int = 4,
    b2: str = "B2",
    m3: str = "M3",
) -> dict[str, Any]:
    crits = []
    auroc = est_report.get("auroc_q0")
    max_ece = est_report.get("max_ece")
    sp = (est_report.get("gain_spearman") or {}).get("pooled")
    crits.append(_crit("a1_auroc_q0>=0.75", auroc is not None and auroc >= 0.75, value=auroc))
    crits.append(
        _crit(
            "a2_ece<=0.05_all_h",
            max_ece is not None and max_ece <= 0.05,
            value=max_ece,
            per_h={h: v["ece"] for h, v in est_report.get("per_h", {}).items()},
        )
    )
    crits.append(_crit("a3_gain_spearman>=0.3", sp is not None and sp >= 0.3, value=sp))
    comps = []
    level = bonferroni_level(k)
    for (split, batch), g in pq.groupby(["split", "batch"]):
        for a in sorted(x for x in g.alpha.unique() if x <= ALPHA_MAX):
            c = compare_methods(g, m3, b2, a, rng, n_boot, level)
            comps.append(
                {"split": split, "batch": batch, "alpha": float(a), **c, "pass": c["diff"] >= 0.02 and c["lo"] > 0}
            )
    main = [c for c in comps if c["batch"] == "all"] or comps
    crits.append(
        _crit("b_M3_beats_B2_by_2pp_some_alpha<=0.5", any(c["pass"] for c in main), level=level, comparisons=comps)
    )
    pr = validate_report.get("pass_rate")
    crits.append(_crit("c_validate_logs_pass_rate>=0.95", pr is not None and pr >= 0.95, value=pr))
    return {"gate": "G1", "pass": all(c["pass"] for c in crits), "criteria": crits}


def gate_g2(
    pq: pd.DataFrame, rng: np.random.Generator, n_boot: int = 2000, k: int = 4, b2: str = "B2", m2: str = "M2"
) -> dict[str, Any]:
    level = bonferroni_level(k)
    comps = []
    for (split, batch), g in pq.groupby(["split", "batch"]):
        for a in sorted(x for x in g.alpha.unique() if x <= ALPHA_MAX):
            c = compare_methods(g, m2, b2, a, rng, n_boot, level)
            comps.append({"split": split, "batch": batch, "alpha": float(a), **c, "pass": c["lo"] > 0})
    crit = _crit(
        "M2_beats_B2_some_alpha<=0.5_bonferroni", any(c["pass"] for c in comps), level=level, comparisons=comps
    )
    return {"gate": "G2", "pass": crit["pass"], "criteria": [crit]}


def gate_g3(
    online_dirs: list[Path], rng: np.random.Generator, n_sets: int = 2000, max_gap: float = 0.03
) -> dict[str, Any]:
    cells = []
    sim_ref = []
    for d in online_dirs:
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        cells += s["cells"]
        sim_ref.append(pd.read_parquet(d / "sim_reference.parquet"))
    ref = pd.concat(sim_ref, ignore_index=True)
    df = pd.DataFrame(cells)
    exceeded = bool(df.exceeded.any()) if "exceeded" in df else True
    configs = []
    for key, g in df.groupby(["alpha", "method"]):
        alpha, method = cast(tuple[Any, Any], key)
        r = ref[(ref.alpha == alpha) & (ref.method == method)].accuracy.to_numpy()
        lo, hi = acceptance_band(r, len(g), rng, n_sets)
        live = float(g.accuracy.mean())
        configs.append(
            {
                "alpha": float(alpha),
                "method": method,
                "n_seeds": len(g),
                "live_mean": live,
                "sim_mean": float(r.mean()),
                "band": [lo, hi],
                "in_band": lo <= live <= hi,
                "gap": abs(live - float(r.mean())),
                "gap_ok": abs(live - float(r.mean())) <= max_gap,
            }
        )
    k = len(configs)
    outside = sum(not c["in_band"] for c in configs)
    allowed = 0 if k == 2 else max(1, int(np.floor(0.1 * k)))
    crits = [
        _crit("a_no_run_exceeds_B", not exceeded),
        _crit("b_in_acceptance_band", outside <= allowed, outside=outside, allowed=allowed, k=k),
        _crit("c_live_vs_sim_gap<=3pp", all(c["gap_ok"] for c in configs)),
    ]
    return {"gate": "G3", "pass": all(c["pass"] for c in crits), "criteria": crits, "configs": configs}


def gate_markdown(res: dict[str, Any]) -> str:
    lines = [
        f"# Cổng {res['gate']}: {'ĐẠT' if res['pass'] else 'KHÔNG ĐẠT'}",
        "",
        "Lệnh này chỉ báo số liệu; con người quyết định đi tiếp (PLAN mục 7.4).",
        "",
        "| Tiêu chí | Đạt | Giá trị |",
        "|---|---|---|",
    ]
    for c in res["criteria"]:
        val = c.get("value", c.get("outside", ""))
        lines.append(f"| {c['criterion']} | {'có' if c['pass'] else 'không'} | {val} |")
    for c in res["criteria"]:
        if c.get("comparisons"):
            lines += [
                "",
                f"## {c['criterion']} (mức tin cậy {c['level']:.4f})",
                "",
                "| split | lô | alpha | chênh lệch | CI | đạt |",
                "|---|---|---|---|---|---|",
            ]
            for x in c["comparisons"]:
                lines.append(
                    f"| {x['split']} | {x['batch']} | {x['alpha']} | {x['diff']:+.4f} | "
                    f"[{x['lo']:+.4f}, {x['hi']:+.4f}] | {'có' if x['pass'] else 'không'} |"
                )
    if res.get("configs"):
        lines += ["", "| alpha | phương pháp | chạy thật | mô phỏng | dải | trong dải |", "|---|---|---|---|---|---|"]
        for c in res["configs"]:
            lines.append(
                f"| {c['alpha']} | {c['method']} | {c['live_mean']:.4f} | {c['sim_mean']:.4f} | "
                f"[{c['band'][0]:.4f}, {c['band'][1]:.4f}] | {'có' if c['in_band'] else 'không'} |"
            )
    return "\n".join(lines) + "\n"
