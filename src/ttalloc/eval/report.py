"""Bảng và hình chỉ từ dữ liệu trong `runs/` (PLAN P2.T18). Mỗi bảng ghi ba dạng: Markdown, CSV, LaTeX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttalloc.eval.metrics import auc_over_alpha, by_difficulty, cell_table, tokens_saved_vs
from ttalloc.eval.stats import compare_methods


def load_runs(run_dirs: list[Path]) -> pd.DataFrame:
    parts = []
    for d in run_dirs:
        pq = pd.read_parquet(d / "per_question.parquet")
        parts.append(pq.assign(run=d.name))
    if not parts:
        raise ValueError("không có thư mục chạy nào")
    return pd.concat(parts, ignore_index=True)


def write_table(df: pd.DataFrame, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.csv", out / f"{name}.md", out / f"{name}.tex"]
    df.to_csv(paths[0], index=False)
    paths[1].write_text(_markdown(df), encoding="utf-8")
    paths[2].write_text(df.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    return paths


def _markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def comparison_table(pq: pd.DataFrame, ref: str, rng: np.random.Generator, n_boot: int, level: float) -> pd.DataFrame:
    rows = []
    for (split, batch), g in pq.groupby(["split", "batch"]):
        for m in sorted(set(g.method) - {ref}):
            for a in sorted(g.alpha.unique()):
                try:
                    c = compare_methods(g, m, ref, a, rng, n_boot, level)
                except ValueError:
                    continue
                rows.append(
                    {
                        "split": split,
                        "batch": batch,
                        "method": m,
                        "vs": ref,
                        "alpha": a,
                        "diff": c["diff"],
                        "ci_lo": c["lo"],
                        "ci_hi": c["hi"],
                    }
                )
    return pd.DataFrame(rows)


def auc_table(cells: pd.DataFrame, ref: str = "B2") -> pd.DataFrame:
    rows = []
    for (split, batch, method), g in cells.groupby(["split", "batch", "method"]):
        row: dict[str, Any] = {
            "split": split,
            "batch": batch,
            "method": method,
            "auc": auc_over_alpha(g.alpha.to_numpy(), g.accuracy.to_numpy()),
        }
        r = cells[(cells.split == split) & (cells.batch == batch) & (cells.method == ref)]
        if method != ref and len(r) >= 2:
            saved = tokens_saved_vs(
                r.tokens.to_numpy(), r.accuracy.to_numpy(), g.tokens.to_numpy(), g.accuracy.to_numpy()
            )
            row["tokens_saved_mean"] = float(np.nanmean(saved)) if np.isfinite(saved).any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy(cells: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for (split, batch), g in cells.groupby(["split", "batch"]):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for m, gm in g.groupby("method"):
            gm = gm.sort_values("alpha")
            ax.errorbar(gm.alpha, gm.accuracy, yerr=gm.accuracy_std.fillna(0), marker="o", capsize=3, label=m)
        ax.set_xlabel("alpha (B / C_full)")
        ax.set_ylabel("độ chính xác")
        ax.set_title(f"{split} / {batch}")
        ax.grid(alpha=0.3)
        ax.legend()
        p = out / f"accuracy_{split}_{batch}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_reliability(est_eval: dict[str, Any], out: Path) -> Path | None:
    rel = est_eval.get("reliability")
    if not rel:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    for h, rows in rel.items():
        df = pd.DataFrame(rows)
        ax.plot(df.p_mean, df.y_mean, marker=".", label=f"h={h}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("tỉ lệ đúng")
    ax.legend(fontsize=7)
    p = out / "reliability.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_allocation(diff: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for (split, batch, alpha), g in diff.groupby(["split", "batch", "alpha"]):
        piv = g.pivot_table(index="diff_bin", columns="method", values="tokens")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        piv.plot.bar(ax=ax)
        ax.set_ylabel("token trung bình mỗi câu")
        ax.set_title(f"{split} / {batch} / alpha={alpha}")
        p = out / f"allocation_{split}_{batch}_a{alpha}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def build_report(
    run_dirs: list[Path],
    out: Path,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
    ref: str = "B2",
    nat_acc: dict[str, float] | None = None,
    est_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pq = load_runs(run_dirs)
    out.mkdir(parents=True, exist_ok=True)
    cells = cell_table(pq)
    files = write_table(cells, out, "accuracy")
    files += write_table(auc_table(cells, ref), out, "auc")
    if ref in set(pq.method):
        files += write_table(comparison_table(pq, ref, rng, n_boot, level), out, f"diff_vs_{ref}")
    diff = by_difficulty(pq, nat_acc)
    files += write_table(diff, out, "by_difficulty")
    figs = plot_accuracy(cells, out) + plot_allocation(diff, out)
    if est_eval:
        r = plot_reliability(est_eval, out)
        if r:
            figs.append(r)
    manifest = {
        "runs": [str(d) for d in run_dirs],
        "tables": [str(p) for p in files],
        "figures": [str(p) for p in figs],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
