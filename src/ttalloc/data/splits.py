"""Chia tập, khử trùng lặp và khóa danh sách ID (PLAN P0.T7, mục 4.1)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ttalloc.config import CONFIGS_DIR, REPO_ROOT
from ttalloc.types import Question

SPLITS_DIR = REPO_ROOT / "splits"
LEVELS = (1, 2, 3, 4, 5)


class SplitsTamperedError(RuntimeError):
    pass


# ---------------------------------------------------------------- khử trùng lặp

_PRESENTATION = re.compile(r"\\(?:left|right|displaystyle|textstyle|quad|qquad|,|!|;|:)|\$|\\\[|\\\]|\\\(|\\\)|~")
_ASY = re.compile(r"\[asy\].*?\[/asy\]", re.S)
# n-gram tính trên từ (chữ và số), bỏ ký hiệu: nếu tính cả ký hiệu LaTeX thì những cụm phổ biến như
# `\frac{1}{2}` cũng đủ 13 token và gây khớp nhầm hàng loạt (chạy thử trên dữ liệu thật, 03/11/2026; ADR-009).
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Chữ thường, bỏ mã hình vẽ [asy], ký hiệu LaTeX trình bày và khoảng trắng thừa."""
    s = _PRESENTATION.sub(" ", _ASY.sub(" ", text.lower()))
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return re.sub(r"\s+", " ", s).strip()


def _ngrams(norm: str, n: int) -> set[str]:
    toks = _TOKEN.findall(norm)
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class DedupIndex:
    """So khớp chính xác sau chuẩn hóa, rồi so khớp n-gram.

    Hai câu trùng theo n-gram khi số n-gram chung chiếm ít nhất `min_overlap` số n-gram của câu ngắn hơn.
    Một vài n-gram chung thường chỉ là câu mẫu ("where m and n are relatively prime positive integers"), không
    phải cùng một đề (ADR-009).
    """

    n: int = 13
    min_overlap: float = 0.5
    exact: dict[str, str] = field(default_factory=dict)
    grams: dict[str, list[str]] = field(default_factory=dict)
    n_grams: dict[str, int] = field(default_factory=dict)

    def add(self, q: Question) -> None:
        norm = normalize_text(q.problem)
        self.exact.setdefault(re.sub(r"\s", "", norm), q.qid)
        grams = _ngrams(norm, self.n)
        self.n_grams[q.qid] = len(grams)
        for g in grams:
            self.grams.setdefault(g, []).append(q.qid)

    def match(self, q: Question) -> str | None:
        """qid của câu đã có mà q trùng với, hoặc None."""
        norm = normalize_text(q.problem)
        hit = self.exact.get(re.sub(r"\s", "", norm))
        if hit is not None:
            return hit
        grams = _ngrams(norm, self.n)
        shared: dict[str, int] = {}
        for g in grams:
            for other in self.grams.get(g, ()):
                shared[other] = shared.get(other, 0) + 1
        best: tuple[float, str] | None = None
        for other, k in shared.items():
            ratio = k / max(1, min(len(grams), self.n_grams[other]))
            if ratio >= self.min_overlap and (best is None or ratio > best[0]):
                best = (ratio, other)
        return best[1] if best else None


# ---------------------------------------------------------------- dựng các tập


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    n: int | None = None  # None: lấy toàn bộ
    stratify: str | None = None  # "difficulty": chia đều theo mức 1–5


@dataclass
class SplitResult:
    name: str
    qids: list[str]
    counts: dict[str, int]
    dedup_removed: dict[str, int]
    seed: int
    sources: list[dict[str, Any]]
    # Cặp (câu bị loại, câu đã có mà nó trùng với), để con người kiểm tra trước khi khóa.
    dedup_pairs: list[tuple[str, str]] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            "name": self.name,
            "seed": self.seed,
            "sources": self.sources,
            "counts": self.counts,
            "dedup_removed": self.dedup_removed,
            "dedup_pairs": [list(p) for p in self.dedup_pairs],
            "n": len(self.qids),
            "qids": self.qids,
        }
        return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def load_splits_cfg(path: Path = CONFIGS_DIR / "splits.yaml") -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: cần một mapping")
    return data


def _source_rng(seed: int, split: str, dataset: str) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}/{split}/{dataset}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _sample(cands: list[Question], spec: SourceSpec, rng: np.random.Generator, where: str) -> list[Question]:
    cands = sorted(cands, key=lambda q: q.qid)
    if spec.n is None:
        return cands
    if spec.stratify is None:
        if len(cands) < spec.n:
            raise ValueError(f"{where}: cần {spec.n} câu, chỉ có {len(cands)}")
        idx = rng.choice(len(cands), size=spec.n, replace=False)
        return [cands[i] for i in sorted(idx)]
    if spec.stratify != "difficulty":
        raise ValueError(f"{where}: stratify không hỗ trợ: {spec.stratify}")
    if spec.n % len(LEVELS):
        raise ValueError(f"{where}: n={spec.n} không chia đều cho {len(LEVELS)} mức")
    per = spec.n // len(LEVELS)
    out: list[Question] = []
    for lv in LEVELS:
        pool = [q for q in cands if q.difficulty == lv]
        if len(pool) < per:
            raise ValueError(f"{where}: mức {lv} cần {per} câu, chỉ có {len(pool)}")
        idx = rng.choice(len(pool), size=per, replace=False)
        out.extend(pool[i] for i in sorted(idx))
    return out


def build_splits(
    pool: dict[str, list[Question]], cfg: dict[str, Any], skip: set[str] | None = None
) -> dict[str, SplitResult]:
    """Hàm thuần. Dựng các tập theo `order`; tập không phải `test_*` loại câu trùng với mọi câu đã chọn trước đó.

    `skip`: các tập bỏ qua hoàn toàn (không dựng, không dùng để khử trùng lặp), ví dụ test_ood khi chưa có HF_TOKEN.
    """
    seed = int(cfg["seed"])
    dedup = cfg.get("dedup", {})
    index = DedupIndex(n=int(dedup.get("ngram", 13)), min_overlap=float(dedup.get("min_overlap", 0.5)))
    used: set[str] = set()
    results: dict[str, SplitResult] = {}
    for split in cfg["order"]:
        if skip and split in skip:
            continue
        specs = [SourceSpec(**s) for s in cfg["splits"][split]]
        chosen: list[Question] = []
        counts: dict[str, int] = {}
        removed: dict[str, int] = {}
        pairs: list[tuple[str, str]] = []
        for spec in specs:
            if spec.dataset not in pool:
                raise KeyError(f"thiếu dữ liệu của tập nguồn {spec.dataset} cho {split}")
            cands = [q for q in pool[spec.dataset] if q.qid not in used]
            if not split.startswith("test"):
                kept = []
                for q in cands:
                    hit = index.match(q)
                    if hit is None:
                        kept.append(q)
                    else:
                        pairs.append((q.qid, hit))
                removed[spec.dataset] = len(cands) - len(kept)
                cands = kept
            picked = _sample(cands, spec, _source_rng(seed, split, spec.dataset), f"{split}/{spec.dataset}")
            for q in picked:
                index.add(q)
                used.add(q.qid)
            chosen.extend(picked)
            counts[spec.dataset] = len(picked)
        results[split] = SplitResult(
            name=split,
            qids=sorted(q.qid for q in chosen),
            counts=counts,
            dedup_removed=removed,
            seed=seed,
            sources=[dict(s) for s in cfg["splits"][split]],
            dedup_pairs=sorted(pairs),
        )
    return results


# ---------------------------------------------------------------- ghi, khóa và kiểm tra


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_splits(results: dict[str, SplitResult], out_dir: Path = SPLITS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, res in results.items():
        path = out_dir / f"{name}.json"
        path.write_bytes(res.to_json().encode("utf-8"))
        (out_dir / f"{name}.sha256").write_text(f"{_sha256(path)}  {name}.json\n", encoding="utf-8")
        written.append(path)
    return written


def read_split(name: str, splits_dir: Path = SPLITS_DIR, verify: bool = True) -> list[str]:
    path = splits_dir / f"{name}.json"
    if verify:
        verify_split(name, splits_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["qids"])


def verify_split(name: str, splits_dir: Path = SPLITS_DIR) -> None:
    path = splits_dir / f"{name}.json"
    sha_path = splits_dir / f"{name}.sha256"
    if not path.exists() or not sha_path.exists():
        raise SplitsTamperedError(f"thiếu {path.name} hoặc {sha_path.name}")
    expected = sha_path.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise SplitsTamperedError(f"{path.name}: mã băm {actual[:12]} khác bản đã khóa {expected[:12]}")


def verify_all(splits_dir: Path = SPLITS_DIR) -> dict[str, int]:
    """Kiểm tra mã băm của mọi tập và việc các tập không giao nhau. Gọi ở đầu mọi pipeline."""
    names = sorted(p.stem for p in splits_dir.glob("*.json"))
    if not names:
        raise SplitsTamperedError(f"{splits_dir}: chưa có tập nào")
    seen: dict[str, str] = {}
    sizes = {}
    for name in names:
        qids = read_split(name, splits_dir, verify=True)
        sizes[name] = len(qids)
        for qid in qids:
            if qid in seen:
                raise SplitsTamperedError(f"{qid} có ở cả {seen[qid]} và {name}")
            seen[qid] = name
    return sizes
