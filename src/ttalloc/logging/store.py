"""Lưu dữ liệu ghi theo phần, ghi nguyên tử và chạy tiếp được sau khi bị ngắt (PLAN P1.T7).

Bố cục `data/logs/<model>/<split>/`:
- `traces/part-XXXXX.parquet`, `points/part-XXXXX.parquet`: mỗi phần gồm các chuỗi đã xong trọn vẹn;
- `manifest.jsonl`: mỗi dòng là một phần đã ghi xong, kèm danh sách khóa `(qid, seed)`. Chỉ các phần có trong tệp kê
  khai mới được đọc; phần ghi dở (tiến trình bị ngắt trước khi kê khai) bị bỏ qua và ghi đè ở lần sau;
- `meta.json`: `log_hash` và config ghi. Không trộn dữ liệu có `log_hash` khác nhau trong một thư mục.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ttalloc.context import DATA_DIR
from ttalloc.logging.schema import POINTS_SCHEMA, TRACES_SCHEMA, to_table, validate_table
from ttalloc.types import TraceKey

LOGS_DIR = DATA_DIR / "logs"


class LogHashMismatch(RuntimeError):
    pass


def atomic_write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class LogStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def for_split(cls, model: str, split: str, logs_dir: Path = LOGS_DIR) -> LogStore:
        return cls(logs_dir / model / split)

    # ---------------------------------------------------------------- meta

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def meta(self) -> dict[str, Any] | None:
        return json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else None

    def init(self, log_hash: str, meta: dict[str, Any]) -> None:
        old = self.meta()
        if old is not None:
            if old["log_hash"] != log_hash:
                raise LogHashMismatch(
                    f"{self.root}: log_hash đã ghi {old['log_hash']} khác hiện hành {log_hash}; ghi vào thư mục khác"
                )
            return
        atomic_write_text(json.dumps({"log_hash": log_hash, **meta}, indent=2, ensure_ascii=False), self.meta_path)

    def log_hash(self) -> str:
        m = self.meta()
        if m is None:
            raise FileNotFoundError(f"chưa có dữ liệu ghi ở {self.root}")
        return str(m["log_hash"])

    # ---------------------------------------------------------------- kê khai

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        out = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # dòng cuối ghi dở khi bị ngắt
                    continue
        return out

    def completed_keys(self) -> set[TraceKey]:
        return {TraceKey(q, int(s)) for e in self.manifest() for q, s in e["keys"]}

    def _next_part(self) -> int:
        done = [int(e["part"]) for e in self.manifest()]
        return max(done, default=-1) + 1

    def write_part(self, traces: list[dict[str, Any]], points: list[dict[str, Any]]) -> int:
        """Ghi một phần gồm các chuỗi đã xong; trả về số thứ tự phần."""
        if not traces:
            return -1
        keys = sorted({(r["qid"], int(r["seed"])) for r in traces})
        done = self.completed_keys()
        dup = [k for k in keys if TraceKey(*k) in done]
        if dup:
            raise ValueError(f"khóa đã ghi trước đó: {dup[:3]}")
        part = self._next_part()
        name = f"part-{part:05d}.parquet"
        atomic_write_table(to_table(traces, TRACES_SCHEMA), self.root / "traces" / name)
        atomic_write_table(to_table(points, POINTS_SCHEMA), self.root / "points" / name)
        line = json.dumps({"part": part, "file": name, "n": len(keys), "keys": keys}, ensure_ascii=False)
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return part

    # ---------------------------------------------------------------- đọc

    def _read(self, kind: str, schema: pa.Schema, columns: list[str] | None = None) -> pa.Table:
        files = [self.root / kind / e["file"] for e in self.manifest()]
        if columns is not None:
            sub = pa.schema([schema.field(c) for c in columns])
            if not files:
                return sub.empty_table()
            return pa.concat_tables([pq.read_table(f, columns=columns, schema=schema) for f in files])
        if not files:
            return schema.empty_table()
        table = pa.concat_tables([pq.read_table(f, schema=schema) for f in files])
        validate_table(table, schema)
        return table

    def read_traces(self, columns: list[str] | None = None) -> pa.Table:
        return self._read("traces", TRACES_SCHEMA, columns)

    def read_points(self, columns: list[str] | None = None) -> pa.Table:
        return self._read("points", POINTS_SCHEMA, columns)
