"""Atomic, verified row checkpoints at the CLI I/O boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class RowCheckpoint:
    def __init__(
        self, path: Path, identity: dict[str, Any], keys: list[tuple[str, int]], resume: Path | None = None
    ) -> None:
        self.path = path
        self.identity = identity
        self.expected = set(keys)
        if len(self.expected) != len(keys):
            raise ValueError("duplicate requested checkpoint keys")
        self.rows: list[dict[str, Any]] = []
        self.seconds = 0.0
        if resume is not None:
            obj = json.loads(resume.read_text(encoding="utf-8"))
            payload = obj["payload"]
            if obj["sha256"] != _digest(payload) or payload["identity"] != identity:
                raise ValueError("checkpoint checksum or config/selection mismatch")
            self.rows = payload["rows"]
            self.seconds = float(payload["seconds"])
        self._validate(self.rows, self.seconds)
        self.add([], 0.0)

    def _validate(self, rows: list[dict[str, Any]], seconds: float) -> None:
        keys = [(str(r["qid"]), int(r["seed"])) for r in rows]
        if len(set(keys)) != len(keys) or not set(keys) <= self.expected:
            raise ValueError("duplicate or unexpected checkpoint row key")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid checkpoint elapsed time")

    def add(self, rows: list[dict[str, Any]], seconds: float) -> None:
        merged = self.rows + rows
        elapsed = self.seconds + seconds
        self._validate(merged, elapsed)
        payload = {"identity": self.identity, "rows": merged, "seconds": elapsed}
        blob = json.dumps({"payload": payload, "sha256": _digest(payload)}, ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(blob, encoding="utf-8")
        tmp.replace(self.path)
        self.rows, self.seconds = merged, elapsed

    def complete(self) -> None:
        if {(str(r["qid"]), int(r["seed"])) for r in self.rows} != self.expected:
            raise ValueError("checkpoint incomplete; refusing final result")
