"""P0.T1: mọi module CPU import được trên máy không có vLLM."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

import ttalloc

ALLOWED_VLLM_IMPORTERS = {"ttalloc.engine.vllm_generator"}


class _BlockVllm(MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise ImportError(f"vllm bị chặn trong test: {fullname}")
        return None


def test_import_cpu_modules_without_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name)
    blocker = _BlockVllm()
    sys.meta_path.insert(0, blocker)
    try:
        names = [m.name for m in pkgutil.walk_packages(ttalloc.__path__, prefix="ttalloc.")]
        assert "ttalloc.config" in names
        for name in names:
            if name in ALLOWED_VLLM_IMPORTERS:
                continue
            importlib.import_module(name)
    finally:
        sys.meta_path.remove(blocker)
