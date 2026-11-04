from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    for name, desc in [("gpu", "cần GPU và vLLM"), ("slow", "chạy lâu"), ("net", "cần mạng")]:
        config.addinivalue_line("markers", f"{name}: {desc}")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def qwen3_tokenizer():  # type: ignore[no-untyped-def]
    from ttalloc.config import load_model_cfg
    from ttalloc.context import load_tokenizer

    return load_tokenizer(load_model_cfg("qwen3_1_7b"), FIXTURES / "tokenizers" / "qwen3_1_7b")


@pytest.fixture(scope="session")
def r1_tokenizer():  # type: ignore[no-untyped-def]
    # Qua đúng hàm nạp của mã chính, để test bắt được lỗi tokenizer theo phiên bản transformers (ADR-013).
    from ttalloc.config import load_model_cfg
    from ttalloc.context import load_tokenizer

    return load_tokenizer(load_model_cfg("r1_distill_qwen_1_5b"), FIXTURES / "tokenizers" / "r1_distill_qwen_1_5b")
