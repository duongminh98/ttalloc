"""Schema pyarrow của các bảng dữ liệu ghi và kết quả chạy (PLAN mục 4.2–4.5)."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

TRACES_SCHEMA = pa.schema(
    [
        pa.field("qid", pa.string(), nullable=False),
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("log_hash", pa.string(), nullable=False),
        pa.field("chain_ids", pa.list_(pa.int32()), nullable=False),
        pa.field("chain_logprobs", pa.list_(pa.float16()), nullable=False),
        pa.field("chain_len", pa.int32(), nullable=False),
        pa.field("natural_end", pa.bool_(), nullable=False),
        pa.field("end_reason", pa.string(), nullable=False),  # think_end | eos | max_len
        pa.field("nat_answer_text", pa.string()),
        pa.field("nat_answer_extracted", pa.string()),
        pa.field("nat_answer_prompt_tokens", pa.int32()),
        pa.field("nat_answer_output_tokens", pa.int32()),
        pa.field("nat_correct", pa.bool_()),
        pa.field("gold", pa.string(), nullable=False),
    ]
)

POINTS_SCHEMA = pa.schema(
    [
        pa.field("qid", pa.string(), nullable=False),
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("point_idx", pa.int32(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),  # start | marker | length
        pa.field("marker", pa.string()),
        pa.field("pos", pa.int32(), nullable=False),
        pa.field("gen_end", pa.int32(), nullable=False),
        pa.field("n_markers", pa.int32(), nullable=False),
        pa.field("local_conf_min", pa.float32()),
        pa.field("probe_prompt_tokens", pa.int16()),
        pa.field("probe_output_tokens", pa.int16()),
        pa.field("probe_output_ids", pa.list_(pa.int32())),
        pa.field("probe_logprobs", pa.list_(pa.float32())),
        pa.field("probe_answer_raw", pa.string()),
        pa.field("probe_answer_key", pa.string()),
        pa.field("probe_conf_geo", pa.float32()),
        pa.field("probe_conf_arith", pa.float32()),
        pa.field("probe_has_think_end", pa.bool_()),
        pa.field("answer_prompt_tokens", pa.int16(), nullable=False),
        pa.field("answer_output_tokens", pa.int16(), nullable=False),
        pa.field("answer_text", pa.string(), nullable=False),
        pa.field("answer_extracted", pa.string()),
        pa.field("answer_correct", pa.bool_(), nullable=False),
    ]
)

COST_FIELDS = ("think", "probe_prompt", "probe_output", "answer_prompt", "answer_output")

PER_QUESTION_SCHEMA = pa.schema(
    [
        pa.field("method", pa.string(), nullable=False),
        pa.field("alpha", pa.float64(), nullable=False),
        pa.field("draw", pa.int32(), nullable=False),
        pa.field("qid", pa.string(), nullable=False),
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("correct", pa.bool_(), nullable=False),
        *[pa.field(name, pa.int64(), nullable=False) for name in COST_FIELDS],
        pa.field("stop_point_idx", pa.int32()),
        pa.field("stop_kind", pa.string()),
        pa.field("n_probes", pa.int32(), nullable=False),
        pa.field("n_rounds", pa.int32(), nullable=False),
        # Thêm ở P2.T15 để một lần chạy chứa nhiều tập, lô và mức ngân sách.
        pa.field("split", pa.string()),
        pa.field("batch", pa.string()),
        pa.field("budget", pa.int64()),
        pa.field("dataset", pa.string()),
        pa.field("difficulty", pa.int32()),
    ]
)

TIMING_SCHEMA = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("t_advance", pa.float64(), nullable=False),
        pa.field("t_probe", pa.float64(), nullable=False),
        pa.field("t_answer", pa.float64(), nullable=False),
        pa.field("t_controller", pa.float64(), nullable=False),
        pa.field("n_active", pa.int32(), nullable=False),
        pa.field("prefix_cache_hit_rate", pa.float64()),
    ]
)

# Câu trả lời dạng văn bản của các lệnh baseline (P1.T5); dùng cho audit-grading (P0.T10).
ANSWERS_SCHEMA = pa.schema(
    [
        pa.field("qid", pa.string(), nullable=False),
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("problem", pa.string(), nullable=False),
        pa.field("gold", pa.string(), nullable=False),
        pa.field("answer_text", pa.string(), nullable=False),
        pa.field("extracted", pa.string()),
        pa.field("correct", pa.bool_(), nullable=False),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {
    "traces": TRACES_SCHEMA,
    "points": POINTS_SCHEMA,
    "per_question": PER_QUESTION_SCHEMA,
    "timing": TIMING_SCHEMA,
    "answers": ANSWERS_SCHEMA,
}


class SchemaError(ValueError):
    pass


def to_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Dựng bảng từ danh sách dòng; báo lỗi nếu thiếu trường, thừa trường hoặc sai kiểu."""
    names = set(schema.names)
    for i, row in enumerate(rows):
        extra = set(row) - names
        missing = {f.name for f in schema if not f.nullable} - set(row)
        if extra or missing:
            raise SchemaError(f"dòng {i}: thừa {sorted(extra)}, thiếu {sorted(missing)}")
    try:
        return pa.Table.from_pylist(rows, schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise SchemaError(str(exc)) from exc


def validate_table(table: pa.Table, schema: pa.Schema) -> None:
    """Kiểm tra bảng khớp schema (tên, thứ tự, kiểu) và không có null ở trường bắt buộc."""
    if not table.schema.equals(schema, check_metadata=False):
        raise SchemaError(f"schema khác:\n{table.schema}\n---\n{schema}")
    for f in schema:
        if not f.nullable and table.column(f.name).null_count:
            raise SchemaError(f"trường bắt buộc {f.name} có giá trị null")
