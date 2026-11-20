# Add only seeds 2–3 to the existing 400-question val equivalence run.
source ops/vast/jobs/common.sh
first="${TTALLOC_EQUIVALENCE_FIRST:?}"
python3 ops/vast/merge_equivalence.py "$first"
TTALLOC_DTYPE=bfloat16 python3 -m pytest -m gpu tests/gpu -q
step equivalence_extension
run_log="$(mktemp)"
trap 'rm -f "$run_log"' EXIT
resume=()
if [ -n "${TTALLOC_EQUIVALENCE_RESUME:-}" ]; then
  resume=(--resume-from "$TTALLOC_EQUIVALENCE_RESUME")
fi
ttalloc equivalence --exp configs/experiments/p1_safe.yaml --n 400 --seeds 2-3 "${resume[@]}" | tee "$run_log"
run="$(sed -n 's/^thư mục chạy: //p' "$run_log")"
test -d "$run"
python3 ops/vast/merge_equivalence.py "$first" "$run"
