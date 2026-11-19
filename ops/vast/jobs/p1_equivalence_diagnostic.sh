# Small val-only comparison of answer formats on matched segmented reasoning.
source ops/vast/jobs/common.sh
first="${TTALLOC_EQUIVALENCE_FIRST:?}"
second="${TTALLOC_EQUIVALENCE_SECOND:?}"
python3 ops/vast/equivalence_diagnostic.py "$first" "$second" --check-selection
TTALLOC_DTYPE=bfloat16 python3 -m pytest -m gpu tests/gpu -q
step equivalence_diagnostic
python3 ops/vast/equivalence_diagnostic.py "$first" "$second"
