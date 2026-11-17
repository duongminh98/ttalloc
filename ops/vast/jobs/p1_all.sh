# P1 GPU steps with checkpointed stages and explicit test_main approval.
source ops/vast/jobs/common.sh
need_approval
TTALLOC_DTYPE=bfloat16 python3 -m pytest -m gpu tests/gpu -q
python3 ops/vast/p1_run.py
