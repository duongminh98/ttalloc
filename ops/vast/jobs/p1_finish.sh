# Run the remaining val-only steps under the approved ADR-023 continuation.
source ops/vast/jobs/common.sh
need_approval
python3 ops/vast/p1_run.py --finish-reviewed-equivalence
