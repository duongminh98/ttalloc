# Dùng chung cho các công việc trong ops/vast/jobs (source từ mỗi tệp). Mỗi lệnh lỗi làm công việc dừng ngay.
set -euo pipefail
EXP=configs/experiments/qwen3.yaml
step() { echo; echo "=== $(date '+%F %T') $*"; python3 ops/hf_sync.py set-status "$TTALLOC_JOB" running "$*" >/dev/null 2>&1 || true; }
need_approval() {
  if [ -z "${TTALLOC_APPROVED_BY:-}" ]; then
    echo "LỖI: công việc này chạm test_* nên cần TTALLOC_APPROVED_BY (PLAN mục 7.5)"; exit 2
  fi
}
mkdir -p runs
ttalloc splits verify
