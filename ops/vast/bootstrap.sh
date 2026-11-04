#!/usr/bin/env bash
# Chạy trên máy Vast. onstart chép tệp này ra /root/bootstrap.sh rồi chạy từ đó, để việc giải nén bản mã sửa lỗi đè
# lên /root/ttalloc không làm hỏng script đang chạy. Không chạy trên máy cá nhân.
#
# Biến môi trường (đặt bởi ops/vast/vast.py create):
#   HF_TOKEN, TTALLOC_HF_REPO   đồng bộ qua Hugging Face (ops/hf_sync.py)
#   TTALLOC_JOB                 tên công việc: ops/vast/jobs/<TTALLOC_JOB>.sh
#   TTALLOC_APPROVED_BY         người duyệt, chỉ cho công việc chạm test_* (PLAN mục 7.5)
#   TTALLOC_SYNC_EVERY          số giây giữa hai lần đẩy dữ liệu lên (mặc định 600)
#   TTALLOC_LOG_EVERY           số giây giữa hai lần đẩy đuôi log lên status/<job>.log (mặc định 60; bỏ qua nếu log không đổi)
#   TTALLOC_FIX_WAIT            khi công việc lỗi: chờ bản mã sửa (push-code) tối đa bấy nhiêu giây rồi chạy lại
#                               (mặc định 2700; 0 để thoát ngay). Logger chạy tiếp từ các khóa đã ghi.
#   TTALLOC_MAX_ATTEMPTS        số lần chạy tối đa của công việc (mặc định 4)
#   TTALLOC_SELF_DESTROY=1 và VAST_API_KEY: tự xóa máy khi kết thúc (tùy chọn; mặc định `vast.py watch` xóa)
set -uo pipefail
cd /root/ttalloc
JOB="${TTALLOC_JOB:?thiếu TTALLOC_JOB}"
SYNC_EVERY="${TTALLOC_SYNC_EVERY:-600}"
LOG_EVERY="${TTALLOC_LOG_EVERY:-60}"
FIX_WAIT="${TTALLOC_FIX_WAIT:-2700}"
MAX_ATTEMPTS="${TTALLOC_MAX_ATTEMPTS:-4}"
LOG=/root/job.log
SYNC_PATHS="data/logs data/questions data/estimators data/tables runs docs/test-log.md"
export PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0

hf() { python3 ops/hf_sync.py "$@"; }
status() { hf set-status "$JOB" "$1" "${2:-}" >/dev/null 2>&1 || true; }
sync_up() { hf push $SYNC_PATHS -m "sync $JOB $(date +%H:%M)" >/dev/null 2>&1 || echo "sync lỗi (thử lại lần sau)"; }
put_log() { hf put-log "$JOB" "$LOG" >/dev/null 2>&1 || true; }
install() {
  # Cài ttalloc và các phụ thuộc còn thiếu; không đụng vào torch và vllm có sẵn trong image.
  find ops -name '*.sh' -exec sed -i 's/\r$//' {} +
  python3 -m pip install -q -e . pytest hypothesis 2>&1 | tail -n 5
}

python3 -m pip install -q "huggingface_hub>=0.25" || exit 1
status starting "cài gói"
nvidia-smi || true
install
python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" \
  || { status failed "import vllm lỗi"; put_log; exit 1; }

# Lấy lại dữ liệu đã có (logger chạy tiếp từ các khóa đã ghi nếu máy trước bị ngắt).
hf pull data runs || true

( while true; do sleep "$SYNC_EVERY"; sync_up; done ) &
SYNC_PID=$!
( while true; do sleep "$LOG_EVERY"; put_log; done ) &
LOG_PID=$!

RC=1
for ATTEMPT in $(seq 1 "$MAX_ATTEMPTS"); do
  REV="$(hf code-rev 2>/dev/null)"
  status running "lần $ATTEMPT: ops/vast/jobs/$JOB.sh (mã ${REV:0:12})"
  bash "ops/vast/jobs/$JOB.sh"
  RC=$?
  sync_up
  put_log
  [ "$RC" -eq 0 ] && break
  [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ] && break
  [ "$FIX_WAIT" -le 0 ] && break
  echo "=== $(date '+%F %T') công việc lỗi (mã $RC); chờ bản mã sửa tối đa ${FIX_WAIT}s"
  status waiting_fix "lần $ATTEMPT lỗi (mã $RC); chờ push-code tối đa ${FIX_WAIT}s"
  put_log
  WAITED=0; NEW=""
  while [ "$WAITED" -lt "$FIX_WAIT" ]; do
    sleep 60; WAITED=$((WAITED + 60))
    NEW="$(hf code-rev 2>/dev/null)"
    [ -n "$NEW" ] && [ "$NEW" != "$REV" ] && break
    NEW=""
  done
  [ -z "$NEW" ] && { echo "không có bản sửa sau ${FIX_WAIT}s"; break; }
  echo "=== $(date '+%F %T') có bản mã mới ${NEW:0:12}; cài lại và chạy tiếp"
  hf pull-code /root/ttalloc && install
done

kill "$SYNC_PID" "$LOG_PID" 2>/dev/null
cp "$LOG" "runs/vast-$JOB.log" 2>/dev/null && hf push "runs/vast-$JOB.log" >/dev/null 2>&1
put_log
if [ "$RC" -eq 0 ]; then status done "xong"; else status failed "mã thoát $RC; xem runs/vast-$JOB.log"; fi

if [ "${TTALLOC_SELF_DESTROY:-0}" = "1" ] && [ -n "${VAST_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ]; then
  curl -s -X DELETE -H "Authorization: Bearer $VAST_API_KEY" "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/"
fi
