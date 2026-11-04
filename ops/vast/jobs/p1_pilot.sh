# P1 bước chạy 4–5: ghi thử 100 câu val × 2 seed với answer.mode=full, hai study, validate-logs. Khoảng 2–3 giờ.
# Nếu sau study chọn `short`, xóa data/logs/qwen3_1_7b/val (cả trên Hugging Face) trước khi ghi thật ở P2,
# vì log_hash khác (I9). Nếu chọn `full`, dữ liệu này được dùng lại.
source ops/vast/jobs/common.sh
SET="--set answer.mode=full"
step "log val 100 x2"
ttalloc log --exp "$EXP" $SET --split val --limit 100 --seeds 0-1
step "validate-logs val"
ttalloc validate-logs --exp "$EXP" $SET --split val
step "study answer-mode"
ttalloc study answer-mode --exp "$EXP" $SET --n 100 --seeds 0
step "study probe-noise"
ttalloc study probe-noise --exp "$EXP" $SET --n 100 --seeds 0
