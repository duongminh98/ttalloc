# P1 bước chạy 4–5: ghi thử 100 câu val × 2 seed với answer.mode=short (đã chốt, ADR-014), hai study, validate-logs.
# Cùng log_hash với P2 nên dữ liệu này được dùng lại (logger bỏ qua khóa đã ghi). study answer-mode vẫn so full với
# short trên chuỗi dài thật để báo cáo.
source ops/vast/jobs/common.sh
step "log val 100 x2"
ttalloc log --exp "$EXP" --split val --limit 100 --seeds 0-1
step "validate-logs val"
ttalloc validate-logs --exp "$EXP" --split val
step "study answer-mode"
ttalloc study answer-mode --exp "$EXP" --n 100 --seeds 0
step "study probe-noise"
ttalloc study probe-noise --exp "$EXP" --n 100 --seeds 0
