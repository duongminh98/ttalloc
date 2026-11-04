# P2 bước chạy 1: dữ liệu thăm dò. train 500 câu × 4 seed, toàn bộ val × 8 seed; validate-logs.
# Cần answer.mode đã chốt trong configs/experiments/qwen3.yaml (ADR). Chạy tiếp được nếu máy bị ngắt.
source ops/vast/jobs/common.sh
step "log val x8"
ttalloc log --exp "$EXP" --split val --seeds 0-7
step "validate-logs val"
ttalloc validate-logs --exp "$EXP" --split val
step "log train 500 x4"
ttalloc log --exp "$EXP" --split train --limit 500 --seeds 0-3
step "validate-logs train"
ttalloc validate-logs --exp "$EXP" --split train
