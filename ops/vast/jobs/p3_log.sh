# P3 bước chạy 1: ghi nốt dữ liệu. train tới TTALLOC_TRAIN_LIMIT câu × 4 seed (bỏ qua khóa đã ghi), test_main × 8 seed.
# TTALLOC_TRAIN_LIMIT lấy từ `ttalloc estimator learning-curve` cuối P2 (ADR-015): 500, 1000 hoặc 1500; truyền bằng
#   vast.py create --job p3_log --env TTALLOC_TRAIN_LIMIT=<số> ...
# Chỉ sinh và ghi, không đánh giá chính sách. Cần người duyệt GPU trước khi chạy ([dừng] ở PLAN).
source ops/vast/jobs/common.sh
: "${TTALLOC_TRAIN_LIMIT:?đặt TTALLOC_TRAIN_LIMIT theo learning curve cuối P2 (ADR-015)}"
step "log train ${TTALLOC_TRAIN_LIMIT} x4"
ttalloc log --exp "$EXP" --split train --limit "$TTALLOC_TRAIN_LIMIT" --seeds 0-3
step "validate-logs train"
ttalloc validate-logs --exp "$EXP" --split train
step "log test_main x8"
ttalloc log --exp "$EXP" --split test_main --seeds 0-7
step "validate-logs test_main"
ttalloc validate-logs --exp "$EXP" --split test_main