# P3 bước chạy 1: ghi nốt dữ liệu. train tới đủ 1.500 câu × 4 seed (bỏ qua khóa đã ghi), test_main × 8 seed.
# Chỉ sinh và ghi, không đánh giá chính sách. Cần người duyệt GPU trước khi chạy ([dừng] ở PLAN).
source ops/vast/jobs/common.sh
step "log train 1500 x4"
ttalloc log --exp "$EXP" --split train --limit 1500 --seeds 0-3
step "validate-logs train"
ttalloc validate-logs --exp "$EXP" --split train
step "log test_main x8"
ttalloc log --exp "$EXP" --split test_main --seeds 0-7
step "validate-logs test_main"
ttalloc validate-logs --exp "$EXP" --split test_main
