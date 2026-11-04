# P4 bước chạy 1–2: chạy thật thử trên val, lưới E5 qua eval-test (config đã khóa), rồi so thời gian với DEER
# chính thức (P4.T5). Cần: bộ ước lượng và bảng trong data/ (đã đẩy lên Hugging Face), configs/locked/e5_* đã commit.
source ops/vast/jobs/common.sh
need_approval
step "online pilot val"
ttalloc online run --exp configs/experiments/online_val_pilot.yaml
step "eval-test E5"
ttalloc eval-test --exp configs/locked/e5_online_qwen3_1_7b.yaml --approved-by "$TTALLOC_APPROVED_BY"
step "P4.T5 baseline full MATH-500"
ttalloc baseline full --exp "$EXP" --split test_main --datasets math500 --seeds 0 --approved-by "$TTALLOC_APPROVED_BY"
step "P4.T5 baseline deer MATH-500"
ttalloc baseline deer --exp "$EXP" --split test_main --datasets math500 --seeds 0 --approved-by "$TTALLOC_APPROVED_BY"
