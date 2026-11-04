# P1 bước chạy 1–2 và P1.T6: tái lập baseline trên test_main (ngoại lệ duy nhất của mục 7.5, cần người duyệt),
# thông lượng, kiểm định tương đương trên val. Khoảng 6–8 giờ RTX 3090.
source ops/vast/jobs/common.sh
need_approval
step "baseline full MATH-500 x1"
ttalloc baseline full --exp "$EXP" --split test_main --datasets math500 --seeds 0 --profile deer_paper \
  --approved-by "$TTALLOC_APPROVED_BY"
step "baseline deer MATH-500 x1"
ttalloc baseline deer --exp "$EXP" --split test_main --datasets math500 --seeds 0 --profile deer_paper \
  --approved-by "$TTALLOC_APPROVED_BY"
step "baseline full AIME 2024 x4"
ttalloc baseline full --exp "$EXP" --split test_main --datasets aime_2024 --seeds 0-3 --profile deer_paper \
  --approved-by "$TTALLOC_APPROVED_BY"
step "baseline deer AIME 2024 x4"
ttalloc baseline deer --exp "$EXP" --split test_main --datasets aime_2024 --seeds 0-3 --profile deer_paper \
  --approved-by "$TTALLOC_APPROVED_BY"
step "baseline throughput"
ttalloc baseline throughput --exp "$EXP" --n 256
step "equivalence val 400 x2"
ttalloc equivalence --exp "$EXP" --n 400 --seeds 0-1
