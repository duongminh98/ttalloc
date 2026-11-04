# Kiểm tra máy vừa thuê: test GPU (P1.T4, P4.T3) và thông lượng (P1 bước chạy 2). Khoảng 20–30 phút.
source ops/vast/jobs/common.sh
step "pytest -m gpu"
TTALLOC_DTYPE=bfloat16 python3 -m pytest -m gpu tests/gpu -q -rA | tee runs/vast-gpu-tests.txt
step "baseline throughput"
ttalloc baseline throughput --exp "$EXP" --n 256
