# Tiến độ

"Mã xong" nghĩa là mã và test CPU đã viết và đạt; chưa commit, chưa review (P0a).

| Task | Trạng thái | Commit | Ghi chú |
|---|---|---|---|
| P0.T1 | mã xong | | Khung repo |
| P0.T2 | mã xong | | Config |
| P0.T3 | mã xong | | Thư mục chạy |
| P0.T4 | mã xong | | Kiểu dữ liệu và schema |
| P0.T5 | mã xong | | Dữ liệu tổng hợp |
| P0.T6 | mã xong | | Bộ nạp dữ liệu; fixture lấy từ datasets-server của Hugging Face |
| P0.T7 | mã xong | | Đã chạy thử trên dữ liệu thật vào thư mục tạm (ADR-009); chưa ghi `splits/` |
| P0.T8 | mã xong | | Lời nhắc |
| P0.T9 | mã xong | | Chấm điểm |
| P0.T10 | mã xong | | Công cụ kiểm tra chấm điểm |
| P0, bước chạy 1 | xong | | 04/11/2026: `p0_smoke` trên Kaggle T4 đạt 5/5 bước; chốt vLLM 0.32.0 (ADR-002) |
| P0, bước chạy 2 | xong | | `bfloat16` cho Vast (RTX 3090); float16 chỉ trên T4 cho test (ADR-002) |
| P0, bước chạy 3 | xong | | 04/11/2026: `splits build` đủ 5 tập (train 1.500, val 400, test_main 560, test_clean 63 theo ADR-004, test_ood 198); `splits verify` OK. Người dùng duyệt khóa ID. Tệp chỉ chứa ID, không có nội dung câu hỏi |
| P0a (review) | chưa | | |
| P1.T1–T3 | mã xong | | Generator, ScriptedGenerator, cắt đoạn có peek (ADR-010) |
| P1.T4 | mã xong | | VllmGenerator. 04/11/2026: Kaggle `p1_gpu_tests` lần 6 trên T4 (float16) đạt hết: GPU 7/7, CPU 255 (1 bỏ qua vì không có git). Qua 6 lần đã sửa: tokenizer R1 (ADR-013), đọc tỉ lệ trúng prefix cache (vLLM ghi 97,5%), test cần git, trình theo dõi. 05/11/2026 lần 7 (sau các sửa của tập dượt): GPU 7/7, CPU 264 đạt. Vast `gpu_check` (bf16) chưa chạy |
| P1.T5 | mã xong | | `baseline full/deer/throughput`; DEER đếm token bằng tokenize lại (ADR-010) |
| P1.T6 | mã xong | | `equivalence` |
| P1.T7–T11 | mã xong | | LiveSource, logger chạy tiếp được, study answer-mode/probe-noise, validate-logs |
| P1, bước chạy 1–5 | chưa | | Vast: `gpu_check`, `p1_baseline` (cần duyệt test_main), `p1_pilot` |
| P2.T1–T6 | mã xong | | Sự kiện, dãy chuẩn, đặc trưng, bộ ước lượng, rho, đánh giá |
| P2.T7–T8 | mã xong | | Quỹ, bộ bảo vệ (ADR-012), Runner; property test I1–I3 |
| P2.T9–T11 | mã xong | | LoggedSource, bảng tính sẵn, RhoProvider, B2/M2/M3 |
| P2.T13–T15 | mã xong | | Hiệu chỉnh B2 (ADR-011), thành phần lô, `sim run` |
| P2.T17–T20 | mã xong | | Thống kê, báo cáo, cổng G1/G2, study curves |
| P2, bước chạy 1–4 | chưa | | Vast `p2_log`; phần còn lại chạy trên CPU (`configs/experiments/g1.yaml`) |
| P3.T3–T6 | mã xong | | Ablation (19 config ở `configs/experiments/e4/`), `tune`, `lock`, `eval-test`, config E1–E4 |
| P3, bước chạy 1–4 | chưa | | Vast `p3_log`; phần còn lại trên CPU |
| P4.T1–T4 | mã xong | | `online run`, kiểm thử phát lại (`tests/integration/test_p4_replay.py`), smoke GPU, `e5_online.yaml`, `gate g3` |
| P4.T5 | mã xong | | Dùng `baseline full/deer` trong công việc `p4_online` |
| P4, bước chạy 1–2 | chưa | | Vast `p4_online` (cần duyệt test_main) |
| Vận hành | mã xong | | `ops/vast/vast.py`, `ops/hf_sync.py`, `ops/vast/jobs/*.sh`, `ops/kaggle/p1_gpu_tests.py`; skill `vast-run`, `phase-runner` |
| Tập dượt Kaggle | xong | 040db2f | 04–05/11/2026: `ops/kaggle/rehearsal.py` chạy mọi lệnh `ttalloc` của `ops/vast/jobs` ở cỡ nhỏ trên T4 (val, train; l_log 1.024); 29/29 bước đạt (lần 5 tới E2, lần 7 nối tiếp từ dữ liệu lần 5 cho P4). Đã sửa: logger chạy quá sức chứa KV (256 chuỗi, KV 74k token, trúng prefix cache 13%) → giới hạn theo KV và gom mỗi vòng một lượt (log val 151 → 88 phút, cùng dữ liệu); online run treo do fork vLLM sau LightGBM; α tập dượt khả thi; test-log đúng 6 cột; log trực tiếp, dừng từ xa, chạy tiếp từ dữ liệu cũ. Số liệu tập dượt không dùng cho khóa luận |
