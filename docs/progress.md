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
| P0, bước chạy 3 | chưa | | `ttalloc splits build` → **[dừng]** khóa ID; `test_ood` cần `HF_TOKEN` |
| P0a (review) | chưa | | |
| P1.T1–T3 | mã xong | | Generator, ScriptedGenerator, cắt đoạn có peek (ADR-010) |
| P1.T4 | mã xong | | VllmGenerator; test GPU ở `tests/gpu` chưa chạy (Kaggle `p1_gpu_tests`, Vast `gpu_check`) |
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
