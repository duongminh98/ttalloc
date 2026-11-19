# Runbook: chạy trên máy có GPU

Tài liệu này cho biết **chạy lệnh nào, ở đâu, theo thứ tự nào** khi có GPU. Đặc tả đầy đủ ở `docs/PLAN.md`
(mục 6, phần **Bước chạy** của từng phase). Bước có **[dừng]** cần con người xác nhận trước khi chạy tiếp.

Nguyên tắc chung:

- **Kaggle (T4, miễn phí):** phát triển và test GPU. Không sinh số liệu đưa vào khóa luận.
- **Vast.ai (RTX 3090, bf16):** tái lập baseline, ghi dữ liệu, chạy thật. Thuê máy tốn tiền: luôn hỏi người dùng
  trước khi tạo máy, và xóa máy (destroy) ngay khi xong việc. Công cụ: `ops/vast/vast.py`, skill `vast-run`.
- **CPU (máy cá nhân, Windows được):** học bộ ước lượng, dựng bảng, mô phỏng, chỉnh tham số, khóa config, báo cáo,
  cổng. Không cần GPU. Trên Windows đặt `$env:UV_LINK_MODE='copy'` trước `uv run`.
- Dữ liệu đi giữa các máy qua một dataset riêng tư trên Hugging Face (`ops/hf_sync.py`), chỉ cần HTTPS.
- Mọi lệnh chạy từ gốc repo.

## Phương án đang thực hiện: tối thiểu 10 USD, dự phòng 15 USD

**Dự toán cập nhật 18/11/2026 sau `gpu_check` (ADR-020):** job đã xong, GPU bf16 7/7, máy đã destroy. Giá đã thuê gồm 60 GB disk là 0,234444 USD/giờ, mạng mỗi chiều 0,0065104 USD/GB. Dùng giá này cho mọi đợt và giữ giả định logger 800 token/giây: còn khoảng **40,22 giờ GPU / 10,18 USD**, tổng gồm `gpu_check` khoảng **40,62 giờ / 10,28 USD**. 500–1.200 token/giây cho phần logger tạo các kịch bản tổng **13,89–8,28 USD** khi độ dài giả định vẫn 6.300 token; không phải khoảng bảo đảm. Throughput đã đo 681/1.578/2.169/2.296 token/giây ở lô 8/32/128/256 chỉ là sinh liền mạch tối đa 4.096 token/câu, chưa thay được pilot dài. Chi tiết và thời gian CPU chưa đo xem mục 3.3.1 của `docs/Ke-hoach-thuc-hien-de-tai.md`. Các giá 0,15/0,24 USD/giờ trong phần ngày 17/11 dưới đây là giả định cũ; kiểm tra lại giá thật trước mỗi lần thuê.

Cập nhật 17/11/2026, theo ADR-019 và mục 3.3 của `docs/Ke-hoach-thuc-hien-de-tai.md`:

- Một Qwen3-1.7B, bf16, `answer.mode=short`; dùng một RTX 3090 trước. Không thuê thêm A100 chỉ để so GPU trong gói này.
- Train 500 câu × 4 seed; val 400 × 8; test_main 560 × 8: 9.680 chuỗi ghi. Pool train 1.500 ID đã khóa giữ nguyên; `--limit 500` lấy theo `hash_order` hiện hành. Không dựng/khóa lại split.
- Mốc khoảng 10 USD (260.000 đồng) có điều kiện; chuẩn bị 15 USD (390.000–400.000 đồng) có dự phòng. Quy đổi dùng 26.000 đồng/USD để lập kế hoạch, không phải tỷ giá hiện hành. Giá đầu vào dự toán đã gồm 60 GB disk: 0,15 USD/giờ cho ghi có thể bị ngắt; 0,24 USD/giờ cho việc theo yêu cầu. Phải lấy báo giá cụ thể trước mỗi lần thuê.
- Cài đặt/kiểm GPU khoảng 2 giờ tổng các đợt, baseline + equivalence 6–8 giờ, study 1–2 giờ; ghi P2 khoảng 14 giờ, ghi P3 khoảng 12 giờ, chạy thật P4 khoảng 4–5 giờ. Các mốc ghi giả định 6.300 token/chuỗi, phụ phí token sinh 10%, tốc độ hiệu dụng 800 token/giây và thời gian ghi lại 10%; phải thay bằng số đo pilot chuỗi dài. Không cộng lại 200 chuỗi pilot trong số chuỗi P2.
- Giữ đầy đủ seed, chuỗi tối đa 32.768 token, 200 lần rút mô phỏng, 50 cấu hình tune mỗi phương pháp và kiểm chứng E5 tối thiểu 6 lần. Học/tune/mô phỏng/E1–E4/bảng hình/cổng chạy trên CPU; destroy GPU trước khi chuyển sang các bước này hoặc chờ phê duyệt.
- Mở rộng chưa tính tiền: train trên 500, `test_clean`, `test_ood`, model thứ hai, lưới E5 lớn hơn. Nếu learning curve yêu cầu tăng hoặc dự báo chi vượt khoản được duyệt/số dư, dừng và cập nhật dự toán trước khi tiếp tục; không ép tham số để vừa tiền.
- Cập nhật kế hoạch không cho phép tự thuê máy, vượt cổng hoặc chạy tập kiểm tra. Vẫn thực hiện các điểm `[dừng]` và ghi phê duyệt theo đúng phạm vi.

## 0. Chuẩn bị một lần trên máy cá nhân

1. Cài [uv](https://docs.astral.sh/uv/) và git. Clone repo, rồi chạy:
   ```bash
   uv sync --group ops
   uv run pytest            # test CPU, phải đạt hết trước khi dùng GPU
   ```
2. **Kaggle:** tạo tài khoản, xác minh số điện thoại (bắt buộc để bật GPU và internet). Vào Settings → API →
   "Create New Token", được tệp `kaggle.json`. Đặt vào `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\`),
   hoặc đặt hai biến môi trường `KAGGLE_USERNAME`, `KAGGLE_KEY`. Không đưa tệp này vào git (đã có trong `.gitignore`).
3. **Hugging Face:** tạo token quyền **ghi** (Settings → Access Tokens), đặt biến `HF_TOKEN`; đặt
   `TTALLOC_HF_REPO=<user>/ttalloc-data` (dataset riêng tư, `ops/hf_sync.py` tự tạo nếu chưa có). Với GPQA ở
   `test_ood`: chấp nhận thêm điều khoản của `Idavidrein/gpqa` trên trang dataset.
4. **Vast.ai:** tài khoản đã có thì kiểm tra API key và số dư; nếu chưa có thì tạo tài khoản và cấu hình `VAST_API_KEY`. Chuẩn bị khoảng 15 USD cho gói tối thiểu, không phải bảo đảm đủ cho toàn bộ phần mở rộng. Nạp tiền không đồng nghĩa đã duyệt một offer cụ thể.
5. Đặt biến môi trường ở mức **user**, không ghi vào tệp trong repo, không dán vào chat. Trên Windows (PowerShell):
   ```powershell
   [Environment]::SetEnvironmentVariable('HF_TOKEN', '<token>', 'User')   # tương tự cho TTALLOC_HF_REPO, VAST_API_KEY
   ```
   Mở lại terminal để biến có hiệu lực.

### Cách chạy một công việc GPU trên Vast (dùng cho mọi bước ghi "Vast" bên dưới)

```bash
uv run --group ops python ops/hf_sync.py push-code                  # mã nguồn hiện tại -> Hugging Face
uv run --group ops python ops/vast/vast.py search [--interruptible]  # chọn offer; HỎI người dùng trước khi thuê
uv run --group ops python ops/vast/vast.py create --offer <id> --job <job> [--approved-by <tên>]
uv run --group ops python ops/vast/vast.py watch <id> --job <job> --destroy-when-done
uv run --group ops python ops/hf_sync.py pull data runs docs/test-log.md
uv run --group ops python ops/vast/vast.py status                   # xác nhận không còn máy nào
```

`<job>` là một tệp trong `ops/vast/jobs/`. Máy tự cài gói, tải lại dữ liệu đã có, chạy công việc, đẩy `data/` và
`runs/` lên mỗi 10 phút, ghi trạng thái `status/<job>.json`. Máy bị thu hồi thì tạo máy mới cùng `--job`: logger bỏ
qua các khóa đã ghi.

## P0. Nền tảng

| Bước | Lệnh | Ở đâu | Ghi chú |
|---|---|---|---|
| 1 | `uv run --group ops python ops/kaggle/push.py p0_smoke` | Máy cá nhân → Kaggle T4 | Đóng gói mã nguồn thành dataset riêng tư `<user>/ttalloc-src`, chạy `ops/kaggle/p0_smoke.py` trên GPU, chờ xong rồi tải `p0_smoke.json` về `ops/kaggle/_build/out-p0_smoke/`. Mất khoảng 15–25 phút, chủ yếu để cài vLLM |
| 1b | Đọc `p0_smoke.json`; cập nhật ADR-002 trong `docs/decisions.md` | — | Nếu bước `pip_install` hoặc `import_vllm` lỗi do CUDA (driver của Kaggle không hỗ trợ CUDA 13), thử lại với phiên bản khác: đặt biến `TTALLOC_VLLM` trong `ops/kaggle/p0_smoke.py` |
| 2 | Chốt `engine.dtype` | — | `bfloat16` cho Vast (RTX 3090). T4 không có bf16 nên smoke test dùng `float16`; dữ liệu thật không ghi trên T4. Ghi ADR |
| 3 | `uv run ttalloc splits build --skip test_ood` (thêm `test_ood` khi đã có `HF_TOKEN`) | Bất kỳ máy nào có mạng | **[dừng]** Con người xem `splits/*.json` (số câu, `dedup_pairs`), xác nhận khóa ID, rồi commit `splits/` |
| — | `uv run ttalloc splits verify` | | Chạy ở đầu mọi pipeline; thoát với mã lỗi nếu `splits/` bị sửa |

Đọc `p0_smoke.json`:

- `gpu_info.result`: tên GPU, driver, phiên bản CUDA tối đa driver hỗ trợ.
- `import_vllm.result`: phiên bản vLLM, torch, có bf16 hay không.
- `engine_checks.result`: các giả định cho P1 — `segment.stop_reason` phải là chuỗi dừng (ví dụ `"\n\nWait"`) khi đoạn dừng ở từ chuyển ý; `segment.n_logprobs` phải bằng `segment.n_tokens`; `probe_repeat_seconds` nhỏ hơn rõ rệt so với `probe.seconds` nếu prefix caching hoạt động; `throughput_batch32.tok_per_s` cho biết tốc độ trên T4.

## P1. Engine và công cụ ghi

Viết tắt: `E=configs/experiments/qwen3.yaml` (config chung của Qwen3-1.7B).

| Bước | Lệnh | Ở đâu | Ghi chú |
|---|---|---|---|
| 0 | `uv run --group ops python ops/kaggle/push.py p1_gpu_tests` | Kaggle T4 | Test GPU (`tests/gpu`, float16) và toàn bộ test CPU trên Linux. Đọc `ops/kaggle/_build/out-p1_gpu_tests/p1_gpu_tests.json` |
| 0b | công việc `gpu_check` | Một RTX 3090 trên Vast, theo yêu cầu, ≈ 0,5 giờ chạy job, chưa gồm cài đặt | Test GPU bf16 và throughput; đối chiếu số đo với dự toán. Gói tối thiểu không yêu cầu thuê thêm A100 (ADR-019) |
| 1 | công việc `p1_baseline` với `--approved-by <tên>` | Vast, theo yêu cầu, ≈ 6–8 giờ | **[dừng]** xin duyệt dùng `test_main` trước. `baseline full`/`deer` (`--profile deer_paper`): MATH-500 ×1, AIME 2024 ×4; `throughput`; `equivalence` (val 400 × 2 seed). So với dòng Qwen3-1.7B, bảng 2 bài DEER: **[dừng]** nếu MATH-500 lệch > 2 điểm phần trăm |
| 2 | `uv run ttalloc audit-grading --run runs/<baseline-full-test_main...> --n 100 --out audit.csv` rồi `uv run ttalloc audit-score audit.csv` | CPU | **[dừng]** chấm tay cột `manual_correct` (1/0); đạt khi khớp ≥ 99% |
| 3 | Đọc `runs/<equivalence...>/equivalence.json` | — | `pass` phải là true. Không kết luận được thì chạy lại với `--seeds 0-3` (PLAN P1.T6) |
| 4 | công việc `p1_pilot` | Vast, theo yêu cầu; study dự trù ≈ 1–2 giờ, ghi thử nằm trong khối lượng P2 | Ghi thử val 100 × 2 seed bằng `short` đã chọn (ADR-014), chạy hai study trên chuỗi dài thật. Tái sử dụng cùng `log_hash` ở P2; không xóa log short hợp lệ. Nếu cần đổi dạng trả lời phải duyệt và dự toán lại |
| 5 | Đọc `data/logs/qwen3_1_7b/val/validate.json` và log thời gian | — | `pass_rate`, `token_ratio`, độ dài trung bình và tốc độ hiệu dụng. **[dừng]** cập nhật dự toán từ pilot trước khi duyệt ghi lớn; không dùng throughput ngắn làm bảo đảm cho chuỗi dài |

Theo [ADR-023](../docs/decisions.md#adr-023-nới-biên-độ-chính-xác-cho-quyết-định-chạy-tiếp-p1), dùng job `p1_finish` để tiếp tục từ kết quả 4 seed đã xác minh. Job kiểm tra các checkpoint trước khi chạy pilot, `validate-logs` và hai study. Khi tiếp tục một job bị ngắt, dùng lại `TTALLOC_P1_SESSION` để giữ các stage đã đồng bộ.

## P2. Thăm dò

| Bước | Lệnh | Ở đâu | Ghi chú |
|---|---|---|---|
| 1 | công việc `p2_log` (`--interruptible` được) | Một RTX 3090, dự toán ≈ 14 giờ theo giả định; thay bằng số đo P1 | **[dừng]** duyệt ghi lớn và chi phí. val × 8 seed, train 500 câu × 4 seed, `validate-logs`; logger bỏ qua 200 chuỗi pilot hợp lệ. Đồng bộ xong thì destroy máy, không giữ GPU chờ CPU |
| 2 | `uv run --group ops python ops/hf_sync.py pull data` | CPU | Lấy dữ liệu ghi về máy cá nhân |
| 3 | `uv run ttalloc estimator train --exp E` | CPU | Lưu ở `data/estimators/qwen3_1_7b/<mã băm>/` |
| 4 | `uv run ttalloc estimator eval --exp E --split val` | CPU | E1 trên val: `runs/<estimator-eval-val...>/eval.json` |
| 4b | `uv run ttalloc estimator learning-curve --exp E --sizes 250,500` | CPU | ADR-015: xác định 500 câu đã đủ hay cần tăng. Nếu đề xuất tăng, báo chi phí và xin duyệt phạm vi mới trước ghi thêm (ADR-019) |
| 5 | `uv run ttalloc table build --exp configs/experiments/g1.yaml --split val` | CPU | Bảng tính sẵn theo mã băm |
| 6 | `uv run ttalloc sim run --exp configs/experiments/g1.yaml` | CPU | B2, M2, M3 × lô all/balanced/easy/hard/u × alpha ≤ 0,5 × 200 lần rút. `sim.workers` chỉ có tác dụng trên Linux (fork) |
| 7 | `uv run ttalloc study curves --exp E --split val` | CPU | P2.T20 |
| 8 | `uv run ttalloc gate g1 --runs runs/<sim-g1...> --est-eval runs/<estimator-eval-val...>/eval.json --validate data/logs/qwen3_1_7b/val/validate.json` | CPU | **[dừng]** người dùng quyết định đi tiếp |

## P3. Mô phỏng đầy đủ

| Bước | Lệnh | Ở đâu | Ghi chú |
|---|---|---|---|
| 1 | công việc `p3_log` với `--env TTALLOC_TRAIN_LIMIT=500` (`--interruptible` được khi tìm offer) | Một RTX 3090, dự toán ≈ 12 giờ theo giả định; thay bằng số đo P1/P2 | **[dừng]** duyệt GPU và chi phí còn lại. Giữ train đã ghi ở P2, chỉ ghi thêm test_main × 8 seed; không tự tăng tới 1.500. Nếu learning curve yêu cầu tăng, dừng và dự toán lại trước. Job chỉ ghi, không đánh giá |
| 2 | `uv run ttalloc estimator train --exp E` và `uv run ttalloc estimator train --exp E --set estimator.kind=logreg`; `estimator eval --split val` cho từng loại | CPU | Chọn loại có `mean_log_loss` thấp hơn trên val, điền `estimator.kind` vào `qwen3.yaml`, ghi ADR |
| 3 | `uv run ttalloc table build --exp configs/experiments/tune_val.yaml --split val` rồi `uv run ttalloc tune --exp configs/experiments/tune_val.yaml` | CPU | 50 cấu hình mỗi phương pháp. Điền thư mục chạy vào `tuned_from` của `configs/experiments/e2_main.yaml` |
| 4 | `uv run ttalloc lock --exp configs/experiments/e2_main.yaml` | CPU | **[dừng]** người dùng xem `configs/locked/e2_main_qwen3_1_7b.yaml` rồi commit |
| 5 | `uv run ttalloc eval-test --exp configs/locked/e2_main_qwen3_1_7b.yaml --approved-by <tên>` | CPU | E2 (và đọc E3 theo thứ tự B2 → M2 → M3). Lệnh tự dựng bảng test_main nếu chưa có |
| 6 | `uv run ttalloc estimator eval --split test_main --exp configs/locked/e2_main_qwen3_1_7b.yaml` | CPU | E1 trên test_main (chỉ sau khi khóa) |
| 7 | Với mỗi `configs/experiments/e4/<x>.yaml`: `estimator train` (nếu tệp ghi "học lại"), `lock`, `eval-test` | CPU | E4. Có 19 biến thể; chạy theo nhóm người dùng chọn |
| 8 | `uv run ttalloc report --runs runs/<eval-test-e2...> [runs/<eval-test-e4...> ...] --est-eval runs/<estimator-eval-test_main...>/eval.json --out reports/p3` | CPU | Bảng và hình |
| 9 | `uv run ttalloc gate g2 --runs runs/<eval-test-e2...>` | CPU | **[dừng]** |

## P4. Chạy thật

| Bước | Lệnh | Ở đâu | Ghi chú |
|---|---|---|---|
| 1 | `uv run ttalloc lock --exp configs/experiments/e5_online.yaml` | CPU | **[dừng]** người dùng xem rồi commit `configs/locked/e5_online_qwen3_1_7b.yaml` |
| 2 | `uv run --group ops python ops/hf_sync.py push data/estimators data/tables runs` | CPU | Máy GPU cần bộ ước lượng, bảng và `tuned_from` |
| 3 | công việc `p4_online` với `--approved-by <tên>` | Vast, theo yêu cầu, ≈ 4–5 giờ | **[dừng]** duyệt GPU trước. Chạy thử val (`online_val_pilot.yaml`), lưới E5 (6 lần), so thời gian với DEER chính thức (P4.T5) |
| 4 | `uv run ttalloc gate g3 --runs runs/<eval-test-e5...>` | CPU | **[dừng]** |

## Khi có sự cố

- `log_hash ... khác config hiện hành (I9)`: config ghi dữ liệu (dp.markers, log.*, probe, answer, local_conf, lời nhắc,
  dtype) đã đổi so với lúc ghi. Không sửa tay meta.json; hoặc trả config về như cũ, hoặc ghi lại vào thư mục mới.
- `answer.mode chưa được chọn`: kiểm tra đang dùng `configs/experiments/qwen3.yaml` với `answer.mode=short` đã chốt theo ADR-014; không tự chuyển về `full` hoặc xóa dữ liệu đã ghi. Nếu config/lời nhắc phải đổi thì duyệt và tính lại chi phí ghi.
- `InfeasibleBudget`: alpha quá nhỏ so với phần dự trữ (N × (c_ans_max + b_min + c_probe_max)).
- `budget_underused` trong summary của B2: ngay cả hạn mức tốt nhất cũng không tiêu hết 98% B (ADR-011); báo cáo
  kèm số này.
