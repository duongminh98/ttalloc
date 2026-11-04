---
name: vast-run
description: Thuê máy RTX 3090 trên Vast.ai để chạy một công việc GPU của ttalloc (tái lập baseline, ghi dữ liệu, chạy thật E5), theo dõi log liên tục qua Hugging Face, tự sửa lỗi để máy chạy tiếp, và xóa máy khi xong. Dùng khi một bước chạy trong ops/RUNBOOK.md ghi "Vast". Không dùng cho test GPU nhỏ (dùng kaggle-gpu-test).
---

# Chạy một công việc trên Vast.ai, theo dõi và tự sửa

Máy cá nhân chỉ ra mạng qua HTTPS (SSH bị chặn), nên máy Vast tự chạy:
- onstart tải mã nguồn từ dataset Hugging Face riêng tư, chạy `ops/vast/bootstrap.sh`, rồi `ops/vast/jobs/<job>.sh`;
- dữ liệu được đẩy lên mỗi 10 phút; đuôi log (`status/<job>.log`) và trạng thái (`status/<job>.json`) mỗi 3 phút;
- công việc lỗi thì máy **không tắt** mà chuyển sang `waiting_fix` và chờ tối đa 45 phút. Khi có bản mã mới (`hf_sync.py push-code`), máy tự giải nén, cài lại và chạy lại công việc. Logger bỏ qua các khóa đã ghi, nên không mất dữ liệu. Tối đa 4 lần chạy mỗi máy.

## Quy tắc bắt buộc
- **Thuê máy tốn tiền: luôn hỏi người dùng trước khi `create`**, nêu offer, giá USD/giờ, công việc, số giờ ước tính (ops/RUNBOOK.md). Chỉ thêm `--yes` sau khi người dùng đồng ý trong cuộc trò chuyện này. Sửa lỗi để máy *đang chạy* chạy tiếp thì không cần hỏi lại; tạo máy *mới* thì phải hỏi lại.
- Xong việc thì **destroy**, không chỉ stop. Trước khi kết thúc lượt làm việc, chạy `vast.py status` để chắc không còn máy nào chạy ngoài ý muốn. Nếu không sửa kịp trong lúc máy chờ (`waiting_fix`), destroy máy, đừng để nó chờ tốn tiền.
- Không in, ghi vào tệp, commit hay yêu cầu dán token (`VAST_API_KEY`, `HF_TOKEN`). Trong PowerShell của agent, nạp biến mức user trước mỗi lệnh, ví dụ `$env:HF_TOKEN=[Environment]::GetEnvironmentVariable('HF_TOKEN','User')`.
- Công việc chạm `test_*` (`p1_baseline`, `p4_online`) cần `--approved-by <tên>` do **người dùng** cung cấp (PLAN mục 7.5).
- Bước có **[dừng]** trong RUNBOOK: dừng lại, báo kết quả, chờ người dùng.

## Chuẩn bị
- `uv run pytest` đạt; `splits/` đã khóa (công việc nào cũng chạy `ttalloc splits verify`).
- `p2_log`, `p3_log`: `answer.mode` đã chốt trong `configs/experiments/qwen3.yaml`.
- `p4_online`: `configs/locked/e5_online_qwen3_1_7b.yaml` có trong mã; đã `python ops/hf_sync.py push data/estimators data/tables runs`.
- Đẩy mã: `uv run --group ops python ops/hf_sync.py push-code`.
- Tìm máy: `uv run --group ops python ops/vast/vast.py search [--interruptible]` (máy có thể bị ngắt rẻ hơn, hợp cho công việc ghi dữ liệu). Báo 3 lựa chọn rẻ nhất và **hỏi xác nhận**.
- Tạo máy: `uv run --group ops python ops/vast/vast.py create --offer <id> --job <job> [--approved-by <tên>] --yes`. Ghi lại id máy.

## Vòng theo dõi – sửa
1. Theo dõi **chạy nền** (`run_in_background`):
   `uv run --group ops python ops/vast/vast.py watch <id> --job <job> --fail-fast --destroy-when-done --poll 120`
   Lệnh in dòng log mới mỗi 2 phút và thoát với mã:
   - `0`: xong (đã destroy) → bước 4.
   - `2`: công việc lỗi; máy đang `waiting_fix` → bước 2 ngay.
   - `4`: log có dấu hiệu sự cố (traceback, OOM, lỗi CUDA, `LỖI`, `BudgetExceeded`…; `ops/watch_rules.py`) trong khi công việc có thể vẫn chạy → đọc kỹ: lỗi đã làm công việc dừng thì sang bước 2; chỉ là cảnh báo thì chạy lại `watch` (không cần `--fail-fast` cho đến dòng tiếp theo).
   - `3`: máy không còn (bị thu hồi) → hỏi người dùng có thuê máy mới cùng `--job` không.
   - `5`: log đứng yên quá 40 phút → xem `vast.py logs <id>` và `hf_sync.py tail <job>`; treo thật thì destroy và báo người dùng.
2. Sửa lỗi:
   - Đọc `python ops/hf_sync.py tail <job> -n 200`. Xác định nguyên nhân gốc.
   - Sửa trên máy cục bộ, thêm test CPU tái hiện lỗi nếu lỗi nằm trong mã, chạy `uv run pytest` và `uv run ruff check src tests ops`.
   - `uv run --group ops python ops/hf_sync.py push-code`. Máy đang chờ sẽ tự chạy lại trong vòng 1 phút. Quay lại bước 1.
   - Tự lặp tối đa **3 lần cho cùng một lỗi**, và luôn trong thời gian máy chờ (45 phút). Quá giới hạn, hoặc lỗi cần quyết định về phương pháp, thì destroy máy rồi hỏi người dùng, kèm chẩn đoán.
3. Kiểm số liệu trong lúc chạy và sau khi xong (so với `ops/RUNBOOK.md`): thông lượng (token/giây) trong khoảng dự kiến; `validate-logs` có `pass_rate` ≥ 0,95; tỉ lệ chuỗi chạm `L_log` và tỉ lệ không trích được đáp án không bất thường; baseline MATH-500 lệch bài DEER không quá 2 điểm phần trăm; `equivalence.json` có `pass`. Số liệu lệch thì tìm lỗi trong mã trước. **Không** chỉnh tham số hay ngưỡng để số liệu "đẹp" hơn, không chạy lại trên `test_*` để chọn kết quả (mục 7.5). Số liệu đúng là quyết định của người dùng ở điểm [dừng].
4. Tải kết quả: `uv run --group ops python ops/hf_sync.py pull data runs docs/test-log.md`. Kiểm `git diff docs/test-log.md`.
5. `vast.py status` xác nhận không còn máy. Cập nhật `docs/progress.md`. Báo người dùng: thời gian, chi phí ước tính (giờ × giá), các lỗi đã gặp và cách sửa, kết quả chính, bước tiếp theo.
