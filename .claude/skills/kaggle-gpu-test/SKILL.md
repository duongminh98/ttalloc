---
name: kaggle-gpu-test
description: Chạy một script GPU của repo ttalloc trên Kaggle (T4), theo dõi log liên tục trong lúc chạy, tự sửa lỗi và chạy lại. Dùng khi cần test GPU, smoke test vLLM, hoặc chạy bước chạy nhỏ của P0/P1 trên Kaggle (không dùng cho ghi dữ liệu thật hay chạy thật; các việc đó chạy trên Vast, skill vast-run).
---

# Chạy script GPU trên Kaggle, theo dõi và tự sửa

## Chuẩn bị
- `uv run pytest` đạt (test CPU). Không đẩy mã đang lỗi lên GPU.
- Có Kaggle token: biến user `KAGGLE_API_TOKEN` (token kiểu mới `KGAT_...`), hoặc `KAGGLE_USERNAME` + `KAGGLE_KEY`, hoặc `~/.kaggle/kaggle.json`. Trong PowerShell của agent, nạp lại biến mức user trước mỗi lệnh: `$env:KAGGLE_API_TOKEN=[Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')`. Không in token, không yêu cầu dán token vào chat.
- Script nằm ở `ops/kaggle/<tên>.py`, ghi báo cáo từng bước (`ok`, `error`, `traceback`) vào `/kaggle/working/<tên>.json`, không chạm `test_*`. Kaggle hiện dùng Python 3.13, nên `pip install -e` cần `--ignore-requires-python`. T4 không có bf16 gốc (vLLM cần compute capability ≥ 8.0): dùng float16.

## Vòng chạy – theo dõi – sửa
1. Đẩy (không chờ): `uv run --group ops python ops/kaggle/push.py <tên> --no-wait` (thêm `--no-upload` nếu mã không đổi).
2. Theo dõi **chạy nền** (`run_in_background`), để nhận thông báo khi lệnh thoát:
   `uv run --group ops python ops/kaggle/push.py <tên> --watch-only --fail-fast --poll 60`
   Lệnh tải log và báo cáo mỗi phút ngay khi kernel đang chạy (`kaggle kernels output` lấy được kết quả từng phần), in dòng log mới và bước lỗi, rồi thoát với mã:
   - `0`: xong, mọi bước đạt → sang bước 4.
   - `2`: có bước lỗi → bước 3.
   - `4`: log có dấu hiệu sự cố (traceback, OOM, lỗi CUDA, lỗi pip; quy tắc ở `ops/watch_rules.py`) → bước 3.
   - `1`: kernel lỗi hoặc bị hủy → bước 3.
   Trong lúc chờ, có thể đọc tệp tạm `ops/kaggle/_build/out-<tên>/<tên>.json` và `ttalloc-<tên>.log` để xem tiến độ.
3. Sửa lỗi:
   - Đọc `error`/`traceback` của bước lỗi và log. Xác định nguyên nhân gốc (môi trường Kaggle, phiên bản gói, lỗi mã), không vá triệu chứng.
   - Sửa trên máy cục bộ; thêm test CPU tái hiện lỗi nếu lỗi nằm trong mã `src/` hoặc `ops/`; chạy `uv run pytest` và `uv run ruff check src tests ops`.
   - Đẩy lại (bước 1, có upload). Kernel cũ đang chạy dở với mã lỗi sẽ bị phiên bản mới thay thế.
   - Tự lặp tối đa **3 lần cho cùng một lỗi**. Quá 3 lần, hoặc lỗi cần quyết định về phương pháp (đổi phiên bản vLLM, đổi tham số trong PLAN, đổi đặc tả), thì dừng và hỏi người dùng, kèm chẩn đoán.
4. Kiểm số liệu, không chỉ kiểm "chạy được": so với kỳ vọng ở `ops/RUNBOOK.md` (ví dụ `p0_smoke`: `segment.stop_reason` là chuỗi dừng, `n_logprobs == n_tokens`, lần gợi lặp lại nhanh hơn rõ rệt nhờ prefix caching; `p1_gpu_tests`: mọi test GPU đạt). Số liệu lệch kỳ vọng thì coi như lỗi: tìm nguyên nhân trong mã trước. **Không** chỉnh tham số hay ngưỡng để số liệu "đẹp" hơn; điểm [dừng] vẫn là quyết định của người dùng.
5. Báo người dùng: bước nào đạt, lỗi nào đã gặp và đã sửa ra sao, số lần chạy lại, thời gian GPU đã dùng. Cập nhật `docs/progress.md` và ADR liên quan trong `docs/decisions.md`.

## Hạn mức
Kaggle cho khoảng 30 giờ GPU mỗi tuần, tính cả thời gian kernel chạy. Không đẩy lặp lại khi chưa sửa gì. Mỗi lần chạy lại, nhắc tổng thời gian đã dùng.
