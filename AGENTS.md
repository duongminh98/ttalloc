# Quy tắc cho agent

Đặc tả đầy đủ ở [docs/PLAN.md](docs/PLAN.md). Tệp này chỉ tóm tắt những quy tắc phải nhớ khi làm việc trong repo.

## Nguồn chuẩn

1. Đọc `docs/PLAN.md` trước khi viết mã. Ngữ nghĩa phương pháp ở `docs/Bao-cao-Phuong-phap-chi-tiet.md`.
2. Khi mâu thuẫn: `docs/PLAN.md` được ưu tiên. Mọi mâu thuẫn hoặc chỗ chưa rõ ghi vào `docs/decisions.md` kèm cách xử lý. Không lệch đặc tả mà không ghi lại.

## Quy trình cho mỗi task (PLAN mục 0.2)

1. Đọc đặc tả task ở PLAN mục 6 và các mục được dẫn.
2. Viết mã và mọi test được liệt kê. Được thêm test, không được bỏ test.
3. Chạy `uv run ruff check`, `uv run mypy src`, `uv run pytest`. Task có test GPU thì chạy `pytest -m gpu` trên máy GPU (xem `ops/RUNBOOK.md`).
4. Tự review theo checklist ở PLAN mục 8.2.
5. Commit với thông điệp `[P<k>.T<n>] <mô tả ngắn>`, rồi cập nhật `docs/progress.md`.

## Điểm phải dừng chờ con người (PLAN mục 0.4)

- Bước có **[dừng]** trong đặc tả phase.
- Tác vụ GPU lớn: ghi dữ liệu đầy đủ, lưới chạy thật.
- Khóa hoặc thay đổi danh sách ID của các tập.
- Khóa cấu hình, hoặc chạy bất kỳ lệnh nào trên `test_*`.
- Kết thúc một phase, kể cả khi cổng đạt.
- Thay đổi định nghĩa đếm token (PLAN mục 2.6) hoặc một bất biến (PLAN mục 2.7).
- Thuê máy trên Vast.ai (tốn tiền).

## Quy ước viết mã (PLAN mục 2.3)

- Type hint đầy đủ; dữ liệu bất biến dùng `@dataclass(frozen=True)`.
- Không có trạng thái toàn cục; mọi nguồn ngẫu nhiên là `numpy.random.Generator` được truyền vào.
- Chỉ `engine/vllm_generator.py` được import `vllm`, và import bên trong hàm.
- Không viết cứng tham số; mọi tham số đến từ config và có trong `config_hash`.
- Lõi tính toán là hàm thuần; đọc ghi tệp và gọi GPU chỉ ở lớp ngoài (CLI, runner, logger).
- Kết quả ghi vào thư mục mới trong `runs/`, không ghi đè.

## Môi trường

- Máy phát triển: Windows, Python 3.12 qua `uv`, không có GPU. Test mặc định: `-m "not gpu and not slow and not net"`.
- Máy GPU: Kaggle (phát triển, test GPU) và Vast.ai RTX 3090 (ghi dữ liệu, chạy thật). Cách chạy ở `ops/RUNBOOK.md`.
