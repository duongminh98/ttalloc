---
name: kaggle-gpu-test
description: Chạy smoke test, kiểm thử GPU hoặc tập dượt nhỏ của repo ttalloc trên Kaggle, theo dõi đầu ra và sửa lỗi có giới hạn. Dùng cho script ops/kaggle; không dùng để ghi dữ liệu đầy đủ, chạy lưới thí nghiệm thật hoặc đánh giá test_*.
---

# Kiểm thử ttalloc trên Kaggle

Chạy từ gốc repo. Đọc `AGENTS.md`, phần tương ứng của `docs/PLAN.md`,
`ops/RUNBOOK.md`, script `ops/kaggle/<tên>.py` và tiến độ hiện tại. Người dùng
yêu cầu test GPU cho phép thực hiện workflow test này; chuyển đổi skill hoặc
hỏi trạng thái không tự cho phép đẩy kernel lên Kaggle.

## Chuẩn bị

- `uv run pytest` phải đạt trước khi đẩy mã. Khi sửa mã, chạy đủ
  `uv run ruff check`, `uv run mypy src` và kiểm thử của task.
- Dùng thông tin xác thực đã cấu hình: `KAGGLE_API_TOKEN`, hoặc cặp
  `KAGGLE_USERNAME`/`KAGGLE_KEY`, hoặc `~/.kaggle/kaggle.json`.
  Trên Windows, nạp từ mức User khi tiến trình chưa kế thừa, chẳng hạn:
  `$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN', 'User')`.
  Không in, ghi vào tệp mới, commit hay yêu cầu dán token vào chat.
- Script ghi kết quả từng bước và lỗi vào `/kaggle/working/<tên>.json`;
  kiểm tra không chạm tập `test_*`. Đầu ra tập dượt không phải số liệu nghiên cứu.
- T4 dùng `float16`, không có bf16 gốc. Không trộn dữ liệu này với dữ liệu bf16
  của GPU thuê. Nếu runtime Kaggle dùng Python 3.13, bản cài editable cần
  `--ignore-requires-python` theo script hiện hành; không thay ràng buộc Python
  toàn repo để né lỗi môi trường.

## Chạy và theo dõi bằng Codex

```text
uv run --group ops python ops/kaggle/push.py <tên> --no-wait
uv run --group ops python ops/kaggle/push.py <tên> --watch-only --fail-fast --poll 30
```

Thêm `--no-upload` khi mã không đổi và thực sự cần chạy lại; không đẩy lặp vô cớ.
Với lệnh watcher dài, dùng `exec_command` để nhận `session_id`, lấy đầu ra tiếp
bằng `write_stdin` với mỗi lần chờ không quá 60 giây, và cập nhật người dùng.
Không dùng `run_in_background` của Claude. Nếu lớp `functions.exec` trả cell ID,
tiếp tục cell bằng `functions.wait`; không nhầm với session ID shell.

Watcher lấy log và báo cáo mỗi 30 giây, có thể lấy kết quả khi kernel đang chạy.
Đọc `ops/kaggle/_build/out-<tên>/<tên>.json` và `ttalloc-<tên>.log` khi cần.

| Mã thoát | Xử lý |
|---|---|
| 0 | Mọi bước đạt; kiểm tra số liệu và thu thập kết quả. |
| 2 | Có bước lỗi; đọc báo cáo để sửa nguyên nhân. |
| 4 | Có dấu hiệu lỗi trong log; xác định lỗi thật trước khi chạy lại. |
| 1 | Kernel lỗi hoặc bị hủy; kiểm tra log, hạn mức và trạng thái. |

## Sửa lỗi và báo cáo

1. Phân biệt lỗi môi trường, phụ thuộc và lỗi mã bằng `error`, `traceback`, log.
   Thêm test CPU tái hiện khi sửa lỗi trong `src/` hoặc `ops/`; chạy đủ các kiểm tra.
2. Chỉ đẩy bản mới sau khi đã sửa. Việc đẩy kernel mới có thể thay thế lượt đang
   chạy; không đẩy đè một lượt khác của người dùng.
3. Tối đa 3 vòng sửa cho cùng lỗi. Quá giới hạn, hết hạn mức hoặc cần thay đổi
   phương pháp/phiên bản đã chốt thì dừng vòng lặp và báo chẩn đoán. Nếu cần hủy
   kernel do workflow này tạo, dùng `push.py <tên> --stop` và xác nhận trạng thái;
   không bỏ mặc một lượt chạy không còn hữu ích tiêu hạn mức.
4. Kiểm số liệu theo RUNBOOK, không chỉ mã thoát: điểm dừng, số logprob,
   cache, kết quả test GPU và chỉ số của tập dượt. Phép đo quá ngắn chưa chứng
   minh cache nhanh hơn; báo thiếu bằng chứng thay vì sửa ngưỡng để đạt.
5. Báo các bước đạt, lỗi/sửa, số vòng chạy lại, thời gian GPU và đường dẫn đầu ra.
   Cập nhật `docs/progress.md`, ADR nếu có quyết định cần ghi lại. Giữ nguyên các
   điểm `[dừng]` theo PLAN, kể cả khi kiểm thử đã đạt.

Hạn mức và phiên bản runtime Kaggle có thể thay đổi. Kiểm hạn mức thực tế trước
lượt dài, không coi con số khoảng 30 giờ/tuần trong skill Claude là bảo đảm.
