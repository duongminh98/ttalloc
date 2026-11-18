---
name: phase-runner
description: Xác định tiến độ và thực hiện bước chạy kế tiếp của repo ttalloc theo PLAN và ops/RUNBOOK.md, chọn CPU, Kaggle hoặc Vast.ai và tôn trọng các điểm dừng. Dùng khi được yêu cầu chạy bước tiếp theo, làm tiếp một phase hoặc kiểm tra đang ở bước nào; hỏi tiến độ không tự cho phép chạy thí nghiệm.
---

# Điều phối các bước thực nghiệm ttalloc

Làm việc từ gốc repo chứa `docs/PLAN.md`. Đọc `AGENTS.md`, `docs/progress.md`,
`ops/RUNBOOK.md`, rồi PLAN mục 0, 6 và đặc tả bước liên quan. Đọc các ADR liên quan
trong `docs/decisions.md` trước khi quyết định theo cấu hình cũ.

1. Xác định bước chưa hoàn thành có đủ điều kiện bắt đầu. Phân biệt “mã xong” với
   “đã chạy”, “đã review” và “đã được duyệt kết thúc phase”. Khi chỉ hỏi trạng thái,
   báo vị trí, bằng chứng và bước còn thiếu; không tự khởi chạy.
2. Khi được yêu cầu thực hiện, chọn môi trường theo RUNBOOK:
   - CPU Windows: học bộ ước lượng, bảng tính sẵn, mô phỏng, tune, báo cáo và cổng.
     Dùng `uv run ttalloc ...`; có thể đặt `$env:UV_LINK_MODE='copy'`.
     Nếu thiếu dữ liệu, tải bằng `uv run --group ops python ops/hf_sync.py pull data runs`
     khi thuộc phạm vi công việc đã yêu cầu.
   - Kaggle: đọc và dùng [kaggle-gpu-test](../kaggle-gpu-test/SKILL.md).
   - Vast.ai: đọc và dùng [vast-run](../vast-run/SKILL.md).
3. Trước thao tác trên mô hình hoặc dữ liệu thật, `uv run pytest` và
   `uv run ttalloc splits verify` phải đạt. Khi có sửa mã, chạy cả
   `uv run ruff check`, `uv run mypy src` và kiểm thử bắt buộc theo task.
4. Trước bước `[dừng]`, ghi dữ liệu GPU lớn, thuê máy, khóa/thay đổi split,
   khóa config, chạy `test_*` hoặc kết thúc phase, kiểm tra quyền đã có trong
   cuộc trò chuyện cho đúng phạm vi. Nếu chưa có, chuẩn bị đầu vào cụ thể để
   người dùng duyệt rồi dừng phần phụ thuộc. Không dùng phê duyệt lịch sử trong
   skill Claude làm quyền mới; không hỏi lại nếu đã được duyệt rõ ràng.
5. Tập kiểm tra chỉ qua `ttalloc eval-test` với config đã khóa và tên người duyệt;
   ngoại lệ baseline P1.T5 theo PLAN mục 7.5. Trước `lock`, chuẩn bị cho người dùng
   xem config đã phân giải; không khóa trước khi có quyền. Không chỉnh tham số
   từ kết quả kiểm tra hoặc lấy số liệu tập dượt làm kết luận nghiên cứu.
6. `gate` chỉ tính và báo bảng tiêu chí. Dù đạt, chưa có quyền thì không vượt
   điểm dừng hoặc tự kết thúc phase. Nếu không đạt, báo rõ chỉ số và phần việc
   có thể làm tiếp trong phạm vi đã duyệt.
7. Sau bước chạy, ghi đường dẫn `runs/...`, cấu hình, số liệu chính và trạng thái
   vào `docs/progress.md`. Ghi quyết định hoặc mâu thuẫn vào `docs/decisions.md`.
   Không suy ra một cổng đạt khi chưa có đầu ra. Với task implement, thực hiện
   quy trình review/commit của PLAN; không áp dụng quy tắc “không commit” của
   skill Claude nếu trái chỉ dẫn hiện hành.

Giữ kết quả ở thư mục chạy mới. Không thực hiện lại bước tốn tiền hoặc trên tập
kiểm tra chỉ để có kết quả tốt hơn. Nếu bước đang yêu cầu nằm ngoài P0–P4, lấy
đúng đặc tả phase đó từ PLAN thay vì suy rộng lưới thí nghiệm chính.
