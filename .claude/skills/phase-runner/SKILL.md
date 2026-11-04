---
name: phase-runner
description: Đi tiếp bước chạy kế tiếp của khóa luận ttalloc (P0–P4) theo ops/RUNBOOK.md, chọn đúng nơi chạy (máy cá nhân CPU, Kaggle, Vast), dừng ở các điểm [dừng]. Dùng khi người dùng hỏi "chạy bước tiếp theo", "làm tiếp P2", hoặc muốn biết đang ở đâu.
---

# Đi tiếp theo RUNBOOK

1. Đọc `docs/progress.md` (trạng thái task và bước chạy) và `ops/RUNBOOK.md` (thứ tự bước, nơi chạy, điều kiện). Xác định bước kế tiếp chưa xong mà điều kiện bắt đầu đã thỏa. Nếu bước trước có **[dừng]** mà chưa có quyết định của người dùng, dừng và hỏi.
2. Chọn nơi chạy theo cột "Ở đâu":
   - **CPU (máy cá nhân):** chạy thẳng bằng `uv run ttalloc ...` (đặt `$env:UV_LINK_MODE='copy'` trên Windows). Các bước học bộ ước lượng, dựng bảng, mô phỏng, chỉnh tham số, khóa, báo cáo, cổng đều chạy được trên CPU. Dữ liệu ghi phải có trong `data/` (tải bằng `python ops/hf_sync.py pull data runs`).
   - **Kaggle:** dùng skill `kaggle-gpu-test`.
   - **Vast:** dùng skill `vast-run` (luôn hỏi trước khi thuê máy).
3. Trước mọi bước trên model hoặc dữ liệu thật: `uv run pytest` đạt; `uv run ttalloc splits verify` đạt.
4. Tập kiểm tra (`test_*`): chỉ qua `ttalloc eval-test` với config đã khóa, hoặc lệnh `baseline` có `--approved-by` (ngoại lệ P1.T5). Tên người duyệt do người dùng cung cấp. `lock` là bước **[dừng]**: cho người dùng xem config đã phân giải trước khi khóa.
5. Lệnh `gate` chỉ báo số liệu. Không tự quyết định đi tiếp, kể cả khi cổng đạt (PLAN mục 7.4): báo bảng tiêu chí và chờ người dùng.
6. Sau mỗi bước: ghi thư mục `runs/...` sinh ra, số liệu chính, và cập nhật `docs/progress.md` (cột Ghi chú). Quyết định của người dùng ở điểm [dừng] ghi vào `docs/decisions.md` dưới dạng ADR.
7. Không commit nếu người dùng chưa yêu cầu.
