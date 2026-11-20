---
name: vast-run
description: Chạy và theo dõi một công việc GPU của repo ttalloc trên Vast.ai qua HTTPS và Hugging Face, sửa lỗi có giới hạn và thu hồi máy khi xong. Dùng cho baseline, ghi dữ liệu hoặc chạy thật theo ops/RUNBOOK.md; test GPU nhỏ dùng kaggle-gpu-test. Chỉ tìm giá khi người dùng yêu cầu ước tính, không tự thuê máy.
---

# Chạy công việc ttalloc trên Vast.ai

Thực hiện lệnh từ gốc repo có `docs/PLAN.md` và `ops/vast/vast.py`.
Đọc `AGENTS.md`, PLAN mục 0.4, đặc tả bước đang làm, `ops/RUNBOOK.md`
và `docs/progress.md`. PLAN là nguồn chuẩn khi tài liệu khác mâu thuẫn.

## Phạm vi và quyền thực hiện

- Đề nghị báo giá hoặc chuyển đổi skill không cho phép tạo máy. Trước khi thuê,
  trình bày offer, loại GPU, giá gồm ổ đĩa, chi phí mạng, job, số giờ và tổng tiền
  ước tính. Dùng phê duyệt còn hiệu lực trong cuộc trò chuyện cho đúng phạm vi;
  không hỏi lại khi đã được duyệt. Thuê máy mới hoặc vượt phạm vi chi phí đã duyệt
  cần xin duyệt. Chỉ thêm `--yes` sau khi có quyền thuê cụ thể.
- Các bước `[dừng]`, tác vụ GPU lớn, khóa cấu hình và chạy trên `test_*` cần có
  phê duyệt theo PLAN. Không coi các câu “đã duyệt ngày 04/11/2026” trong skill
  Claude là quyền cho phiên hiện tại. `--approved-by` ghi đúng người đã duyệt.
- Chỉ sửa để job đang được phép chạy tiếp trong cùng phạm vi. Không thay ngưỡng,
  định nghĩa token, bất biến, dataset hoặc tham số phương pháp để vượt cổng.
- Chỉ destroy đúng ID máy của công việc này. Việc dọn máy sau job nằm trong vòng
  đời thuê đã duyệt; không destroy máy khác của tài khoản.

## Chuẩn bị và khởi chạy

Nạp các biến `HF_TOKEN`, `TTALLOC_HF_REPO`, `VAST_API_KEY` từ môi trường đã cấu hình;
trên Windows có thể nạp từng biến từ mức User nếu tiến trình chưa kế thừa:

```powershell
$env:HF_TOKEN = [Environment]::GetEnvironmentVariable('HF_TOKEN', 'User')
$env:TTALLOC_HF_REPO = [Environment]::GetEnvironmentVariable('TTALLOC_HF_REPO', 'User')
$env:VAST_API_KEY = [Environment]::GetEnvironmentVariable('VAST_API_KEY', 'User')
```

Không in giá trị, lưu vào file, commit hoặc yêu cầu dán token vào chat.
Thiếu quyền hoặc thông tin xác thực thì báo thiếu gì; không tạo thông tin giả.

- Chạy `uv run pytest` và `uv run ttalloc splits verify` trước dữ liệu thật.
- `p2_log`, `p3_log`: kiểm tra dạng trả lời đã chốt; `p3_log` nhận
  `TTALLOC_TRAIN_LIMIT` từ learning curve theo ADR-015.
- `p4_online`: cần config khóa đã duyệt, bộ ước lượng và bảng tính sẵn tương ứng.
- Chỉ đẩy mã/dữ liệu cần cho job đã được yêu cầu; kiểm tra nội dung trước khi upload.

```text
uv run --group ops python ops/hf_sync.py push-code
uv run --group ops python ops/vast/vast.py search --limit 3
uv run --group ops python ops/vast/vast.py search --interruptible --limit 3
uv run --group ops python ops/vast/vast.py create --offer <offer> --job <job> --yes
```

Thêm `--approved-by <tên>` khi job yêu cầu; `--env TTALLOC_TRAIN_LIMIT=<n>` cho
`p3_log`. Gói tối thiểu hiện hành theo ADR-019 kiểm tra một RTX 3090 trước,
không thuê thêm A100 chỉ để so tốc độ. Chỉ đổi GPU khi có số đo, dự toán và quyền
thực hiện phù hợp; nguyên tắc chi phí mỗi token của ADR-016 vẫn áp dụng.
Ghi ID máy, offer, đơn giá và giờ bắt đầu, không kèm bí mật.

## Theo dõi bằng Codex

Máy cá nhân chỉ dùng HTTPS: onstart tải mã từ Hugging Face, chạy bootstrap rồi job.
Dữ liệu đồng bộ khoảng 10 phút/lần; đuôi log mỗi phút khi đổi. Job lỗi có thể chờ
bản sửa tối đa 45 phút (`waiting_fix`), tối đa 4 lượt thực thi mỗi máy.

```text
uv run --group ops python ops/vast/vast.py watch <id> --job <job> --fail-fast --destroy-when-done --poll 60
```

Dùng `exec_command` với thời gian trả về ngắn, giữ `session_id` khi lệnh còn chạy,
rồi lấy đầu ra bằng `write_stdin`. Không dùng `run_in_background` của Claude.
Các lần chờ không quá 60 giây; cập nhật người dùng trong lúc chờ. Nếu tools được
gọi qua `functions.exec` và trả cell ID, dùng `functions.wait` cho cell đó;
cell ID khác session ID của tiến trình shell.

| Mã thoát | Xử lý |
|---|---|
| 0 | Job xong; kiểm tra kết quả và trạng thái máy. |
| 2 | Job lỗi hoặc `waiting_fix`; đọc log, sửa trong thời hạn chờ. |
| 3 | Máy không còn; báo mất máy, chỉ thuê lại khi có quyền phù hợp. |
| 4 | Đọc log để phân biệt lỗi dừng job với cảnh báo; chỉ chạy tiếp khi hiểu nguyên nhân. |
| 5 | Log đứng yên; kiểm tra log và trạng thái, destroy đúng máy nếu xác nhận treo. |

## Sửa lỗi và kết thúc

1. Đọc `uv run --group ops python ops/hf_sync.py tail <job> -n 200` và
   `uv run --group ops python ops/vast/vast.py logs <id> --tail 200`.
2. Sửa nguyên nhân trên máy cục bộ. Nếu sửa mã, thêm kiểm thử tái hiện cần thiết;
   chạy `uv run ruff check`, `uv run mypy src`, `uv run pytest`.
3. Đẩy bản sửa bằng `push-code`; máy đang chờ sẽ tự tải lại. Tối đa 3 vòng sửa
   cho cùng lỗi và không vượt thời hạn chờ. Quá giới hạn hoặc cần quyết định
   phương pháp thì dừng job, destroy đúng máy đã thuê và báo chẩn đoán.
4. Kiểm thông lượng, `validate-logs`, số chuỗi bị cắt, khả năng trích đáp án,
   kiểm định tương đương và độ lệch baseline theo RUNBOOK. Số liệu không đạt
   không cho phép chỉnh ngưỡng hay chọn lại kết quả trên tập kiểm tra.
5. Sau job, tải kết quả trước khi giải phóng máy khi có thể:
   `uv run --group ops python ops/hf_sync.py pull data runs docs/test-log.md`.
   Nếu watcher đã destroy, lấy từ bản đồng bộ Hugging Face và báo thiếu dữ liệu nếu có.
6. Gọi `uv run --group ops python ops/vast/vast.py status` để xác nhận đúng ID
   không còn tính phí. Nếu watcher chưa dọn, dùng `destroy <id>`, không chỉ stop.
   Không để máy lỗi chờ hết hạn trong khi đã biết không thể sửa kịp.
7. Cập nhật tiến độ, đường dẫn kết quả, quyết định và báo thời gian, chi phí,
   lỗi đã sửa cùng bước kế tiếp. Commit theo PLAN cho task có mã đã hoàn thành;
   không commit dữ liệu sinh hay các thay đổi không thuộc công việc.
