# Báo giá tiếp tục P1 — 18/11/2026

**Chưa thuê máy.** Người dùng đã duyệt ngoại lệ Full MATH-500 (ADR-022) và yêu cầu tìm offer/trình tiền trước khi chạy. Giữ kết quả 88,2% theo `grade` và 87,0% theo DEER trên văn bản đã lưu; không sinh lại Full. Ngoại lệ không áp dụng cho các cổng khác.

## Số dư và offer trực tiếp

Snapshot API Vast lúc **13:43:45 ngày 18/11/2026, giờ Việt Nam**: credit **4,61088126994 USD**, **0 instance**. Bằng chứng: `runs/p1-continuation-quote-20261118/vast-snapshot.json` và `quote.json`. Truy vấn on-demand, một RTX 3090, verified, reliability ≥98%, RAM ≥32 GB, CPU hiệu dụng ≥8, CUDA ≥13, đủ 60 GB disk. Giá và tình trạng offer có thể đổi; kiểm tra lại trước khi thuê.

| Offer | Vị trí | GPU + 60 GB disk / giờ | Tin cậy | Down / Up, USD/GB |
|---|---|---:|---:|---:|
| **44880351 — đề xuất** | Pháp | **0,243333 USD** | **99,8624%** | 0,00911458 / 0,01171875 |
| 56377751 — rẻ hơn | Trung Quốc | 0,205556 USD | 99,5078% | 0,015625 / 0,01692708 |

Offer Pháp có 24 GB VRAM, CPU hiệu dụng 16 core, RAM khoảng 62,8 GiB, bandwidth VRAM 805,5 GB/s; mạng down/up 926/793 Mbps, driver 580.159.03, verified. Mạng nhanh hơn và điểm tin cậy cao hơn offer rẻ nhất; chênh dự toán khoảng 0,11–0,21 USD cho lượt này. Chưa có benchmark trên host mới, không cam kết throughput bằng host cũ.

**Cách tính disk:** API `dph_total` đang dùng dung lượng mặc định 8 GB, không phải 60 GB. Báo giá tính lại `dph_base + storage_cost × 60 / 720`: offer Pháp 0,2266667 + 0,2 × 60 / 720 = 0,2433333 USD/giờ. Không lấy 0,2288889 USD/giờ hiển thị trong search làm giá thuê 60 GB.

## Phần P1 còn thiếu và thời gian

| Công việc | Phạm vi giữ nguyên | Giờ ước tính |
|---|---|---:|
| Khởi tạo, GPU regression, throughput | Host mới; throughput 4 cỡ lô, tối đa 256 câu val | 0,4–0,6 |
| DEER MATH-500 | 500 câu ×1; nhóm checkpoint 32 câu, có nhiều lần khởi tạo engine | 1,2–1,6 |
| Full + DEER AIME 2024 | 30 câu ×4 lượt cho mỗi cách | 0,9–1,2 |
| Equivalence | val 400 ×2 seed, Full và segmented/probe | 3,6–4,7 |
| Pilot và validate | val 100 ×2 seed | 0,6–0,8 |
| Answer-mode và probe-noise | 100 câu val ×1 cho từng study | 1,3–1,6 |
| **Tổng nếu mọi bước được phép đi tiếp** | Không tính lại Full MATH-500 hoặc audit đã hoàn tất | **8,0–10,5** |

Đây là dự toán, không phải số đo từng stage. Cơ sở: Full dài đã chạy ~0,91 giờ cho 500 câu/2,50 triệu token, khoảng 767 token/giây; giữ giả định kế hoạch 800 token/giây cho logger và dự toán baseline/equivalence/study cũ, cộng chi phí khởi tạo host và các nhóm DEER. Phép đo Full không thay thế throughput logger có nhánh phụ. Độ dài val lấy mẫu ở temperature 0,6, chi phí KV, biên dịch và các nhánh study có thể làm thời gian vượt khoảng này. Khi có pilot dài sẽ tính lại, không tự nâng trần.

Các cổng vẫn có thể kết thúc lượt sớm: DEER MATH-500 lệch >2 điểm; equivalence không đạt; lỗi dữ liệu, lỗi đồng bộ, không có tiến triển hoặc nguy cơ hết hạn mức. Không tự tăng equivalence lên 4 seed trong báo giá này. Full MATH-500 được đánh dấu **chấp nhận ngoại lệ**, không giả làm `gate pass` và không chạy lại. Cần cập nhật driver để nạp quyết định này có phạm vi và kiểm thử trước thuê; chưa sửa driver hoặc tạo trạng thái chạy giả trong lần báo giá.

## Dự toán tiền và đề nghị duyệt

Với offer Pháp, giả định kế hoạch **25 GB tải xuống + 5 GB tải lên** cho mã/model/môi trường/dữ liệu/sync: phí mạng khoảng **0,28646 USD**. Chưa đo lưu lượng thực; không coi đây là phí cố định hay bảo đảm đủ khi tải ảnh/môi trường lớn.

| Thời gian thuê | GPU + disk | Mạng theo giả định | Khoản chạy mới | P1 cộng khoản cũ |
|---|---:|---:|---:|---:|
| 8 giờ | 1,94667 | 0,28646 | **2,23313** | **2,51286** |
| 10 giờ | 2,43333 | 0,28646 | **2,71979** | **2,99953** |
| 10,5 giờ, chỉ là kịch bản | 2,55500 | 0,28646 | **2,84146** | **3,12120** |

Khoản P1 đã ghi nhận **0,2797374973 USD** vẫn cộng dồn. Tổng trần giữ **3,25 USD**, nên số tiền danh nghĩa còn lại **2,9702625027 USD**. Credit hiện tại đủ; không cần nạp thêm để duyệt phạm vi này. Khoản gpu_check trước P1 là khoản riêng, không trừ lần nữa từ trần P1.

**Đề nghị duyệt cụ thể:** offer 44880351, một RTX 3090 on-demand, disk 60 GB; phần P1 còn thiếu ở trên; **cửa sổ mới tối đa 10 giờ từ lúc thuê mới**, giữ **trần P1 cộng dồn 3,25 USD**, gồm 0,2797374973 USD cũ. Không tự chuyển offer khi offer mất; báo lại nếu cần thay giá/phạm vi. Đây là đề nghị gia hạn cửa sổ sau khi deadline cũ **13:00:31 ngày 18/11** đã qua, không âm thầm reset 12 giờ cũ hoặc đặt lại ngân sách.

Guard giữ biên **0,25 USD**, do đó hướng tới dừng khi tổng P1 tiến gần **3,00 USD** hoặc sớm hơn nếu dự báo stage tiếp theo không đủ ngân sách/thời gian. Dùng cả credit thực tế và ước tính theo thời gian; phí API có thể cập nhật trễ. Khi hết cửa sổ/nguy cơ vượt tiền: checkpoint, sync có giới hạn thời gian rồi destroy đúng máy, kể cả lỗi sync; không để máy treo. Xong sớm thì destroy ngay. 10 giờ không bảo đảm hoàn tất nếu rơi vào kịch bản 10,5 giờ hoặc chậm hơn; giữ kết quả từng stage thay vì chi thêm ngoài trần.

Sau khi người dùng duyệt khoản thuê mới: hoàn thiện và kiểm thử driver chấp nhận ngoại lệ đúng một Full run, lưu raw output cho lượt sau và báo song song hai bộ chấm; xác nhận CPU checks/splits; kiểm lại offer; thiết lập deadline/cost guard mới có tính khoản cũ rồi mới thuê. Không bắt đầu P2 hoặc ghi P1 hoàn tất nếu chưa đủ kết quả.
