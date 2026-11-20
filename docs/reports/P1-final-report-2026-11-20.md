# Báo cáo tổng kết thực nghiệm P1

**Ngày xác nhận kết thúc: 20/11/2026, giờ Việt Nam.** P1 đã hoàn tất các lượt chạy baseline, kiểm định Full–Segmented, ghi pilot và hai study. Người dùng xác nhận kết thúc P1 với các ngoại lệ đã ghi nhận ở ADR-021–023. Kết quả nổi bật là 200/200 chuỗi pilot hợp lệ. Kiểm định tương đương chưa đạt biên độ chính xác định trước ±3 điểm phần trăm. Quyết định chạy tiếp dùng biên ±3,1 điểm phần trăm chọn sau khi xem dữ liệu. So sánh B2 với M2 dưới cùng ngân sách token thuộc P3 và P4. [Kế hoạch P1](../PLAN.md#p1-engine-và-công-cụ-ghi) · [Hồ sơ xác nhận](../progress.md) · [ADR-023](../decisions.md#adr-023-nới-biên-độ-chính-xác-cho-quyết-định-chạy-tiếp-p1)

## 1. Thiết lập và phạm vi

- **Mô hình và môi trường.** Qwen/Qwen3-1.7B ở revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, bfloat16 trên RTX 3090 với vLLM 0.32.0. Cấu hình `p1_safe.yaml`, hash `3c9fbf102c27`, dùng lô 32 cho baseline, cửa sổ 32 chuỗi cho logger và ghi tiến độ sau mỗi nhóm tối thiểu 8 chuỗi hoàn tất. [Cấu hình](../../configs/experiments/p1_safe.yaml)
- **Phạm vi dữ liệu.** MATH-500 và AIME 2024 thuộc `test_main` dùng cho baseline với quyền đã duyệt. Equivalence, pilot và hai study dùng `val`. Pilot dùng 100 câu × seed 0–1. Mỗi study dùng 100 câu × seed 0. [Quy trình](../../ops/RUNBOOK.md#p1-engine-và-công-cụ-ghi) · [ADR-021](../decisions.md#adr-021-p1-có-checkpoint-tự-thu-hồi-và-audit-ai-theo-ủy-quyền-trực-tiếp)
- **Giới hạn sinh.** Baseline đối chiếu DEER dùng profile `deer_paper`, giải mã greedy và giới hạn 16.384 token. Logger dùng giới hạn 32.768 token cho chuỗi suy luận. Dạng trả lời `short` đã được chọn tại ADR-014. [Đối chiếu DEER](P1-DEER-comparison-2026-11-18.md) · [ADR-014](../decisions.md#adr-014-chốt-dạng-câu-trả-lời-short)

## 2. Baseline, audit và thông lượng

| Bộ câu hỏi | Cách sinh | Số lượt | Đúng theo `grade` | Đúng theo bộ chấm DEER | Token trung bình/lượt |
|---|---|---:|---:|---:|---:|
| MATH-500 | Full | 500 × 1 | 441/500 (88,2%) | 435/500 (87,0%) trên văn bản đã lưu | 5.001,25 |
| MATH-500 | DEER | 500 × 1 | 436/500 (87,2%) | 437/500 (87,4%) | 3.191,16 |
| AIME 2024 | Full | 30 × 4 | 44/120 (36,67%) | 46/120 (38,33%) | 13.494,85 |
| AIME 2024 | DEER | 30 × 4 | 40/120 (33,33%) | 40/120 (33,33%) | 10.856,13 |

Full sinh liền mạch. DEER dùng mã chính thức tại commit `c9dd19fbffa27f841cfe47502d015b63811b4d1b`. Số token Full lấy từ engine, còn số token DEER đến từ việc mã hóa lại văn bản upstream theo tokenizer ghim. Đơn vị AIME là lượt câu–seed trên 30 câu. [Mã DEER đã ghim](https://github.com/iie-ycx/DEER/tree/c9dd19fbffa27f841cfe47502d015b63811b4d1b)

Full MATH-500 đạt 88,2% theo bộ chấm của đề tài, cao hơn mốc công bố 85,6% **2,6 điểm phần trăm**, vượt biên tái lập ±2 điểm. Chấm lại văn bản đã lưu bằng bộ chấm DEER cho 87,0%. Tám nhãn thay đổi, làm chênh lệch ròng 1,2 điểm phần trăm. Sáu trường hợp thiếu phần suy luận của phản hồi gốc khiến điểm DEER trên toàn bộ đầu ra gốc chưa xác định được. ADR-022 duyệt tiếp tục với cả hai số và giới hạn dữ liệu này. [Báo cáo đối chiếu](P1-DEER-comparison-2026-11-18.md) · [Bài DEER, Bảng 2](https://arxiv.org/html/2504.15895v3)

**Audit chấm điểm.** AI đối chiếu 100 mẫu lấy từ Full MATH-500 sau khi ẩn nhãn tự động. Hai bên đồng thuận **100/100** và cùng ghi nhận 90 câu đúng. Kết quả đạt ngưỡng 99% theo ngoại lệ audit AI đã duyệt. Phạm vi là đáp án cuối của mẫu đã chọn. [Phương pháp và hồ sơ audit](P1-status-2026-11-18.md#3-ai-audit-100-mẫu)

**Thông lượng sinh liền mạch trên `val`.** Mỗi câu có giới hạn 4.096 token trong benchmark này. Tốc độ tăng rõ từ lô 8 lên 32 và tăng ít hơn ở các lô lớn hơn. Số đo hiệu dụng của logger chuỗi dài nằm ở mục 4.

| Kích thước lô | Tổng token sinh | Thời gian (giây) | Token/giây |
|---:|---:|---:|---:|
| 8 | 32.768 | 37,26 | 879,4 |
| 32 | 111.142 | 68,09 | 1.632,4 |
| 128 | 365.322 | 208,14 | 1.755,1 |
| 256 | 750.554 | 423,50 | 1.772,3 |

## 3. Kiểm định Full–Segmented

Full sinh chuỗi liền mạch. Segmented sinh theo từng đoạn và gọi các nhánh phụ tại điểm quyết định. Kiểm định dùng cùng câu hỏi và seed cho hai cách sinh. Sau khi lấy trung bình qua các seed của từng câu, tính chênh lệch Segmented trừ Full và bootstrap ghép cặp theo câu hỏi với 2.000 lần lấy mẫu. Cổng định trước đòi cả hai CI90 nằm trọn trong biên ±3 điểm phần trăm về độ chính xác và ±5% về độ dài suy luận trung bình. [Mã kiểm định](../../src/ttalloc/baseline/equivalence.py)

| Mẫu | Đúng Full | Đúng Segmented | Chênh độ chính xác, điểm % (CI90) | Chênh độ dài (CI90) |
|---|---:|---:|---:|---:|
| 400 câu × 2 seed | 88,875% | 88,000% | −0,875 (−2,750 đến +0,875) | −4,831% (−9,801% đến −0,056%) |
| 400 câu × 4 seed | 89,1875% | 87,8125% | −1,375 (−3,0625 đến +0,1875) | −1,551% (−4,770% đến +1,638%) |

Lượt hai seed đạt biên độ chính xác nhưng chưa đạt biên độ dài, nên bổ sung seed 2–3 theo kế hoạch và giữ lại seed 0–1. Với bốn seed, độ dài trung bình là 4.738,60 token cho Full và 4.665,10 token cho Segmented. CI90 độ dài đạt. Cận dưới của CI90 độ chính xác vượt biên −3 đúng **0,0625 điểm phần trăm**. Do đó báo cáo gốc ghi `pass: false`. Sau khi xem kết quả, người dùng duyệt ±3,1 điểm phần trăm cho quyết định chạy tiếp P1. Cả hai CI90 nằm trong các biên của quyết định này. [ADR-023](../decisions.md#adr-023-nới-biên-độ-chính-xác-cho-quyết-định-chạy-tiếp-p1)

**Chẩn đoán hậu kiểm.** So sánh hai cách trả lời trên cùng tiền tố Segmented cho sáu câu MATH lệch ở cả bốn seed và sáu câu đối chứng theo độ dài. Dạng `short` đúng 5/24 ở nhóm lệch và 24/24 ở nhóm đối chứng. Trong 19 lượt sai của nhóm lệch, 11 lượt có đáp án trích được nhưng sai và 8 lượt có khung chưa đóng. Phép chấm thăm dò nội dung trong khung chưa đóng khớp đáp án chuẩn ở 2/8 lượt. Sinh tiếp tự nhiên chạm giới hạn 256 token ở 44/48 lượt. Các nhóm được chọn từ kết quả quan sát, nên số liệu mô tả lỗi trên các câu đã chọn.

## 4. Pilot và kiểm tra log

| Chỉ số trên 100 câu `val` × 2 seed | Kết quả |
|---|---:|
| Chuỗi đúng cấu trúc và qua kiểm tra | 200/200 (100%) |
| Số điểm được ghi | 5.328 |
| Điểm trung bình mỗi chuỗi | 26,64 |
| Độ dài chuỗi trung bình / trung vị | 5.614,63 / 2.391,5 token |
| Chuỗi chạm giới hạn | 1/200 (0,50%) |
| Không trích được đáp án ở điểm ghi / điểm kết thúc tự nhiên | 1,14% / 2,01% |
| Tổng token sinh chia cho token chuỗi chính | 1,06374 |
| Thời gian toàn bước pilot | 3.908 giây (65,1 phút) |
| Thông lượng hiệu dụng trên toàn bước | khoảng 306 token sinh/giây |

`validate-logs` kiểm tra cấu trúc, thứ tự điểm, số token và giới hạn vận hành, rồi ghi `validate.json`. Báo cáo được đối chiếu lại trên CPU từ dữ liệu đồng bộ. Log hash `4c7298df1c40` xác định phiên bản log pilot. Khi P2 chạy với cùng log hash, 200 khóa hợp lệ này được tái sử dụng trong 3.200 chuỗi `val`. Tốc độ 306 token/giây là số đo trên mẫu pilot và máy đã dùng, gồm thời gian của toàn bước ghi. [Quy trình tái sử dụng pilot](../../ops/RUNBOOK.md)

## 5. Hai study trên chuỗi dài

### Dạng trả lời

| Vị trí | Số điểm | Cùng nhãn đúng/sai | Không trích được, Full / Short | Đúng, Full / Short | Chi phí token trung bình, Full / Short |
|---|---:|---:|---:|---:|---:|
| Điểm trong chuỗi | 2.512 | 65,13% | 97,77% / 1,07% | 2,23% / 36,54% | 257,03 / 14,38 |
| Kết thúc tự nhiên | 100 | 18,00% | 97,00% / 2,00% | 3,00% / 85,00% | 255,70 / 13,10 |

Ở đây Full và Short là hai **dạng trả lời tại cùng vị trí** của study. Chi phí gồm token lời nhắc trả lời và token đầu ra. Trong giới hạn token của study, phần lớn đầu ra Full thiếu đáp án trích được. Quy tắc tự động ở mốc 98% đề xuất Full vì hai dạng có mức đồng thuận thấp. Các số liệu về khả năng trích đáp án, độ chính xác và chi phí củng cố quyết định dùng Short đã duyệt ở ADR-014. [Quyết định và đối chiếu](../decisions.md#adr-014-chốt-dạng-câu-trả-lời-short)

### Độ nhạy với lời nhắc gợi

Thay mẫu lời nhắc tại cùng điểm cho khóa đáp án khác nhau ở **36,94%** trong 2.182 điểm. Tương quan Pearson giữa hai số đo độ tự tin là **0,843**. Tỉ lệ dưới đây mô tả mức đổi đáp án theo vị trí tương đối trong chuỗi.

| Vị trí tương đối | Số điểm | Đáp án khác nhau |
|---|---:|---:|
| 0–20% | 383 | 51,96% |
| 20–40% | 444 | 44,82% |
| 40–60% | 458 | 39,96% |
| 60–80% | 450 | 32,22% |
| 80–100% | 447 | 17,90% |

## 6. Chi phí và bàn giao

Vast ghi nhận chi phí cộng dồn **4,523524 USD** tại lần đối chiếu sau khi thu hồi GPU, dưới trần 4,90 USD. Guard ghi `done` lúc **04:00:17 ngày 20/11/2026** giờ Việt Nam, và Vast xác nhận **0 máy đang thuê**. Pilot được lưu đủ trước khi gặp HTTP 429 do giới hạn tạo commit của Hugging Face. Phiên tiếp theo dùng lại checkpoint đã xác minh, đẩy log mỗi 5 phút và đồng bộ dữ liệu mỗi 10 phút.

Theo phép ngoại suy từ pilot, với độ dài giả định 6.300 token, hệ số token sinh 1,06374, dự phòng thời gian ghi 10% và giá GPU gồm đĩa 0,234444 USD/giờ, phần ghi dữ liệu còn lại của P2 khoảng **33,50 giờ / 7,85 USD** cho 5.000 chuỗi. Phần ghi P3 khoảng **30,01 giờ / 7,04 USD** cho 4.480 chuỗi. Hai ước tính chỉ gồm GPU và đĩa. Tốc độ logger và độ dài chuỗi có thể thay đổi theo tập. P2 chờ duyệt riêng cho lượt ghi lớn. [Mô hình dự toán](../Ke-hoach-thuc-hien-de-tai.md#331-dự-toán-cập-nhật-sau-gpu_check-ngày-18112026) · [Quy trình P2](../../ops/RUNBOOK.md#p2-thăm-dò)

## 7. Tra cứu dữ liệu gốc

Các đường dẫn dưới đây nằm trong kho hiện vật Hugging Face riêng của dự án. Bảng tổng hợp dùng JSON và CSV của các lượt đã lưu.

| Kết quả | Đường dẫn trong kho hiện vật |
|---|---|
| Full MATH-500 và đối chiếu bộ chấm | `runs/20261117-181157_baseline-full-test_main-deer_paper_3c9fbf102c27/` và `runs/p1-deer-comparison-20261118/` |
| DEER MATH-500 | `runs/20261118-191850_baseline-deer-test_main-deer_paper_3c9fbf102c27/` |
| Full AIME 2024 | `runs/20261118-203418_baseline-full-test_main-deer_paper_3c9fbf102c27/` |
| DEER AIME 2024 | `runs/20261119-025608_baseline-deer-test_main-deer_paper_3c9fbf102c27/` |
| Audit và thông lượng | `runs/p1-20261118-010031/ai-audit-summary.json` và `runs/20261119-034439_baseline-throughput_3c9fbf102c27/throughput.json` |
| Equivalence | `runs/20261119-035954_equivalence_3c9fbf102c27/equivalence.json` và `runs/20261119-153426_equivalence_3c9fbf102c27/equivalence_4seed.json` |
| Chẩn đoán hậu kiểm | `runs/20261119-161719_equivalence-diagnostic_3c9fbf102c27/` |
| Pilot | `data/logs/qwen3_1_7b/val/` |
| Hai study | `runs/20261119-194301_study-answer-mode_3c9fbf102c27/` và `runs/20261119-203712_study-probe-noise_3c9fbf102c27/` |
| Tiến độ job và biên nhận thu hồi GPU | `runs/p1-cloud-20261120-003133/stages.json` và `runs/guard-61504389.json` |
