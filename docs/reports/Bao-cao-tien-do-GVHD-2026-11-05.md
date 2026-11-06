# Báo cáo tiến độ khóa luận

**Sinh viên:** Dương Tuấn Minh
**Ngày:** 05/11/2026
**Đề tài:** Phân bổ ngân sách token khi suy luận cho một lô câu hỏi trên mô hình suy luận lớn
**Mã nguồn:** https://github.com/duongminh98/ttalloc

## 1. Tóm tắt

Em đã viết xong mã nguồn cho toàn bộ pipeline thực nghiệm chính, từ chuẩn bị dữ liệu tới chạy thật trên GPU. Em cũng đã chạy thử toàn bộ pipeline ở quy mô nhỏ trên GPU miễn phí của Kaggle, và mọi bước đều chạy hết. Bước tiếp theo là chạy ở quy mô thật trên GPU thuê để có số liệu cho khóa luận. Hiện em chưa có kết quả nào để so sánh với baseline.

## 2. Phần mã đã hoàn thành

Mô hình chính là Qwen3-1.7B, sinh văn bản bằng vLLM.

| Thành phần | Nội dung | Trạng thái |
|---|---|---|
| Dữ liệu và chia tập | Nạp MATH, DeepScaleR, AMC 2023, MATH-500, AIME 2024/2025, GPQA-Diamond; khử trùng lặp; chia 5 tập: train 1.500, val 400, test_main 560, test_clean 63, test_ood 198 câu | Xong, danh sách câu đã khóa |
| Chấm điểm | Trích đáp án, so khớp biểu thức toán và đáp án trắc nghiệm | Xong |
| Engine sinh theo đoạn | Cắt chuỗi suy luận tại các điểm quyết định; tại mỗi điểm, gợi đáp án thử trên một nhánh phụ mà không làm thay đổi chuỗi chính | Xong |
| Baseline | Tái lập suy luận đầy đủ và DEER (dừng sớm theo độ tự tin) | Xong |
| Ghi dữ liệu | Ghi chuỗi suy luận, đáp án thử và câu trả lời tại các điểm quyết định; chạy tiếp được nếu bị ngắt | Xong |
| Bộ ước lượng | Đặc trưng; LightGBM và hiệu chuẩn isotonic để dự đoán xác suất trả lời đúng theo số token suy luận thêm | Xong |
| Bộ điều khiển | Phân bổ theo giá token (phương pháp đề xuất M2); bảo đảm không vượt tổng ngân sách | Xong |
| Mô phỏng và đánh giá | Mô phỏng trên dữ liệu đã ghi; so sánh M2 với baseline B2 (DEER với hạn mức được hiệu chỉnh); khoảng tin cậy bootstrap; ablation; chỉnh tham số trên val rồi khóa trước khi chạy trên tập kiểm tra | Xong |
| Chạy thật | Bộ điều khiển chạy trực tiếp trên vLLM | Xong |
| Hạ tầng | Thuê GPU trên Vast.ai, đồng bộ dữ liệu qua Hugging Face, chạy thử trên Kaggle | Xong |

Mã có 262 unit test và 7 test trên GPU, tất cả đều đạt. Các baseline phụ (chia đều, tuần tự, CoDE-Stop, oracle…) và mô hình thứ hai em để sang giai đoạn mở rộng.

## 3. Các lần chạy thử

| Nơi chạy | Phần cứng | Nội dung | Quy mô |
|---|---|---|---|
| Máy cá nhân | CPU | Unit test; mô phỏng trên dữ liệu tổng hợp | 262 test |
| Kaggle | GPU T4 16 GB | Kiểm tra cài đặt và test GPU | 7 test GPU |
| Kaggle | GPU T4 16 GB | Chạy thử toàn bộ pipeline, 30 bước, khoảng 1,5 giờ | val 400 câu × 2 lần sinh; train 100 câu × 2; chuỗi tối đa 1.024 token (thật là 32.768); baseline và chạy thật trên 8 câu |

Các lần chạy thử này chỉ để kiểm tra pipeline chạy đúng. Quy mô quá nhỏ và chuỗi bị cắt ngắn nên em không dùng số liệu của chúng trong khóa luận.

Qua chạy thử, em đã sửa một số lỗi và điều chỉnh pipeline:
- Chọn câu trả lời ngắn khi dừng (tối đa 32 token). Câu trả lời dài thường bị cắt trước khi kịp đưa ra đáp án. Thay đổi này làm thời gian ghi dữ liệu giảm khoảng một nửa.
- Cỡ tập train được quyết định theo learning curve: bắt đầu từ 500 câu, chỉ tăng thêm khi độ chính xác của bộ ước lượng còn cải thiện rõ. Ngưỡng em đã đặt trước khi có dữ liệu thật.
- Giới hạn số chuỗi sinh cùng lúc theo bộ nhớ KV cache của GPU.

## 4. Kế hoạch tiếp theo

Các bước dưới đây chạy trên GPU thuê, RTX 3090 hoặc A100; em sẽ chọn sau khi đo tốc độ thực tế. Em ước tính tổng cộng khoảng 35–45 giờ GPU.

| Bước | Nội dung | Kết quả cần có |
|---|---|---|
| 1. Kiểm tra máy và baseline | Chạy test GPU trên máy thuê; tái lập DEER trên MATH-500 và AIME 2024; kiểm tra chấm điểm tự động bằng cách chấm tay 100 câu | Baseline lệch số công bố không quá 2 điểm % trên MATH-500; chấm tự động khớp chấm tay ít nhất 99% |
| 2. Thăm dò | Ghi dữ liệu val (8 lần sinh) và train 500 câu (4 lần sinh); huấn luyện bộ ước lượng; mô phỏng trên val | Cổng G1: bộ ước lượng đủ tốt (AUROC ≥ 0,75, ECE ≤ 0,05), và M3 (M2 với xác suất đúng lấy từ đường cong thực nghiệm) hơn B2 ít nhất 2 điểm % ở ít nhất một mức ngân sách |
| 3. Mô phỏng đầy đủ | Ghi dữ liệu test_main; chỉnh tham số trên val, khóa cấu hình; chạy so sánh chính, phân rã và ablation trên test_main | Cổng G2: M2 hơn B2 có ý nghĩa thống kê ở ít nhất một mức ngân sách ≤ 50% |
| 4. Chạy thật | M2 và B2 trên MATH-500, ngân sách 50%, 3 seed | Cổng G3: không vượt ngân sách; kết quả chạy thật lệch mô phỏng không quá 3 điểm % |
| 5. Mở rộng (nếu kịp) | Tập test_clean, test_ood, thành phần và kích thước lô, các baseline phụ | |
| 6. Hoàn thiện | Tái lập bảng và hình; viết khóa luận | |

Sau mỗi cổng G1, G2, G3, em sẽ báo kết quả để thầy/cô cho ý kiến trước khi làm bước tiếp theo. Nếu cổng G1 không đạt, em sẽ xem lại đặc trưng và bộ ước lượng trước khi ghi thêm dữ liệu.
