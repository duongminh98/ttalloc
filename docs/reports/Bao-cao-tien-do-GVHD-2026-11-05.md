# Báo cáo tiến độ khóa luận

**Sinh viên:** Dương Tuấn Minh
**Ngày:** 05/11/2026 (cập nhật 06/11/2026)
**Đề tài:** Phân bổ ngân sách token khi suy luận cho một lô câu hỏi trên mô hình suy luận lớn
**Mã nguồn:** https://github.com/duongminh98/ttalloc

## 1. Tóm tắt

- Em đã viết xong mã nguồn cho toàn bộ pipeline thực nghiệm chính: chuẩn bị dữ liệu, sinh chuỗi suy luận, ghi dữ liệu, bộ ước lượng, bộ điều khiển ngân sách, mô phỏng, đánh giá và chạy thật trên GPU.
- Em đã chạy thử toàn bộ pipeline ở quy mô nhỏ trên GPU miễn phí của Kaggle (Tesla T4). Lần chạy thử cuối chạy hết 30/30 bước trong 92 phút.
- Qua các lần chạy thử, em đã sửa một số lỗi. Ba thay đổi quan trọng nhất là: đổi sang dạng câu trả lời ngắn, chọn cỡ tập train theo learning curve, và điều chỉnh cách nạp chuỗi theo bộ nhớ GPU. Nhờ đó, thời gian ghi dữ liệu dự kiến giảm khoảng một nửa.
- **Em chưa có số liệu để kết luận cho khóa luận.** Lần chạy thử có chuỗi suy luận bị cắt ở 1.024 token (thật là 32.768) và chỉ có 2 lần sinh mỗi câu (thật là 4–8). Các số ở mục 6 chỉ dùng để kiểm tra pipeline chạy đúng.
- Bước tiếp theo là chạy ở quy mô thật trên GPU thuê (Vast.ai). Em ước tính cần khoảng 35–45 giờ GPU, tương đương khoảng 5–8 USD.

## 2. Bài toán và phương pháp

**Bài toán.** Cho một lô N câu hỏi và một tổng ngân sách B token dùng chung cho cả lô, cần quyết định câu nào được suy luận tiếp và câu nào dừng để trả lời, sao cho số câu trả lời đúng là nhiều nhất. Ngân sách được cho theo tỉ lệ α so với tổng số token khi để mô hình suy luận hết mọi câu: B = α × C_full.

**Phương pháp đề xuất (M2):**

1. Chuỗi suy luận được cắt tại các **điểm quyết định**. Đó là chỗ mô hình viết một từ chuyển ý ("Wait", "Hmm", "Alternatively", "But", "Let me check"…) ngay sau một dòng trống, và cách điểm trước một khoảng tối thiểu.
2. Tại mỗi điểm, em **gợi một đáp án thử** trên một nhánh phụ. Nhánh phụ dùng lại bộ nhớ đệm (prefix cache) của phần đã sinh, nên tốn ít token và không làm thay đổi chuỗi chính. Độ tự tin của đáp án thử được tính từ xác suất của các token đáp án.
3. Một **bộ ước lượng** (LightGBM + hiệu chuẩn isotonic) dự đoán q̂(h): xác suất câu trả lời đúng nếu suy luận thêm h token.
4. Từ q̂, em tính **lợi ích trên mỗi token** ρ = max theo h của (q̂(h) − q̂(0)) / (h·(1 + ω)), trong đó ω là tỉ lệ token gợi trên token suy luận, ước lượng từ dữ liệu ghi.
5. Bộ điều khiển đặt một **giá token** μ chung cho cả lô, dựa trên ngân sách còn lại. Câu nào có ρ ≥ μ thì được suy luận tiếp, câu nào thấp hơn thì dừng và trả lời. Một bộ bảo vệ luôn giữ đủ phần dự trữ để mọi câu còn chạy đều trả lời được, nên tổng chi tiêu không bao giờ vượt B.

**Các phương pháp so sánh:**

| Ký hiệu | Phương pháp |
|---|---|
| B0 | Suy luận đầy đủ, không dừng sớm |
| B2 | DEER: dừng khi độ tự tin của đáp án thử ≥ λ. Em hiệu chỉnh thêm một hạn mức chung cho mỗi câu, để B2 tiêu đúng ngân sách như M2 |
| M2 | Phương pháp đề xuất |
| M3 | Như M2, nhưng q̂ lấy từ đường cong thực nghiệm (các lần sinh khác của cùng câu) thay cho bộ ước lượng. Dùng để xem phương pháp có tiềm năng hay không, nếu bộ ước lượng tốt |

## 3. Dữ liệu

### 3.1. Nguồn dữ liệu

Mô hình chính là **Qwen3-1.7B** (chế độ thinking), sinh văn bản bằng vLLM 0.32.0. Tham số lấy mẫu: temperature 0,6, top_p 0,95, top_k 20, theo khuyến nghị của Qwen.

Em dùng 9 bộ dữ liệu, tải từ Hugging Face. Mỗi bộ được ghim theo một phiên bản (commit) cố định để tái lập được.

| Bộ dữ liệu | Loại | Số câu dùng | Vai trò |
|---|---|---|---|
| MATH (train, 7 chủ đề, mức 1–5) | Toán | 1.260 | train và val |
| DeepScaleR | Toán | 600 | train |
| AMC 2023 | Toán | 40 | val |
| MATH-500 | Toán | 500 | test_main |
| AIME 2024 | Toán | 30 | test_main |
| AIME 2025 | Toán | 30 | test_main |
| AIME 2026 | Toán | 30 | test_clean |
| HMMT 02/2026 | Toán | 33 | test_clean |
| GPQA-Diamond | Trắc nghiệm khoa học | 198 | test_ood |

### 3.2. Chia tập và khử trùng lặp

- Em dựng các tập theo thứ tự test_main → test_clean → test_ood → val → train. Mỗi tập dựng sau loại mọi câu trùng với các tập đã dựng trước. Hai câu bị coi là trùng nếu khớp chính xác sau khi chuẩn hóa, hoặc có từ 50% số 13-gram chung trở lên.
- Câu MATH được lấy đều theo 5 mức khó: 180 câu mỗi mức ở train, 72 câu mỗi mức ở val.
- Seed chia tập được cố định. Danh sách ID của mỗi tập được khóa bằng mã băm SHA-256 và kiểm tra lại ở đầu mọi pipeline.

| Tập | Số câu | Thành phần | Số câu bị loại do trùng |
|---|---|---|---|
| train | 1.500 | MATH 900 + DeepScaleR 600 | DeepScaleR 1.392; MATH 40 |
| val | 400 | AMC 2023 40 + MATH 360 | MATH 30 |
| test_main | 560 | MATH-500 500 + AIME 2024 30 + AIME 2025 30 | 0 |
| test_clean | 63 | AIME 2026 30 + HMMT 02/2026 33 | 0 |
| test_ood | 198 | GPQA-Diamond | 0 |

DeepScaleR có 1.392 câu trùng với các tập kiểm tra và val, nên bước khử trùng lặp là cần thiết.

- test_clean gồm các đề ra đời sau Qwen3 (04/2025), nên mô hình không thể đã thấy chúng khi huấn luyện.
- test_ood dùng để kiểm tra phương pháp ngoài miền toán học.

### 3.3. Lời nhắc và chấm điểm

- **Lời nhắc:** em dùng chat template của Qwen3 với chế độ thinking, yêu cầu đặt đáp án trong `\boxed{}`. Với câu trắc nghiệm, thứ tự các lựa chọn được xáo theo seed cố định, và có lời nhắc gợi đáp án thử riêng.
- **Chấm điểm:** lấy khung `\boxed{}` cuối cùng (có xử lý ngoặc lồng nhau). Đáp án toán được so khớp bằng `math_verify`, nên ví dụ `0.5`, `1/2` và `\frac12` được coi là bằng nhau. Câu trắc nghiệm được so theo chữ cái.
- Em đã viết công cụ rút ngẫu nhiên 100 câu để chấm tay, nhằm kiểm tra bộ chấm tự động. Bước chấm tay sẽ làm trên đầu ra của baseline ở máy thuê.

## 4. Phần mã đã hoàn thành

| Thành phần | Nội dung | Trạng thái |
|---|---|---|
| Dữ liệu | Nạp 9 bộ dữ liệu, khử trùng lặp, chia và khóa 5 tập, dựng lời nhắc, chấm điểm, công cụ chấm tay, dựng lô câu hỏi theo thành phần (đều, theo mức khó, trộn nguồn) | Xong; danh sách ID đã khóa |
| Engine sinh theo đoạn | Cắt chuỗi tại điểm quyết định; gợi đáp án thử trên nhánh phụ (giải mã tham lam, dừng ở dấu `}`); tính độ tự tin; sinh câu trả lời ép buộc khi dừng; đếm chi phí theo 5 loại token | Xong |
| Baseline | Suy luận đầy đủ; chạy mã DEER chính thức rồi chấm lại bằng bộ chấm của đề tài; đo tốc độ sinh; kiểm định tương đương giữa sinh liền mạch và sinh theo đoạn | Xong |
| Ghi dữ liệu | Ghi chuỗi suy luận, đáp án thử và câu trả lời tại mọi điểm quyết định; chạy tiếp được nếu máy bị ngắt; kiểm tra tính hợp lệ của dữ liệu ghi | Xong |
| Đặc trưng | Độ tự tin hiện tại và lớn nhất, xu hướng độ tự tin, mức nhất quán giữa các đáp án thử, số lần đổi đáp án, độ tự tin cục bộ thấp nhất, độ dài và dạng đáp án, vị trí trong chuỗi, mật độ từ chuyển ý | Xong |
| Bộ ước lượng | LightGBM + hiệu chuẩn isotonic cho từng h; chia fold theo câu hỏi để tránh rò rỉ; đánh giá bằng AUROC, ECE, Brier, Spearman; learning curve để chọn cỡ train | Xong |
| Bộ điều khiển | Quỹ ngân sách chung, bộ bảo vệ, cửa sổ trượt khi giới hạn số câu chạy đồng thời; các policy B2, M2, M3 | Xong |
| Mô phỏng | Phát lại dữ liệu đã ghi thay cho GPU, chạy hàng trăm lần rút trên CPU; dùng chung giao diện với chạy thật | Xong |
| Đánh giá | Độ chính xác theo α; khoảng tin cậy bootstrap ghép cặp; hiệu chỉnh Bonferroni; cổng G1, G2, G3; sinh bảng và hình | Xong |
| Thí nghiệm | Chỉnh tham số trên val bằng tìm kiếm ngẫu nhiên; khóa cấu hình trước khi chạy trên tập kiểm tra; 19 cấu hình ablation | Xong |
| Chạy thật | Bộ điều khiển điều khiển vLLM trực tiếp, đo thời gian từng phần | Xong |
| Hạ tầng | Thuê và điều khiển GPU trên Vast.ai; đồng bộ dữ liệu qua Hugging Face; chạy thử trên Kaggle | Xong |

**19 cấu hình ablation**, chia theo nhóm:
- bỏ từng nhóm đặc trưng (nhất quán, xu hướng, độ tự tin cục bộ, dạng đáp án);
- bỏ hiệu chuẩn;
- thay LightGBM bằng hồi quy logistic;
- chỉ dùng một tầm h;
- bỏ giá sàn, bỏ trần token mỗi câu;
- thay đổi khoảng cách giữa các điểm quyết định và số lần gợi;
- hai cách trả lời khác khi dừng: lấy đáp án thử tốt nhất, hoặc bỏ phiếu.

**Kiểm thử:** 269 unit test chạy trên CPU và 7 test chạy trên GPU, tất cả đều đạt. Các test kiểm tra các tính chất bắt buộc của hệ thống:
- không bao giờ vượt ngân sách;
- mỗi câu được trả lời đúng một lần;
- gợi đáp án thử không làm thay đổi chuỗi chính;
- chính sách không dùng thông tin tương lai;
- cùng seed thì cùng kết quả;
- chạy thật và mô phỏng cho cùng quyết định khi phát lại cùng dữ liệu.

Các baseline phụ (chia đều, tuần tự, CoDE-Stop, oracle…) và mô hình thứ hai (DeepSeek-R1-Distill-Qwen-1.5B) em để sang giai đoạn mở rộng.

## 5. Các lần chạy thử

| Ngày | Nơi chạy | Nội dung | Kết quả |
|---|---|---|---|
| 04/11 | Kaggle, T4 16 GB | Cài đặt vLLM, nạp Qwen3-1.7B, sinh thử | 5/5 bước đạt; chốt vLLM 0.32.0 |
| 04/11 | Máy cá nhân, CPU | Chia và khóa 5 tập | Đạt |
| 04–05/11 | Kaggle, T4 | Test GPU, 7 lần | Lần cuối: 7/7 test GPU đạt |
| 04–05/11 | Kaggle, T4 | Chạy thử toàn bộ pipeline (tập dượt), 8 lần | Lần 8: 30/30 bước đạt, 92 phút |

Tập dượt chạy đúng các lệnh mà các bước trên máy thuê sẽ gọi, gồm: kiểm tra GPU, baseline, ghi dữ liệu val và train, huấn luyện và đánh giá bộ ước lượng, mô phỏng, cổng G1, chỉnh tham số, khóa cấu hình, đánh giá, cổng G2, chạy thật, cổng G3.

## 6. Kết quả lần chạy thử cuối (tập dượt lần 8, 05/11/2026)

### 6.1. Cấu hình

| | Tập dượt | Chạy thật (dự kiến) |
|---|---|---|
| GPU | Tesla T4, float16 | RTX 3090 hoặc A100, bfloat16 |
| Độ dài chuỗi tối đa | 1.024 token | 32.768 token |
| Dữ liệu ghi val | 400 câu × 2 lần sinh | 400 câu × 8 lần sinh |
| Dữ liệu ghi train | 100 câu × 2 lần sinh | 500–1.500 câu × 4 lần sinh |
| Tầm h của bộ ước lượng | 256, 512 | 512 … 8.192 |
| Số lần rút trong mô phỏng | 20 | 200 |
| Chạy thật | 8 câu, 1 seed | MATH-500, 3 seed |
| Tập đánh giá | val dùng thay cho tập kiểm tra | test_main |

**Lưu ý quan trọng khi đọc kết quả:** vì chuỗi bị cắt ở 1.024 token, 89% số chuỗi chạm độ dài tối đa trước khi mô hình suy luận xong. Do đó các số dưới đây không phản ánh hành vi thật của mô hình khi được suy luận đầy đủ.

### 6.2. Baseline và tốc độ

- **Baseline trên 8 câu val:** suy luận đầy đủ đúng 4/8 câu (trung bình 1.799 token mỗi câu); DEER đúng 6/8 câu (trung bình 1.348 token). Mẫu quá nhỏ, chỉ để kiểm tra lệnh chạy và bộ chấm.
- **Tốc độ sinh trên T4:** 254 token/giây với lô 8 chuỗi; 499 token/giây với lô 32 chuỗi.
- **Kiểm định tương đương** (8 câu): chênh lệch độ dài trung bình −3,3%, nhưng khoảng tin cậy quá rộng để kết luận. Kiểm định thật sẽ chạy trên 400 câu val.

### 6.3. Ghi dữ liệu

| | val | train |
|---|---|---|
| Số chuỗi | 800 | 200 |
| Số điểm quyết định | 4.707 (5,9 điểm mỗi chuỗi) | 1.167 (5,8 điểm mỗi chuỗi) |
| Tỉ lệ dữ liệu hợp lệ | 100% | 100% |
| Tỉ lệ đáp án thử không trích được | 0,3% | 2,1% |
| Tổng token sinh / token chuỗi chính | 1,06 | 1,08 |
| Thời gian ghi | 45 phút | 13,5 phút |

Phần gợi đáp án thử và câu trả lời chỉ tốn thêm khoảng 6–8% so với token của chuỗi chính.

### 6.4. Dạng câu trả lời và độ nhiễu của lời nhắc gợi

**So sánh hai dạng câu trả lời khi dừng**, trên 36 điểm quyết định và 2 chuỗi tự kết thúc:

| | Dạng `full` (tối đa 256 token) | Dạng `short` (tối đa 32 token) |
|---|---|---|
| Không trích được đáp án tại điểm quyết định | 92% | 0% |
| Độ chính xác tại điểm quyết định | 8% | 42% |
| Chi phí trung bình | 254 token | 14 token |

Theo quy tắc ban đầu, em chỉ chọn `short` khi hai dạng cho cùng kết quả chấm ở ít nhất 98% số điểm. Quy tắc này không áp dụng được, vì dạng `full` gần như không viết kịp đáp án. Vì vậy em chọn dạng `short` và ghi lại quyết định này.

**Độ nhiễu của lời nhắc gợi:** em gợi cùng một điểm bằng hai mẫu lời nhắc khác nhau. Hai lần gợi cho đáp án khác nhau ở 22% số điểm, và tương quan giữa hai độ tự tin là 0,57. Tỉ lệ khác nhau giảm dần theo vị trí trong chuỗi: 50% ở đầu chuỗi, còn 8% ở 20% cuối chuỗi.

### 6.5. Bộ ước lượng (đánh giá trên val)

| h (token suy luận thêm) | AUROC | ECE | Brier |
|---|---|---|---|
| 0 | 0,857 | 0,065 | 0,153 |
| 256 | 0,837 | 0,074 | 0,166 |
| 512 | 0,821 | 0,085 | 0,171 |

- **Khả năng phân biệt câu đúng và sai tốt:** AUROC 0,86, cao hơn ngưỡng 0,75.
- **Hiệu chuẩn chưa đạt:** ECE 0,065–0,085, cao hơn ngưỡng 0,05.
- **Dự đoán lợi ích khi suy luận thêm còn yếu:** tương quan Spearman giữa lợi ích dự đoán và lợi ích thực nghiệm là 0,18, thấp hơn ngưỡng 0,3.
- **Đặc trưng quan trọng nhất:** độ tự tin lớn nhất và độ tự tin hiện tại, tiếp theo là độ tự tin cục bộ thấp nhất và độ dài đáp án.
- **Learning curve:** tăng train từ 50 lên 100 câu chỉ làm AUROC tăng 0,0002, còn log-loss tăng thêm 1,5%. Theo ngưỡng đặt trước, không cần tăng thêm dữ liệu. Ở quy mô thật, em sẽ bắt đầu từ 500 câu và áp dụng cùng quy tắc này.

### 6.6. Đường cong độ chính xác theo số token suy luận (val)

Tỉ lệ đúng trung bình khi dừng sau b token:

| Mức khó | b = 0 | 256 | 512 | 1.024 |
|---|---|---|---|---|
| MATH mức 1 | 0,42 | 0,60 | 0,85 | 0,95 |
| MATH mức 2 | 0,29 | 0,40 | 0,61 | 0,87 |
| MATH mức 3 | 0,25 | 0,34 | 0,49 | 0,77 |
| MATH mức 4 | 0,18 | 0,17 | 0,33 | 0,65 |
| MATH mức 5 | 0,10 | 0,08 | 0,14 | 0,28 |
| AMC 2023 (40 câu) | 0,05 | 0,15 | 0,23 | 0,41 |

- 51% số câu tăng độ chính xác ít nhất 20 điểm % khi được suy luận thêm.
- 3,8% số câu có độ chính xác ở cuối chuỗi thấp hơn mức tốt nhất ít nhất 20 điểm %, tức suy luận thêm lại làm hại (overthinking).
- Câu dễ đạt gần mức tối đa sớm, còn câu khó cần nhiều token hơn. Đây là điều kiện để việc phân bổ lại token có lợi.

### 6.7. Mô phỏng: so sánh B2, M2, M3 trên val

Kết quả trên toàn bộ 400 câu val, 20 lần rút:

| α | Phương pháp | Độ chính xác | Token đã dùng / ngân sách |
|---|---|---|---|
| 0,5 | B2 (DEER hiệu chỉnh) | 45,5% | 0,98 |
| 0,5 | M2 (đề xuất) | 46,8% | 0,98 |
| 0,5 | M3 (đường cong thực nghiệm) | 51,5% | 0,80 |
| 0,8 | B2 | 62,7% | 0,90 |
| 0,8 | M2 | 62,1% | 0,93 |
| 0,8 | M3 | 51,5% | 0,50 |

Chênh lệch so với B2, với khoảng tin cậy bootstrap 95% ghép cặp theo câu hỏi:

| α | So sánh | Chênh lệch | Khoảng tin cậy 95% |
|---|---|---|---|
| 0,5 | M2 − B2 | +1,2 điểm % | [−1,2; +4,0] |
| 0,8 | M2 − B2 | −0,6 điểm % | [−1,9; +0,7] |
| 0,5 | M3 − B2 | +6,0 điểm % | [+2,9; +9,2] |
| 0,8 | M3 − B2 | −11,1 điểm % | [−13,9; −8,6] |

Kết quả trên lô 100 câu MATH chia đều theo mức khó cũng tương tự: ở α = 0,5, M2 − B2 = +1,3 điểm % và M3 − B2 = +6,0 điểm %.

Nhận xét:
- **Ngân sách được giữ:** tỉ lệ token đã dùng trên ngân sách dưới 1 ở mọi cấu hình.
- **Chi phí gợi:** phần gợi đáp án thử chiếm khoảng 6% tổng token.
- **M2 so với B2:** ở α = 0,5, M2 nhỉnh hơn B2 khoảng 1 điểm %, nhưng khoảng tin cậy còn chứa 0.
- **M3 ở α = 0,5:** hơn B2 rõ, 6 điểm %.
- **M3 ở α = 0,8:** dừng sớm và chỉ dùng một nửa ngân sách, nên thua B2. Với chỉ 2 lần sinh mỗi câu, đường cong thực nghiệm của M3 được ước lượng từ đúng một chuỗi còn lại, nên rất nhiễu. Ở quy mô thật (8 lần sinh mỗi câu), đường cong sẽ ổn định hơn.

### 6.8. Các cổng đánh giá

Em đặt các ngưỡng của cổng trước khi có dữ liệu. Ở tập dượt, việc kiểm tra cổng chỉ để chắc rằng lệnh tính được và báo cáo đúng.

**Cổng G1 (thăm dò): không đạt**

| Tiêu chí | Ngưỡng | Tập dượt | Đạt |
|---|---|---|---|
| AUROC của q̂(0) | ≥ 0,75 | 0,857 | Có |
| ECE ở mọi h | ≤ 0,05 | 0,085 | Không |
| Spearman của lợi ích dự đoán | ≥ 0,3 | 0,18 | Không |
| M3 hơn B2 ít nhất 2 điểm % ở một α ≤ 0,5 (khoảng tin cậy Bonferroni 98,75%) | cận dưới > 0 | +6,0 điểm %, [+1,4; +9,5] | Có |
| Tỉ lệ chuỗi hợp lệ | ≥ 95% | 100% | Có |

**Cổng G2 (so sánh chính): không đạt.** Ở tập dượt, em dùng val thay cho tập kiểm tra. Kết quả M2 − B2 = +1,1 điểm % ở α = 0,5, khoảng tin cậy 98,75% là [−2,3; +4,2], còn chứa 0.

**Cổng G3 (chạy thật so với mô phỏng): không đạt.** Chạy thật trên 8 câu val, α = 0,5, 1 seed:

| Phương pháp | Độ chính xác chạy thật | Mô phỏng | Dải chấp nhận | Token đã dùng / ngân sách |
|---|---|---|---|---|
| B2 | 2/8 | 0,55 | [0,50; 0,625] | 0,91 |
| M2 | 4/8 | 0,60 | [0,50; 0,625] | 0,93 |

- Không lần chạy thật nào vượt ngân sách.
- Chạy thật khớp mô phỏng ở M2 nhưng không khớp ở B2. Với 8 câu và 1 seed, mỗi câu tương ứng 12,5 điểm %, nên độ lệch này chưa có ý nghĩa.
- **Thời gian:** mỗi lần chạy thật mất khoảng 28 giây. Gợi đáp án thử chiếm 3–4% thời gian; bộ điều khiển gần như không tốn thời gian (dưới 0,1%).

### 6.9. Kết luận từ tập dượt

- Toàn bộ pipeline chạy được từ đầu đến cuối trên GPU, và dữ liệu ghi ra hợp lệ.
- Các ràng buộc của hệ thống được giữ: không vượt ngân sách, chi phí gợi nhỏ.
- Có tín hiệu ban đầu rằng phân bổ lại token có thể có lợi: đường cong khác nhau rõ giữa các mức khó, và M3 hơn B2 ở α = 0,5. Tuy vậy, chuỗi bị cắt ngắn và dữ liệu ít, nên đây chưa phải bằng chứng.
- Điểm yếu cần theo dõi ở quy mô thật:
  - hiệu chuẩn của bộ ước lượng (ECE);
  - khả năng dự đoán lợi ích khi suy luận thêm (Spearman).

  Nếu cổng G1 thật vẫn không đạt ở hai tiêu chí này, em sẽ xem lại đặc trưng và bộ ước lượng trước khi ghi thêm dữ liệu.

## 7. Lỗi phát hiện qua chạy thử và cách xử lý

| Vấn đề | Cách xử lý | Kết quả |
|---|---|---|
| Bộ ghi dữ liệu nạp 256 chuỗi cùng lúc, vượt sức chứa bộ nhớ KV của GPU (khoảng 74 nghìn token), nên prefix cache chỉ trúng 13% | Chỉ nạp chuỗi mới khi còn chỗ trong bộ nhớ; gom các bước của mỗi vòng vào một lượt gọi | Prefix cache trúng 97%; ghi dữ liệu val từ 151 xuống 88 phút; dữ liệu ghi ra không đổi |
| Câu trả lời dạng dài bị cắt trước khi viết đáp án | Đổi sang dạng ngắn (tối đa 32 token) | Tỉ lệ không trích được đáp án từ 92% xuống 0%; ghi dữ liệu val từ 88 xuống 45 phút, train từ 24 xuống 13,5 phút |
| Chạy thật bị treo do xung đột giữa vLLM và thư viện LightGBM khi tạo tiến trình | Khởi tạo vLLM trước | Chạy hết |
| Tokenizer của mô hình thứ hai (R1-Distill) bị dựng sai với `transformers` 5.16–5.19, làm mất dấu cách và tách sai từ | Kiểm tra tokenizer khi nạp bằng cách mã hóa rồi giải mã lại một câu mẫu; sai thì nạp lại theo cách đúng | Đã xử lý; mô hình chính không bị ảnh hưởng |
| Bảng tổng hợp cộng dồn token khi gộp nhiều lần chạy có cùng cấu hình, nên tỉ lệ chi tiêu bị báo gấp đôi | Tách bảng theo từng lần chạy | Đã sửa và thêm test |
| Cỡ tập train ban đầu là 1.500 câu, tốn nhiều giờ GPU | Chọn cỡ train theo learning curve: bắt đầu 500 câu, chỉ tăng khi AUROC tăng ít nhất 0,01 hoặc log-loss giảm ít nhất 1%. Ngưỡng đặt trước khi có dữ liệu thật | Có thể bớt tới 4.000 chuỗi train |
| Giá thuê GPU theo giờ không phản ánh tốc độ | Chọn GPU theo chi phí trên mỗi token, sau khi đo tốc độ trên RTX 3090 và A100 | Sẽ chọn sau bước kiểm tra máy |

## 8. Kế hoạch tiếp theo

Các bước dưới đây chạy trên GPU thuê. Em ước tính tổng cộng 35–45 giờ GPU; trên RTX 3090 tương đương khoảng 5–8 USD. Em sẽ ước lại chi phí sau bước 1.

| Bước | Nội dung | Kết quả cần có |
|---|---|---|
| 1. Kiểm tra máy và baseline | Chạy test GPU trên RTX 3090 và A100, đo tốc độ và chọn GPU; tái lập DEER trên MATH-500 và AIME 2024; chấm tay 100 câu; ghi thử 200 chuỗi val | Baseline lệch số công bố không quá 2 điểm % trên MATH-500; chấm tự động khớp chấm tay ít nhất 99% |
| 2. Thăm dò | Ghi dữ liệu val (8 lần sinh) và train 500 câu (4 lần sinh); huấn luyện bộ ước lượng; mô phỏng trên val | **Cổng G1:** AUROC ≥ 0,75, ECE ≤ 0,05, Spearman ≥ 0,3; M3 hơn B2 ít nhất 2 điểm % ở ít nhất một α ≤ 0,5 |
| 3. Mô phỏng đầy đủ | Ghi dữ liệu test_main (8 lần sinh); chỉnh tham số trên val rồi khóa cấu hình; chạy so sánh chính, phân tích theo độ khó và ablation trên test_main | **Cổng G2:** M2 hơn B2 có ý nghĩa thống kê ở ít nhất một α ≤ 0,5, khoảng tin cậy hiệu chỉnh Bonferroni |
| 4. Chạy thật | M2 và B2 trên MATH-500, α = 0,5, 3 seed; so thời gian với DEER chính thức | **Cổng G3:** không vượt ngân sách; chạy thật lệch mô phỏng không quá 3 điểm % |
| 5. Mở rộng (nếu kịp) | test_clean, test_ood, thành phần và kích thước lô, các baseline phụ, mô hình thứ hai | |
| 6. Hoàn thiện | Sinh lại toàn bộ bảng và hình từ dữ liệu đã lưu; viết khóa luận | |

Sau mỗi cổng G1, G2, G3, em sẽ báo kết quả để thầy/cô cho ý kiến trước khi làm bước tiếp theo.
