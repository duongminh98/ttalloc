# Đối chiếu baseline P1 với DEER — 18/11/2026

**Kết luận:** đã xác định một nguyên nhân có số đo: khác bộ chấm làm điểm trên cùng 500 đáp án thay đổi từ **441/500 (88,2%)** thành **435/500 (87,0%)**. Giải thích được **1,2 trong 2,6 điểm phần trăm** chênh lệch với công bố 85,6%. Chưa xác định nguyên nhân của 1,4 điểm còn lại. Đây là đối chiếu trên CPU; không thuê máy, sinh lại đáp án, đổi config, seed hoặc ngưỡng.

**Chưa tự mở lại cổng P1.** PLAN P1.T5 yêu cầu chấm bằng `grade`, và bước chạy 1 dừng nếu lệch quá 2 điểm; điểm 88,2% theo định nghĩa đó vẫn ngoài biên. Điểm đối chiếu 87,0% nằm trong biên nhưng không được âm thầm thay thế tiêu chí đã định trước. Ngoài ra, thiếu một phần đầu ra để tái hiện hoàn toàn bộ chấm tác giả. P1 vẫn chưa hoàn tất.

## 1. Dữ liệu và nguồn

- Đầu ra giữ nguyên: `runs/20261117-181157_baseline-full-test_main-deer_paper_3c9fbf102c27/`, config_hash `3c9fbf102c27`.
- Bằng chứng đối chiếu mới: `runs/p1-deer-comparison-20261118/`; `manifest.json` băm các tệp kết quả.
- Mã DEER được tải ở commit đã ghim **c9dd19fbffa27f841cfe47502d015b63811b4d1b**; không dùng HEAD thay thế.
- [Bài DEER arXiv v3](https://arxiv.org/html/2504.15895v3), Bảng 2 và 13: Qwen3-1.7B Vanilla MATH-500 85,6%, trung bình 5.160 token. Phụ lục K mô tả greedy và giới hạn 16.384 cho thí nghiệm chính.
- [check.py của DEER](https://github.com/iie-ycx/DEER/blob/c9dd19fbffa27f841cfe47502d015b63811b4d1b/check.py): trích đáp án từ `generated_responses`, rồi gọi `check_is_correct`.
- [parser.py](https://github.com/iie-ycx/DEER/blob/c9dd19fbffa27f841cfe47502d015b63811b4d1b/utils/parser.py) và [grader.py](https://github.com/iie-ycx/DEER/blob/c9dd19fbffa27f841cfe47502d015b63811b4d1b/utils/grader.py): dùng nguyên bản cho lần đối chiếu, có SHA-256 trong summary.

## 2. Cấu hình và dữ liệu

| Thành phần | Kết quả đối chiếu | Giới hạn |
|---|---|---|
| Đề MATH-500 | 500/500 nội dung khớp dữ liệu `data/math/test.jsonl` của DEER sau bỏ khoảng trắng ngoài | Không thay ID/split |
| Đáp án chuẩn | 500/500 khớp nguyên văn | Không thay gold |
| Prompt | 500/500 prompt dựng lại khớp cách ghép system/user và chat template trong script DEER | Đối chiếu CPU bằng tokenizer ghim; không có dump prompt ID từ server |
| Thinking | `enable_thinking=True`; bằng với mặc định template ghim | Template không chèn `<think>` đầu khi thinking bật; không coi việc thiếu tiền tố này là lỗi |
| Giải mã Full | temperature=0, top_p=1, top_k=-1 | Khớp greedy trong bài; không chọn seed theo kết quả |
| Giới hạn token | 16.384 | Khớp mô tả bài và mặc định script; shell ví dụ DEER ghi 16.000, nên tài liệu gốc chưa hoàn toàn nhất quán. Không tự đổi về 16.000 |
| Precision | bfloat16 | Khớp ví dụ script; GPU khác môi trường tác giả |
| Model | Qwen/Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | DEER dùng đường dẫn model local, không công bố checksum trọng số của lượt ra 85,6%; không chứng minh cùng bytes |
| Runtime | vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, RTX 3090; lô 32 | Repo DEER khai vLLM<=0.6.1, không có môi trường khóa tái hiện cho kết quả Qwen3. Đây là khác biệt, chưa phải nguyên nhân được chứng minh |
| Độ dài | 5.001,25 token/câu, so với 5.160 trong bài | Đếm token engine ở đề tài và tokenize lại văn bản trong `check.py` khác cách đo; không đồng nhất hai số |

Mã `baseline/full.py` đọc phần sau `</think>` đầu tiên nếu có, trích khung cuối rồi gọi `grade` (`math-verify` 0.9.0). DEER tìm khung cuối trên toàn bộ response, chuẩn hóa bằng `strip_string` và dùng bộ so sánh LIMO/SymPy. Vì vậy so trực tiếp hai độ chính xác chưa có cùng thước chấm.

## 3. Chấm đối chiếu toàn bộ đáp án đã lưu

`regrade.py` đọc checkpoint đã hoàn tất, gọi nguyên bản `extract_answer` và `check_is_correct` từ DEER. Không chỉnh nội dung câu trả lời. Kết quả ở `regrade-results.json`, tổng hợp ở `regrade-summary.json`.

| Chỉ số | Kết quả |
|---|---:|
| Số khóa `(qid, seed)` | 500 |
| Đúng theo `grade` của đề tài, giữ nguyên | 441 = 88,2% |
| Đúng theo DEER trên `answer_text` đã lưu | 435 = 87,0% |
| Đúng ở đề tài, sai ở DEER | 7 |
| Sai ở đề tài, đúng ở DEER | 1 |
| Tổng nhãn khác | 8 |
| Chênh ròng do bộ chấm | 6 câu = 1,2 điểm phần trăm |
| Chênh còn lại so với công bố | +1,4 điểm phần trăm |

Đã chạy thêm một lần không giới hạn thời gian để loại khả năng kết quả thấp do timeout/spawn trên Windows: **500/500 nhãn trùng**. Lần đầu dùng timeout mặc định của DEER. Không dùng nhiều lần chấm để chọn điểm thuận lợi.

Kiểm tra toàn vẹn: checksum checkpoint hợp lệ, 500 khóa duy nhất; tính lại `grade` trên CPU khớp toàn bộ 500 nhãn cũ và vẫn là 441 đúng. Mười một tệp upstream đã tải khớp Git blob SHA ở cây commit đã ghim. Không thay nguồn/chấm nghiên cứu; không chạy lại bộ test phần mềm vì phiên này chỉ thêm báo cáo và phân tích đầu ra cũ.

Tám trường hợp đổi nhãn:

| QID, bỏ tiền tố `math500/test/` | Đáp án sinh / chuẩn | `grade` / DEER | Nhận xét |
|---|---|---|---|
| algebra/101.json | `[-2,7]` / `x ∈ [-2,7]` | đúng / sai | Cùng tập nghiệm |
| geometry/473.json | `864` / `864 inches²` | đúng / sai | Đề đã yêu cầu đơn vị |
| intermediate_algebra/183.json | `[2,∞)` / `(2,∞)` | sai / đúng | DEER chấp nhận nhầm khác biệt đầu mút; không lấy nhãn DEER làm chân lý toán học |
| number_theory/1257.json | `204` / `204_5` | đúng / sai | Đề yêu cầu cơ số 5 |
| number_theory/368.json | `4343` / `4343_6` | đúng / sai | Đề yêu cầu cơ số 6 |
| number_theory/516.json | `2516` / `2516_8` | đúng / sai | Đề yêu cầu cơ số 8 |
| prealgebra/105.json | `east` / `\text{east}` | đúng / sai | Cùng hướng đông |
| prealgebra/1114.json | `15` / `15 cm²` | đúng / sai | Đề cho kích thước theo cm |

Ở cả 8 trường hợp, nội dung đáp án được hai bên trích ra giống nhau; nhãn đổi do chuẩn hóa/so sánh. Đây không phải bằng chứng bộ chấm đề tài cộng nhầm 6 câu: 7 nhãn bổ sung có lý do ngữ nghĩa, còn DEER có một dương tính sai rõ ở khoảng mở/đóng.

## 4. Giới hạn tái hiện

`answers.parquet` và checkpoint chỉ giữ `answer_text` sau `</think>` đầu tiên, không giữ toàn bộ token/suy luận trước đó. Có **6 câu** đã sinh think-end nhưng phần trả lời bị cắt ở trần 16.384 và không có đáp án trích được. DEER có thể lấy khung trong phần suy luận đã mất ở những câu này; không đủ dữ liệu để kiểm tra. Với các response có khung hợp lệ ở phần trả lời, khung đó nằm sau phần đã bỏ; các response không có think-end giữ nguyên văn bản sinh.

Vì vậy 87,0% là điểm trên phần văn bản còn lưu, **không khẳng định là điểm DEER trên toàn bộ raw response**. Nếu chỉ xét bất định sáu câu này, khoảng số đếm có thể là 435–441/500 (87,0–88,2%); đó là khoảng giới hạn do thiếu dữ liệu, không phải khoảng tin cậy thống kê. Biên này cắt qua ngưỡng 87,6%, nên chưa chứng minh cổng gốc đạt bằng chấm lại.

Môi trường chấm gốc cũng có mâu thuẫn: `requirements.txt` ghim SymPy 1.12/ANTLR 4.11.1 nhưng latex2sympy2 1.9.1 yêu cầu ANTLR 4.7.2; cài trực tiếp thất bại. Bản đối chiếu tải hai runtime riêng: nạp parser latex2sympy2 với 4.7.2 trước, rồi SymPy parse_latex với 4.11.1. Hai parser kiểm tra phân số 1/2 thành công; không sửa hàm chấm/chuẩn hóa upstream. Đây là lớp tương thích được công khai, không phải tái dựng chính xác môi trường tác giả. Lặp không timeout xác nhận nhãn ổn định trong môi trường đối chiếu này.

## 5. Có thể tiếp tục P1 không?

**Có cơ sở để đề xuất tiếp tục sau một quyết định rõ về điều kiện tái lập; chưa đủ để tự đánh dấu cổng đạt.** Điểm chính là khác định nghĩa chấm, không có bằng chứng dữ liệu sai hoặc prompt sai. Không cần sinh lại Full chỉ để kéo điểm về 85,6%.

Đề xuất cụ thể để người dùng xem xét:

1. Giữ 88,2% là kết quả chính theo `grade`; giữ 87,0% là đối chiếu DEER trên dữ liệu còn lưu, cùng giới hạn 6 câu. Chấp nhận ngoại lệ tái lập Full có giải trình, thay vì đổi ±2 điểm hoặc tuyên bố tái lập chính xác.
2. Nếu duyệt ngoại lệ, chạy phần P1 còn thiếu với cùng cấu hình đã duyệt; Full và DEER đều báo điểm theo `grade`, thêm điểm theo mã tác giả để đối chiếu. Không thay bộ chấm nghiên cứu để được điểm thuận lợi. Lưu raw output/token và phiên bản bộ chấm từ các lượt sau.
3. Không ghi P1 hoàn tất khi chưa có DEER/AIME, equivalence và pilot/study/validate. Nếu DEER hoặc equivalence không đạt thì dừng theo tiêu chí riêng; không suy từ Full.
4. Thuê lại GPU cần dự toán/duyệt ở thời điểm thực hiện. Trần lịch sử 3,25 USD/12 giờ là cộng dồn theo deadline gốc, không tự gia hạn hoặc đặt lại khi thuê mới. Lần này không chi thêm GPU.

Không tự áp dụng đề xuất này vào PLAN hay driver. ADR-022 ghi đây là quyết định đang chờ; mã và cổng hiện hành giữ nguyên. Phần công việc hiện tại hoàn thành ở đối chiếu, không phải chạy tiếp GPU.
