# Kế hoạch thực hiện đề tài

**Sinh viên:** Dương Tuấn Minh  
**MSSV:** 23021937  
**Đề tài:** Phân bổ thích ứng tài nguyên suy luận cho mô hình suy luận lớn  
*(Adaptive Test-Time Compute Allocation for Large Reasoning Models)*  
**Phiên bản:** 2.1. Kế hoạch chia theo phase, không gắn với mốc thời gian, và được rà soát lại sau mỗi cổng quyết định (mục 8).

**Thay đổi ở bản 2.1 (03/11/2026):** model chính đổi từ R1-Distill-Qwen-7B sang Qwen3-1,7B để giảm chi phí GPU. Chỉ giữ một baseline là B2 (DEER với giới hạn nâng lên); các phương pháp so sánh còn lại chuyển sang phần để sau (mục 7.2, việc 5.1).

**Tài liệu liên quan:**
- [Báo cáo phương pháp tiếp cận](Bao-cao-Phuong-phap-tiep-can.md): tổng quan phương pháp.
- [Báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md): thiết kế đầy đủ, tham số khởi đầu, rủi ro kỹ thuật.
- [Kế hoạch implement](Ke-hoach-implement.md): đặc tả mã nguồn để AI agent cài đặt, kiểm thử và review, chia theo cùng các phase với kế hoạch này.
- [Ghi chép research](Research-notes-phan-bo-token.md): danh sách kiểm tra cho thiết kế thực nghiệm ở [mục 8](Research-notes-phan-bo-token.md#L327).

---

## 1. Tổng quan

Đề tài được thực hiện qua bảy phase, từ P0 đến P6. Mỗi phase có mục tiêu, điều kiện bắt đầu, danh sách công việc, đầu ra và tiêu chí hoàn thành. Một phase chỉ kết thúc khi đạt tiêu chí hoàn thành, và phase sau chỉ bắt đầu khi phase trước đã kết thúc. Ba cổng quyết định G1–G3 nằm ở cuối P2, P3 và P4; mỗi cổng kiểm tra một giả định then chốt trước khi chuyển sang phần việc tốn kém hơn.

Bốn nguyên tắc xuyên suốt:

1. **Mô phỏng trước, hệ thống sau.** Phần lớn thí nghiệm chạy trên các chuỗi suy luận đã ghi lại, nên chi phí thấp và lặp nhanh. Phần cài đặt trên vLLM dùng để xác nhận kết quả mô phỏng và đo thời gian chạy thực tế.
2. **Một mã chính sách cho hai môi trường.** Bộ mô phỏng và bộ chạy thật gọi cùng một mã chính sách, nên kết quả của hai bên so sánh trực tiếp được.
3. **Tập kiểm tra chỉ được dùng khi cấu hình đã khóa.** Mọi tham số được chọn trên tập validation. Ngoại lệ duy nhất là lần tái lập baseline ở P1, và không tham số nào được chọn từ kết quả đó.
4. **Mã nguồn do AI agent viết theo kế hoạch implement.** Sinh viên duyệt tại các điểm dừng, chấm tay khi cần và ra quyết định ở các cổng.

| Phase | Nội dung chính | Kết thúc khi | Đầu ra |
|---|---|---|---|
| P0. Nền tảng | Hạ tầng, repo, dữ liệu, chia tập, chấm điểm | Danh sách ID đã khóa; vLLM chạy được trên máy chủ GPU | D1 |
| P1. Engine và công cụ ghi | Sinh theo đoạn, gợi đáp án thử ở nhánh phụ, tái lập baseline, ghi dữ liệu | Baseline khớp số công bố; các kiểm thử của công cụ ghi đều đạt | D2 |
| P2. Thăm dò | Dữ liệu thăm dò, bộ ước lượng và bộ mô phỏng phiên bản đầu, đo dư địa | Có quyết định ở cổng **G1** | D3 |
| P3. Mô phỏng đầy đủ | Dữ liệu đầy đủ, mọi phương pháp, chỉnh tham số, khóa cấu hình, E1–E4 | Có quyết định ở cổng **G2** | D4 |
| P4. Chạy thật | Bộ phân bổ trên vLLM, đo chi phí hệ thống (E5) | Có quyết định ở cổng **G3** | D5 |
| P5. Mở rộng | Các phương pháp so sánh để sau, model thứ hai, tập sạch, tập ngoài miền, thành phần lô (E6) | Các phần được giao của P5 có kết quả | D6 |
| P6. Hoàn thiện | Chốt số liệu, viết khóa luận, đóng gói, chuẩn bị bảo vệ | Có bản nộp cuối | D7, D8 |

Một số việc chạy song song với các phase:

- Trong một phase, GPU ghi dữ liệu ở chế độ chạy nền trong lúc agent viết mã cho các phần chưa cần tới dữ liệu đó.
- Khóa luận được viết dần theo mục 11: chương nào đã đủ nội dung thì viết, không chờ tới P6.
- Các phương pháp so sánh để sau (việc 5.1) chạy trong mô phỏng trên dữ liệu đã ghi, không cần GPU, nên có thể bổ sung bất cứ lúc nào sau khi khóa cấu hình ở P3.

Trong các bảng công việc ở mục 6, cột *Thực hiện* ghi **Agent** cho việc do AI agent làm theo kế hoạch implement, và **SV** cho việc do sinh viên làm. Cột *Kế hoạch implement* ghi task (ví dụ P1.T5) hoặc bước chạy tương ứng trong phase cùng tên của kế hoạch implement.

## 2. Mục tiêu và phạm vi

### 2.1. Câu hỏi nghiên cứu và tiêu chí thành công

Các ngưỡng dưới đây là đề xuất, cần thống nhất với giảng viên hướng dẫn trước khi bắt đầu P0.

| Câu hỏi | Thí nghiệm | Tiêu chí thành công |
|---|---|---|
| **RQ1.** Có ước lượng được lợi ích của việc suy luận thêm ngay trong quá trình sinh không? | E1 | AUROC của $\hat q(0)$ ≥ 0,75; ECE ≤ 0,05 với mọi $h$ sau hiệu chuẩn; lợi ích dự đoán $\hat q(h) - \hat q(0)$ có tương quan Spearman ≥ 0,3 với lợi ích thực nghiệm |
| **RQ2.** Với cùng ngân sách $B$, phương pháp có trả lời đúng nhiều câu hơn các cách dừng sớm theo từng câu không? | E2 | Hơn DEER với giới hạn nâng lên (B2) ít nhất 2 điểm phần trăm, với khoảng tin cậy hiệu chỉnh Bonferroni không chứa 0, ở ít nhất một mức ngân sách $\alpha \le 0{,}5$ trên model chính; hoặc tiết kiệm ít nhất 15% token ở cùng độ chính xác |
| **RQ3.** Phần cải thiện đến từ đâu? | E3, E4 | Hoàn thành các ablation và phân rã B2 → M2 → M3; mỗi kết luận có kiểm định thống kê. Phân rã đầy đủ (mục 7.2) làm khi có các phương pháp để sau |
| Tính khả thi khi chạy thật | E5 | Đạt cổng G3 (mục 8); thời gian chạy và chi phí phụ được đo và báo cáo |
| Tính tổng quát | E6 | Có kết quả trên model thứ hai, tập sạch và tập ngoài miền, kể cả khi kết quả âm tính |

Nếu RQ2 cho kết quả âm tính, đề tài vẫn có kết quả hợp lệ khi có phân tích nguyên nhân (mục 8). Khi đó, đóng góp chính chuyển sang bộ dữ liệu ghi lại, bộ mô phỏng, và phân tích điều kiện để quỹ chung mang lại lợi ích.

### 2.2. Thứ tự ưu tiên

Khi cần thu hẹp phạm vi, ví dụ vì thiếu GPU, vì một cổng không đạt, hoặc vì gần đến hạn nộp, phần việc được cắt từ dưới lên.

| Mức | Nội dung |
|---|---|
| Bắt buộc | Model Qwen3-1,7B (chế độ thinking); MATH-500 và AIME 2024–2025; baseline B2, phương pháp M2 và M3 (mục 7.2); mô phỏng đầy đủ; chạy thật trên một tập con cấu hình; các ablation chính |
| Nên có | Tập sạch AIME 2026 và HMMT 02/2026; GPQA-Diamond; các thành phần lô; các phương pháp để sau B0, B1, M1, O1 |
| Có thể | Các phương pháp để sau B3, B4, B5 và Dynasor-CoT; model thứ hai; chạy bất đồng bộ; chạy lần lượt với điều chỉnh giá; ngân sách khởi đầu theo độ khó dự đoán |
| Ngoài phạm vi | Huấn luyện lại mô hình suy luận; phục vụ trực tuyến với ràng buộc độ trễ |

### 2.3. Giả định của kế hoạch

- GPU lấy từ hai nguồn (mục 3.1): Kaggle miễn phí cho phát triển và test, GPU thuê theo giờ trên Vast.ai cho ghi dữ liệu và chạy thật. Máy Windows hiện tại chỉ dùng để viết và phân tích, vì vLLM chỉ chạy trên Linux hoặc WSL2.
- Mã nguồn do AI agent viết theo kế hoạch implement. Sinh viên đọc báo cáo review và duyệt tại các điểm dừng ([mục 0.4 của kế hoạch implement](Ke-hoach-implement.md#L34)).
- Hạn nộp khóa luận do khoa quy định. Các phase không gắn với mốc thời gian; nếu hạn nộp đến trước khi xong mọi phase, cắt phạm vi theo mục 2.2 và ưu tiên hoàn thành P0–P4.

## 3. Hạ tầng và tài nguyên

### 3.1. Phần cứng

| Nguồn | Cấu hình | Phạm vi sử dụng |
|---|---|---|
| Kaggle (miễn phí) | 2 × T4 16 GB; khoảng 30 giờ GPU mỗi tuần, mỗi phiên tối đa 12 giờ (kiểm tra lại quy định hiện hành trước P0) | Phát triển, test GPU, ghi thử. Colab miễn phí là dự phòng |
| Vast.ai (thuê theo giây) | 1 × RTX 3090 24 GB, hoặc RTX 4090 / A100 40 GB nếu cần nhanh hơn | Tái lập baseline, ghi dữ liệu, chạy thật |

Lý do chọn: Qwen3-1,7B chiếm khoảng 4 GB trọng số nên chạy được trên GPU 16–24 GB. KV cache của model này khoảng 112 KB mỗi token (28 lớp × 8 KV head × 128 × 2 × 2 byte), nên với chuỗi dài, tốc độ sinh phụ thuộc chủ yếu vào băng thông bộ nhớ. RTX 3090 có băng thông khoảng 936 GB/s, gần gấp ba T4 (320 GB/s), và hỗ trợ bf16 (T4 không hỗ trợ). Giá tham khảo trên Vast.ai ngày 03/11/2026, lấy từ API công khai, máy đã xác minh, một GPU:

| GPU | Theo yêu cầu (USD/giờ) | Có thể bị ngắt (USD/giờ) |
|---|---|---|
| RTX 3090 24 GB | ≈ 0,19–0,20 | ≈ 0,08–0,13 |
| RTX 4090 24 GB | ≈ 0,36–0,54 | ≈ 0,20–0,36 |
| RTX A6000 48 GB | ≈ 0,40 | — |
| A100 40 GB | ≈ 0,44–0,60 | ≈ 0,29 |
| A100 80 GB | — | ≈ 0,52 |
| H100 80 GB | ≈ 2,40 | ≈ 1,00 |

Giá thay đổi theo cung cầu từng ngày; chọn máy đã xác minh, độ tin cậy cao. Loại có thể bị ngắt dùng được cho ghi dữ liệu vì logger chạy tiếp từ các khóa đã ghi (việc 1.6).

**Cấu hình thuê trên Vast.ai** (dùng để tính chi phí ở mục 3.3):

| Mục | Giá trị |
|---|---|
| Máy | 1 × RTX 3090; đã xác minh; độ tin cậy ≥ 0,98; băng thông bộ nhớ Vast đo ≥ 780 GB/s; `cuda_max_good` ≥ 13.0; ≥ 8 vCPU; ≥ 32 GB RAM; mạng tải xuống ≥ 300 Mbps |
| Container image | `vllm/vllm-openai:v0.32.0` (CUDA 13.0, 8,6 GB nén); phiên bản chính thức chốt ở việc 0.1 |
| Chế độ chạy | SSH; Vast thay entrypoint của image, onstart script clone repo và cài gói |
| Ổ đĩa | Container disk 60 GB; không dùng volume của Vast. Dữ liệu ghi được đồng bộ lên một dataset riêng tư trên Hugging Face sau mỗi lô |
| Giá ngày 03/11/2026, máy khớp cấu hình trên, đã gồm 60 GB ổ đĩa | Theo yêu cầu 0,2356 USD/giờ; có thể bị ngắt 0,1317 USD/giờ |

Xóa máy (destroy) khi xong mỗi đợt; máy chỉ dừng (stop) vẫn bị tính tiền ổ đĩa.

**Kiểu số:** mọi lần ghi dữ liệu và chạy thật dùng cùng một kiểu số, chốt ở P0 và ghi vào `docs/decisions.md`: bf16 trên GPU thuê; chỉ dùng fp16 nếu buộc phải ghi trên T4.

Ổ đĩa cần khoảng 60 GB: khoảng 20 GB image đã giải nén, 4 GB trọng số model, dưới 10 GB dữ liệu ghi lại, phần còn lại cho môi trường, kết quả và bộ nhớ đệm.

### 3.2. Phần mềm

- Linux, CUDA theo image vLLM đã chốt (mục 3.1), Python 3.12, quản lý môi trường bằng `uv`.
- vLLM được cố định một phiên bản từ P0 và không nâng cấp giữa chừng. Danh sách thư viện ở [mục 2.2 của kế hoạch implement](Ke-hoach-implement.md#L128).
- Dữ liệu ghi lại không đưa vào git; chỉ lưu tệp kê khai kèm mã băm.
- Mã tham khảo: repo [DEER](https://github.com/iie-ycx/DEER) cho lời nhắc gợi đáp án thử và cách tính độ tự tin; repo [Dynasor](https://github.com/hao-ai-lab/Dynasor) cho cách gợi đáp án qua prefix caching.

### 3.3. Ước tính khối lượng tính toán

Chi phí ghi một chuỗi suy luận dài $L$ token xấp xỉ $\kappa L$, trong đó $\kappa \approx 1{,}2$–$2$ là hệ số cho phần gợi đáp án thử và câu trả lời ép buộc tại mọi điểm quyết định. Giá trị của $\kappa$ phụ thuộc vào dạng câu trả lời được chọn ở P1. Bảng dưới dùng các giả định sau; tất cả được đo lại ở P1 (việc 1.3 và 1.9):

- $\kappa = 1{,}9$; mỗi chuỗi khi ghi tốn khoảng 12.000 token.
- Độ dài trung bình một chuỗi của Qwen3-1,7B khoảng 6.300 token, ước lượng từ độ dài trong bảng 2 của bài DEER (MATH 5.160, AMC 8.637, AIME 13.758 token) và tỉ lệ các nguồn trong mỗi tập.
- Thông lượng của Qwen3-1,7B trên một RTX 3090, lấy mức thận trọng: 1.000–1.300 token/giây khi ghi; 700–1.000 token/giây khi chạy thật, do phải chờ đồng bộ mỗi vòng. Trần lý thuyết theo băng thông bộ nhớ là khoảng 1.400–1.700 token/giây.
- Ghi dữ liệu dùng máy có thể bị ngắt, cộng 10% cho phần phải chạy lại; các việc còn lại dùng máy theo yêu cầu. Giá theo mục 3.1; tỷ giá Vietcombank bán ra ngày 03/11/2026 là 25.970 đồng/USD.

| Phase | Hạng mục | Khối lượng | Token (triệu) | Loại máy | Giờ RTX 3090 | USD | Nghìn đồng |
|---|---|---|---|---|---|---|---|
| P0–P4 | Phát triển, test GPU, ghi thử để gỡ lỗi, smoke test (P0 ≈ 3–5 giờ, P1 ≈ 10–18, P2–P3 ≈ 0–2, P4 ≈ 4–7) | Tập nhỏ | — | Kaggle | ≈ 17–32 giờ hạn mức Kaggle, tính cả thời gian phiên mở | 0 | 0 |
| P1 | Tái lập baseline theo cấu hình bài DEER (việc 1.2) | MATH-500 một lượt; AIME 2024 bốn lượt; model gốc và DEER | ≈ 7 | Theo yêu cầu | 1,5–2,5 | | |
| P1 | Đo thông lượng (việc 1.3) | Tập con `val`, bốn kích thước lô | ≈ 2 | Theo yêu cầu | 0,5 | | |
| P1 | Kiểm định tương đương (việc 1.5) | 400 câu `val` × 2 seed, sinh liền mạch và sinh theo đoạn | ≈ 15 | Theo yêu cầu | 3–4 | | |
| P1 | Ghi thử, chọn dạng câu trả lời, đo nhiễu lời nhắc (việc 1.7, 1.8) | 100 câu `val` × 2 seed | ≈ 3,5 | Theo yêu cầu | 0,75–1 | | |
| **P1** | | | **≈ 27** | | **≈ 6–8** | **1,41–1,88** | **37–49** |
| **P2** | Ghi dữ liệu thăm dò | 500 câu `train` × 4 seed; 400 câu `val` × 8 seed | **≈ 62** | Có thể bị ngắt | **≈ 14,7–19,1** | **1,94–2,52** | **50–65** |
| **P3** | Ghi nốt dữ liệu | 1.000 câu `train` còn lại × 4 seed; 560 câu `test_main` × 8 seed | **≈ 102** | Có thể bị ngắt | **≈ 23,9–31,1** | **3,15–4,10** | **82–106** |
| P4 | Chạy thật E5 (việc 4.4) | MATH-500 × $\alpha = 0{,}5$ × {B2, M2} × 3 seed: 6 lần | ≈ 8 | Theo yêu cầu | 2,2–3,1 | | |
| P4 | So thời gian với DEER chính thức và sinh liền mạch (việc 4.5) | MATH-500 một lượt | ≈ 4 | Theo yêu cầu | 0,8–1,1 | | |
| **P4** | | | **≈ 12** | | **≈ 3–4,2** | **0,71–0,99** | **18–26** |
| Dự phòng | Kiểm định tương đương thêm 2 seed; tái lập lại nếu lệch; chạy lại E5 một lần | | | Theo yêu cầu | ≈ 8–10 | 1,88–2,36 | 49–61 |
| Khởi tạo máy | Khoảng 8 lần thuê, mỗi lần 15–20 phút tải image, model và cài gói | | | Theo yêu cầu | ≈ 2–3 | 0,47–0,71 | 12–18 |
| Mạng | Tải image và model mỗi lần thuê; đẩy dữ liệu ghi lên Hugging Face | ≈ 110 GB | | | | ≈ 0,3 | ≈ 8 |
| **Tổng P0–P4** | | | **≈ 200** | | **≈ 58–75** | **≈ 9,9–12,9** | **≈ 256–335** |
| P5 (nếu làm) | Ghi `test_clean` và `test_ood` (việc 5.2) | 60 + 198 câu × 8 seed | ≈ 37–45 | Có thể bị ngắt | ≈ 9–14 | ≈ 1,2–1,9 | ≈ 31–49 |
| **Tổng cả đề tài** | | | | | **≈ 67–89** | **≈ 11,1–14,8** | **≈ 288–384** |

Nếu thanh toán bằng thẻ của ngân hàng Việt Nam, cộng thêm phí giao dịch ngoại tệ khoảng 1–3%. Nạp trước 15 USD là đủ cho cả đề tài. Nếu thông lượng đo ở việc 1.3 đạt trần lý thuyết, tổng giảm khoảng 25%.

Thời gian thực: P2 ghi liên tục khoảng một ngày, P3 khoảng 1–1,5 ngày, chạy nền trong lúc viết mã. Có thể thuê hai máy cùng lúc để rút ngắn một nửa thời gian mà tổng tiền không đổi.

Các phương pháp để sau (việc 5.1), mô phỏng, huấn luyện bộ ước lượng và chỉnh tham số chạy trên CPU, không tốn giờ GPU.

Phương án khác:
- **Chỉ dùng Kaggle:** khoảng 110–150 giờ Kaggle, tức 4–5 tuần hạn mức. Phải ghi bằng fp16 vì T4 không hỗ trợ bf16, và thời gian đo được ở E5 phản ánh T4.
- **Giảm seed** (`train` × 2, `val` × 4, `test_main` × 4): tổng P0–P4 giảm còn khoảng 7,3–9,6 USD (190–250 nghìn đồng); đổi lại bộ ước lượng học trên ít chuỗi hơn và đường cong của M3 nhiễu hơn.
- **A100 40 GB:** nhanh hơn RTX 3090 khoảng 1,7 lần nhờ băng thông bộ nhớ lớn hơn; tổng tiền tương đương hoặc cao hơn một chút.

## 4. Dữ liệu

### 4.1. Phân chia

| Tập | Vai trò | Nguồn | Số câu | Seed | Ghi chú |
|---|---|---|---|---|---|
| `train` | Huấn luyện bộ ước lượng | MATH tập train, đủ 5 mức độ khó (`EleutherAI/hendrycks_math`): 900 câu; DeepScaleR gồm AIME 1984–2023, AMC trước 2023, Omni-MATH, STILL (`agentica-org/DeepScaleR-Preview-Dataset`): 600 câu | 1.500 | 4 | Lấy mẫu phân tầng theo độ khó |
| `val` | Chỉnh mọi tham số; cổng G1 | AMC 2023 (`math-ai/amc23`): 40 câu; MATH tập train, giữ riêng: 360 câu | 400 | 8 | |
| `test_main` | Kiểm tra chính | MATH-500 (`HuggingFaceH4/MATH-500`); AIME 2024 (`HuggingFaceH4/aime_2024`) và AIME 2025 (`MathArena/aime_2025`) | 560 | 8 | Cả AIME 2024 và AIME 2025 đều ra đời trước Qwen3 (04/2025), nên có thể đã lọt vào dữ liệu huấn luyện |
| `test_clean` | Kiểm tra sạch | AIME 2026 và HMMT 02/2026 (MathArena) | 60 | 8 | Ra đời sau Qwen3; cần kiểm tra tên dataset trên Hugging Face |
| `test_ood` | Kiểm tra ngoài miền | GPQA-Diamond (`Idavidrein/gpqa`, cần chấp nhận điều khoản) | 198 | 8 | Trắc nghiệm khoa học; bộ ước lượng chỉ học trên toán |

### 4.2. Quy tắc quản lý dữ liệu

- Tách theo câu hỏi, không theo mẫu. Danh sách ID của từng tập được cố định ở P0, lưu kèm mã băm, và không thay đổi về sau.
- Khử trùng lặp giữa tập huấn luyện, validation và mọi tập kiểm tra: chuẩn hóa văn bản rồi so khớp chính xác, cộng thêm so khớp 13-gram.
- Dữ liệu thăm dò ở P2 là tập con của dữ liệu đầy đủ, nên không phải ghi lại ở P3.
- Thành phần lô được dựng từ các câu có mức độ khó của MATH: cân bằng, chủ yếu dễ, chủ yếu khó và hình chữ U. Có thêm một lô trộn MATH-500 với AIME. Tỉ lệ cụ thể ở [P2.T14 của kế hoạch implement](Ke-hoach-implement.md#L955).
- Đáp án trắc nghiệm của GPQA được chấm bằng trích xuất chữ cái, không dùng math-verify. Thứ tự các lựa chọn được xáo cố định theo từng câu.

### 4.3. Cấu hình sinh

Model chính là Qwen3-1,7B ở chế độ thinking. Có hai cấu hình: một cho ghi dữ liệu và chạy thật, một chỉ để tái lập số công bố ở P1 (việc 1.2).

| Tham số | Ghi dữ liệu và chạy thật | Tái lập baseline ở P1 (cấu hình bài DEER) |
|---|---|---|
| Lấy mẫu | temperature 0,6; top_p 0,95; top_k 20 (khuyến nghị của Qwen3) | Giải mã tham lam |
| Lời nhắc | `Please reason step by step, and put your final answer within \boxed{}.`; chat template với `enable_thinking=True` | Như cột bên trái |
| Độ dài tối đa | $L_{\text{log}}$ = 32.768 token | 16.384 token |
| Kiểu số | Theo mục 3.1, cố định cho mọi lần ghi và chạy thật | Như cột bên trái |
| Nhánh gợi đáp án thử | Lời nhắc cố định theo DEER; giải mã tham lam; `logprobs=1`; tối đa 20 token | Script DEER chính thức |

Số công bố để đối chiếu nằm ở bảng 2 của bài DEER, dòng Qwen3-1.7B: model gốc đạt 85,6% trên MATH (5.160 token) và 30,0% trên AIME (13.758 token); DEER đạt 85,6% (2.463 token) và 30,0% (7.943 token).

Nếu sau này thêm model thứ hai thuộc họ R1-Distill: lấy mẫu với temperature 0,6, top_p 0,95; phần trả lời bắt đầu bằng `<think>\n`.

Giá trị khởi đầu của các tham số phương pháp ($b_{\min}$, $b_{\max}$, $H$, $L_{\max}$, $m$, $k$, $r_{\text{ans}}$) lấy theo [mục 10 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L286). Giá trị dùng trong mã ở [mục 9.1 của kế hoạch implement](Ke-hoach-implement.md#L1261).

## 5. Mã nguồn

Toàn bộ đặc tả mã nguồn, gồm kiến trúc, dữ liệu ghi lại, thuật toán, test và quy trình review, nằm ở [kế hoạch implement](Ke-hoach-implement.md). Ba điểm chính:

- Bộ mô phỏng và bộ chạy thật dùng chung một vòng lặp điều khiển và cùng mã chính sách. Hai bên chỉ khác nguồn của đoạn suy luận kế tiếp: đọc từ dữ liệu ghi lại, hoặc sinh bằng vLLM ([mục 3.1 của kế hoạch implement](Ke-hoach-implement.md#L193)).
- Mọi số token được đếm theo một định nghĩa chuẩn gồm năm thành phần: suy luận, lời nhắc gợi, đáp án thử, lời nhắc trả lời và câu trả lời ([mục 2.6](Ke-hoach-implement.md#L157)). Tổng chi phí của mọi lần chạy không vượt $B$ (bất biến I1 ở [mục 2.7](Ke-hoach-implement.md#L173)).
- Mỗi nhóm task kết thúc bằng một lượt review độc lập, và agent dừng ở các điểm cần con người xác nhận ([mục 0.4](Ke-hoach-implement.md#L34)).

## 6. Kế hoạch theo phase

### P0. Nền tảng

**Mục tiêu:** có môi trường chạy được, repo nền tảng, dữ liệu đã chia và khóa, và bộ chấm điểm.

**Bắt đầu khi:** đã xong các việc ở mục 13.

| ID | Công việc | Thực hiện | Kế hoạch implement |
|---|---|---|---|
| 0.1 | Cài môi trường GPU; cố định phiên bản vLLM và ghi lý do chọn | Agent; SV kiểm tra | Bước chạy 1 |
| 0.2 | Khung repo, config, thư mục chạy, kiểu dữ liệu, dữ liệu tổng hợp cho test | Agent | P0.T1–T5 |
| 0.3 | Bộ nạp dữ liệu; tải model và các tập ở mục 4.1 | Agent | P0.T6, bước chạy 2 |
| 0.4 | Chia tập và khử trùng lặp; duyệt và khóa danh sách ID | Agent; SV duyệt | P0.T7, bước chạy 3 |
| 0.5 | Lời nhắc, bộ chấm điểm và công cụ xuất mẫu để chấm tay | Agent | P0.T8–T10 |

**Đầu ra:** D1.

**Tiêu chí hoàn thành:** review P0a đạt; danh sách ID đã khóa; vLLM chạy được trên máy chủ GPU.

Đặc tả mã nguồn: [P0 trong kế hoạch implement](Ke-hoach-implement.md#L730).

### P1. Engine và công cụ ghi

**Mục tiêu:** có công cụ sinh chuỗi suy luận theo đoạn, gợi đáp án thử ở nhánh phụ và ghi đủ dữ liệu cho cả bộ ước lượng lẫn bộ mô phỏng; có số liệu tham chiếu của baseline.

**Bắt đầu khi:** P0 kết thúc.

| ID | Công việc | Thực hiện | Kế hoạch implement |
|---|---|---|---|
| 1.1 | Sinh theo đoạn trên vLLM: gửi lại tiền tố dưới dạng token id, bật prefix caching; kết thúc đoạn tại từ chuyển ý ở đầu đoạn ("Wait", "Alternatively", "Hmm", "But", "Let me check") hoặc sau $L_{\max}$ token. Nhánh gợi đáp án thử; câu trả lời ép buộc; đếm token theo năm thành phần | Agent | P1.T1–T4 |
| 1.2 | Chạy model gốc, không dừng sớm, và script DEER chính thức với Qwen3-1,7B, theo cấu hình bài DEER (mục 4.3): MATH-500 một lượt, AIME 2024 bốn lượt. So với bảng 2 của bài DEER | Agent; SV duyệt việc dùng tập kiểm tra | P1.T5, bước chạy 1 |
| 1.3 | Đo thông lượng theo kích thước lô, trên GPU sẽ dùng để ghi dữ liệu | Agent | P1.T5, bước chạy 2 |
| 1.4 | Chấm tay 100 mẫu ngẫu nhiên để kiểm tra bộ chấm điểm | SV | Bước chạy 3 |
| 1.5 | Kiểm định tương đương giữa sinh liền mạch và sinh theo đoạn có nhánh phụ | Agent | P1.T6 |
| 1.6 | Lưu trữ, logger chạy tiếp được sau khi bị ngắt, kiểm tra dữ liệu ghi. Khi ghi, điểm ghi dày hơn mức chính sách cần: hai điểm liên tiếp cách nhau hơn 128 và không quá 256 token | Agent | P1.T7, P1.T8, P1.T11 |
| 1.7 | Chọn dạng câu trả lời: đầy đủ (tối đa 256 token) hay rút gọn (chỉ phần đáp án trong khung). Nếu hai dạng cho cùng kết quả chấm ở ít nhất 98% số điểm, chọn dạng rút gọn để giảm $\kappa$. Dạng đã chọn được dùng thống nhất cho ghi dữ liệu, mô phỏng và chạy thật | Agent đề xuất; SV duyệt | P1.T9, bước chạy 4 |
| 1.8 | Đo nhiễu do lời nhắc: tại cùng một điểm, gợi bằng hai mẫu lời nhắc khác nhau trên 100 chuỗi | Agent | P1.T10, bước chạy 4 |
| 1.9 | Cập nhật $\kappa$ và thông lượng ở mục 3.3 | SV | Bước chạy 2 và 5 |

**Đầu ra:** D2.

**Tiêu chí hoàn thành:**

- **Baseline:** trên MATH-500, độ chính xác của model gốc và của DEER chênh với số công bố không quá 2 điểm phần trăm. AIME 2024 chỉ có 30 câu nên chỉ báo cáo, không dùng làm tiêu chí. Nếu chênh nhiều hơn, tìm nguyên nhân trước khi ghi dữ liệu.
- **Chấm điểm:** bộ chấm tự động khớp với chấm tay ở ít nhất 99% trong 100 mẫu.
- **Tương đương:** khoảng tin cậy 90% của chênh lệch giữa sinh liền mạch và sinh theo đoạn nằm trong ±3 điểm phần trăm về độ chính xác và ±5% về độ dài trung bình. Lỗi thường gặp: thêm lại BOS hoặc chat template khi gửi lại tiền tố; tokenize lại chuỗi ký tự làm đổi ranh giới token.
- **Đếm token:** khớp tuyệt đối với số token vLLM trả về.
- **Mật độ điểm ghi:** khoảng cách giữa hai điểm liên tiếp không vượt mức tối đa khi ghi.
- Review P1a và P1b đạt.

Đặc tả mã nguồn: [P1 trong kế hoạch implement](Ke-hoach-implement.md#L794).

### P2. Thăm dò

**Mục tiêu:** kiểm tra ba giả định ở [mục 14 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L334) trước khi đầu tư cài đặt đầy đủ: bộ ước lượng có tín hiệu, có dư địa để cải thiện, và mô phỏng hợp lệ.

**Bắt đầu khi:** P1 kết thúc.

| ID | Công việc | Thực hiện | Kế hoạch implement |
|---|---|---|---|
| 2.1 | Ghi dữ liệu thăm dò với Qwen3-1,7B: 500 câu `train` × 4 seed; toàn bộ `val` × 8 seed. GPU chạy nền trong lúc agent viết mã cho các việc sau | Agent | Bước chạy 1 |
| 2.2 | Trích đặc trưng; bộ ước lượng v0 (LightGBM, hiệu chuẩn isotonic theo từng $h$); E1 trên `val` | Agent | P2.T1–T6, bước chạy 2 |
| 2.3 | Bộ phân bổ, bảo đảm ngân sách cứng, bộ mô phỏng; baseline B2, phương pháp M2 và M3; dựng thành phần lô | Agent | P2.T7–T15 |
| 2.4 | Độ đo, thống kê, báo cáo, cổng G1 | Agent | P2.T16–T19 |
| 2.5 | Vẽ đường cong xác suất đúng theo số token của từng câu; đo tỉ lệ câu được lợi và tỉ lệ câu bị giảm độ chính xác khi suy luận thêm | Agent | P2.T20 |
| 2.6 | Đo dư địa theo $\alpha$ và theo thành phần lô, với các lô dựng từ phần MATH của `val` | Agent | Bước chạy 3 |
| 2.7 | Viết báo cáo thăm dò; cập nhật tham số trong báo cáo phương pháp chi tiết | SV | — |
| 2.8 | Quyết định G1 cùng giảng viên hướng dẫn | SV | Bước chạy 4 |

**Đầu ra:** D3.

**Tiêu chí hoàn thành:** báo cáo thăm dò trả lời được cả ba tiêu chí của G1 (mục 8); quyết định G1 được ghi lại.

Đặc tả mã nguồn: [P2 trong kế hoạch implement](Ke-hoach-implement.md#L883).

### P3. Mô phỏng đầy đủ

**Mục tiêu:** có kết quả đầy đủ của model chính trên mô phỏng, đủ để trả lời RQ1–RQ3.

**Bắt đầu khi:** đã có quyết định G1. Nếu G1 dẫn tới thu hẹp phạm vi, cập nhật kế hoạch này và kế hoạch implement trước khi bắt đầu.

| ID | Công việc | Thực hiện | Kế hoạch implement |
|---|---|---|---|
| 3.1 | Ghi nốt dữ liệu của model chính: `train` tới đủ 1.500 câu × 4 seed và `test_main` × 8 seed; `test_clean` và `test_ood` ghi ở việc 5.2. GPU chạy nền suốt phase | Agent; SV duyệt dùng GPU | Bước chạy 1 |
| 3.2 | *(Để sau, chuyển sang việc 5.1.)* Các phương pháp so sánh còn lại | — | P5.T1 |
| 3.3 | Cài đặt các biến thể cho ablation (mục 7.4) | Agent | P3.T3 |
| 3.4 | Bộ ước lượng v1 trên dữ liệu đầy đủ, với đủ đặc trưng ở [mục 4.3 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L120); so sánh LightGBM với logistic regression theo từng $h$, chọn theo log-loss trên `val` | Agent | Bước chạy 2 |
| 3.5 | Chỉnh tham số trên `val`: mọi phương pháp có cùng ngân sách chỉnh (50 cấu hình tìm ngẫu nhiên); một bộ tham số chung cho mọi mức $B$ | Agent | P3.T4, bước chạy 3 |
| 3.6 | Khóa cấu hình và ghi vào sổ kiểm tra; chạy E1–E4 trên tập kiểm tra chính | Agent; SV duyệt cấu hình khóa | P3.T5–T6, bước chạy 3 và 4 |
| 3.7 | Phân tích lỗi: tỉ lệ cắt nhầm theo độ khó; các trường hợp điển hình của từng loại lỗi | SV | — |
| 3.8 | Quyết định G2 cùng giảng viên hướng dẫn | SV | Bước chạy 4 |

**Đầu ra:** D4.

**Tiêu chí hoàn thành:** có bảng và hình của E1–E4 cho model chính, sinh lại được bằng một lệnh; quyết định G2 được ghi lại.

Đặc tả mã nguồn: [P3 trong kế hoạch implement](Ke-hoach-implement.md#L999).

### P4. Chạy thật

**Mục tiêu:** có bộ phân bổ chạy thật trên vLLM, cho kết quả khớp với mô phỏng và có số đo chi phí hệ thống.

**Bắt đầu khi:** đã có quyết định G2. Nếu G2 không đạt và M3 cũng không hơn B2, P4 chỉ làm ở mức minh chứng khả thi.

| ID | Công việc | Thực hiện | Kế hoạch implement |
|---|---|---|---|
| 4.1 | Vòng lặp phân bổ quanh vLLM: mỗi vòng gửi các câu tiếp tục suy luận kèm hạn mức `max_tokens`, chờ mọi câu tới điểm quyết định, gợi đáp án thử cho cả lô trong một lần gọi, rồi ra quyết định. Bảo đảm ngân sách cứng; cửa sổ trượt khi số câu vượt sức chứa KV cache, theo [mục 7.4 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L231) | Agent | P4.T1 |
| 4.2 | Kiểm thử phát lại: một engine giả phát lại các chuỗi đã ghi; quyết định của bộ chạy thật phải trùng hoàn toàn với bộ mô phỏng | Agent | P4.T2 |
| 4.3 | Kiểm thử ngân sách trên GPU với $B$ rất nhỏ | Agent | P4.T3 |
| 4.4 | Chạy lưới E5 với Qwen3-1,7B; đo thời gian hoàn thành lô, thông lượng, thời gian chờ đồng bộ, tỉ lệ trúng prefix cache, thời gian của bộ ước lượng và bộ phân bổ mỗi vòng | Agent; SV duyệt dùng GPU | P4.T4, bước chạy 1 |
| 4.5 | So thời gian chạy với script DEER chính thức và với sinh liền mạch, trên cùng phần cứng | Agent | P4.T5, bước chạy 2 |
| 4.6 | Quyết định G3 | SV | Bước chạy 1 |

**Đầu ra:** D5.

**Tiêu chí hoàn thành:** có quyết định G3; kiểm thử phát lại và kiểm thử ngân sách chạy tự động.

Đặc tả mã nguồn: [P4 trong kế hoạch implement](Ke-hoach-implement.md#L1041).

### P5. Mở rộng

**Mục tiêu:** bổ sung các phương pháp so sánh để sau; kiểm tra tính tổng quát của kết quả (E6).

**Bắt đầu khi:** đã có quyết định G3. Phần việc được chọn theo thứ tự ưu tiên ở mục 2.2. Riêng việc 5.1 có thể làm sớm hơn, ngay sau khi khóa cấu hình ở P3, vì chỉ chạy trong mô phỏng.

| ID | Công việc | Thực hiện | Kế hoạch implement |
|---|---|---|---|
| 5.1 | Các phương pháp so sánh để sau (mục 7.2): B0, B1, M1, O1; nếu còn thời gian thì B3, B4, B5 và Dynasor-CoT. Chạy trong mô phỏng trên dữ liệu đã ghi với cấu hình đã khóa, không cần GPU; bổ sung vào E2 và phân rã đầy đủ ở E3 | Agent; SV duyệt cấu hình khóa | P5.T1, bước chạy 1 |
| 5.2 | Tập sạch AIME 2026 và HMMT 02/2026; GPQA-Diamond với bộ ước lượng học trên toán | Agent; SV duyệt dùng GPU | Bước chạy 2 |
| 5.3 | Thành phần lô (mục 4.2) và kích thước lô $N \in \{32, 128, 500\}$ | Agent | Bước chạy 3 |
| 5.4 | (Có thể) Model thứ hai, chọn theo kinh phí: R1-Distill-Qwen-1,5B chạy được trên GPU 16–24 GB; R1-Distill-Qwen-7B hoặc Qwen3-8B cần GPU thuê lớn hơn. Kiểm tra lại dạng câu trả lời; ghi dữ liệu; bộ ước lượng và hiệu chuẩn riêng; chỉnh tham số và khóa cấu hình; E2 và E4 rút gọn trên mô phỏng | SV quyết định; Agent; SV duyệt dùng GPU và cấu hình khóa | P5.T2, bước chạy 4 |
| 5.5 | (Có thể) Một hướng mở rộng ở [mục 7.5 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L235), chỉ khi P0–P4 đều đạt mà không phải cắt phạm vi | SV quyết định; Agent làm khi đã có đặc tả | P5.T3 |

**Đầu ra:** D6.

**Tiêu chí hoàn thành:** mọi phần được giao của P5 có kết quả, kể cả khi kết quả âm tính.

Đặc tả mã nguồn: [P5 trong kế hoạch implement](Ke-hoach-implement.md#L1073).

### P6. Hoàn thiện

**Mục tiêu:** có bản khóa luận hoàn chỉnh, gói tái lập, và sẵn sàng bảo vệ.

**Bắt đầu khi:** P5 kết thúc, hoặc phần còn lại của P5 đã được cắt theo mục 2.2.

| ID | Công việc | Thực hiện | Kế hoạch implement |
|---|---|---|---|
| 6.1 | Chốt toàn bộ số liệu; sinh lại mọi bảng và hình từ `runs/` | SV chốt; Agent sinh lại | P6.T1 |
| 6.2 | Viết các chương còn lại theo mục 11 | SV | — |
| 6.3 | Gửi bản thảo cho giảng viên hướng dẫn; chỉnh sửa theo góp ý | SV | — |
| 6.4 | Đóng gói mã nguồn, dữ liệu ghi lại và hướng dẫn tái lập | Agent | P6.T1 |
| 6.5 | Chuẩn bị slide; tập dượt bảo vệ | SV | — |

**Đầu ra:** D7, D8.

**Tiêu chí hoàn thành:** bản nộp cuối đã được chỉnh theo góp ý của giảng viên hướng dẫn; gói tái lập sinh lại đúng mọi bảng và hình trong khóa luận.

Đặc tả mã nguồn: [P6 trong kế hoạch implement](Ke-hoach-implement.md#L1113).

## 7. Thiết kế thực nghiệm

### 7.1. Danh sách thí nghiệm

| Mã | Thí nghiệm | Trả lời cho | Phase |
|---|---|---|---|
| E1 | Chất lượng bộ ước lượng: AUROC, ECE, biểu đồ độ tin cậy, mức quan trọng của đặc trưng | RQ1 | P2 (`val`), P3 (tập kiểm tra) |
| E2 | So sánh chính: đường độ chính xác–token của B2, M2, M3; các phương pháp để sau được bổ sung ở việc 5.1 | RQ2 | P3, P5 |
| E3 | Phân rã đóng góp của từng thành phần: hiện tại B2 → M2 → M3; đầy đủ nhóm ① đến ⑤ sau việc 5.1 | RQ3 | P3, P5 |
| E4 | Ablation thành phần và độ nhạy tham số | RQ3 | P3 |
| E5 | Chạy thật trên vLLM: đối chiếu với mô phỏng, đo thời gian và chi phí hệ thống | Khả thi | P4 |
| E6 | Tập sạch, tập ngoài miền, thành phần và kích thước lô; model thứ hai nếu làm | Tổng quát | P5 |

### 7.2. Các phương pháp so sánh

Hiện tại chỉ làm ba phương pháp: baseline **B2**, phương pháp đề xuất **M2**, và **M3**. M3 không phải baseline mà là công cụ chẩn đoán: cổng G1 dùng M3 để đo dư địa, cổng G2 dùng M3 để biết nút thắt nằm ở bộ ước lượng hay ở quy tắc phân bổ (mục 8). M3 chạy trong mô phỏng nên không tốn GPU. Các phương pháp còn lại để sau (việc 5.1).

| Nhóm | Mã | Cấu hình | Vai trò | Trạng thái |
|---|---|---|---|---|
| ① Chia đều | B0 | Không dừng sớm, mỗi câu tối đa $B/N$ token | Mốc dưới | Để sau |
| | B1 | DEER, mỗi câu tối đa $B/N$ token; ngưỡng $\lambda$ chỉnh trên validation, giá trị khởi đầu 0,95 | Dừng sớm theo từng câu | Để sau |
| ② Tận dụng token dư, không có giá | B2 | DEER với giới hạn mỗi câu nâng lên sao cho tổng bằng $B$ | **Baseline chính** | **Đang làm** |
| | B3 | CoDE-Stop với giới hạn nâng lên | Baseline mạnh thứ hai | Để sau |
| | B4 | DEER cộng quỹ chung; câu nào hết hạn mức trước thì được cấp thêm trước | Giá trị của việc xếp hạng | Để sau |
| | B5 | Xử lý lần lượt từng câu, token dư dồn sang câu sau | Cách làm đơn giản nhất khi không có quỹ | Để sau |
| ③ Giá cố định | M1 | Giá $\mu$ cố định cho từng $B$, hiệu chỉnh để tổng chi tiêu xấp xỉ $B$, không cập nhật trong lúc chạy | Giá trị của việc cập nhật giá | Để sau |
| ④ Phương pháp đề xuất | M2 | Giá $\mu$ cập nhật theo vòng đồng bộ | | **Đang làm** |
| | M3 | Như M2, nhưng thay $\hat q$ bằng đường cong thực nghiệm tính từ các seed khác của cùng câu | Trần của bộ ước lượng | **Đang làm** |
| ⑤ Oracle | O1 | Biết trước kết quả chấm tại mọi điểm của chuỗi đang xét | Trần lý thuyết | Để sau |

**Quy tắc công bằng:** mọi phương pháp dùng cùng điểm quyết định, cùng lời nhắc gợi, cùng dạng câu trả lời, cùng cơ chế bảo đảm ngân sách cứng và cùng số cấu hình khi chỉnh tham số. Token gợi đáp án thử được tính cho mọi phương pháp có gợi. Với Qwen3, baseline DEER giữ điều kiện bổ sung của code gốc: model phải sinh `</think>` ngay sau đáp án thử.

**Hiệu chỉnh theo ngân sách:** hạn mức của B2 và giá của M1 được hiệu chỉnh trên chính tập đang đánh giá, chỉ dựa vào tổng chi tiêu, không dùng nhãn đúng sai. Cách này có lợi cho baseline và được nêu rõ trong báo cáo kết quả.

**Cách tính O1:** mỗi câu đóng góp 0 hoặc 1 câu đúng, nên bài toán của oracle giải chính xác bằng cách sắp các câu theo chi phí tăng thêm để đạt một câu trả lời đúng (so với chi phí tối thiểu), rồi chọn lần lượt tới khi hết ngân sách.

**Cách đọc E3** (hiện tại chỉ có hai bước B2 → M2 và M2 → M3; các bước còn lại có sau việc 5.1):
- ① → ②: lợi ích của việc tận dụng token dư.
- ② → ③: lợi ích của việc xếp hạng theo lợi ích trên mỗi token, với một mức giá.
- ③ → M2: lợi ích của việc cập nhật giá theo tình trạng quỹ.
- M2 → M3: phần mất do bộ ước lượng chưa chính xác.
- M3 → ⑤: phần không thể đạt được khi không biết trước kết quả của chuỗi đang chạy.

### 7.3. Lưới cấu hình

- Ngân sách $B = \alpha \cdot C_{\text{full}}$, với $\alpha \in \{0{,}2;\ 0{,}3;\ 0{,}4;\ 0{,}5;\ 0{,}7;\ 1{,}0\}$ và $C_{\text{full}}$ là tổng token khi mọi câu chạy tới hết, không dừng sớm. Kết quả được báo cáo kèm $B/N$ tính theo token.
- **Mô phỏng:** toàn bộ lưới model × tập dữ liệu × $\alpha$ × phương pháp; mỗi ô 200 lần rút seed.
- **Chạy thật (E5):**
  - Phương án tối thiểu: Qwen3-1,7B × MATH-500 × $\alpha = 0{,}5$ × {B2, M2} × 3 seed: 6 lần, ở P4.
  - Thêm một lượt script DEER chính thức và một lượt sinh liền mạch trên MATH-500 để so thời gian.
  - Nếu kết quả tốt và còn kinh phí, mở rộng sang $\alpha \in \{0{,}3;\ 0{,}5;\ 0{,}7\}$ và lô trộn MATH-500 với AIME (tối đa 36 lần); mỗi 6 lần thêm khoảng 3 giờ RTX 3090.
  - Thời gian đo được gắn với GPU đã dùng (mục 3.1) và được báo cáo kèm tên GPU.

### 7.4. Ablation (E4)

- Chỉ một tầm nhìn, $H = \{1\text{k}\}$ (quy tắc một bước của đề xuất ban đầu).
- Bỏ từng nhóm đặc trưng: độ nhất quán; xu hướng; độ tự tin cục bộ thấp nhất; dạng đáp án.
- Bỏ bước hiệu chuẩn.
- Bỏ giá sàn $\mu_{\min}$; bỏ trần $b_{\max}$.
- Cách trả lời khi dừng: câu trả lời ép buộc, đáp án thử tự tin nhất, hoặc bỏ phiếu có trọng số ([mục 6.1 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L194)).
- Dạng bộ ước lượng: LightGBM hoặc logistic regression theo từng $h$.
- Độ nhạy theo $L_{\max}$, $m$, $k$ và $b_{\min}$.

### 7.5. Độ đo

| Độ đo | Định nghĩa |
|---|---|
| Độ chính xác tại $B$ | Tỉ lệ câu trả lời đúng, với tổng token không vượt $B$ |
| Đường độ chính xác–token và AUC | Độ chính xác theo $\alpha$; diện tích dưới đường, chuẩn hóa về khoảng [0, 1] |
| Token tiết kiệm ở cùng độ chính xác | So với B2, nội suy trên đường độ chính xác–token |
| Tỉ lệ cắt nhầm | Tỉ lệ câu bị dừng với đáp án sai, trong khi nếu chạy tới hết thì đúng |
| Độ chính xác theo nhóm độ khó | Theo mức độ khó của MATH, hoặc theo tỉ lệ đúng của model gốc |
| Chất lượng bộ ước lượng | AUROC, ECE, điểm Brier, biểu đồ độ tin cậy, theo từng $h$ |
| Tỉ trọng chi phí gợi | Phần trăm token dành cho lời nhắc gợi và đáp án thử |
| Chi phí hệ thống | Thời gian hoàn thành lô, thông lượng, thời gian chờ đồng bộ, tỉ lệ trúng prefix cache, thời gian của bộ ước lượng và bộ phân bổ mỗi vòng |

Mọi số token được báo cáo đều gồm cả lời nhắc gợi, đáp án thử và câu trả lời cuối.

### 7.6. Giao thức mô phỏng

1. Mỗi lần mô phỏng, với mỗi câu, rút ngẫu nhiên một seed trong các chuỗi đã ghi. Các phương pháp dùng chung các lần rút, nên so sánh được theo cặp.
2. Chính sách chỉ quan sát đáp án thử tại những điểm mà nó thực sự dừng lại để xét, và chỉ những lần gợi đó bị tính chi phí.
3. Khi dừng tại điểm $\tau$, kết quả chấm lấy từ câu trả lời ép buộc đã ghi tại $\tau$. Nếu chuỗi tự kết thúc trước đó, lấy câu trả lời tự nhiên.
4. Nếu hạn mức của một đoạn kết thúc giữa hai điểm đã ghi, điểm dừng được làm tròn xuống điểm đã ghi gần nhất.
5. Điều kiện hợp lệ: điểm quyết định của chính sách là tập con của các điểm đã ghi, và $b_{\max} \le L_{\text{log}} - c_{\text{ans,max}}$, với $c_{\text{ans,max}}$ là chi phí tối đa của một câu trả lời, gồm cả lời nhắc. Mỗi khi đổi $L_{\max}$ hoặc $m$, kiểm tra lại điều kiện này.
6. Với M3, đường cong thực nghiệm của một câu được tính từ các seed còn lại (bỏ seed đang mô phỏng), để không dùng thông tin của chính chuỗi đang chạy.

### 7.7. Thống kê và kỷ luật với tập kiểm tra

- Mô phỏng: 200 lần rút seed cho mỗi cấu hình; khoảng tin cậy 95% bằng bootstrap cặp theo câu hỏi. Khi kết luận trên nhiều mức $\alpha$, như ở cổng G1 và G2, dùng khoảng tin cậy hiệu chỉnh Bonferroni.
- So sánh hai phương pháp trên cùng các lần chạy: kiểm định McNemar theo từng câu.
- Chạy thật: 3 seed cho mỗi cấu hình; báo cáo trung bình và độ lệch chuẩn.
- AIME 2024 và 2025 được gộp (60 câu) để giảm nhiễu; tập sạch được báo cáo riêng.
- Tập kiểm tra chỉ được chạy khi cấu hình đã khóa; ngoại lệ duy nhất là lần tái lập baseline ở P1. Mỗi lần chạy được ghi vào `docs/test-log.md` (cấu hình, mã băm, người duyệt, kết quả). Không chỉnh tham số sau khi xem kết quả kiểm tra.

## 8. Cổng quyết định

| Cổng | Vị trí | Tiêu chí | Nếu không đạt |
|---|---|---|---|
| G1 | Cuối P2 | (a) Bộ ước lượng v0 đạt ngưỡng RQ1 trên validation. (b) M3 hơn B2 ít nhất 2 điểm phần trăm ở ít nhất một mức $\alpha \le 0{,}5$, với cận dưới của khoảng tin cậy hiệu chỉnh Bonferroni lớn hơn 0. (c) Ít nhất 95% chuỗi đủ điều kiện mô phỏng (mục 7.6) | (a): thêm dữ liệu và đặc trưng, hoặc thu gọn mục tiêu dự đoán về $\hat q(0)$ và một tầm $h$. (b): thu hẹp phạm vi về vùng có dư địa (ngân sách eo hẹp, lô trộn độ khó), hoặc chuyển trọng tâm sang tiết kiệm token ở cùng độ chính xác. (c): ghi lại với điểm dày hơn hoặc $L_{\text{log}}$ lớn hơn |
| G2 | Cuối P3 | M2 hơn B2 ở ít nhất một mức $\alpha \le 0{,}5$ trên tập kiểm tra chính, với khoảng tin cậy hiệu chỉnh Bonferroni không chứa 0 | So M2 với M3. Nếu M3 hơn nhiều, nút thắt là bộ ước lượng: cải thiện thêm một vòng trên validation, khóa lại với mã băm mới, chạy lại trên tập kiểm tra và báo cáo cả hai kết quả. Nếu M3 cũng không hơn B2, báo cáo kết quả âm tính kèm phân tích; P4 chỉ làm ở mức minh chứng khả thi |
| G3 | Cuối P4 | (a) Không lần chạy nào vượt $B$. (b) Ở cả hai cấu hình (B2 và M2), độ chính xác trung bình 3 seed nằm trong dải chấp nhận của mô phỏng. (c) Ở cả hai cấu hình, độ chính xác khi chạy thật lệch mô phỏng không quá 3 điểm phần trăm. Nếu mở rộng lưới E5, tiêu chí (b) đổi thành: số cấu hình nằm ngoài dải không quá $\max(1, \lfloor 0{,}1 K \rfloor)$, với $K$ là số cấu hình | Kiểm tra lại cách đếm token, vị trí điểm quyết định và việc KV cache bị đẩy ra. Nếu vẫn không đạt sau một vòng sửa, dùng mô phỏng làm kết quả chính và chạy thật làm minh chứng khả thi |

Cách tính chi tiết ở [mục 7.4 của kế hoạch implement](Ke-hoach-implement.md#L1181). Ở mỗi cổng, agent chỉ báo cáo đạt hay không kèm số liệu. Quyết định đi tiếp do sinh viên đưa ra cùng giảng viên hướng dẫn và được ghi vào `docs/decisions.md`.

## 9. Rủi ro thực thi

Các rủi ro kỹ thuật của bản thân phương pháp đã nêu ở [mục 13 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L321). Bảng dưới chỉ gồm rủi ro trong quá trình thực hiện.

| Rủi ro | Dấu hiệu sớm | Cách xử lý |
|---|---|---|
| GPU thuê bị ngắt hoặc máy chủ kém ổn định | Phiên Kaggle hết 12 giờ; máy Vast.ai bị thu hồi hoặc chậm bất thường | Logger chạy tiếp từ các khóa đã ghi; sao lưu dữ liệu sau mỗi phiên; chọn máy đã xác minh, độ tin cậy cao; cắt phạm vi theo mục 2.2 |
| Kết quả phụ thuộc kiểu số hoặc loại GPU | Tái lập ở P1 lệch số công bố; chuỗi lặp hoặc ra ký tự rác khi chạy fp16 | Cố định một kiểu số cho mọi lần ghi và chạy thật (mục 3.1); không trộn dữ liệu ghi bằng hai kiểu số |
| vLLM thay đổi hành vi (logprobs, prefix cache) | Kết quả lệch sau khi cài lại môi trường | Cố định phiên bản; mọi lời gọi đi qua lớp bọc `engine/`; kiểm thử hồi quy |
| Việc nối đoạn làm thay đổi chuỗi chính | Kiểm định tương đương ở P1 thất bại | Làm việc trên token id; không thêm lại BOS hoặc chat template |
| Chấm đáp án sai | Độ chính xác lệch số công bố; chấm tay không khớp | Chấm tay 100 mẫu ở P1; bổ sung quy tắc cho các dạng đáp án đặc biệt |
| Dữ liệu ghi quá ngắn so với $b_{\max}$ | Nhiều chuỗi chạm $L_{\text{log}}$ | Theo dõi tỉ lệ chạm trần khi kiểm tra dữ liệu ghi; ghi bổ sung phần đuôi |
| Rò rỉ tập kiểm tra vào quá trình chỉnh tham số | Tham số bị đổi sau khi xem kết quả kiểm tra | Sổ kiểm tra; lệnh đánh giá từ chối cấu hình chưa khóa |
| Qwen3 tự tin quá mức | ECE cao | Hiệu chuẩn isotonic theo từng $h$; nếu thêm model thứ hai thì dùng bộ ước lượng và hiệu chuẩn riêng cho model đó |
| Model nhỏ cho tín hiệu yếu | AUROC của $\hat q(0)$ thấp ở cổng G1 | Xử lý theo nhánh (a) của G1; nếu vẫn không đạt, cân nhắc thuê GPU lớn hơn để chuyển sang model 4B–8B |
| Thời gian chờ mỗi vòng quá lớn | Thời gian chạy tăng mạnh so với B2 | Báo cáo đầy đủ; chạy bất đồng bộ là hướng mở rộng |
| Mã do agent viết lệch đặc tả | Test đạt nhưng lần vết bằng tay không khớp; review phát hiện lỗi blocker | Review độc lập ở mọi điểm review; lần vết bằng tay một lần chạy nhỏ; sinh viên đọc báo cáo review trước khi duyệt điểm dừng |
| Một phase kéo dài hoặc phải lặp lại nhiều lần | Phase chưa đạt tiêu chí hoàn thành sau hai vòng sửa; gần đến hạn nộp mà chưa xong P4 | Cắt phạm vi theo mục 2.2; ưu tiên hoàn thành P0–P4 |

## 10. Sản phẩm bàn giao

| Mã | Sản phẩm | Phase |
|---|---|---|
| D1 | Môi trường, repo nền tảng, danh sách chia tập đã khóa, bộ chấm điểm | P0 |
| D2 | Bảng tái lập baseline, ước tính tài nguyên đã cập nhật, công cụ ghi chuỗi suy luận, bộ kiểm thử, tài liệu mô tả dữ liệu ghi lại | P1 |
| D3 | Báo cáo thăm dò và quyết định G1 | P2 |
| D4 | Kết quả mô phỏng của model chính (E1–E4) và quyết định G2 | P3 |
| D5 | Bộ phân bổ trực tuyến, kết quả E5 và quyết định G3 | P4 |
| D6 | Kết quả mở rộng (E6) | P5 |
| D7 | Bản thảo khóa luận hoàn chỉnh | P6 |
| D8 | Bản nộp cuối, slide bảo vệ, gói mã nguồn và dữ liệu để tái lập | P6 |

## 11. Khóa luận: cấu trúc và phase viết

| Chương | Nội dung | Viết ở | Nguồn |
|---|---|---|---|
| 1. Mở đầu | Bối cảnh, bài toán, đóng góp | P6 | [Báo cáo phương pháp tiếp cận](Bao-cao-Phuong-phap-tiep-can.md) |
| 2. Tổng quan | Mô hình suy luận lớn, dừng sớm, phân bổ tính toán | Song song với P2–P3, trong lúc GPU ghi dữ liệu | [Báo cáo Dự án Công nghệ](Bao-cao-Du-An-Cong-Nghe-v3.md), [ghi chép research](Research-notes-phan-bo-token.md) |
| 3. Phương pháp | Bộ theo dõi, bộ ước lượng, giá token, bộ phân bổ | Sau G2, song song với P4 | [Báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md), cập nhật sau G2 |
| 4. Cài đặt | Công cụ ghi, bộ mô phỏng, bộ chạy thật trên vLLM | Sau G3, song song với P5 | Mã nguồn; D2 và D5 |
| 5. Thực nghiệm | E1–E6 | Phần E1–E4 sau G2; hoàn thành ở P6 | Thư mục `runs/` |
| 6. Kết luận | Kết quả, hạn chế, hướng phát triển | P6 | |

## 12. Quy trình làm việc

- **Khi bắt đầu một phase:** giao phase cho agent bằng câu mẫu ở [mục 9.5 của kế hoạch implement](Ke-hoach-implement.md#L1355). Agent làm lần lượt các task và bước chạy của phase, và dừng ở các điểm dừng.
- **Tại mỗi điểm dừng:** đọc báo cáo của agent và báo cáo review liên quan; ghi quyết định vào `docs/decisions.md`, hoặc vào `docs/test-log.md` nếu liên quan tới tập kiểm tra.
- **Khi kết thúc một phase:** viết một báo cáo ngắn gồm tiêu chí hoàn thành, đầu ra, các quyết định đã đưa ra và việc còn tồn đọng, rồi gửi giảng viên hướng dẫn.
- **Trước mỗi buổi trao đổi với giảng viên hướng dẫn:** cập nhật `docs/nhat-ky.md` gồm phase hiện tại, việc đã xong, kết quả, vướng mắc và việc kế tiếp.
- **Mỗi lần chạy:** một thư mục riêng trong `runs/` theo [mục 2.4 của kế hoạch implement](Ke-hoach-implement.md#L143), chứa cấu hình, mã commit, phiên bản thư viện, thông tin GPU và các độ đo. Không ghi đè kết quả cũ.
- **Bảng và hình:** sinh lại được bằng một lệnh từ `runs/`; không chép số liệu bằng tay.
- **Sao lưu:** dữ liệu ghi lại được sao lưu ra ổ ngoài hoặc lưu trữ đám mây sau mỗi đợt ghi.

## 13. Việc cần làm trước khi bắt đầu P0

- [ ] Thống nhất với giảng viên hướng dẫn: hạn nộp khóa luận, các ngưỡng ở mục 2.1 và mục 8, nguồn GPU, việc đổi model chính sang Qwen3-1,7B.
- [ ] Tạo tài khoản Kaggle và xác minh số điện thoại (để dùng GPU và internet); thử chạy vLLM với Qwen3-1,7B trên T4.
- [ ] Tạo tài khoản Vast.ai, nạp một khoản nhỏ; thử thuê một RTX 3090 trong vài phút để kiểm tra driver, CUDA và dung lượng đĩa (khoảng 50 GB).
- [ ] Chuẩn bị tài khoản Hugging Face cho máy chủ; chấp nhận điều khoản của GPQA.
- [ ] Tạo repo `ttalloc`; chép kế hoạch implement, kế hoạch này và báo cáo phương pháp chi tiết vào `docs/` theo [mục 2.1 của kế hoạch implement](Ke-hoach-implement.md#L86).
- [ ] Clone repo DEER; đọc phần gợi đáp án thử và phần tính độ tự tin, để duyệt quyết định của agent về lời nhắc gợi.
- [ ] Giao P0 cho agent.
