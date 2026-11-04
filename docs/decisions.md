# Quyết định (ADR)

Mỗi chỗ lệch đặc tả, mâu thuẫn hoặc chỗ chưa rõ được ghi ở đây (PLAN mục 0.1, mẫu ở mục 9.4).

## ADR-001: Viết mã P0 trên máy không có GPU
- Task: P0 (điều kiện bắt đầu)
- Bối cảnh: PLAN P0 ghi điều kiện bắt đầu là "máy có GPU đã cài driver và CUDA". Máy phát triển là Windows ở công ty, không có GPU và không SSH được ra máy thuê.
- Quyết định: viết và kiểm thử toàn bộ phần CPU của P0 trên máy Windows (Python 3.12 qua `uv`). Bước chạy 1 của P0 (cài vLLM, kiểm tra `import vllm`) chạy sau trên Kaggle bằng `ops/kaggle/p0_smoke.py`.
- Hệ quả: P0 chưa kết thúc cho tới khi bước chạy 1–3 xong và P0a đạt.

## ADR-002: Phiên bản vLLM
- Task: P0, bước chạy 1
- Bối cảnh: cần cố định một phiên bản vLLM (PLAN mục 2.2). Ngày 03/11/2026, bản mới nhất là v0.33.0 (ra 02/11/2026); v0.32.0 ra 18/10/2026. Image `vllm/vllm-openai:v0.32.0` dựng trên CUDA 13.0.2, có kiến trúc 7.5 (T4) và 8.6 (RTX 3090).
- Quyết định (tạm thời): `vllm==0.32.0`, vì đã ra được hai tuần. Chốt sau khi `ops/kaggle/p0_smoke.py` chạy được trên T4 của Kaggle. Nếu driver của Kaggle không chạy được bản CUDA 13, chọn một phiên bản có bản dựng CUDA 12.x và dùng cùng phiên bản đó trên Vast (khi đó lọc máy Vast với `cuda_max_good >= 12.8`).
- Hệ quả: chưa chốt; cập nhật ADR này sau khi chạy smoke test.

## ADR-003: Phân tầng theo độ khó khi lấy mẫu MATH
- Task: P0.T7
- Bối cảnh: PLAN mục 4.1 ghi "`train` và `val` được lấy mẫu phân tầng theo độ khó" nhưng không nói chia đều hay theo tỉ lệ.
- Quyết định: chia đều theo mức 1–5: `train` 180 câu MATH mỗi mức (900), `val` 72 câu mỗi mức (360). Câu không có mức (`Level ?`) bị loại khỏi phần lấy mẫu phân tầng. DeepScaleR không có mức nên lấy mẫu ngẫu nhiên.
- Hệ quả: phân phối độ khó của phần MATH đều nhau, đủ câu cho các thành phần lô `balanced`, `easy`, `hard`, `u` (P2.T14). Khử trùng lặp chỉ loại câu khỏi `train` và `val`; các tập kiểm tra lấy toàn bộ.

## ADR-004: Kích thước `test_clean`
- Task: P0.T6, P0.T7
- Bối cảnh: kế hoạch ghi `test_clean` có 60 câu. Trên Hugging Face, `MathArena/aime_2026` có 30 câu và `MathArena/hmmt_feb_2026` có 33 câu.
- Quyết định: lấy toàn bộ, `test_clean` có 63 câu.
- Hệ quả: chi phí ghi `test_clean` tăng khoảng 5% so với ước tính.

## ADR-005: Vị trí lời nhắc hướng dẫn
- Task: P0.T8
- Bối cảnh: PLAN mục 9.2 cho nội dung lời nhắc nhưng không nói đặt ở đâu. Script DEER chính thức (`vllm-deer-qwen3.py`, commit `c9dd19f`) đặt hướng dẫn ở system message và đề bài ở user message, với cả Qwen3 và R1-Distill.
- Quyết định: theo DEER (`configs/prompts/deer.yaml`, `instruction_role: system`), để tái lập ở P1 dùng đúng prompt của DEER.
- Hệ quả: khác khuyến nghị của Qwen3 (đặt mọi hướng dẫn ở user message). Đổi được bằng `instruction_role: user`; đổi thì phải ghi lại dữ liệu vì `log_hash` thay đổi.

## ADR-006: Lời nhắc gợi đáp án thử cho câu trắc nghiệm
- Task: P0.T8 (chốt ở P1.T3)
- Bối cảnh: với GPQA, DEER dùng lời nhắc gợi "I believe the final answer, rather than the option, is \boxed", tức đòi nội dung đáp án chứ không phải chữ cái, trong khi hướng dẫn của ta yêu cầu ghi chữ cái.
- Quyết định (tạm thời): dùng cùng lời nhắc gợi với toán (`\n**Final Answer**\n\boxed`). Chốt lại khi cài A1b ở P1.T3.
- Hệ quả: ảnh hưởng `test_ood` (P5) nếu đổi; không ảnh hưởng các tập toán.

## ADR-007: LightGBM trên Windows
- Task: chuẩn bị cho P2.T4
- Bối cảnh: `lib_lightgbm.dll` cần `vcomp140.dll` (OpenMP của Visual C++), máy phát triển không có và không có quyền admin để cài.
- Quyết định: trước khi import `lightgbm` trên Windows, thêm thư mục `sklearn/.libs` (có sẵn `vcomp140.dll`) vào đường dẫn DLL bằng `os.add_dll_directory`. Cài trong `estimator/` ở P2.
- Hệ quả: chỉ ảnh hưởng máy Windows; Linux không đổi.

## ADR-008: Liên kết giữa các tài liệu trong `docs/`
- Task: P0.T1
- Bối cảnh: PLAN mục 2.1 chép kế hoạch implement thành `docs/PLAN.md`, nhưng kế hoạch thực hiện đề tài liên kết tới tệp tên `Ke-hoach-implement.md`.
- Quyết định: giữ `docs/PLAN.md` theo đặc tả. Các liên kết `Ke-hoach-implement.md#L...` trong `docs/Ke-hoach-thuc-hien-de-tai.md` trỏ tới nội dung của `docs/PLAN.md`.
- Hệ quả: bấm vào các liên kết đó trong repo sẽ không mở được tệp; đọc `docs/PLAN.md` ở cùng số dòng.

## ADR-009: Tiêu chí khử trùng lặp
- Task: P0.T7
- Bối cảnh: PLAN mục 4.1 yêu cầu khử trùng lặp bằng "chuẩn hóa văn bản và trùng n-gram" nhưng không nói đơn vị n-gram và ngưỡng. Chạy thử trên dữ liệu thật với 13-gram ký tự và quy tắc "chung một 13-gram là trùng" loại 10/40 câu của `amc23` và 1.022 câu của MATH. Xem các cặp bị loại: phần lớn chỉ chung câu mẫu ("can be written as $\frac{m}{n}$ where $m$ and $n$ are relatively prime…") hoặc mã hình `[asy]`, không phải cùng một bài.
- Quyết định: chuẩn hóa văn bản (chữ thường, bỏ khối `[asy]…[/asy]` và ký hiệu LaTeX chỉ để trình bày, gộp khoảng trắng), tách thành từ (`[a-z0-9]+`), lấy 13-gram từ. Một câu bị coi là trùng với câu kiểm tra nếu văn bản chuẩn hóa giống hệt, hoặc ít nhất 50% số 13-gram của câu ngắn hơn trong cặp xuất hiện ở câu kia (`configs/splits.yaml`: `dedup.ngram: 13`, `dedup.min_overlap: 0.5`). Mọi cặp bị loại được ghi vào `dedup_pairs` của tệp split để con người xem lại ở bước chạy 3.
- Hệ quả: chạy thử loại 30 câu khỏi `val` và 40 câu MATH cùng 1.392 câu DeepScaleR khỏi `train`. Các câu DeepScaleR bị loại là bài MATH chép lại, tức trùng thật; `train` vẫn đủ 1.500 câu vì lấy mẫu sau khi khử trùng lặp.

## ADR-010: Cắt đoạn ở từ chuyển ý, đếm token của DEER chính thức, phạm vi độ tự tin
- Task: P1.T2, P1.T3, P1.T5
- Bối cảnh: (1) vLLM dừng ngay sau token khớp chuỗi dừng. Với `"\n\nBut"`, ta không biết token kế tiếp là `"ter"` ("Butter") hay là ranh giới từ. (2) Repo DEER chính thức không trả số token theo đúng tokenizer và cách đếm của ta. (3) A1b chỉ nói độ tự tin tính trên "phần trong khung".
- Quyết định:
  - (1) Khi từ chuyển ý nằm sát cuối văn bản, `classify(final=False)` trả `peek`. LiveSource sinh thêm đúng một token (cùng seed theo vị trí, chỉ dừng ở `</think>`), rồi phân loại lại với `final=True`. Token thêm thuộc chuỗi chính và được tính vào `think`. Token thêm là EOS thì sự kiện là `eos`; là `</think>` thì sự kiện là `natural_end`.
  - (2) `parse_deer_output` tokenize lại văn bản đầu ra bằng tokenizer của model để đếm token suy luận và token trả lời.
  - (3) Phạm vi tính độ tự tin bắt đầu ở token ngay sau `{`, hoặc ở chính token `{` nếu tokenizer gộp `{` với nội dung. Phạm vi kết thúc ở token chứa `}` đóng khung.
  - Khi chạy repo DEER gốc (P1.T5, P4.T5), ta dùng chế độ `baseline.deer_policy = avg2` của repo. B2 trong mô phỏng và chạy thật dùng phạm vi ở điểm (3).
- Hệ quả:
  - Trong test (`test_p1_engine.py`), không còn đoạn nào bị cắt sai ở từ ghép.
  - Mỗi lần peek tốn thêm một lượt gọi engine sinh 1 token.
  - Số token của DEER gốc có thể lệch vài token so với số repo tự báo. Báo cáo dùng số đã tokenize lại.

## ADR-011: Hiệu chỉnh hạn mức của B2, schema kết quả mở rộng, mu_min khởi đầu
- Task: P2.T13, P2.T15, P2.T19
- Bối cảnh:
  - (1) A11 tìm hạn mức chung c bằng chia đôi trên [B/N, c_hi], tức là ngầm giả định tỉ lệ chi tiêu tăng theo c. Thực tế không như vậy: khi c lớn, quỹ chung cạn sớm và nhiều câu bị dừng cùng lúc, nên chi tiêu tại c_hi có thể thấp hơn tại một c vừa phải. Trên dữ liệu tổng hợp, thuật toán cũ báo `budget_underused` ngay, dù có giá trị c đạt mục tiêu.
  - (2) Với mỗi giá trị c thử, phép hiệu chỉnh chạy lại toàn bộ các lần rút.
  - (3) `per_question.parquet` cần thêm cột để báo cáo theo lô và theo độ khó.
  - (4) Cổng G1 chạy trước khi chỉnh tham số, nhưng M2 và M3 cần `mu_min`.
- Quyết định:
  - (1) Nếu tại c_hi chi tiêu chưa đạt 98% B, thử thêm một lưới 6 điểm đều trên [B/N, c_hi], rồi chia đôi trên đoạn đầu tiên vượt mục tiêu. Không điểm nào đạt thì dùng c có chi tiêu cao nhất (hòa thì lấy c nhỏ hơn) và ghi `budget_underused`. Phép chia đôi dừng ngay khi gặp c có chi tiêu trong [0,98B; B].
  - (2) `sim.calib_draws = 50`: hiệu chỉnh trên 50 lần rút đầu, rồi áp dụng cho cả 200 lần rút. Chạy thật dùng `online.calib_draws = 200` lần rút mô phỏng trên đúng lô của chạy thật.
  - (3) `PER_QUESTION_SCHEMA` thêm các cột có thể rỗng: `split`, `batch`, `budget`, `dataset`, `difficulty`.
  - (4) Ở G1, M2 và M3 dùng `mu_min = 1e-5` (1 điểm phần trăm trên 1.000 token), B2 dùng `lambda = 0,95`. Giá trị chính thức chọn bằng `tune` ở P3.
- Hệ quả: hiệu chỉnh tốn tối đa khoảng 8 + 23 lượt chạy, mỗi lượt trên 50 lần rút. Trên dữ liệu tổng hợp (40 câu, alpha 0,3), tỉ lệ chi tiêu đạt 0,99 thay vì dừng ở 0,82.

## ADR-012: Bộ bảo vệ khi phần chia đều nhỏ hơn min_segment
- Task: P2.T7
- Bối cảnh:
  - Theo A8, các câu chưa có rho (mọi câu của B2, và câu mới nạp của M2) được chia đều phần quỹ dùng được khi quỹ không đủ cho tất cả. Nếu phần chia đều nhỏ hơn `guard.min_segment`, mọi câu đó dừng cùng lúc, kể cả khi quỹ còn đủ cho vài câu.
  - Mô phỏng trên dữ liệu tổng hợp cho thấy B2 để thừa tới khoảng 20% B. Nhóm câu đã có rho của M2 được cấp tham lam nên không gặp vấn đề này.
  - Kết quả là B2 bị thiệt một cách không công bằng (mục 7.2).
- Quyết định:
  - Khi phần chia đều trừ chi phí gợi nhỏ hơn `min_segment`, chỉ cấp cho k = floor(available / (min_segment + c_probe_max)) câu đầu theo `tie_key`. Mỗi câu được cấp nhận phần chia đều giữa k câu; các câu còn lại dừng.
  - Câu không còn đủ chỗ tới `b_max` (want < min_segment) dừng trước khi chia.
  - Khi phần chia đều đủ lớn, hành vi giữ nguyên như A8.
- Hệ quả:
  - Bất biến I1 và việc câu mới nạp không bị bỏ đói vẫn giữ nguyên, có test hypothesis ở `test_p2_controller_policies.py`.
  - Chi tiêu của B2 sát B hơn, nên so sánh M2 với B2 không còn thiên lệch do quỹ bị bỏ phí.
  - Mã lệch so với mã mẫu ở A8, nên reviewer cần xem kỹ phần này.
