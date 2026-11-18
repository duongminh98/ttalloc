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
- Hệ quả:
  - **Chốt `vllm==0.32.0`**: ngày 04/11/2026, `p0_smoke` trên Kaggle đạt cả 5 bước.
  - Môi trường Kaggle lúc đó: T4 với driver 580.178.04 (CUDA tối đa 13.0), Python 3.13, torch 2.13.0+cu130.
  - Trên T4, `torch.cuda.is_bf16_supported()` trả True dù T4 (compute capability 7.5) không có bf16 gốc. Vì vậy, kiểu số được chọn theo compute capability (≥ 8.0 thì dùng bf16), và smoke test dùng float16.
  - Số đo phụ trên T4 (float16): sinh song song 32 chuỗi đạt 587 token/giây; nhánh gợi `{59}` tốn 4 token.
  - Một giả định chưa được kiểm: đoạn thử là câu hỏi dễ, nên model tự kết thúc (EOS) sau 441 token mà không gặp từ chuyển ý. Giả định "`stop_reason` là chuỗi dừng" vì thế để `tests/gpu` kiểm tiếp.
  - Thời gian gợi lặp lại (0,11 giây so với 0,10 giây) quá ngắn để kết luận về prefix caching.

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

## ADR-013: Tokenizer của R1-Distill hỏng với transformers 5.16–5.19
- Task: P0.T8, P1.T4
- Bối cảnh:
  - vLLM 0.32.0 yêu cầu `transformers>=5.16.1,<5.20`. Trên Kaggle, pip cài bản trong khoảng này; máy cá nhân lúc đó dùng 5.22.1 nên test CPU vẫn đạt.
  - `tokenizer_config.json` của DeepSeek-R1-Distill-Qwen-1.5B khai `LlamaTokenizerFast`, trong khi `tokenizer.json` là ByteLevel. Với 5.16–5.19, `AutoTokenizer` dựng ra một `LlamaTokenizer` hỏng: mất dấu cách và tách chữ thành từng mẩu ("Please" thành 4 token). Bản 5.22 tự nhận ra lỗi này và chuyển sang `TokenizersBackend`.
  - Lần chạy 3 của `p1_gpu_tests` (04/11/2026) bắt được lỗi ở `test_r1_prompt_ends_with_think`.
  - vLLM giải mã đầu ra và so stop string bằng tokenizer riêng, nạp theo cùng cách. Nếu tokenizer hỏng, marker của R1 sẽ không bao giờ khớp mà không có báo lỗi nào. Qwen3-1.7B (model chính) không bị ảnh hưởng.
- Quyết định:
  - `load_tokenizer` mã hóa rồi giải mã lại một câu mẫu. Nếu không khớp thì nạp lại bằng `PreTrainedTokenizerFast`, tức đọc thẳng `tokenizer.json`. Vẫn không khớp thì báo lỗi.
  - `VllmGenerator` làm cùng phép kiểm tra trên cách vLLM sẽ nạp. Nếu hỏng thì lưu tokenizer đã sửa ra một thư mục tạm và truyền cho vLLM qua `tokenizer=`. Sau khi khởi tạo, kiểm tra lại `llm.get_tokenizer()` và dừng nếu vẫn sai.
  - `pyproject.toml` chốt `transformers>=5.16.1,<5.20` cho khớp với vLLM 0.32.0, để test CPU chạy đúng phiên bản như trên GPU. Fixture tokenizer trong test đi qua `load_tokenizer` của mã chính.
- Hệ quả: dãy token của R1 giờ giống hệt bản 5.22 (đã kiểm tra trên 5.16.1 và 5.19.0, kể cả chat template). Khi dùng R1-Distill ở P5, chạy `tests/gpu` với model đó để xác nhận stop string của vLLM khớp.

## ADR-014: Chốt dạng câu trả lời `short`
- Task: P1.T9
- Bối cảnh:
  - Bản tập dượt trên Kaggle T4 (04–05/11/2026; Qwen3-1.7B float16; `val` 400 câu × 2 seed và `train` 100 câu × 2 seed; chuỗi cắt ở 1.024 token) ghi câu trả lời ép buộc dạng `full` với `r_ans` = 256.
  - Câu trả lời `full` dài trung bình 248–253 token. Ở 85–92% số điểm, câu trả lời chạm `r_ans` trước khi viết `\boxed{}`, nên không trích được đáp án và bị chấm sai.
  - Ở điểm chuỗi tự kết thúc, trường hợp không phụ thuộc việc cắt chuỗi ở 1.024 token, tỉ lệ này vẫn là 67–77%: sau `</think>`, Qwen3 viết lại lời giải rồi mới đưa đáp án.
  - `study answer-mode` trên 8 câu cho: `full` không trích được 92%, đúng 8%; `short` không trích được 0%, đúng 42%; tỉ lệ cùng kết quả chấm 67%.
  - Luật ở P1.T9 (chọn `short` khi tỉ lệ cùng kết quả chấm ≥ 98%) coi `full` là chuẩn. Khi `full` hỏng, luật chọn đúng dạng hỏng.
  - Câu trả lời `full` chiếm khoảng 58% số token khi ghi log: mỗi điểm tốn 256 token, so với khoảng 14 token của `short`.
- Quyết định:
  - `answer.mode: short` (`r_ans` = 32) trong `configs/experiments/qwen3.yaml`, cho mọi lần ghi và chạy thật của Qwen3-1.7B. Người dùng duyệt ngày 05/11/2026.
  - `p1_pilot` ghi luôn bằng `short`, nên dữ liệu đó dùng lại được ở P2. `study answer-mode` vẫn chạy trên chuỗi dài thật, và khóa luận báo cáo cả hai dạng.
  - Số liệu dùng để quyết định chỉ lấy từ `val` và `train`, chưa chạm tập kiểm tra.
- Hệ quả:
  - Thời gian ghi log giảm khoảng 50%. `c_ans_max` nhỏ hơn, nên phần dự trữ của quỹ và nhu cầu KV của mỗi chuỗi cũng nhỏ hơn.
  - Nhãn `y` có nghĩa là "nếu bắt trả lời ngay thì có đúng không". Model không được viết thêm bước nào để chốt đáp án; phần phương pháp phải nói rõ điều này.
  - Với model thứ hai ở P5, vẫn áp luật ở P5 bước 3, nhưng so với dạng `short` của model chính.

## ADR-015: Cỡ `train` theo learning curve
- Task: P2 bước chạy 2, P3 bước chạy 1
- Bối cảnh: `train` chỉ dùng để học bộ ước lượng, nhưng 1.500 câu × 4 seed chiếm khoảng 43% số chuỗi phải ghi. Lợi ích của dữ liệu thêm thường giảm dần.
- Quyết định:
  - Cuối P2 đã có 500 câu × 4 seed. Lệnh `ttalloc estimator learning-curve --sizes 250,500` học bộ ước lượng trên 250 câu đầu và trên 500 câu đầu, theo đúng thứ tự `hash_order` mà `log --limit` dùng (nên hai tập lồng nhau), rồi đánh giá trên `val`.
  - Ghi thêm, gấp đôi số câu (tối đa 1.500), khi việc gấp đôi dữ liệu còn tăng AUROC của $\hat q(0)$ ít nhất 0,01 hoặc giảm log-loss trung bình theo các $h$ ít nhất 1% (tương đối). Nếu không thì dừng ở cỡ hiện tại.
  - Sau khi ghi tới 1.000 câu, kiểm tra lại cặp 500 → 1.000 trước khi đi tới 1.500.
  - Ngưỡng được chốt trước khi có dữ liệu thật, và chỉ dùng `val`.
  - `p3_log` nhận cỡ qua biến `TTALLOC_TRAIN_LIMIT` (`vast.py create --env`), không có giá trị mặc định.
- Hệ quả:
  - Bỏ được tới 4.000 chuỗi. Khóa luận báo cáo kết quả learning curve, kể cả khi quyết định dừng ở 500 câu.
  - AUROC trên `val` có sai số khoảng ±0,01, nên luật có thể ghi thêm vì nhiễu. Cách này thiên về an toàn: ghi thừa chỉ tốn thêm GPU, không làm sai kết quả.

## ADR-016: Chọn loại GPU theo chi phí mỗi token
- Task: P1 bước chạy 1–2
- Bối cảnh:
  - Mục tiêu là giảm tiền thuê GPU. Tốc độ ghi log bị giới hạn bởi băng thông bộ nhớ và dung lượng KV cache: số chuỗi chạy được cùng lúc tỉ lệ với KV cache.
  - Giá trên Vast ngày 05/11/2026 (USD/giờ, máy bid / máy on-demand):
    - RTX 3090: 0,11–0,18 / 0,17–0,27 (24 GB, 805 GB/s);
    - RTX 5090: 0,29–0,38 / 0,53–0,60 (32 GB, 1.456 GB/s);
    - A100 SXM4 80GB: 0,27–0,41, chỉ có máy bid (1.665 GB/s).
- Quyết định:
  - Chạy `gpu_check` (khoảng 15 phút) trên một RTX 3090 và một A100 80GB. Chọn loại có chi phí mỗi token thấp nhất, tức giá mỗi giờ chia cho số token/giây ở lô lớn của `baseline throughput`.
  - Dùng đúng loại đó cho mọi lần ghi log và chạy thật, để số đo thời gian của E5 và kết quả tính toán nhất quán.
  - Các công việc ghi log dùng máy bid (rẻ hơn khoảng 35%), vì logger chạy tiếp được sau khi bị ngắt. E5 (đo thời gian) dùng máy on-demand.
  - Lệnh tìm máy: `vast.py search --gpu <tên> --min-vram-gb <n>`.
- Hệ quả: tiêu chí là tổng tiền, không phải số giờ. GPU nhanh hơn chỉ được chọn khi rẻ hơn tính theo token. Báo cáo E5 ghi rõ tên GPU.

## ADR-017: Chuyển skill Claude Code sang Codex theo phạm vi repo
- Công việc: chuyển ba skill `vast-run`, `phase-runner`, `kaggle-gpu-test` sang `.agents/skills/`, giữ bản Claude làm nguồn tham chiếu.
- Bối cảnh: `vast-run` bản Claude ghi phê duyệt từ ngày 04/11/2026 cho mọi điểm dừng và tập kiểm tra, trái quy tắc dừng trong PLAN mục 0.4 và `AGENTS.md` hiện hành. `phase-runner` bản Claude còn yêu cầu không commit nếu chưa được hỏi, trong khi PLAN mục 0.2 quy định commit cho task implement. Công cụ `run_in_background` của Claude không có trong Codex.
- Quyết định: bản Codex theo PLAN và các quyền thực sự đã có trong cuộc trò chuyện, không tự kế thừa lời khẳng định phê duyệt lịch sử của skill. Giữ đúng phạm vi phê duyệt còn hiệu lực để tránh hỏi lại không cần thiết. Thực hiện quy trình commit theo loại task của PLAN. Thay theo dõi nền bằng phiên `exec_command`/`write_stdin`, hoặc tiếp tục cell `functions.exec` bằng `functions.wait` khi có cell ID.
- Hệ quả: các skill Codex chỉ phục vụ repo ttalloc; giữ lệnh vận hành, giới hạn sửa lỗi, bảo vệ thông tin xác thực và vòng đời thu hồi GPU. Việc chuyển skill không thuê máy, đẩy kernel, đẩy dữ liệu hoặc cấp quyền chạy thí nghiệm.

## ADR-018: Miễn review trước lần kiểm tra thật trên Vast theo yêu cầu người dùng
- Ngày: 17/11/2026.
- Phạm vi: P0a, P1a và P1b, các điểm review đang được trao đổi trước khi chạy trên Vast.
- Bối cảnh: PLAN mục 0.3 và mục 6 yêu cầu review độc lập; P0a chưa được thực hiện theo tiến độ hiện tại. Người dùng chỉ đạo trực tiếp: "bỏ qua công đoạn review đi, đánh dấu các phần đã xong". Chỉ đạo mới này thay yêu cầu review trong phạm vi nêu trên.
- Quyết định: ghi trạng thái review là "được miễn theo yêu cầu người dùng", không ghi review đạt hoặc tạo báo cáo review giả. Chấp nhận P0 hoàn tất theo ngoại lệ do các bước chạy P0 đã ghi nhận xong và người dùng yêu cầu đánh dấu phần đã hoàn thành. Giữ trạng thái "mã xong" cho phần triển khai P1–P4; các lần chạy thật chưa thực hiện vẫn là chưa chạy. Không tự miễn các điểm review P2 trở đi.
- Hệ quả: có thể chuẩn bị bước `gpu_check` trên Vast theo ADR-016 mà không chờ P0a/P1a/P1b. P1 chưa kết thúc vì còn baseline, equivalence, pilot và audit. Các kiểm tra tự động, phê duyệt thuê máy, GPU lớn, tập `test_*`, khóa cấu hình và kết thúc các phase tiếp theo vẫn áp dụng. Quyết định này không cấp quyền thuê máy hay chạy thí nghiệm.

## ADR-019: Phạm vi và ngân sách thực hiện tối thiểu
- Ngày: 17/11/2026.
- Yêu cầu: người dùng thống nhất phương án tối thiểu khoảng 10 USD, chuẩn bị 15 USD và yêu cầu cập nhật kế hoạch thực hiện thực tế. Đây là duyệt kế hoạch, không phải duyệt thuê một máy cụ thể hoặc chạy tập kiểm tra.
- Bối cảnh: PLAN và ADR-015 đã cho chọn cỡ train theo learning curve, nhưng kế hoạch thực hiện và RUNBOOK còn mặc định tăng tới 1.500 câu, dùng hệ số token phụ của dạng trả lời `full` và khẳng định 15 USD đủ cả đề tài. ADR-016 còn yêu cầu thuê thử cả RTX 3090 và A100. Các nội dung này chưa phản ánh phạm vi tối thiểu và `short` đã chọn ở ADR-014.
- Quyết định:
  - Một Qwen3-1.7B, bf16, dạng trả lời short; train sử dụng 500 câu × 4 seed, val 400 × 8, test_main 560 × 8, tổng 9.680 chuỗi ghi. Không thay pool 1.500 ID train đã khóa; lấy tập con qua `hash_order`/`--limit` có sẵn. 200 chuỗi pilot cùng log_hash được tái sử dụng.
  - Giữ learning curve 250 → 500. Không tự tăng lên 1.000/1.500 câu; nếu tiêu chí ADR-015 yêu cầu tăng, dừng trước khi ghi thêm, cập nhật chi phí và xin duyệt. Không bỏ qua kết quả learning curve hoặc giảm seed/giới hạn chuỗi để ép vừa tiền.
  - Gói tối thiểu kiểm tra một RTX 3090; thay yêu cầu thuê thêm A100 ở ADR-016 trong phạm vi này. Nguyên tắc so chi phí mỗi token vẫn giữ; đổi máy/loại GPU phải có lý do và quyền phù hợp.
  - Giữ baseline, audit bộ chấm, kiểm định tương đương, hai study, E1–E4 trên CPU và E5 tối thiểu 6 lần chạy thật cùng đối chiếu baseline. Không suy diễn việc bỏ hàng thời gian GPU khỏi báo cáo thành hủy P4.
  - Giá lập ngân sách gồm GPU và disk giả định 0,15 USD/giờ khi ghi, 0,24 USD/giờ cho việc theo yêu cầu; 6.300 token chuỗi chính, tỉ lệ token sinh 1,10, tốc độ hiệu dụng 800 token/giây, dự phòng ghi lại 10%. Kịch bản khoảng 40,62 giờ, 8,19 USD trước phí thanh toán; mục tiêu khoảng 10 USD và khoản chuẩn bị 15 USD không phải bảo đảm trọn gói. Dùng 26.000 đồng/USD để quy đổi kế hoạch (260.000 và khoảng 390.000–400.000 đồng), không coi là tỷ giá ngân hàng hiện hành.
  - Đo lại pilot chuỗi dài và lấy giá offer thật trước ghi lớn. Khi tổng đã tiêu cộng phần còn lại dự kiến vượt khoản được duyệt/số dư, dừng chi tiêu để điều chỉnh. Destroy máy khi chờ xử lý CPU, chấm tay hoặc phê duyệt.
  - P5, train trên 500, model thứ hai, so nhiều GPU và E5 mở rộng chưa nằm trong khoản này; cần giao việc và dự toán riêng.
- Hệ quả: đồng bộ PLAN, kế hoạch thực hiện, RUNBOOK và skill Vast; giữ nguyên config, split, cổng và sổ tập kiểm tra. Các phê duyệt thuê máy/tác vụ lớn/khóa config/test vẫn thực hiện đúng phạm vi. Không có job GPU nào được chạy bởi thao tác cập nhật tài liệu này.

## ADR-020: Cập nhật dự toán bằng đơn giá và throughput đã đo trên RTX 3090
- Ngày: 18/11/2026.
- Yêu cầu: dùng thông số `gpu_check` vừa chạy để ước tính lại thời gian/chi phí theo kế hoạch.
- Bối cảnh: ADR-019 giả định máy ghi 0,15 USD/giờ và logger 800 token/giây. Offer đã thuê thực tế là 0,234444 USD/giờ gồm 60 GB disk, mạng 0,0065104 USD/GB mỗi chiều. Benchmark trên val giới hạn 4.096 token/câu đo 680,60/1.577,75/2.168,91/2.296,39 token/giây ở lô 8/32/128/256; đây không phải phép đo logger có probe/answer trên chuỗi tối đa 32.768 token.
- Quyết định: lập kịch bản dùng giá đã thuê cho mọi đợt tương lai, không giả định mua được giá interruptible cũ. Giữ độ dài 6.300, hệ số token sinh 1,10, logger 800 token/giây và dự phòng chạy lại 10% làm giả định cơ sở chưa được pilot xác nhận; thêm các kịch bản 500/1.200 token/giây và chuỗi dài 10.000 token. Không chia thời gian mọi job cho throughput benchmark cao nhất. Giữ các quỹ baseline/equivalence, study và E5 cũ vì chưa có số đo thay thế; thời gian CPU, chấm tay, review và viết luận chưa được tính như số đã đo.
- Kết quả: tổng cơ sở 40,62 giờ GPU / 10,28 USD, trong đó còn 40,22 giờ / 10,18 USD; đã chi khoảng 0,103731 USD theo chênh lệch credit sau destroy (có thể cập nhật trễ). Dự phòng mạng tương lai 0,75 USD; pilot 200 chuỗi nằm trong tổng 9.680 chuỗi log, không đếm lại. Giá mới làm tổng tăng so với kịch bản lịch sử 8,19 USD. Phép tính chi tiết ở kế hoạch thực hiện mục 3.3.1.
- Hệ quả: không đổi scope/config/split/định nghĩa token hoặc bất biến, không nạp tiền/thuê máy/chạy thêm. Mốc khoảng 10 USD và chuẩn bị 15 USD vẫn là mục tiêu có điều kiện; cần pilot dài và offer mới trước khi duyệt ghi lớn. Không coi các kịch bản là khoảng tin cậy hoặc cam kết đủ tiền.

## ADR-021: P1 có checkpoint, tự thu hồi và audit AI theo ủy quyền trực tiếp
- Ngày: 18/11/2026.
- Người dùng duyệt toàn P1, quyền dùng test_main, tổng P1 tối đa 3,25 USD/12 giờ, yêu cầu tự lưu/xóa máy khi xong hoặc có rủi ro vượt hạn mức; sau đó yêu cầu AI tự audit 100 mẫu và báo cáo. Đây là quyền thực hiện, không phải khẳng định P1 đã đạt.
- Checkpoint Full/DEER theo nhóm 32 câu; equivalence lưu riêng phía full/segmented; kiểm checksum/config/khóa câu và không sinh lại khóa đã hoàn tất. Chạy tiếp tạo run mới. DEER vẫn dùng repo/commit/script/profile chính thức, tách nhóm đầu vào để có đầu ra sớm; ghi chi phí khởi tạo mỗi nhóm và config_hash, không chọn nhóm theo kết quả. Không chọn tham số phương pháp theo test.
- `p1_safe.yaml` chỉ đổi thông số vận hành: baseline.chunk=32, online.max_num_seqs=32, log.window=32, log.flush_every=8. Kiểm thử xác nhận log_hash bằng qwen3.yaml; không đổi dtype, lời nhắc, sampling, split, seed hoặc token limits. Không thay định nghĩa token/bất biến.
- Supervisor phía GPU giới hạn epoch tuyệt đối, credit giảm và thời gian thuê ước tính, chừa biên 0,25 USD và thời gian sync cuối; kiểm tra tiến triển thật từ generation/checkpoint. Đồng bộ bị giới hạn thời gian, lỗi đồng bộ không cản destroy; local monitor là lớp dự phòng. API/thu phí có thể trễ, không coi đây là cam kết trần chính xác tới từng cent khi dịch vụ lỗi.
- Dừng khi baseline MATH-500 lệch quá 2 điểm phần trăm so với 85,6%, equivalence chưa đạt, lỗi lệnh hoặc mất đồng bộ; không tự tăng seed, đổi ngưỡng, chạy lại test để tìm kết quả tốt hơn. Khi không có vấn đề, tiếp tục pilot/validate/hai study trong hạn mức còn lại. Không tự chạy P2.
- Ngoại lệ theo chỉ đạo mới: kiểm tra 100 mẫu do AI đối chiếu độc lập với đáp án chuẩn thay phần người dùng chấm tay. Ghi rõ là AI audit và lưu từng nhận định, không gọi là kiểm chứng độc lập bởi con người. Chỉ báo hoàn thành P1 khi mọi tiêu chí thực đo còn lại đạt; nếu chưa đạt thì báo kết quả/giới hạn thực tế.


## ADR-022: Kh?c b? ch?m khi ??i chi?u baseline DEER ? ngo?i l? Full ?? ???c duy?t
- Task: P1.T5; y?u c?u ng??i d?ng ng?y 18/11/2026 ??i chi?u c?u h?nh/c?ch ch?m v? xem kh? n?ng ti?p t?c P1.
- B?ng ch?ng: `docs/reports/P1-DEER-comparison-2026-11-18.md`; `runs/p1-deer-comparison-20261118/`. ??/gold 500/500 kh?p DEER; prompt d?ng l?i 500/500 kh?p; greedy/16.384 kh?p m? t? b?i. Ch?m l?i nguy?n b?n DEER tr?n v?n b?n ?? l?u cho 435/500 (87,0%), so v?i `grade` 441/500 (88,2%); 8 nh?n ??i, r?ng 6 c?u (1,2 ?i?m). Hai l??t c?/kh?ng timeout tr?ng 500/500 nh?n.
- M?u thu?n: PLAN P1.T5 y?u c?u `grade` nh?ng so ?i?m v?i c?ng b? d?ng b? ch?m kh?c. Kh?ng t? chuy?n c?ng sang b? ch?m DEER sau khi th?y k?t qu?. S?u c?u thi?u ph?n suy lu?n tr??c think-end n?n ?i?m DEER tr?n raw response ch?a bi?t (gi?i h?n do thi?u d? li?u 87,0?88,2%); kh?ng tuy?n b? c?ng ??t b?ng ?i?m 87,0%.
- Quy?t ??nh hi?n h?nh: gi? nguy?n PLAN, config/seed, c?ng d?ng v? k?t qu? g?c; kh?ng thu? m?y, kh?ng sinh l?i test, kh?ng ch?y P2. M? ch?m upstream kh?ng s?a; c?ng khai l?p t??ng th?ch hai phi?n b?n ANTLR do requirements g?c xung ??t.
- ?? xu?t ?ang ch? ng??i d?ng: ch?p nh?n ngo?i l? t?i l?p Full c? gi?i tr?nh ?? ti?p t?c P1; b?o song song ?i?m nghi?n c?u `grade` v? ?i?m ??i chi?u t?c gi?, kh?ng l?y b? ch?m t?c gi? thay ch?n l? ng? ngh?a. L?u raw output cho l??t sau. ??y ch?a ph?i ph? duy?t ngo?i l?, ho?n t?t phase ho?c gia h?n ng?n s?ch/deadline thu?.

- C?p nh?t ph? duy?t 18/11/2026: ng??i d?ng x?c nh?n ?Oke, duy?t ngo?i l? n?y nh?. Gi? c? hai s? trong b?o c?o, r?i t?m offer GPU v? b?o chi ph? ch?y ti?p ph?n P1 c?n thi?u cho m?nh duy?t.? Ngo?i l? ?p d?ng cho Full MATH-500 ?? ch?y (88,2% theo grade; 87,0% theo DEER tr?n v?n b?n ?? l?u); kh?ng s?a ?i?m th?nh ??t theo c?ng s? h?c, kh?ng sinh l?i Full. ???c chu?n b? ph?n ti?p theo nh?ng ch?a thu? m?y. C?c c?ng DEER/equivalence v? ?i?u ki?n k?t th?c P1 gi? nguy?n. C?a s? 12 gi? c? ?? h?t 13:00:31 ng?y 18/11; c?a s? m?i ph?i n?m trong ?? ngh? thu? m?i, kh?ng t? gia h?n. Gi? chi ph? P1 ?? ghi nh?n 0,2797374973 USD trong tr?n c?ng d?n 3,25 USD.
