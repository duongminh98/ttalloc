# Kế hoạch implement

**Đề tài:** Phân bổ thích ứng tài nguyên suy luận cho mô hình suy luận lớn  
**Dành cho:** AI agent implement, test và review mã nguồn của đề tài.  
**Phiên bản:** 2.0

Tài liệu đặc tả việc xây dựng mã nguồn: kiến trúc, dữ liệu, thuật toán, các phase theo thứ tự phụ thuộc, test bắt buộc và quy trình review. Tài liệu chia việc theo phase, không theo thời gian; các phase P0–P6 trùng với các phase của [kế hoạch thực hiện đề tài](Ke-hoach-thuc-hien-de-tai.md).

---

## 0. Cách dùng tài liệu

### 0.1. Nguồn chuẩn

1. Đọc toàn bộ tài liệu này trước khi viết mã.
2. Ngữ nghĩa của phương pháp nằm ở [Báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md). Mỗi task dẫn tới đúng mục cần đọc.
3. Khi có mâu thuẫn: tài liệu này được ưu tiên, sau đó tới báo cáo phương pháp chi tiết. Mọi mâu thuẫn hoặc chỗ chưa rõ được ghi vào `docs/decisions.md` (mẫu ở mục 9.4) kèm cách xử lý đã chọn. Không lệch khỏi đặc tả mà không ghi lại.

### 0.2. Quy trình cho mỗi task

1. Đọc đặc tả task ở mục 6 và các mục được dẫn.
2. Viết mã và mọi test được liệt kê trong task. Được thêm test, không được bỏ test.
3. Chạy `ruff check`, `mypy src`, `pytest -m "not gpu and not slow and not net"`. Task có test GPU thì chạy thêm `pytest -m gpu` trên máy có GPU.
4. Tự review theo checklist ở mục 8.2.
5. Commit với thông điệp `[P<k>.T<n>] <mô tả ngắn>`, rồi cập nhật `docs/progress.md`.

### 0.3. Quy trình review trong mỗi phase

- Mỗi phase có một hoặc vài điểm review, ghi ở cuối đặc tả của phase (mục 6). Tên điểm review gồm tên phase và một chữ cái, ví dụ P2b.
- Tại mỗi điểm review, một agent review độc lập (ngữ cảnh mới, không phải agent đã viết mã) review toàn bộ thay đổi kể từ điểm review trước, theo mục 8, và ghi báo cáo vào `docs/review/<điểm review>.md`, ví dụ `docs/review/P2b.md`.
- Lỗi mức blocker và major phải được sửa trước khi làm task hoặc bước chạy kế tiếp. Sau khi sửa, reviewer kiểm tra lại các lỗi đó.
- Một phase kết thúc khi mọi điểm review của phase đều đạt, mọi bước chạy đã xong và con người đã xác nhận tiêu chí hoàn thành.

### 0.4. Điểm phải dừng để chờ con người xác nhận

Agent dừng, báo cáo và chờ xác nhận trước khi:

- chạy tiếp qua một bước có **[dừng]** trong đặc tả phase (mục 6);
- chạy tác vụ GPU lớn: ghi dữ liệu đầy đủ, chạy lưới thí nghiệm trực tuyến;
- khóa danh sách ID của các tập dữ liệu, hoặc thay đổi chúng sau khi đã khóa;
- khóa cấu hình, hoặc chạy bất kỳ lệnh nào trên `test_*`, kể cả tái lập baseline ở P1.T5 (mục 7.5);
- kết thúc một phase, kể cả khi cổng G1, G2 hoặc G3 của phase đó đạt (mục 7.4);
- thay đổi định nghĩa đếm token (mục 2.6) hoặc một bất biến (mục 2.7).

---

## 1. Bối cảnh tóm tắt

**Bài toán.** Cho $N$ câu hỏi, một mô hình suy luận lớn (LRM) cố định và tổng ngân sách $B$ token. Trong lúc LRM suy luận, tại các điểm quyết định, một chính sách quyết định câu nào tiếp tục suy luận, câu nào dừng và trả lời. Mục tiêu là tối đa số câu trả lời đúng, với $\sum_i T_i \le B$ ở **mọi** lần chạy ([mục 2 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L15)).

**Phương pháp.**

1. **Điểm quyết định.** Một từ chuyển ý ở đầu đoạn ("Wait", "Alternatively", …) xuất hiện sau ít nhất $m$ token kể từ điểm trước, hoặc khi đã sinh $L_{\max}$ token mà không gặp từ chuyển ý.
2. **Gợi đáp án thử.** Tại điểm quyết định, tạo một nhánh phụ từ tiền tố (dùng prefix caching), chèn lời nhắc, đọc đáp án thử và độ tự tin. Chuỗi chính không thay đổi.
3. **Bộ ước lượng** $f$ (LightGBM và hiệu chuẩn isotonic) dự đoán $\hat q(h)$: xác suất trả lời đúng nếu suy luận thêm $h$ token rồi dừng.
4. **Lợi ích trên mỗi token:** $\rho_i = \max_h \big(\hat q(h) - \hat q(0)\big) / \big(h(1+\omega)\big)$. Tầm $h_i^*$ là giá trị $h$ đạt max.
5. **Vòng đồng bộ.** Xếp các câu theo $\rho$ giảm dần và cộng dồn $h^*$. Giá $\mu$ là $\rho$ của câu đầu tiên làm tổng vượt quỹ. Câu tiếp tục nếu $\rho_i \ge \mu$, $\rho_i > \mu_{\min}$ và $b_i < b_{\max}$; các câu còn lại dừng.
6. **Bảo đảm ngân sách cứng** bằng phần dự trữ cho câu trả lời và hạn mức token cho từng đoạn.

Hệ thống có hai môi trường chạy dùng chung một bộ điều khiển: **mô phỏng**, với nguồn đoạn suy luận là dữ liệu đã ghi, và **chạy thật**, với nguồn đoạn suy luận là vLLM.

**Ký hiệu và tên trong mã:**

| Ký hiệu | Tên trong mã | Ý nghĩa |
|---|---|---|
| $B$ | `budget` | Tổng ngân sách token của một lần chạy |
| $N$ | `n_questions` | Số câu trong lô |
| $\alpha$ | `alpha` | Mức ngân sách tương đối: $B = \alpha \cdot C_{\text{full}}$ |
| $C_{\text{full}}$ | `c_full` | Tổng chi phí khi mọi câu chạy tới hết, không gợi và không dừng sớm (mục 7.3) |
| $b_i$ | `state.gen_end` | Số token chuỗi chính đã sinh cho câu $i$ |
| $m$, $L_{\max}$, $b_{\min}$ | `dp.m`, `dp.l_max`, `dp.b_min` | Tham số điểm quyết định |
| $b_{\max}$ | `policy.b_max` | Trần số token chuỗi chính của một câu |
| $H$ | `policy.horizons` | Tập tầm nhìn |
| $\hat q(h)$ | `qhat` | Xác suất đúng ước lượng, theo từng $h$ |
| $\rho_i$, $h_i^*$ | `rho`, `hstar` | Lợi ích trên mỗi token, tầm đạt max |
| $\mu$, $\mu_{\min}$ | `mu`, `mu_min` | Giá token, giá sàn |
| $\omega$ | `omega` | Số token gợi trung bình trên mỗi token suy luận |
| $R$ | `pool.remaining` | Quỹ còn lại |
| $r_{\text{probe}}$, $r_{\text{ans}}$ | `probe.r_probe`, `answer.r_ans` | Số token sinh tối đa của đáp án thử, của câu trả lời |
| $c_{\text{probe,max}}$, $c_{\text{ans,max}}$ | `c_probe_max`, `c_ans_max` | Chi phí tối đa của một lần gợi, một câu trả lời, gồm cả lời nhắc |

---

## 2. Quy ước chung

### 2.1. Repo

Mã nguồn nằm trong một repo riêng tên `ttalloc`, đặt trên máy có GPU. Tài liệu này được chép vào `docs/PLAN.md` của repo, cùng với báo cáo phương pháp chi tiết và kế hoạch thực hiện đề tài ở `docs/Bao-cao-Phuong-phap-chi-tiet.md` và `docs/Ke-hoach-thuc-hien-de-tai.md` (giữ nguyên tên tệp) để các liên kết trong tài liệu vẫn đúng.

```
ttalloc/
├── pyproject.toml
├── AGENTS.md                 # quy tắc cho agent: mục 0, 2 của tài liệu này
├── configs/
│   ├── base.yaml
│   ├── models/               # qwen3_1_7b.yaml (model chính); model thứ hai thêm ở P5.T2 nếu làm
│   ├── datasets/             # math_train.yaml, deepscaler.yaml, amc23.yaml, math500.yaml, ...
│   ├── prompts/              # instruction, probe, answer (mục 9.2)
│   ├── splits.yaml
│   ├── batches.yaml          # thành phần lô (P2.T14)
│   └── experiments/          # g1.yaml, e2_main.yaml, e5_online.yaml, ...
├── splits/                   # <tên tập>.json và <tên tập>.sha256
├── src/ttalloc/
│   ├── config.py  runs.py  types.py  cli.py
│   ├── data/                 # loaders.py, splits.py, prompts.py, grading.py, batches.py
│   ├── engine/               # generator.py, scripted.py, vllm_generator.py, cut.py, live_source.py
│   ├── baseline/             # full.py, deer_official.py, throughput.py
│   ├── logging/              # schema.py, store.py, logger.py, validate.py, studies.py
│   ├── features/             # events.py, extract.py
│   ├── estimator/            # dataset.py, train.py, rho.py, evaluate.py
│   ├── controller/           # budget.py, runner.py
│   ├── policies/             # base.py, deer.py, price.py; để sau (P5.T1): uniform.py, fifo_pool.py, codestop.py
│   ├── sim/                  # logged_source.py, table.py, providers.py, tune.py, run.py; để sau (P5.T1): oracle.py, sequential.py
│   ├── online/               # run.py, timing.py
│   └── eval/                 # metrics.py, stats.py, report.py, gates.py, curves.py
├── tests/
│   ├── fixtures/             # tokenizer của Qwen3-1.7B, dữ liệu nhỏ
│   ├── synth.py              # sinh dữ liệu tổng hợp đúng schema
│   ├── unit/
│   └── integration/
├── docs/                     # PLAN.md, Bao-cao-Phuong-phap-chi-tiet.md, Ke-hoach-thuc-hien-de-tai.md,
│                             # progress.md, decisions.md, test-log.md, review/
├── third_party/              # repo DEER chính thức ở commit cố định (P1.T5); không đưa vào git
├── data/                     # không đưa vào git: logs/, tables/, estimators/
└── runs/                     # không đưa vào git
```

### 2.2. Công nghệ

- Python 3.12, quản lý bằng `uv`. `pyproject.toml` có ba nhóm phụ thuộc: core, `gpu`, `dev`.
- **core:** numpy, pandas, pyarrow, pydantic v2, pyyaml, typer, transformers, datasets, math-verify, lightgbm, scikit-learn, scipy, statsmodels, matplotlib.
- **gpu:** vllm, cố định một phiên bản. Lý do chọn phiên bản được ghi vào `docs/decisions.md`; không nâng cấp giữa chừng.
- **dev:** pytest, hypothesis, ruff, mypy.

### 2.3. Quy ước viết mã

- Có type hint đầy đủ. Dữ liệu bất biến dùng `@dataclass(frozen=True)`.
- Không có trạng thái toàn cục. Mọi nguồn ngẫu nhiên là `numpy.random.Generator` được truyền vào.
- Chỉ `engine/vllm_generator.py` được import `vllm`, và import bên trong hàm. Mọi module khác import được trên máy không có vLLM.
- Không viết cứng tham số trong mã; mọi tham số đến từ config.
- Lõi tính toán là hàm thuần. Đọc ghi tệp và gọi GPU chỉ nằm ở lớp ngoài (CLI, runner, logger).

### 2.4. Config và thư mục chạy

- Config là YAML, được nạp vào các mô hình pydantic trong `config.py`. Hỗ trợ khóa `extends` (kế thừa tệp khác) và ghi đè từ dòng lệnh dạng `--set dp.m=256`.
- `config_hash` là sha256 của JSON chuẩn hóa (khóa sắp xếp) của config đã phân giải; dùng 12 ký tự đầu.
- Mỗi lần chạy tạo một thư mục `runs/<YYYYmmdd-HHMMSS>_<tên>_<hash>/` chứa `config.yaml`, `meta.json` (commit git, cờ dirty, phiên bản Python và các gói, thông tin GPU), các tệp kết quả và `log.txt`. Không bao giờ ghi đè thư mục đã có.
- Dữ liệu lớn nằm trong `data/`: `data/logs/<model>/<split>/`, `data/tables/<model>/<split>/<hash>.parquet`, `data/estimators/<model>/<hash>/`.

### 2.5. Quy ước test

- Marker của pytest: `gpu` (cần GPU và vLLM), `slow` (chạy lâu), `net` (cần mạng). Mặc định chạy `-m "not gpu and not slow and not net"`.
- Test CPU dùng `ScriptedGenerator` (mục 3.3), dữ liệu tổng hợp từ `tests/synth.py` và tokenizer lưu sẵn trong `tests/fixtures/`.
- Test GPU dùng chính model Qwen3-1.7B, đủ nhỏ để chạy trên T4 của Kaggle.
- `tests/synth.py` sinh các bảng `traces` và `points` đúng schema ở mục 4. Đường cong đúng sai theo số token của mỗi câu có dạng chữ S, với tham số ngẫu nhiên; độ tự tin của đáp án thử tương quan với tính đúng sai.

### 2.6. Đếm token (định nghĩa chuẩn)

Chi phí của câu $i$ là $T_i$ = `think` + `probe_prompt` + `probe_output` + `answer_prompt` + `answer_output`.

| Thành phần | Định nghĩa |
|---|---|
| `think` | Mọi token chuỗi chính mà engine đã sinh cho câu $i$, kể cả token của từ chuyển ý và các token nằm sau vị trí cắt khi câu dừng tại đó |
| `probe_prompt` | Số token lời nhắc gợi đã chèn, cộng trên mọi lần gợi |
| `probe_output` | Số token engine đã sinh ở nhánh gợi, cộng trên mọi lần gợi |
| `answer_prompt` | Số token đã chèn để ép trả lời, gồm cả token `</think>` nếu do hệ thống chèn |
| `answer_output` | Số token engine đã sinh cho câu trả lời cuối |

- Token của đề bài (prompt ban đầu) không tính vào $T_i$.
- Mọi số token lấy từ số token engine thực sự sinh hoặc thực sự được chèn, không suy ra từ văn bản.
- $c_{\text{probe,max}}$ = `len(probe_prompt_ids)` + $r_{\text{probe}}$ + `probe.think_end_lookahead`; $c_{\text{ans,max}}$ = `len(answer_prompt_ids)` + $r_{\text{ans}}$.

### 2.7. Bất biến

Mọi bất biến phải có test kiểm tra. Vi phạm một bất biến luôn là lỗi mức blocker.

| Mã | Bất biến | Test chính |
|---|---|---|
| I1 | Ngân sách: tổng $T_i$ của mọi câu $\le B$ ở mọi lần chạy, cả mô phỏng lẫn chạy thật | Property test ở P2.T7, P2.T8; smoke test GPU ở P4.T3 |
| I2 | Hạch toán: tổng theo từng loại token khớp với số token engine báo; $T_i$ bằng tổng năm thành phần | P1.T3, P2.T8 |
| I3 | Mỗi câu có đúng một câu trả lời cuối, được chấm 0 hoặc 1 | P2.T8 |
| I4 | Không nhìn trước: quyết định tại một điểm chỉ dùng thông tin đã có tới điểm đó (M3 và O1 là ngoại lệ có chủ đích, mục 7.2) | P2.T11, P5.T1 |
| I5 | Không rò rỉ: bộ ước lượng chỉ học trên tập huấn luyện; hiệu chuẩn dùng dự đoán ngoài fold trên tập huấn luyện; chỉnh tham số trên validation; tập kiểm tra chỉ dùng khi cấu hình đã khóa. Hai ngoại lệ có chủ đích: (1) A11 hiệu chỉnh hạn mức và giá trên chính tập đang đánh giá, chỉ dùng chi tiêu, không dùng nhãn; (2) P1.T5 chạy model gốc và DEER chính thức trên `test_main` để đối chiếu với số công bố, có người duyệt, và không tham số nào được chọn từ kết quả đó | P0.T7, P1.T5, P2.T4, P3.T5 |
| I6 | Tái lập: cùng config và cùng seed thì kết quả mô phỏng giống hệt; chạy thật ghi lại seed | P2.T15 |
| I7 | Toàn vẹn chuỗi chính: gợi đáp án thử và trả lời không làm thay đổi chuỗi chính | P1.T3, P1.T4 |
| I8 | Mô phỏng hợp lệ: mọi sự kiện mà mô phỏng cần đều có trong dữ liệu ghi lại, và $b_{\max} \le L_{\text{log}} - c_{\text{ans,max}}$ | P1.T11, P2.T9 |
| I9 | Nhất quán: lời nhắc, dạng câu trả lời và tham số sinh giống nhau khi ghi dữ liệu, mô phỏng và chạy thật; mô phỏng và chạy thật từ chối chạy nếu mã băm khác | P1.T8, P2.T9, P4.T1 |

---

## 3. Kiến trúc

### 3.1. Thành phần

```
                      ┌────────────────────────┐
  danh sách câu  ───► │  Runner (controller)   │ ◄── Policy.decide()
  ngân sách B         │  Pool, guard, vòng lặp │ ◄── RhoProvider (ước lượng ρ, h*)
                      └───────────┬────────────┘
                                  │ SegmentSource
                 ┌────────────────┴─────────────────┐
         LoggedSource (mô phỏng)            LiveSource (chạy thật)
         đọc data/logs                      Generator: VllmGenerator / ScriptedGenerator
```

- **Runner** điều khiển vòng đồng bộ, quản lý quỹ và áp dụng bảo đảm ngân sách. Mô phỏng và chạy thật dùng cùng một Runner.
- **Logger** (ghi dữ liệu) dùng trực tiếp `LiveSource` với chế độ ghi mọi điểm; Logger không đi qua Runner.

### 3.2. SegmentSource

```python
@dataclass(frozen=True)
class TraceKey:
    qid: str
    seed: int

@dataclass(frozen=True)
class AdvanceRequest:
    key: TraceKey
    min_tokens: int          # chỉ chấp nhận từ chuyển ý khi số token đã sinh trong đoạn > min_tokens (A1a)
    max_tokens: int          # hạn mức token chuỗi chính của đoạn
    stop_at_markers: bool    # False: chỉ dừng khi tự kết thúc hoặc hết hạn mức

class SegmentSource(Protocol):
    def start(self, keys: list[TraceKey]) -> None: ...
    def advance(self, reqs: list[AdvanceRequest]) -> list[SegmentResult]: ...
    def probe(self, keys: list[TraceKey]) -> list[ProbeResult]: ...
    def answer(self, keys: list[TraceKey]) -> list[AnswerResult]: ...
```

- `advance` sinh tiếp chuỗi chính từ vị trí `gen_end` hiện tại. Đoạn kết thúc ở sự kiện xảy ra sớm nhất trong ba sự kiện: một từ chuyển ý hợp lệ (nếu `stop_at_markers`), chuỗi tự kết thúc (`</think>` hoặc EOS), hoặc đã sinh đủ `max_tokens`. `SegmentResult` gồm `kind` (`marker`, `length`, `natural_end`, `eos`, `truncated`), `pos`, `gen_end`, `n_new_tokens` và `marker`.
- `probe` gợi đáp án thử tại `pos` của điểm hiện tại, theo A1b. `ProbeResult` gồm đáp án thô, khóa chuẩn hóa, `conf_geo`, `conf_arith`, logprob từng token, số token lời nhắc và số token sinh.
- `answer` sinh câu trả lời cuối theo A1c: ép kết thúc suy luận tại `pos` của điểm hiện tại; nếu model đã tự sinh `</think>` thì chỉ chèn phần còn lại của lời nhắc trả lời. `AnswerResult` gồm văn bản, đáp án trích ra, `correct` (có thể là `None`; khi đó Runner chấm), số token lời nhắc và số token sinh. Sau `answer`, câu kết thúc.
- `pos` là độ dài tiền tố dùng để gợi và trả lời; `gen_end` là số token chuỗi chính đã sinh. Luôn có `pos <= gen_end`.

### 3.3. Generator

```python
@dataclass(frozen=True)
class GenRequest:
    prompt_ids: list[int]
    max_tokens: int
    stop: list[str]
    stop_token_ids: list[int]
    temperature: float
    top_p: float
    top_k: int
    logprobs: int | None

@dataclass(frozen=True)
class GenOutput:
    token_ids: list[int]
    logprobs: list[float] | None    # logprob của token đã sinh
    finish_reason: str              # "stop" | "length"
    stop_reason: str | int | None   # chuỗi dừng, token dừng, hoặc None nếu gặp EOS

class Generator(Protocol):
    def generate(self, reqs: list[GenRequest]) -> list[GenOutput]: ...
```

- `VllmGenerator` gọi `LLM.generate` một lần cho cả danh sách: mỗi request dùng `TokensPrompt` và một `SamplingParams` riêng, với `include_stop_str_in_output=True`. Engine tạo với `enable_prefix_caching=True`. Việc chuyển `GenRequest` thành tham số của vLLM nằm trong hàm thuần `to_sampling_kwargs(req) -> dict` ở `engine/generator.py`, để test được trên CPU.
- **Không dùng tham số `min_tokens` của vLLM.** Tham số này chặn cả EOS và mọi `stop_token_ids` (gồm `</think>`) cho tới khi đủ số token, nên làm thay đổi phân phối của chuỗi. Khoảng cách tối thiểu giữa hai điểm quyết định do LiveSource tự áp dụng (A1a).
- `ScriptedGenerator` (dùng cho test) nhận một hàm `continuation(prompt_ids) -> list[int]` trả về "chuỗi tiếp nối thật". Nó mô phỏng `max_tokens`, chuỗi dừng (so trên văn bản giải mã, kể cả khi chuỗi dừng nằm bên trong một token dài hơn), `stop_token_ids` và EOS, và trả logprob theo kịch bản cho trước.

### 3.4. Trạng thái câu hỏi, chính sách và RhoProvider

```python
@dataclass
class QuestionState:
    key: TraceKey
    gen_end: int = 0             # b_i
    pos: int = 0
    point_kind: str = "start"    # start | marker | length
    visited: list[int] = field(default_factory=list)   # thứ tự điểm đã ghé
    costs: Costs = field(default_factory=Costs)        # theo năm loại ở mục 2.6
    features: FeatureState = ...
    qhat: np.ndarray | None = None
    rho: float | None = None
    hstar: int | None = None

@dataclass(frozen=True)
class Decision:
    action: Literal["continue", "stop"]
    min_tokens: int = 0
    max_tokens: int = 0          # hạn mức mong muốn; guard có thể giảm

class Policy(Protocol):
    uses_probes: bool            # B0: False
    uses_rho: bool
    stop_at_markers: bool        # B0: False
    def initial_request(self, s: QuestionState, pool: PoolView) -> Decision: ...
    def decide(self, active: list[QuestionState], pool: PoolView) -> dict[TraceKey, Decision]: ...

class RhoProvider(Protocol):
    def fill(self, states: list[QuestionState]) -> None: ...   # ghi qhat, rho, hstar vào state
```

- Policy chỉ đọc `QuestionState` và `PoolView`; không ghi.
- Có ba RhoProvider: `EstimatorRhoProvider` gọi bộ ước lượng; `TableRhoProvider` đọc bảng tính sẵn (mục 4.4) và quay về gọi bộ ước lượng khi lịch sử điểm đã ghé khác dãy chuẩn; `CurveRhoProvider` dùng đường cong thực nghiệm cho M3.
- Mọi phép sắp xếp có thể hòa (theo $\rho$, theo thứ tự nạp, theo thứ tự chia đều) phá thế hòa bằng `tie_key(key, run_seed)`: mã băm ổn định của khóa câu hỏi và seed của lần chạy. Không phá thế hòa bằng `qid`, vì như vậy tên tập dữ liệu sẽ quyết định câu nào được ưu tiên.

---

## 4. Dữ liệu

### 4.1. Câu hỏi và các tập

```python
@dataclass(frozen=True)
class Question:
    qid: str                 # "<dataset>/<id gốc hoặc chỉ số>"
    dataset: str
    problem: str
    options: list[str] | None
    gold: str
    answer_type: Literal["math", "choice"]
    difficulty: int | None   # mức 1–5 của MATH, nếu có
```

| Tên tập | Nguồn | Số câu | Seed | Vai trò |
|---|---|---|---|---|
| `train` | MATH tập train, đủ 5 mức (`EleutherAI/hendrycks_math`): 900 câu; DeepScaleR (`agentica-org/DeepScaleR-Preview-Dataset`): 600 câu | 1.500 | 4 | Huấn luyện bộ ước lượng |
| `val` | AMC 2023 (`math-ai/amc23`): 40 câu; MATH tập train, tách riêng khỏi `train`: 360 câu | 400 | 8 | Chỉnh tham số, cổng G1 |
| `test_main` | MATH-500 (`HuggingFaceH4/MATH-500`); AIME 2024 (`HuggingFaceH4/aime_2024`); AIME 2025 (`MathArena/aime_2025`) | 560 | 8 | Kết quả chính |
| `test_clean` | AIME 2026 và HMMT 02/2026 của MathArena (kiểm tra tên dataset trước khi tải) | 60 | 8 | Tập ra đời sau các model |
| `test_ood` | GPQA-Diamond (`Idavidrein/gpqa`, cấu hình `gpqa_diamond`) | 198 | 8 | Ngoài miền |

- `train` và `val` được lấy mẫu phân tầng theo độ khó, với seed cố định trong `configs/splits.yaml`.
- Khử trùng lặp giữa `train`, `val` và mọi tập kiểm tra: chuẩn hóa văn bản (chữ thường, bỏ khoảng trắng thừa và ký hiệu LaTeX trình bày), so khớp chính xác, rồi so khớp 13-gram. Câu trùng bị loại khỏi `train` và `val`.
- Với GPQA, thứ tự các lựa chọn được xáo bằng seed cố định theo `qid` (giống nhau cho mọi seed sinh); `gold` là chữ cái của đáp án đúng sau khi xáo.

### 4.2. Bảng `traces`

Mỗi dòng là một chuỗi suy luận đã ghi. Lưu dạng parquet, chia thành nhiều tệp `part-*.parquet`.

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `qid`, `seed` | string, int32 | Khóa |
| `model`, `split` | string | |
| `log_hash` | string | Mã băm của config ghi dữ liệu và lời nhắc (I9) |
| `chain_ids` | list<int32> | Token chuỗi chính, gồm `</think>` nếu tự kết thúc |
| `chain_logprobs` | list<float16> | Logprob của từng token chuỗi chính (có thể rỗng nếu tắt) |
| `chain_len` | int32 | `len(chain_ids)` |
| `natural_end` | bool | Chuỗi tự kết thúc trong $L_{\text{log}}$, bằng `</think>` hoặc EOS |
| `end_reason` | string | `think_end`, `eos` hoặc `max_len` |
| `nat_answer_text`, `nat_answer_extracted` | string | Câu trả lời tại điểm tự kết thúc, sinh theo A1c (nếu `natural_end`) |
| `nat_answer_prompt_tokens`, `nat_answer_output_tokens` | int32 | Số token của câu trả lời tự nhiên |
| `nat_correct` | bool, có thể null | |
| `gold` | string | |

### 4.3. Bảng `points`

Mỗi dòng là một điểm đã ghi của một chuỗi. Mỗi chuỗi có điểm `start` (`point_idx = 0`, `pos = gen_end = 0`): chỉ có câu trả lời ép buộc, không có đáp án thử. Điểm này dùng khi một câu phải trả lời trước khi suy luận.

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `qid`, `seed`, `point_idx` | string, int32, int32 | Khóa; `point_idx` tăng theo `gen_end` |
| `kind` | string | `start`, `marker` hoặc `length` |
| `marker` | string, có thể null | Từ chuyển ý |
| `pos`, `gen_end` | int32 | Như mục 3.2 |
| `n_markers` | int32 | Số từ chuyển ý ở đầu đoạn trong `chain_ids[:pos]` |
| `local_conf_min` | float32, có thể null | Mục A5 |
| `probe_prompt_tokens`, `probe_output_tokens` | int16 | |
| `probe_output_ids` | list<int32> | |
| `probe_logprobs` | list<float32> | |
| `probe_answer_raw`, `probe_answer_key` | string | Đáp án thử thô và khóa chuẩn hóa |
| `probe_conf_geo`, `probe_conf_arith` | float32 | Trung bình nhân và trung bình cộng của xác suất |
| `probe_has_think_end` | bool | Nhánh gợi có sinh `</think>` (điều kiện bổ sung của DEER với Qwen3) |
| `answer_prompt_tokens`, `answer_output_tokens` | int16 | Câu trả lời ép buộc tại điểm này |
| `answer_text`, `answer_extracted` | string | |
| `answer_correct` | bool | |

### 4.4. Bảng điểm tính sẵn cho mô phỏng

`data/tables/<model>/<split>/<hash>.parquet`. Mã băm gồm config điểm quyết định, config đặc trưng và mã băm bộ ước lượng. Mỗi dòng ứng với một điểm thuộc dãy chuẩn (mục A3) của một chuỗi, gồm: khóa, chỉ số của điểm trước và điểm sau trong dãy chuẩn, véc-tơ đặc trưng, `qhat` cho mọi $h$ trong `estimator.h_train` và cho $h = 0$.

### 4.5. Kết quả của một lần chạy

- `per_question.parquet`: `method`, `alpha`, `draw`, `qid`, `seed`, `correct`, năm thành phần chi phí, `stop_point_idx`, `stop_kind`, `n_probes`, `n_rounds`.
- `summary.json`: độ chính xác, tổng token theo từng loại, $B$, $B/N$ và mã băm của mọi đầu vào.
- `timing.parquet` (chỉ khi chạy thật): mỗi vòng gồm thời gian sinh đoạn, thời gian gợi, thời gian trả lời, thời gian của bộ điều khiển, số câu đang chạy và tỉ lệ trúng prefix cache nếu có.

---

## 5. Đặc tả thuật toán

### A1. Chạy thật: sinh đoạn, gợi đáp án thử, trả lời (`engine/cut.py`, `engine/live_source.py`)

**A1a. Sinh đoạn.** `LiveSource.advance` gửi `GenRequest` với `stop = ["\n\n" + w for w in dp.markers]` và `stop_token_ids = [think_end_id]`. Prompt của mọi request bắt đầu bằng `prompt_ids` của đề bài, tiếp theo là chuỗi chính đã có. Không dựa vào văn bản đã bị vLLM cắt; vị trí điểm luôn tính lại trên token id.

```python
MARKER_RE = re.compile(r"\s*(Wait|Alternatively|Hmm|But|Let me check)\b")

def classify(chain_ids, start, seg_start, out, tok, req):
    """start: gen_end ở đầu đoạn; seg_start: vị trí bắt đầu của lần gửi này
    (lớn hơn start nếu đã gửi lại). Trả về một sự kiện, hoặc RESUME nếu phải sinh tiếp."""
    seg = out.token_ids
    gen_end = seg_start + len(seg)
    full = chain_ids[:seg_start] + seg
    if think_end_id in seg:                                       # tự kết thúc
        return natural_end(pos=seg_start + seg.index(think_end_id) + 1, gen_end=gen_end)
    if out.finish_reason == "stop" and out.stop_reason is None:    # EOS mà không có </think>
        return eos(pos=gen_end - 1, gen_end=gen_end)
    if out.finish_reason == "stop" and isinstance(out.stop_reason, str):
        for t in range(gen_end - 1, max(seg_start - 1, gen_end - 9), -1):   # tìm ngược tối đa 8 token
            head = tok.decode(full[max(0, t - 4):t])
            if head.endswith("\n\n") and MARKER_RE.match(tok.decode(full[t:gen_end])):
                if gen_end - start <= req.min_tokens:
                    return RESUME                                  # từ chuyển ý đến quá sớm
                return marker(pos=t, gen_end=gen_end)
        return RESUME                      # khớp chuỗi dừng nhưng không phải từ chuyển ý, ví dụ "Butter"
    return length(pos=newline_cutback(full, start, gen_end, tok), gen_end=gen_end)

def newline_cutback(full, start, gen_end, tok):
    lo = gen_end - (gen_end - start) // 4                         # chỉ lùi trong 25% cuối đoạn
    for t in range(gen_end, lo, -1):
        if tok.decode(full[t - 1:t]).endswith("\n"):
            return t
    return gen_end
```

- Với `RESUME`, LiveSource nối các token vừa sinh vào chuỗi chính rồi gửi tiếp với `max_tokens` còn lại (`req.max_tokens - (gen_end - start)`), ngay trong cùng lời gọi `advance`. Các câu cần gửi lại được gom thành một lô mới; prefix caching làm cho việc gửi lại rẻ. Nếu hết `max_tokens` thì sự kiện là `length`.
- Nhờ vậy, ngữ nghĩa khoảng cách luôn là: chỉ chấp nhận từ chuyển ý khi số token đã sinh trong đoạn lớn hơn `min_tokens`. Logger và A2 dùng đúng ngữ nghĩa này.
- Khi giải mã luôn dùng `skip_special_tokens=False` và `clean_up_tokenization_spaces=False`.

**A1b. Gợi đáp án thử.**

- Prompt là `prompt_ids + chain_ids[:pos] + probe_prompt_ids`, với `temperature=0`, `logprobs=1`, `stop=["}"]`.
- Sau mỗi lần sinh, đếm ngoặc kể từ `\boxed{` cuối cùng. Nếu khung chưa đóng, chưa gặp EOS và tổng số token đã sinh nhỏ hơn `r_probe`, gửi tiếp phần còn lại (gom lô như A1a). Như vậy nhánh gợi dừng ngay khi khung đóng, kể cả với đáp án có ngoặc lồng nhau như `\frac{1}{2}`.
- Đáp án thử là nội dung trong khung. Độ tự tin tính trên các token từ token đầu tiên sau dấu `{` mở khung tới token chứa dấu `}` đóng khung (gồm cả token đó): `conf_geo = exp(mean(logprob))`, `conf_arith = mean(exp(logprob))`.
- Khi `probe.think_end_lookahead > 0` (dùng cho Qwen3), sau khi khung đóng, sinh thêm tối đa số token đó để ghi `probe_has_think_end`.
- Mọi token sinh ở nhánh gợi tính vào `probe_output`. `probe_prompt` tính `len(probe_prompt_ids)` một lần cho mỗi lần gợi, không tính lại khi gửi tiếp.
- Nếu code DEER chính thức dùng quy tắc dừng hoặc phạm vi token khác, làm theo code DEER và ghi ADR.

**A1c. Câu trả lời cuối.** Luôn dùng `temperature=0` và `max_tokens = r_ans`.

| Tình huống | Prompt | Token chèn được tính |
|---|---|---|
| Dừng tại một điểm (`start`, `marker`, `length`), hoặc `truncated` | `prompt_ids + chain_ids[:pos] + answer_prompt_ids` | `len(answer_prompt_ids)` |
| Tự kết thúc bằng `</think>` | `prompt_ids + chain_ids[:pos] + answer_suffix_ids`. `answer_suffix_ids` là `answer_prompt_ids` bỏ token `</think>` ở đầu, vì model đã tự sinh token này | `len(answer_suffix_ids)` |
| Kết thúc bằng EOS mà không có `</think>` | Như dừng tại một điểm, với `pos = gen_end - 1` | `len(answer_prompt_ids)` |

Logger, mô phỏng và chạy thật dùng đúng các quy tắc này, nên chi phí của mọi câu trả lời không vượt $c_{\text{ans,max}}$.

### A2. Sự kiện kế tiếp trên chuỗi đã ghi (`features/events.py`)

Hàm này là nguồn chung cho `LoggedSource.advance` (P2.T9) và cho việc chọn dãy chuẩn (A3). Không được cài đặt hai lần.

```python
def next_event(trace, points, cur_idx, s, min_tokens, max_tokens, stop_at_markers):
    """s = gen_end hiện tại; points sắp theo gen_end; cur_idx = điểm đang đứng."""
    lim = s + max_tokens
    marker = None
    if stop_at_markers:
        marker = first(p for p in points[cur_idx + 1:]
                       if p.kind == "marker" and p.gen_end - s > min_tokens and p.gen_end <= lim)
    nat = trace.chain_len if trace.natural_end else None
    if nat is not None and nat <= lim and (marker is None or nat <= marker.gen_end):
        return Event("natural_end", gen_end=nat)
    if marker is not None:
        return Event("marker", point=marker)
    last = last_or_none(p for p in points[cur_idx + 1:] if s < p.gen_end <= lim)
    if last is not None:
        return Event("length", point=last)              # làm tròn xuống điểm đã ghi gần nhất
    if not trace.natural_end and trace.chain_len <= lim:
        return Event("truncated", point=points[cur_idx])  # chuỗi ghi đã chạm L_log
    raise SimValidityError(trace.key, s, lim)            # vi phạm I8
```

Chi phí `think` của sự kiện bằng `gen_end` mới trừ `s`. Với sự kiện `length`, mô phỏng làm tròn xuống điểm đã ghi gần nhất; chạy thật dừng đúng ở hạn mức. Đây là sai khác đã biết (mục 7.6).

### A3. Dãy điểm chuẩn của chính sách (`features/events.py`)

```python
def canonical_points(trace, points, dp) -> list[int]:
    out, cur, s = [], 0, 0
    min_tok, max_tok = dp.b_min, dp.b_min + dp.l_max
    while True:
        ev = next_event(trace, points, cur, s, min_tok, max_tok, stop_at_markers=True)
        if ev.kind in ("natural_end", "truncated"):
            return out
        out.append(ev.point.point_idx)
        cur, s = index_of(ev.point), ev.point.gen_end
        min_tok, max_tok = dp.m, dp.l_max
```

Dãy chuẩn là dãy điểm mà một chính sách luôn tiếp tục sẽ ghé. Đặc trưng để huấn luyện bộ ước lượng chỉ tính trên dãy này, nên khớp với những gì chính sách thấy khi chạy.

### A4. Nhãn huấn luyện (`estimator/dataset.py`)

```python
def samples_for_trace(trace, points, sub, H_train):
    for j, p in enumerate(sub):                       # sub = canonical_points(...)
        x = features_after(sub[: j + 1])              # A5
        yield x, 0, p.answer_correct
        for h in H_train:
            lim = p.gen_end + h
            if trace.natural_end and trace.chain_len <= lim:
                y = trace.nat_correct
            else:
                y = last(q for q in sub[j:] if q.gen_end <= lim).answer_correct
            yield x, h, y
```

- Trọng số mẫu là `1 / (len(sub) * (1 + len(H_train)))`, để tổng trọng số của mỗi chuỗi bằng 1. Nhóm (group) là `qid`.
- Với $h$ vượt quá phần chuỗi đã ghi mà chuỗi không tự kết thúc, nhãn lấy từ điểm cuối cùng của dãy chuẩn.

### A5. Đặc trưng (`features/extract.py`)

Một lớp `FeatureState` duy nhất được dùng khi huấn luyện, khi mô phỏng và khi chạy thật. Định nghĩa đặc trưng ở [mục 4.3 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L120).

```python
class FeatureState:
    def update(self, gen_end, probe: ProbeResult, n_markers, local_conf_min): ...
    def vector(self) -> dict[str, float]:
        c = [p.conf for p in self.probes]        # conf = conf_geo
        recent = self.probes[-k:]
        return {
            "conf_cur": c[-1],
            "conf_max": max(c),
            "conf_slope_per_1k": ols_slope([p.gen_end / 1000 for p in recent], [p.conf for p in recent])
                                 if len(recent) >= 2 else 0.0,
            "has_slope": float(len(recent) >= 2),
            "consistency": mean(p.key == recent[-1].key for p in recent),
            "n_recent": len(recent),
            "n_changes": sum(a.key != b.key for a, b in pairwise(self.probes)),
            "n_probes": len(c),
            "b": self.probes[-1].gen_end,
            "marker_rate_per_1k": 1000 * n_markers / max(gen_end, 1),
            "local_conf_min": local_conf_min,           # NaN nếu tắt
            "ans_len": recent[-1].answer_tokens,
            **one_hot("ans_type", recent[-1].answer_type),  # integer, fraction, expression, choice, other
        }
```

- `local_conf_min`: giá trị nhỏ nhất, trên mọi cửa sổ trượt `features.local_window` token của chuỗi chính tới `pos`, của trung bình logprob token đã sinh. Tính tăng dần bằng tổng tiền tố. Nếu tắt logprob của chuỗi chính thì giá trị là NaN.
- Khóa đáp án `key` lấy từ `grading.canonical_key`; hai đáp án được coi là trùng khi khóa bằng nhau hoặc `grading.equivalent` trả về True (có cache).

### A6. Lợi ích trên mỗi token (`estimator/rho.py`)

```python
def rho_hstar(qhat: dict[int, float], b: int, horizons, b_max, omega):
    ok = [h for h in horizons if h <= b_max - b]
    if not ok:
        return -inf, None
    gains = [(qhat[h] - qhat[0]) / (h * (1 + omega)) for h in ok]
    k = argmax(gains)
    return gains[k], ok[k]
```

- $\omega$ được ước lượng trên `train` với config điểm quyết định hiện hành: tổng chi phí gợi tại các điểm của dãy chuẩn chia cho tổng token suy luận tới điểm cuối của dãy.
- Trong mã, $\rho$ có đơn vị xác suất trên mỗi token. Khi báo cáo, nhân $10^5$ để ra "điểm phần trăm trên 1.000 token". `mu` và `mu_min` dùng cùng đơn vị với `rho` trong mã.
- Khi không còn tầm hợp lệ, hàm trả về `(-inf, None)`; câu đó luôn dừng.

### A7. Quyết định của M2 (`policies/price.py`)

```python
def decide_m2(active, pool, cfg):
    avail = pool.available()                           # A8
    order = sorted(active, key=lambda s: (-s.rho, tie_key(s.key, cfg.run_seed)))
    cum, mu = 0, 0.0
    for s in order:
        if s.hstar is None:
            continue
        cum += s.hstar
        if cum > avail:
            mu = s.rho                                 # câu ở ranh giới
            break
    return {
        s.key: Decision("continue", cfg.dp.m, cfg.dp.l_max)
        if s.rho >= mu and s.rho > cfg.mu_min and s.gen_end < cfg.b_max
        else Decision("stop")
        for s in active
    }
```

Câu ở ranh giới vẫn tiếp tục (vì $\rho \ge \mu$). Câu tiếp tục chỉ chạy tới điểm quyết định kế tiếp; $h^*$ chỉ dùng để đặt giá ([mục 7.2 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L208)).

### A8. Quỹ và bộ bảo vệ ngân sách (`controller/budget.py`)

```python
class Pool:
    def remaining(self):
        return self.budget - self.spent.total
    def reserve(self):
        return ((self.n_active + self.n_waiting) * self.c_ans_max
                + self.n_waiting * (self.dp.b_min + self.c_probe_max))   # cửa sổ trượt
    def available(self):
        return self.remaining() - self.reserve()
    def charge(self, category, n):
        self.spent[category] += n
        assert self.spent.total <= self.budget               # I1

def guard(decisions, active, pool, policy, cfg):
    avail = pool.available()
    cp = cfg.c_probe_max if policy.uses_probes else 0
    cont = [s for s in active if decisions[s.key].action == "continue"]
    want = {s.key: min(decisions[s.key].max_tokens, cfg.b_max - s.gen_end) for s in cont}
    allot = {}

    def give(s, cap):
        nonlocal avail
        if cap < cfg.guard.min_segment:
            decisions[s.key] = Decision("stop")
            return
        allot[s.key] = (min(decisions[s.key].min_tokens, cap), cap)
        avail -= cap + cp

    # 1) Câu chưa có ρ (vòng đầu, câu mới nạp, hoặc chính sách không dùng ρ): chia đều khi không đủ
    fresh = sorted((s for s in cont if s.rho is None), key=lambda s: tie_key(s.key, cfg.run_seed))
    if sum(want[s.key] + cp for s in fresh) <= avail:
        for s in fresh:
            give(s, want[s.key])
    else:
        share = avail // max(len(fresh), 1)
        for s in fresh:
            give(s, min(want[s.key], share - cp))
    # 2) Câu đã có ρ: cấp theo ρ giảm dần từ phần còn lại
    ranked = sorted((s for s in cont if s.rho is not None),
                    key=lambda s: (-s.rho, tie_key(s.key, cfg.run_seed)))
    for s in ranked:
        give(s, min(want[s.key], avail - cp))
    return decisions, allot
```

- Trước vòng đầu, Runner kiểm tra `pool.available() >= 0` với mọi câu đang chạy và đang chờ; nếu không thỏa thì báo lỗi `InfeasibleBudget`. Từ đó, đầu mỗi vòng luôn có `pool.available() >= 0`.
- Vì mỗi câu trả lời tốn tối đa $c_{\text{ans,max}}$ (đã dự trữ), mỗi lần gợi tốn tối đa $c_{\text{probe,max}}$ và mỗi đoạn sinh tối đa `cap` token chuỗi chính (kể cả khi gửi lại theo A1a), tổng chi tiêu không bao giờ vượt $B$.
- Câu mới được nạp từ hàng chờ được cấp trước, từ chính phần dự trữ vừa được giải phóng của nó, nên không bị bỏ đói. Với chính sách không dùng $\rho$ (B0–B5), mọi câu thuộc nhóm 1, nên luôn được chia đều.
- `cfg.b_max` là trần của chính sách đang chạy (mục 9.1, khóa `policy.b_max_mult`).
- `guard.min_segment` không được nhỏ hơn `log.l_max`. Nhờ vậy, trong mô phỏng mọi đoạn được cấp đều chứa ít nhất một điểm đã ghi (A2). Quy tắc này áp dụng như nhau ở hai môi trường: hạn mức nhỏ hơn `min_segment` thì câu dừng.

### A9. Vòng lặp Runner (`controller/runner.py`)

```python
def run_batch(keys, policy, source, rho_provider, cfg, budget, rng) -> BatchResult:
    pool = Pool(budget, cfg); active = {}; fresh = set()
    waiting = deque(sorted(keys, key=lambda k: tie_key(k, cfg.run_seed)))   # cùng thứ tự nạp ở hai môi trường
    pool.check_feasible(n_total=len(keys))                     # InfeasibleBudget
    admit(waiting, active, fresh, cfg.max_active, source)     # gọi source.start
    while active:
        states = list(active.values())
        dec = {k: policy.initial_request(active[k], pool.view()) for k in fresh}
        rest = [s for s in states if s.key not in fresh]
        if rest:
            if policy.uses_rho:
                rho_provider.fill(rest)
            dec |= policy.decide(rest, pool.view())
        fresh.clear()
        dec, allot = guard(dec, states, pool, policy, cfg)
        stops = [k for k, d in dec.items() if d.action == "stop"]
        finalize(source.answer(stops), pool)                    # trừ quỹ, chấm điểm, rời active
        segs = source.advance([AdvanceRequest(k, *allot[k], policy.stop_at_markers) for k in allot])
        for r in segs:
            pool.charge("think", r.n_new_tokens); active[r.key].apply(r)
        ended = [r.key for r in segs if r.kind in ("natural_end", "eos", "truncated")]
        finalize(source.answer(ended), pool)
        at_point = [r.key for r in segs if r.kind in ("marker", "length")]
        if policy.uses_probes:
            for p in source.probe(at_point):
                pool.charge("probe_prompt", p.prompt_tokens)
                pool.charge("probe_output", p.output_tokens)
                active[p.key].features.update(...)
        admit(waiting, active, fresh, cfg.max_active, source)
        pool.check_round_invariants()
    return result(pool)
```

### A10. Oracle O1 (`sim/oracle.py`)

Với mỗi câu và chuỗi đã rút, tập ứng viên gồm mọi điểm đã ghi (kể cả điểm `start`) và điểm tự kết thúc nếu có. Chi phí dừng tại một ứng viên (không gợi) là `gen_end` của ứng viên đó cộng chi phí câu trả lời đã ghi tại đó. Vì gồm cả điểm `start`, O1 là cận trên của mọi chính sách trong mô phỏng, ở mọi mức ngân sách khả thi.

```python
c_min[i]  = chi phí nhỏ nhất của câu i trên mọi ứng viên
c_star[i] = chi phí nhỏ nhất trên các ứng viên trả lời đúng (inf nếu không có)
slack = B - sum(c_min)                    # >= 0 vì đã qua kiểm tra InfeasibleBudget
inc = sorted(c_star[i] - c_min[i] for i if c_star[i] < inf)
k = số phần tử đầu tiên của inc có tổng tích lũy <= slack
số câu đúng = k
```

Mỗi câu đóng góp 0 hoặc 1 câu đúng, nên chọn tham lam theo chi phí tăng thêm là tối ưu. Test so với vét cạn ở P2.T12.

### A11. Hiệu chỉnh theo ngân sách (`sim/tune.py`)

**Hạn mức theo câu.** Mọi hạn mức theo câu tính trên $T_i$ (mục 2.6). Một chính sách cho câu $i$ hạn mức $a$ sẽ yêu cầu đoạn kế tiếp với `max_tokens = min(dp.l_max, a - spent_i - c_ans_max - c_probe_max)`, và cho câu dừng khi giá trị này nhỏ hơn `guard.min_segment`. B0 không gợi nên bỏ `c_probe_max`.

- **B2:** tìm hạn mức chung $c$ bằng chia đôi trên $[B/N,\ c_{\text{hi}}]$, với $c_{\text{hi}} = L_{\text{log}} + c_{\text{ans,max}} + \lceil L_{\text{log}} / m \rceil \cdot c_{\text{probe,max}}$. Chi tiêu tăng theo $c$.
- **M1:** tìm giá cố định $\mu$ bằng chia đôi trên $\log \mu$, trong khoảng $[\max(\mu_{\min}, 10^{-9}),\ \rho_{\max}]$, với $\rho_{\max}$ là $\rho$ lớn nhất quan sát được. Chi tiêu giảm khi $\mu$ tăng.
- **Mục tiêu và điều kiện dừng:** chi tiêu trung bình qua các lần rút nằm trong $[0{,}98B;\ B]$. Dừng khi đạt mục tiêu, khi khoảng tìm hẹp hơn 1% của khoảng ban đầu, hoặc sau 30 vòng.
- **Khi không đạt được mục tiêu:** nếu ngay tại cận cho chi tiêu lớn nhất ($c = c_{\text{hi}}$, hoặc $\mu$ ở cận dưới) mà chi tiêu vẫn dưới $0{,}98B$, dùng cận đó và ghi `budget_underused: true` cùng tỉ lệ chi tiêu vào `summary.json`. Trường hợp này thường gặp ở $\alpha$ lớn.
- Việc hiệu chỉnh làm trên chính tập đang đánh giá và chỉ dùng chi tiêu, không dùng nhãn đúng sai (ngoại lệ có chủ đích của I5). Cách này thiên vị có lợi cho baseline; phải ghi rõ trong báo cáo kết quả.

### A12. Thống kê (`eval/stats.py`)

```python
# a[d, q], b[d, q]: đúng sai của hai phương pháp ở lần rút d, câu q (cùng các lần rút)
D = (a - b).mean(axis=0)                                  # theo từng câu
boot = [D[rng.integers(0, Q, Q)].mean() for _ in range(n_boot)]
ci = percentile(boot, [2.5, 97.5])
```

- Mô phỏng: bootstrap cặp theo câu hỏi, `n_boot = 2000`.
- Chạy thật: kiểm định McNemar theo từng câu cho từng seed (`statsmodels`, dạng exact khi số cặp lệch nhỏ).
- **So sánh trên nhiều mức $\alpha$** (G1, G2): dùng khoảng tin cậy hiệu chỉnh Bonferroni, mức $1 - 0{,}05/K$, với $K$ là số mức $\alpha$ trong họ so sánh.
- **Dải chấp nhận của G3:** với mỗi cấu hình chạy thật, lấy ngẫu nhiên 2.000 bộ 3 lần rút khác nhau từ mô phỏng có cùng $B$, cùng `max_active` và cùng thứ tự nạp; tính độ chính xác trung bình của mỗi bộ. Dải là khoảng phân vị 2,5–97,5% của các giá trị đó, và được so với trung bình 3 seed của chạy thật.

---

## 6. Các phase và task

Công việc chia thành bảy phase, P0 đến P6, cùng cách chia với [kế hoạch thực hiện đề tài](Ke-hoach-thuc-hien-de-tai.md). Phase sau chỉ bắt đầu khi phase trước đã kết thúc và con người đã xác nhận (mục 0.4).

Đặc tả của mỗi phase gồm:

- **Mục tiêu** và **điều kiện bắt đầu**.
- **Task:** phần viết mã, theo thứ tự phụ thuộc. Mỗi task ghi tệp, cách làm, test bắt buộc và tiêu chí hoàn thành (DoD). Tiêu chí chung cho mọi task: lint, kiểm tra kiểu và test CPU đều đạt.
- **Bước chạy:** các lệnh chạy trên model và dữ liệu thật, theo thứ tự. Mỗi bước ghi rõ điều kiện để bắt đầu: sau task nào, sau điểm review nào, hoặc sau bước nào. Bước có **[dừng]** cần con người xác nhận trước khi chạy tiếp. Bước ghi dữ liệu dài có thể chạy nền trên GPU trong lúc viết mã các task chưa cần tới dữ liệu đó.
- **Điểm review** (mục 0.3) và **tiêu chí hoàn thành** của phase.

| Phase | Mục tiêu | Task | Kết thúc khi |
|---|---|---|---|
| P0. Nền tảng | Repo, config, dữ liệu, chia tập, chấm điểm | P0.T1–T10 | Danh sách ID đã khóa |
| P1. Engine và công cụ ghi | Sinh theo đoạn trên vLLM, tái lập baseline, ghi dữ liệu | P1.T1–T11 | Baseline khớp số công bố; kiểm định tương đương đạt; đã chọn dạng câu trả lời |
| P2. Thăm dò | Bộ ước lượng, bộ điều khiển, mô phỏng, các phương pháp cần cho G1 | P2.T1–T20 | Có quyết định ở cổng G1 |
| P3. Mô phỏng đầy đủ | Các phương pháp còn lại, biến thể ablation, chỉnh tham số, khóa cấu hình, E1–E4 | P3.T1–T6 | Có quyết định ở cổng G2 |
| P4. Chạy thật | Runner trên vLLM, lưới E5 | P4.T1–T5 | Có quyết định ở cổng G3 |
| P5. Mở rộng | Model thứ hai, tập sạch, tập ngoài miền, thành phần và kích thước lô (E6) | P5.T1–T2 | Mọi phần được giao của E6 có kết quả |
| P6. Hoàn thiện | Tái lập toàn bộ bảng và hình, đóng gói | P6.T1 | Gói tái lập sinh lại đúng mọi bảng và hình |

### P0. Nền tảng

**Mục tiêu:** repo chạy được test, có config, dữ liệu đã chia và khóa, chấm điểm đúng.

**Điều kiện bắt đầu:** máy có GPU đã cài driver và CUDA; `docs/` đã có các tài liệu ở mục 2.1.

**P0.T1. Khung repo.**
- Tệp: `pyproject.toml`, `src/ttalloc/__init__.py`, `AGENTS.md`, `docs/progress.md`, `docs/decisions.md`, `docs/test-log.md`, `.gitignore` (`data/`, `runs/`, `third_party/`), `tests/conftest.py` (đăng ký marker).
- Test: `test_import_cpu_modules_without_vllm`: chặn import `vllm`, rồi import mọi module trừ `engine/vllm_generator.py`.

**P0.T2. Config.**
- Tệp: `config.py`, `configs/base.yaml`, `configs/models/*.yaml`, `configs/datasets/*.yaml`.
- Các mô hình pydantic: `ModelCfg`, `DatasetCfg`, `DecisionPointCfg`, `LogCfg`, `ProbeCfg`, `AnswerCfg`, `FeatureCfg`, `EstimatorCfg`, `PolicyCfg`, `GuardCfg`, `SimCfg`, `OnlineCfg`, `ExperimentCfg`. Giá trị mặc định theo mục 9.1. Khóa chỉ dùng ở phase sau, ví dụ `baseline.*` hay `batch`, được thêm khi làm task tương ứng.
- Kiểm tra hợp lệ: `log.m <= dp.m`; `log.l_max <= dp.l_max`; `guard.min_segment >= log.l_max`; `policy.horizons` là tập con của `estimator.h_train`; $b_{\max} \le L_{\text{log}} - c_{\text{ans,max}}$ (kiểm khi đã có tokenizer).
- Test: `test_config_extends_and_override`, `test_config_hash_stable_under_key_order`, `test_config_validators_reject_bad_values`.

**P0.T3. Thư mục chạy.**
- Tệp: `runs.py`. Theo mục 2.4.
- Test: `test_rundir_unique_and_never_overwritten`, `test_meta_has_git_and_versions`.

**P0.T4. Kiểu dữ liệu và schema.**
- Tệp: `types.py` (các lớp ở mục 3.2–3.4 và `Costs`), `logging/schema.py` (schema pyarrow của mục 4.2–4.5).
- Test: `test_costs_total_is_sum_of_five`, `test_schema_roundtrip`.

**P0.T5. Dữ liệu tổng hợp cho test.**
- Tệp: `tests/synth.py`. Hàm `make_logs(n_questions, seeds, rng, **knobs)` trả về hai bảng `traces` và `points` đúng schema, có đủ các trường hợp: chuỗi tự kết thúc, chuỗi chạm $L_{\text{log}}$, chuỗi có đoạn dài không có từ chuyển ý.
- Test: `test_synth_conforms_to_schema`, `test_synth_deterministic`.

**P0.T6. Bộ nạp dữ liệu.**
- Tệp: `data/loaders.py`, `configs/datasets/*.yaml`. Mỗi tập trả về danh sách `Question` (mục 4.1). `qid` phải ổn định qua các lần nạp.
- Test (dùng tệp mẫu offline trong `tests/fixtures/`): `test_loader_fields_<dataset>`, `test_qid_stable`.

**P0.T7. Chia tập.**
- Tệp: `data/splits.py`; lệnh `ttalloc splits build` và `ttalloc splits verify`.
- `build` sinh `splits/<tên>.json` và `splits/<tên>.sha256` theo `configs/splits.yaml`. `verify` được gọi ở đầu mọi pipeline và dừng ngay nếu mã băm không khớp.
- Test: `test_splits_disjoint`, `test_splits_deterministic`, `test_dedup_detects_planted_duplicate`, `test_verify_detects_tampering`.
- DoD: các tệp `splits/` đã sinh và commit. **Dừng chờ xác nhận khóa danh sách ID.**

**P0.T8. Lời nhắc.**
- Tệp: `data/prompts.py`, `configs/prompts/*.yaml`. Hàm `build_prompt_ids(question, model_cfg, tokenizer) -> list[int]`. Với R1-Distill: dùng chat template và bảo đảm tiền tố kết thúc bằng `<think>\n`. Với Qwen3: `enable_thinking=True`. Nội dung lời nhắc theo mục 9.2. Với câu trắc nghiệm, xáo lựa chọn theo mục 4.1.
- Test: `test_r1_prompt_ends_with_think`, `test_qwen3_prompt_has_thinking`, `test_choice_prompt_lists_options`, `test_choice_shuffle_deterministic_and_gold_remapped`.

**P0.T9. Chấm điểm.**
- Tệp: `data/grading.py`. Các hàm:
  - `extract_boxed(text) -> str | None`: lấy khung `\boxed{...}` cuối cùng, xử lý ngoặc lồng nhau, chấp nhận `\boxed 5` và `\fbox{}`.
  - `grade(pred, gold, answer_type) -> bool`: toán dùng `math_verify.parse` và `verify`; trắc nghiệm trích chữ cái.
  - `canonical_key(ans) -> str` và `equivalent(a, b) -> bool` (có cache).
  - `classify_answer(ans) -> integer | fraction | expression | choice | other`.
- Test dạng bảng: khung lồng nhau, nhiều khung (lấy khung cuối), không có khung, ngoặc không cân; `0.5` tương đương `\frac12` và `1/2`; `2` khác `3`; trích `\boxed{B}` và `Answer: (C)`; phân loại dạng đáp án.

**P0.T10. Công cụ kiểm tra chấm điểm.**
- Lệnh `ttalloc audit-grading --run <thư mục chạy> --n 100`, hoặc `--model --split` để lấy từ dữ liệu ghi: lấy mẫu ngẫu nhiên, xuất CSV gồm đề, đáp án chuẩn, đáp án trích ra và kết luận, để chấm tay.
- Test: trên một thư mục chạy tổng hợp, CSV có đủ cột và đúng số dòng.

**Bước chạy:**

1. Sau P0.T1: cài nhóm phụ thuộc `gpu` trên máy GPU; ghi phiên bản vLLM đã chọn và lý do vào `docs/decisions.md`; kiểm tra `import vllm`.
2. Sau P0.T6: tải model Qwen3-1.7B và các tập ở mục 4.1. Chốt `engine.dtype` (mục 9.1) theo GPU sẽ dùng để ghi; ghi ADR.
3. Sau P0a: `ttalloc splits build` → **[dừng]** khóa danh sách ID.

**Điểm review:** P0a, sau P0.T10.

**Tiêu chí hoàn thành:** P0a đạt; danh sách ID đã khóa và commit; `import vllm` chạy được trên máy GPU.

### P1. Engine và công cụ ghi

**Mục tiêu:** sinh chuỗi suy luận theo đoạn trên vLLM, có nhánh gợi đáp án thử; có số liệu tham chiếu của baseline; có công cụ ghi dữ liệu cho bộ ước lượng và bộ mô phỏng.

**Điều kiện bắt đầu:** P0 đã kết thúc.

**P1.T1. Generator và ScriptedGenerator.**
- Tệp: `engine/generator.py`, `engine/scripted.py`. Theo mục 3.3.
- Test: `test_to_sampling_kwargs_never_sets_min_tokens`, `test_scripted_respects_max_tokens`, `test_scripted_stop_strings_and_token_ids`, `test_scripted_stop_inside_longer_token`, `test_scripted_eos`.

**P1.T2. Xác định điểm sau một đoạn.**
- Tệp: `engine/cut.py`. Theo A1a, cộng hàm `count_markers(text) -> int` (từ chuyển ý ở đầu đoạn, dùng cùng `MARKER_RE`).
- Test (tokenizer trong `tests/fixtures/`):
  - `test_cut_before_marker`: tiền tố tại `pos` giải mã ra văn bản kết thúc bằng `\n\n`, phần sau bắt đầu bằng từ chuyển ý.
  - `test_cut_with_merged_newline_token`: trường hợp token gộp như `.\n\n`.
  - `test_marker_inside_sentence_is_not_point`.
  - `test_butter_is_not_marker`: trả về `RESUME`, không tạo điểm.
  - `test_early_marker_returns_resume`: từ chuyển ý đến khi số token trong đoạn chưa vượt `min_tokens`.
  - `test_newline_cutback_window`, `test_think_end_natural`, `test_eos_without_think`.

**P1.T3. LiveSource.**
- Tệp: `engine/live_source.py`. Lưu cho mỗi khóa: `prompt_ids`, `chain_ids`, `chain_logprobs`, điểm hiện tại, bộ đếm từ chuyển ý, bộ theo dõi `local_conf_min`.
  - `advance`: theo A1a, với `logprobs=0` nếu bật `features.local_conf`; gửi lại các câu nhận `RESUME` trong cùng lời gọi.
  - `probe`: theo A1b.
  - `answer`: theo A1c.
  - Prompt của mọi request bắt đầu bằng `prompt_ids` của đề bài.
- Test (CPU, với ScriptedGenerator):
  - `test_points_match_script`, `test_probe_does_not_change_chain` (I7), `test_cost_categories_match_generated_tokens` (I2);
  - `test_early_marker_resubmits`, `test_rejected_marker_resubmits`, `test_probe_stops_when_box_closes` (đáp án có ngoặc lồng nhau);
  - `test_natural_end_answer_uses_suffix`, `test_eos_forces_answer`, `test_answer_at_start_point`, `test_answer_cost_never_exceeds_c_ans_max`.

**P1.T4. VllmGenerator và test GPU.**
- Tệp: `engine/vllm_generator.py` (import lười).
- Test (`gpu`, Qwen3-1.7B):
  - `test_segments_concat_equals_chain`: ghép các đoạn ra đúng chuỗi token; không lặp BOS hay chat template.
  - `test_advance_resubmits_until_valid_point`: mọi điểm trả về hoặc là từ chuyển ý hợp lệ thỏa điều kiện khoảng cách, hoặc là điểm theo độ dài.
  - `test_probe_does_not_modify_chain`, `test_token_counts_match_engine`, `test_max_tokens_exact`, `test_natural_end_detected`.
  - `test_probe_confidence_matches_manual`: độ tự tin tính từ logprob trả về khớp với tính tay.

**P1.T5. Tái lập baseline.**
- Tệp: `baseline/full.py`, `baseline/deer_official.py`, `baseline/throughput.py`; lệnh `ttalloc baseline full|deer|throughput`.
- `full`: sinh liền mạch qua `VllmGenerator`, không gợi, không dừng sớm. Prompt lấy từ `build_prompt_ids`; lấy mẫu theo mục 9.1; `max_tokens` bằng $L_{\text{log}}$; chỉ dừng ở EOS. Đáp án là khung `\boxed{}` cuối cùng của đầu ra, chấm bằng `grade`. Ghi theo từng câu và seed: độ dài, đáp án trích ra, kết quả chấm. Ghi tổng hợp: độ chính xác, độ dài trung bình và trung vị. Hàm sinh liền mạch được dùng lại ở P1.T6 và P4.T5.
- `deer`: chạy repo DEER chính thức bằng subprocess, trên cùng câu hỏi, model và tham số lấy mẫu. Repo được clone vào `third_party/DEER` ở commit ghi trong config (`baseline.deer_commit`). Đọc đầu ra, chấm lại bằng `grade`; ghi độ chính xác, số token và thời gian theo cùng định dạng với `full`.
- `throughput`: sinh liền mạch trên một tập con của `val` với các kích thước lô 8, 32, 128 và 256; ghi số token mỗi giây.
- Tham số `--profile`: `log` (mặc định) lấy mẫu theo mục 9.1 với `max_tokens` bằng $L_{\text{log}}$; `deer_paper` dùng cấu hình của bài DEER để đối chiếu số công bố: giải mã greedy (temperature 0), `max_tokens` 16384, cùng lời nhắc hướng dẫn ở mục 9.2. `full` và `deer` trong cùng một lần đối chiếu phải dùng cùng profile.
- Tham số `--datasets` giới hạn các tập nguồn trong một split, ví dụ chỉ MATH-500 và AIME 2024 của `test_main`.
- Khi tập là `test_*`, cả ba lệnh đòi tham số `--approved-by <tên>` và ghi một dòng vào `docs/test-log.md` (mục 7.5).
- Test: `test_deer_output_parser` (tệp mẫu trong `tests/fixtures/`), `test_baseline_refuses_test_split_without_approval`; `gpu`: `test_baseline_full_smoke` (Qwen3-1.7B, 8 câu của `val`).

**P1.T6. Kiểm định tương đương.**
- Test (`slow`, `gpu`): 400 câu của `val` × 2 seed (không dùng tập kiểm tra), sinh liền mạch (hàm của P1.T5) và sinh theo đoạn có nhánh phụ. Đạt khi khoảng tin cậy bootstrap 90% của chênh lệch nằm trọn trong biên tương đương: độ chính xác ±3 điểm phần trăm, độ dài trung bình ±5%. Nếu khoảng quá rộng để kết luận, tăng lên 4 seed. Kết quả ghi vào `runs/`.

**P1.T7. Lưu trữ.**
- Tệp: `logging/store.py`. Ghi theo lô vào tệp tạm rồi đổi tên (ghi nguyên tử). Có tệp kê khai các khóa `(qid, seed)` đã hoàn tất; hàm `completed_keys()` dùng để chạy tiếp sau khi bị ngắt.
- Test: `test_atomic_write`, `test_completed_keys_after_partial_run`.

**P1.T8. Logger.**
- Tệp: `logging/logger.py`; lệnh `ttalloc log --model --split --seeds 0-3 [--limit N]`.
- Cách làm:
  - Hàng đợi các khóa `(qid, seed)` chưa hoàn tất; tối đa `log.window` chuỗi chạy đồng thời.
  - Với mỗi chuỗi mới: ghi điểm `start` (một câu trả lời ép buộc tại `pos = 0`).
  - Mỗi vòng: `advance` mọi chuỗi đang chạy với `min_tokens = log.m`, `max_tokens = log.l_max`, `stop_at_markers=True` (A1a). Tại mọi điểm mới: gợi đáp án thử (A1b) và sinh câu trả lời ép buộc (A1c). Chuỗi tự kết thúc bằng `</think>` hoặc EOS: sinh câu trả lời theo A1c, với cùng `r_ans`. Chuỗi chạm $L_{\text{log}}$: đánh dấu `max_len`.
  - Ghi `log_hash` (I9) vào mọi dòng của `traces`.
- Test (CPU, với ScriptedGenerator): `test_logger_end_to_end`, `test_logger_resume_no_duplicates` (giả lập dừng giữa chừng), `test_start_point_present`.

**P1.T9. Chọn dạng câu trả lời.**
- Lệnh `ttalloc study answer-mode --model --n 100`, chạy trên một tập con của `val`. Tại mỗi điểm và tại điểm tự kết thúc, sinh cả dạng `full` và dạng `short` (mục 9.2), rồi báo cáo riêng cho hai loại vị trí: tỉ lệ cùng kết quả chấm, tỉ lệ không trích được đáp án và chi phí trung bình.
- Đề xuất: chọn `short` nếu tỉ lệ cùng kết quả chấm ít nhất 98%. Ghi ADR. **Dừng chờ xác nhận.**

**P1.T10. Đo nhiễu do lời nhắc gợi.**
- Lệnh `ttalloc study probe-noise --model --n 100`: tại cùng điểm, gợi bằng hai mẫu lời nhắc; báo cáo tỉ lệ đáp án khác nhau theo vị trí tương đối trong chuỗi.

**P1.T11. Kiểm tra dữ liệu ghi.**
- Tệp: `logging/validate.py`; lệnh `ttalloc validate-logs`.
- Kiểm tra: đúng schema; điểm sắp theo `gen_end`; `pos <= gen_end`; khoảng cách giữa hai điểm liên tiếp không quá `log.l_max`; số token khớp; tỉ lệ chuỗi chạm $L_{\text{log}}$; tỉ lệ không trích được đáp án; mọi chuỗi có điểm `start`.
- Test: `test_validate_detects_injected_errors` (mỗi loại lỗi một trường hợp).

**Bước chạy:**

1. Sau P1a: **[dừng]** xin duyệt dùng `test_main`, rồi chạy `baseline full` và `baseline deer` với `--profile deer_paper` cho Qwen3-1.7B: MATH-500 một lượt, AIME 2024 bốn lượt (cả hai thuộc `test_main`). So với dòng Qwen3-1.7B ở bảng 2 của bài DEER. **[dừng]** nếu trên MATH-500 lệch quá 2 điểm phần trăm; AIME 2024 chỉ báo cáo.
2. Sau P1a: `baseline throughput` với Qwen3-1.7B, trên GPU sẽ dùng để ghi dữ liệu.
3. Sau bước 1: `audit-grading --n 100` trên đầu ra của bước 1 → **[dừng]** con người chấm tay; đạt khi khớp ít nhất 99%.
4. Sau P1b: ghi thử 100 câu `val` × 2 seed; `study answer-mode`, `study probe-noise` → **[dừng]** chọn dạng câu trả lời.
5. Sau bước 4: `validate-logs` trên dữ liệu ghi thử; báo cáo tỉ lệ giữa tổng số token sinh khi ghi (gồm gợi và trả lời) và số token chuỗi chính.

**Điểm review:** P1a, sau P1.T6; P1b, sau P1.T11.

**Tiêu chí hoàn thành:** P1a và P1b đạt; baseline khớp số công bố; chấm tự động khớp chấm tay ít nhất 99%; kiểm định tương đương đạt; dạng câu trả lời đã chọn và có ADR.

### P2. Thăm dò

**Mục tiêu:** có bộ ước lượng, bộ điều khiển, bộ mô phỏng và các phương pháp cần cho cổng G1; kiểm tra ba giả định ở [mục 14 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L334) trước khi cài đặt đầy đủ.

**Điều kiện bắt đầu:** P1 đã kết thúc.

**P2.T1. Sự kiện và dãy chuẩn.**
- Tệp: `features/events.py`. Theo A2 và A3.
- Test trên dữ liệu tổng hợp: thứ tự ưu tiên giữa tự kết thúc và từ chuyển ý; điều kiện `gen_end - s > min_tokens`; sự kiện `length` làm tròn xuống; sự kiện `truncated`; lỗi `SimValidityError` khi thiếu điểm.

**P2.T2. Đặc trưng.**
- Tệp: `features/extract.py`. Theo A5.
- Test với số liệu tính tay: hệ số góc theo đơn vị 1k token; độ nhất quán với cửa sổ $k$ và `n_recent`; `n_changes`; `local_conf_min` qua tổng tiền tố; mã hóa one-hot dạng đáp án; lần gợi đầu tiên.

**P2.T3. Bộ dữ liệu huấn luyện.**
- Tệp: `estimator/dataset.py`. Theo A4.
- Test: nhãn khi chuỗi tự kết thúc trước `lim`; nhãn khi chuỗi chạm $L_{\text{log}}$; tổng trọng số của mỗi chuỗi bằng 1; cột nhóm là `qid`.

**P2.T4. Huấn luyện và hiệu chuẩn.**
- Tệp: `estimator/train.py`. LightGBM nhị phân với đầu vào là đặc trưng và $h$. `GroupKFold(5)` theo `qid` để lấy dự đoán ngoài fold. Trong mỗi fold, early stopping dùng một phần tách theo nhóm (20% số câu) của phần huấn luyện, không dùng fold đang được dự đoán. Model cuối học trên toàn bộ `train`, với số cây bằng trung vị số cây tốt nhất của các fold. Isotonic regression riêng cho từng $h$, với `out_of_bounds="clip"`, học trên dự đoán ngoài fold.
- Lưu `data/estimators/<model>/<hash>/`: `model.txt`, `isotonic.pkl`, `meta.json` (tên đặc trưng, `h_train`, mã băm của dữ liệu ghi, config điểm quyết định, các độ đo ngoài fold).
- API: `Estimator.load(path)`; `predict(X: DataFrame, hs: list[int]) -> ndarray` có kích thước `(n, len(hs))`.
- Test: `test_group_folds_disjoint` (I5), `test_isotonic_monotone`, `test_predictions_in_unit_interval`, `test_reproducible_with_seed`, `test_artifact_roundtrip`.

**P2.T5. Lợi ích trên mỗi token.**
- Tệp: `estimator/rho.py`. Theo A6, cộng hàm `estimate_omega(logs, dp)`.
- Test: `test_rho_worked_example` dùng số liệu ở [mục 9 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L265) với `horizons = [1000, 2000, 4000]` và $\omega = 0$: $\rho$ lần lượt là 1; 22,5; 10; 0,5 điểm phần trăm trên 1.000 token, và $h^*$ lần lượt là 1000, 2000, 4000, 2000. Thêm: lợi ích âm cho $\rho \le 0$; tầm bị lọc theo $b_{\max}$; không còn tầm nào thì trả về `(-inf, None)`.

**P2.T6. Đánh giá bộ ước lượng (E1).**
- Tệp: `estimator/evaluate.py`; lệnh `ttalloc estimator eval --split val`.
- Độ đo trên `val`: AUROC của $\hat q(0)$ so với $y(0)$; ECE (15 nhóm cùng số mẫu) và điểm Brier cho từng $h$; dữ liệu cho biểu đồ độ tin cậy; tương quan Spearman giữa $\hat q(h) - \hat q(0)$ và lợi ích thực nghiệm tính từ các seed khác của cùng câu; mức quan trọng của đặc trưng.
- Test: ECE gần 0 trên dữ liệu đã hiệu chuẩn hoàn hảo; AUROC bằng 1 trên dữ liệu tách rời hoàn toàn.

**P2.T7. Quỹ và bộ bảo vệ ngân sách.**
- Tệp: `controller/budget.py`. Theo A8.
- Test (hypothesis): với đầu vào ngẫu nhiên, tổng chi tiêu xấu nhất của các phần đã cấp cộng phần dự trữ không vượt `remaining`; không có hạn mức âm hoặc nhỏ hơn `guard.min_segment`; câu chưa có $\rho$ được cấp trước câu đã có $\rho$; trong nhóm đã có $\rho$, câu có $\rho$ cao được ưu tiên; khi ngân sách rất nhỏ thì mọi câu dừng; `InfeasibleBudget` khi `available() < 0` lúc bắt đầu, tính cả phần dự trữ của hàng chờ.

**P2.T8. Runner.**
- Tệp: `controller/runner.py`. Theo A9.
- Test (dùng LoggedSource trên dữ liệu tổng hợp, với nhiều chính sách ngẫu nhiên):
  - `test_budget_never_exceeded` (I1, hypothesis);
  - `test_every_question_answered_once` (I3);
  - `test_cost_categories_sum` (I2);
  - `test_sliding_window_reserve`;
  - `test_sliding_window_newcomers_not_starved`: M2 với `max_active < N`; mọi câu mới nạp đều được cấp đoạn đầu tiên khi phần dự trữ của nó đủ cho `guard.min_segment`;
  - `test_deterministic` (I6).

**P2.T9. LoggedSource.**
- Tệp: `sim/logged_source.py`. `advance` dùng `next_event` (A2). `probe` và `answer` đọc từ bảng `points` và `traces`; chi phí lấy đúng số token đã ghi. Kiểm tra `log_hash` khớp config hiện hành (I9).
- Test: kết quả trên chuỗi tổng hợp có cấu trúc biết trước; câu trả lời tại điểm `start`; từ chối chạy khi `log_hash` khác.

**P2.T10. Bảng tính sẵn và các RhoProvider.**
- Tệp: `sim/table.py` (lệnh `ttalloc table build`), `sim/providers.py`.
- `TableRhoProvider` chỉ dùng giá trị trong bảng khi `state.visited` trùng với tiền tố của dãy chuẩn; nếu không, tính lại bằng bộ ước lượng.
- `CurveRhoProvider` (M3): $q_i(b)$ là tỉ lệ đúng khi dừng tại điểm cuối cùng của dãy chuẩn có `gen_end <= b`, tính trung bình trên các seed **khác** seed đang mô phỏng.
- Test: `test_table_equals_estimator_on_canonical_path`, `test_table_falls_back_when_path_differs`, `test_curve_excludes_current_seed`.

**P2.T11. Chính sách B2, M2, M3.**
- Tệp: `policies/base.py` (registry), `deer.py` (B2; tham số `cap` để P5.T1 dùng lại cho B1), `price.py` (M2, M3). Ngữ nghĩa theo mục 7.2.
- Test:
  - `test_m2_worked_example_round1`: số liệu ở mục 9 của báo cáo phương pháp chi tiết, với `horizons = [1000, 2000, 4000]`, `available = 5080`, $\omega = 0$, $\mu_{\min} = 0$ và $b_{\max}$ đủ lớn. Kỳ vọng: $\mu = 10$ điểm phần trăm trên 1.000 token (tức $10^{-4}$ trong mã); câu 2 và câu 3 tiếp tục; câu 1 và câu 4 dừng.
  - `test_m2_worked_example_round2`: chỉ còn câu 2 và câu 3, `available = 10000`. Câu 2 có $\hat q$ = 0,88; 0,88; 0,87; 0,86 (với $h$ = 0, 1000, 2000, 4000) nên $\rho = 0$ và dừng. Câu 3 có $\hat q$ = 0,15; 0,35; 0,50; 0,70 nên $\rho = 20$ với $h^* = 1000$; tổng cộng dồn không vượt quỹ nên $\mu = 0$, và câu 3 tiếp tục.
  - `test_ties_broken_by_seeded_hash`: hai câu cùng $\rho$; thứ tự đổi theo `run_seed` và không phụ thuộc `qid`.
  - `test_deer_stops_at_threshold`, `test_deer_qwen3_requires_think_end`, `test_allowance_is_on_total_cost` (hạn mức tính trên $T_i$, theo A11).
  - `test_non_anticipation` (I4): chạy chính sách trên dữ liệu đã cắt sau điểm $j$, quyết định tới điểm $j$ phải giống hệt khi có đủ dữ liệu.

**P2.T12.** *(Để sau, chuyển sang P5.T1.)* Oracle O1.

**P2.T13. Hiệu chỉnh theo ngân sách cho B2.**
- Tệp: `sim/tune.py`. Theo A11: hạn mức theo câu tính trên $T_i$; chia đôi tìm hạn mức chung của B2. Phần cho B0, B1, M1 và B4 làm ở P5.T1; tìm kiếm ngẫu nhiên làm ở P3.T4.
- Test: hiệu chỉnh đạt mục tiêu trên dữ liệu tổng hợp; trường hợp không đạt được mục tiêu thì dùng cận và ghi `budget_underused`.

**P2.T14. Thành phần lô.**
- Tệp: `data/batches.py`, `configs/batches.yaml`. Hàm `build_batch(questions, spec, rng) -> list[str]` trả về danh sách `qid` của một lô.
- `spec` gồm `composition`, `n` và `sources` (cho `mix`):
  - `all`: lấy mẫu đều `n` câu của tập; `n` bằng cỡ tập thì lấy toàn bộ.
  - `balanced`, `easy`, `hard`, `u`: chỉ dùng câu có `difficulty` (MATH). Tỉ lệ theo mức 1 đến 5: `balanced` mỗi mức 20%; `easy` mức 1 và mức 2 mỗi mức 35%, ba mức còn lại mỗi mức 10%; `hard` mức 4 và mức 5 mỗi mức 35%, ba mức còn lại mỗi mức 10%; `u` mức 1 và mức 5 mỗi mức 35%, ba mức còn lại mỗi mức 10%. Mặc định `n = 100`. Báo lỗi nếu một mức không đủ câu.
  - `mix`: `sources` là số câu lấy từ mỗi tập nguồn, ví dụ `{aime_2024: 30, aime_2025: 30, math500: 100}`.
- Trong mô phỏng, mỗi lần rút dựng lại lô bằng seed của lần rút, nên mọi phương pháp dùng chung lô ở cùng lần rút. Chạy thật dựng lô một lần bằng `batch.seed`.
- Test: `test_batch_level_proportions`, `test_batch_deterministic`, `test_batch_no_duplicates`, `test_batch_raises_when_level_short`, `test_mix_batch_counts_per_source`.

**P2.T15. Chạy mô phỏng.**
- Tệp: `sim/run.py`; lệnh `ttalloc sim run --exp <yaml>`. Duyệt lưới mô hình × tập × lô × $\alpha$ × phương pháp × lần rút. Lô dựng theo P2.T14 (khóa `batch` trong config thí nghiệm). Mỗi lần rút chọn một seed cho mỗi câu; mọi phương pháp dùng chung các lần rút. Chạy song song theo lần rút. Ghi các tệp ở mục 4.5.
- Hiệu năng: một lần chạy (một phương pháp, một $\alpha$, một lần rút) với $N = 500$ mất dưới 1 giây trên CPU.
- Test: `test_same_seed_same_output_hash` (I6), `test_draws_paired_across_methods`.

**P2.T16. Độ đo.** Tệp `eval/metrics.py`; định nghĩa ở mục 7.3. Test trên dữ liệu nhỏ tính tay.

**P2.T17. Thống kê.** Tệp `eval/stats.py`; theo A12. Test: McNemar khớp `statsmodels`; `slow`: độ phủ của khoảng bootstrap nằm trong 95 ± 3% qua 500 lần lặp tổng hợp.

**P2.T18. Báo cáo.** Tệp `eval/report.py`; lệnh `ttalloc report --runs ... --out reports/`. Sinh bảng (Markdown, CSV, LaTeX) và hình (độ chính xác theo $\alpha$, biểu đồ độ tin cậy, phân bổ token theo độ khó) chỉ từ dữ liệu trong `runs/`. Test: chạy trên một thư mục chạy tổng hợp.

**P2.T19. Cổng G1 và G2.** Tệp `eval/gates.py`; lệnh `ttalloc gate g1|g2 --runs ...`. Xuất `gate_<tên>.json` (đạt hay không, từng tiêu chí, số liệu) và một tóm tắt Markdown. Tiêu chí ở mục 7.4. Test trên số liệu tổng hợp đạt và không đạt.

**P2.T20. Phân tích thăm dò.**
- Tệp: `eval/curves.py`; lệnh `ttalloc study curves --model --split val`.
- Với mỗi câu, tính đường cong thực nghiệm $q_i(b)$ như `CurveRhoProvider`, nhưng lấy trung bình trên mọi seed, với $b$ thuộc $\{0\}$, `estimator.h_train` và $L_{\text{log}}$.
- Báo cáo theo nhóm độ khó (mục 7.3):
  - tỉ lệ câu được lợi khi suy luận thêm: $q_i(L_{\text{log}}) - q_i(0) \ge \delta$;
  - tỉ lệ câu bị giảm độ chính xác khi suy luận thêm: $\max_b q_i(b) - q_i(L_{\text{log}}) \ge \delta$;
  - phân phối của $b$ nhỏ nhất mà $q_i(b) \ge 0{,}9 \max_b q_i(b)$;
  - hình đường cong trung bình theo nhóm độ khó.
- $\delta$ là `study.curves_delta` (mục 9.1).
- Test: trên dữ liệu tổng hợp có đường cong biết trước, các tỉ lệ khớp với tính tay.

**Bước chạy:**

1. Khi bắt đầu phase: ghi dữ liệu thăm dò với Qwen3-1.7B, gồm `train` 500 câu × 4 seed và `val` × 8 seed; `validate-logs`. Bước này chạy nền trong lúc viết mã P2.T1–T6.
2. Sau P2a và bước 1: `estimator train`, `estimator eval` (E1 trên `val`).
3. Sau P2c và bước 2: `table build` cho `val`; `sim run --exp g1` với B2, M2, M3 trên toàn bộ `val` và trên các lô `balanced`, `easy`, `hard`, `u` dựng từ phần MATH của `val`; `study curves --split val`.
4. Sau bước 3: `gate g1` → **[dừng]**.

**Điểm review:** P2a, sau P2.T6; P2b, sau P2.T15; P2c, sau P2.T20.

**Tiêu chí hoàn thành:** P2a, P2b và P2c đạt; có báo cáo cổng G1 và quyết định của con người.

### P3. Mô phỏng đầy đủ

**Mục tiêu:** có các biến thể cho ablation; chỉnh tham số cho B2, M2, M3 và khóa cấu hình; có kết quả E1–E4 của model chính trên tập kiểm tra. Các phương pháp còn lại ở mục 7.2 làm ở P5.T1.

**Điều kiện bắt đầu:** con người đã quyết định sau cổng G1. Nếu phạm vi bị thu hẹp, đặc tả của phase này được cập nhật trước khi làm.

**P3.T1.** *(Để sau, chuyển sang P5.T1.)* Chính sách B3, B4, M1.

**P3.T2.** *(Để sau, chuyển sang P5.T1.)* Chạy lần lượt (B5).

**P3.T3. Biến thể cho ablation.**
- Các khóa config cho E4 (mục 7.3). Giá trị mặc định của mỗi khóa là cấu hình chính.
  - `answer.on_stop`: `forced` (mặc định, A1c), `best_probe` hoặc `vote`. `best_probe` lấy đáp án thử có `conf_geo` cao nhất trong các điểm đã ghé. `vote` bỏ phiếu theo khóa chuẩn hóa, trọng số là `conf_geo`. Hai biến thể này không sinh câu trả lời, nên chi phí trả lời bằng 0; đáp án được chấm bằng `grade`. Câu chưa có lần gợi nào thì dùng `forced`. Câu tự kết thúc vẫn dùng câu trả lời tự nhiên. Chỉ dùng trong mô phỏng.
  - `features.drop_groups`: các nhóm đặc trưng bị bỏ. Nhóm `consistency` gồm `consistency`, `n_recent`, `n_changes`; nhóm `trend` gồm `conf_slope_per_1k`, `has_slope`; nhóm `local_conf` gồm `local_conf_min`; nhóm `ans_type` gồm `ans_len` và các cột `ans_type_*`.
  - `estimator.calibrate`: `false` thì dùng thẳng xác suất của LightGBM, bỏ isotonic.
  - `estimator.kind`: `lightgbm` hoặc `logreg`. `logreg` học một logistic regression riêng cho từng $h$ trên đặc trưng đã chuẩn hóa; giá trị thiếu thay bằng trung vị trên `train`, kèm cột chỉ báo; hiệu chuẩn như LightGBM.
  - Một tầm nhìn: `policy.horizons = [1024]`. Bỏ giá sàn: `policy.mu_min = 0`. Bỏ trần: `policy.b_max_mult = null`, tức $b_{\max} = L_{\text{log}} - c_{\text{ans,max}}$.
  - Độ nhạy theo `dp.l_max`, `dp.m`, `features.k` và `dp.b_min` chỉ cần ghi đè config. Bộ ước lượng và bảng tính sẵn được học lại theo mã băm (mục 4.4).
- Test: `test_on_stop_best_probe_picks_max_conf`, `test_on_stop_vote_weighted`, `test_on_stop_falls_back_without_probe`, `test_drop_groups_removes_columns`, `test_uncalibrated_estimator_skips_isotonic`, `test_logreg_estimator_same_api`.

**P3.T4. Chỉnh tham số.**
- Tệp: `sim/tune.py`; lệnh `ttalloc tune`. Tìm kiếm ngẫu nhiên: mỗi phương pháp có cùng số cấu hình (mặc định 50) trên `val`; mục tiêu là độ chính xác trung bình trên lưới $\alpha$. Chỉ chỉnh các tham số không đòi hỏi học lại bộ ước lượng: $\lambda$, $\mu_{\min}$, $b_{\max}$. Các tham số cấu trúc ($m$, $L_{\max}$, $b_{\min}$) chỉ xét trong phân tích độ nhạy, có học lại bộ ước lượng. `answer.on_stop` không được chỉnh: mọi phương pháp dùng `forced` theo quy tắc công bằng ở mục 7.2; hai biến thể còn lại chỉ xuất hiện trong E4.
- Test: tìm kiếm ngẫu nhiên tái lập được.

**P3.T5. Khóa cấu hình và đánh giá trên tập kiểm tra.**
- Lệnh `ttalloc lock --exp <yaml>`: ghi config đã phân giải vào `configs/locked/` và thêm một dòng (ngày, mã băm, thí nghiệm) vào `docs/test-log.md`.
- Lệnh `ttalloc eval-test --exp <yaml> --approved-by <tên>`: từ chối chạy nếu config chưa được khóa, mã băm không khớp hoặc thiếu người duyệt. Sau các kiểm tra đó, lệnh dựng bảng tính sẵn (P2.T10) cho tập kiểm tra nếu chưa có. Mọi lần chạy trên `test_*` đều đi qua lệnh này, trừ ngoại lệ ở mục 7.5.
- Test: `test_eval_test_refuses_unlocked`, `test_eval_test_refuses_hash_mismatch`.

**P3.T6. Config thí nghiệm E1–E4.** Tệp YAML trong `configs/experiments/` theo mục 7.3; mẫu ở mục 9.3. Config của E5 viết ở P4.T4; config của E6 viết ở P5.T2.

**Bước chạy:**

1. Khi bắt đầu phase: **[dừng]** duyệt GPU, rồi ghi nốt dữ liệu của Qwen3-1.7B: `train` tới đủ 1.500 câu × 4 seed (logger bỏ qua các khóa đã ghi ở P2) và `test_main` × 8 seed; `validate-logs`. Bước này chỉ sinh và ghi, không đánh giá chính sách, và chạy nền trong lúc viết mã P3.T3–T6.
2. Sau P3a và bước 1: học lại bộ ước lượng trên `train` đầy đủ với `lightgbm` và với `logreg`; chọn mô hình có log-loss trung bình theo các $h$ thấp hơn trên `val`; ghi ADR.
3. Sau bước 2: `table build` cho `val`; `tune` trên `val`; `lock` → **[dừng]** duyệt cấu hình khóa.
4. Sau bước 3: `eval-test` cho E1 (trên `test_main`), E2, E3 và E4; `report`; `gate g2` → **[dừng]**.

**Điểm review:** P3a, sau P3.T6.

**Tiêu chí hoàn thành:** P3a đạt; bảng và hình của E1–E4 sinh lại được bằng `ttalloc report`; có báo cáo cổng G2 và quyết định của con người.

### P4. Chạy thật

**Mục tiêu:** bộ phân bổ chạy thật trên vLLM, khớp với mô phỏng, có số đo chi phí hệ thống.

**Điều kiện bắt đầu:** con người đã quyết định sau cổng G2. Nếu G2 không đạt và M3 cũng không hơn B2, phase này chỉ làm ở mức minh chứng khả thi: P4.T1–T3 và phần lưới E5 do con người chọn.

**P4.T1. Runner chạy thật.**
- Tệp: `online/run.py`, `online/timing.py`; lệnh `ttalloc online run --exp <yaml>`. Ghép Runner, `LiveSource` (với `VllmGenerator`), `EstimatorRhoProvider` và chính sách. Lô dựng theo P2.T14. Đo thời gian theo mục 4.5. Ghi cùng định dạng kết quả với mô phỏng. Kiểm tra mã băm lời nhắc và config (I9).

**P4.T2. Kiểm thử phát lại.**
- Test (CPU): dựng `ScriptedGenerator` phát lại đúng các chuỗi tổng hợp (chuỗi chính, đáp án thử và câu trả lời). Chạy Runner với `LiveSource` và với `LoggedSource` trên cùng dữ liệu.
- Điều kiện để hai bên trùng nhau hoàn toàn: dữ liệu tổng hợp có các từ chuyển ý cách nhau trong khoảng (`log.m`, `log.l_max`]; `dp.m = log.m`, `dp.b_min = log.m`, `dp.l_max = log.l_max`; ngân sách đủ lớn để bộ bảo vệ không rút ngắn đoạn nào. Khi đó quyết định, chi phí và kết quả phải giống hệt.
- Thêm một test với ngân sách nhỏ: chỉ so sánh những lần chạy mà mọi điểm của chạy thật đều là điểm đã ghi; mọi lần chạy đều phải thỏa I1.

**P4.T3. Smoke test GPU.**
- Test (`gpu`): Qwen3-1.7B, $N = 8$, ngân sách rất nhỏ và ngân sách vừa, với B2 và M2. Không vượt $B$; mọi câu đều có câu trả lời; số token theo từng loại khớp với engine.

**P4.T4. Lưới E5 và cổng G3.**
- Tệp `configs/experiments/e5_online.yaml` (E5, mục 7.3); lệnh `ttalloc gate g3`. Mỗi cấu hình chạy thật được so với dải chấp nhận của mô phỏng (A12). Mô phỏng dùng để so sánh phải có cùng $B$, cùng `max_active` và cùng thứ tự nạp với chạy thật.

**P4.T5. So thời gian với DEER chính thức.**
- Dùng `baseline deer` và `baseline full` (P1.T5) trên cùng câu hỏi và phần cứng với lưới E5, mỗi lệnh một lượt trên MATH-500; đọc thời gian, số token và độ chính xác, ghi cùng định dạng với `timing.parquet`.

**Bước chạy:**

1. Sau P4a: **[dừng]** duyệt GPU, rồi chạy lưới E5 qua `eval-test`; `gate g3` → **[dừng]**.
2. Sau P4a: so thời gian theo P4.T5.

**Điểm review:** P4a, sau P4.T5.

**Tiêu chí hoàn thành:** P4a đạt; có báo cáo cổng G3 và quyết định của con người.

### P5. Mở rộng

**Mục tiêu:** bổ sung các phương pháp so sánh để sau vào E2 và E3; E6: tập sạch, tập ngoài miền, thành phần và kích thước lô, và model thứ hai nếu con người chọn làm.

**Điều kiện bắt đầu:** con người đã quyết định sau cổng G3 và chỉ rõ những phần của phase này được làm. Riêng P5.T1 có thể bắt đầu ngay sau khi khóa cấu hình ở P3 (P3, bước chạy 3), vì chỉ chạy trong mô phỏng.

**P5.T1. Các phương pháp so sánh để sau.** Ngữ nghĩa theo mục 7.2. Làm theo thứ tự, mỗi nhóm có test riêng:
- **Nhóm 1 (B0, B1, M1, O1):**
  - Tệp: `policies/uniform.py` (B0); `policies/deer.py` (B1, dùng `cap: per_question_uniform`); `policies/price.py` (thêm M1); `sim/oracle.py` (O1, A10); trong `sim/tune.py`, phần A11 cho hạn mức $B/N$ của B0, B1 và giá cố định của M1.
  - Test: `test_b0_uniform_caps`; `test_m1_fixed_price_calibrated` (chi tiêu nằm trong khoảng mục tiêu của A11 trên dữ liệu tổng hợp); `test_oracle_matches_bruteforce` với $N \le 8$, chi phí và kết quả ngẫu nhiên, tập ứng viên gồm cả điểm `start`; `test_oracle_is_upper_bound`: trên cùng lần rút, O1 không thấp hơn bất kỳ chính sách nào đã cài; mở rộng `test_non_anticipation` cho B0, B1 và M1.
- **Nhóm 2 (B3, B4, B5), chỉ khi con người yêu cầu:**
  - Tệp: `policies/codestop.py` (B3), `policies/fifo_pool.py` (B4), `sim/sequential.py` (B5); trong `sim/tune.py`, phần A11 cho hạn mức gốc của B4.
  - Test: `test_fifo_pool_order`; `test_sequential_carryover`; test cho quy tắc CoDE-Stop theo ADR; mở rộng `test_non_anticipation` cho B3 và B4; mở rộng `test_oracle_is_upper_bound` cho mọi phương pháp đã cài.
- **Dynasor-CoT:** chỉ làm khi con người yêu cầu, sau khi đặc tả được thêm vào mục 7.2.
- Mọi phương pháp mới được chỉnh tham số theo P3.T4 với cùng số cấu hình như B2, M2 và M3, rồi khóa thành một config mới (P3.T5); config đã khóa của B2, M2, M3 giữ nguyên.

**P5.T2. Config thí nghiệm E6 và model thứ hai.**
- Tệp YAML trong `configs/experiments/` theo mục 7.3:
  - `test_clean` và `test_ood` với cấu hình đã khóa của từng model;
  - thành phần lô (`balanced`, `easy`, `hard`, `u`, `mix`) và kích thước lô $N \in \{32, 128, 500\}$ trên `test_main`;
  - nếu con người chọn làm model thứ hai: tệp `configs/models/<model>.yaml`; E2 với B2, M2, M3 (và các phương pháp đã có ở P5.T1); E4 với tập con ablation do con người chọn.
- Test: `test_experiment_configs_resolve`: mọi tệp của E1–E6 nạp được và qua kiểm tra hợp lệ.

**P5.T3. Hướng mở rộng (tùy chọn).** Chỉ làm khi con người yêu cầu, sau khi đặc tả bổ sung đã được thêm vào tài liệu này. Các hướng ở [mục 7.5 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L235).

**Bước chạy:**

1. Sau P5.T1 (từng nhóm): `table build` nếu thiếu; `tune` trên `val`; `lock` → **[dừng]** duyệt cấu hình khóa; `eval-test` cho E2 với các phương pháp mới; `report` để có E3 đầy đủ (mục 7.3).
2. Sau P5a: **[dừng]** duyệt GPU, rồi ghi `test_clean` và `test_ood` × 8 seed nếu chưa có; `validate-logs`; `eval-test` với cấu hình đã khóa.
3. Sau P5a: `eval-test` cho thành phần lô và kích thước lô trên `test_main` (mô phỏng, cấu hình đã khóa của Qwen3-1.7B).
4. Nếu con người chọn làm model thứ hai, sau P5a:
   - **[dừng]** duyệt GPU, rồi ghi thử 100 câu `val` × 2 seed và chạy `study answer-mode`, `study probe-noise`. Dùng cùng dạng câu trả lời với model chính nếu tỉ lệ cùng kết quả chấm vẫn đạt ít nhất 98%; nếu không thì **[dừng]**.
   - Ghi `train`, `val` và `test_main` theo số câu và seed ở mục 4.1; `validate-logs`.
   - Học bộ ước lượng và hiệu chuẩn riêng; `table build` cho `val`; `tune` trên `val`; `lock` → **[dừng]**.
   - `eval-test` cho E2 và E4.

**Điểm review:** P5a, sau P5.T1 nhóm 1 (và sau mỗi phần làm thêm của P5.T1–T3).

**Tiêu chí hoàn thành:** P5a đạt; mọi phần được giao của P5 có kết quả trong `runs/`, kể cả khi kết quả âm tính.

### P6. Hoàn thiện

**Mục tiêu:** sinh lại toàn bộ bảng và hình từ `runs/`; đóng gói mã nguồn và dữ liệu để tái lập.

**Điều kiện bắt đầu:** con người xác nhận đã chốt số liệu.

**P6.T1. Tái lập.** Lệnh `ttalloc reproduce --tables` sinh lại toàn bộ bảng và hình từ `runs/`. README mô tả pipeline đầy đủ theo các phase ở mục 6. Có tệp kê khai dữ liệu kèm mã băm.

**Bước chạy:**

1. Sau P6a: `reproduce --tables`; so từng bảng và hình với bản đã báo cáo.

**Điểm review:** P6a, sau P6.T1.

**Tiêu chí hoàn thành:** P6a đạt; mọi bảng và hình sinh lại khớp với bản đã báo cáo.

---

## 7. Vận hành thực nghiệm

### 7.1. Thứ tự chạy

Thứ tự chạy nằm trong phần **Bước chạy** của từng phase (mục 6). Mỗi bước ghi rõ điều kiện để bắt đầu; bước có **[dừng]** cần con người xác nhận trước khi chạy tiếp. Các bước ghi dữ liệu nối tiếp nhau như sau: ghi thử ở P1; dữ liệu thăm dò (`train` 500 câu, toàn bộ `val`) ở P2; phần còn lại của `train` và `test_main` ở P3; `test_clean`, `test_ood` và dữ liệu của model thứ hai (nếu làm) ở P5. Logger chạy tiếp từ các khóa đã ghi, nên không phải ghi lại.

### 7.2. Các phương pháp

Hiện tại chỉ cài B2 (baseline duy nhất), M2 (phương pháp đề xuất) và M3 (công cụ chẩn đoán cho cổng G1 và G2). Các phương pháp còn lại để sau, cài ở P5.T1.

| Mã | Chính sách | Ngữ nghĩa | Môi trường | Cài ở |
|---|---|---|---|---|
| B0 | `uniform` | Không gợi, không dừng sớm; hạn mức mỗi câu là $B/N$, tính trên $T_i$ (A11) | Mô phỏng | Để sau (P5.T1) |
| B1 | `deer` | Gợi tại điểm quyết định; dừng khi độ tự tin $\ge \lambda$; hạn mức mỗi câu là $B/N$ (A11) | Mô phỏng | Để sau (P5.T1) |
| B2 | `deer` | Như B1 nhưng hạn mức mỗi câu là $c$, hiệu chỉnh theo A11 để tổng chi tiêu bằng $B$. **Baseline chính** | Mô phỏng, chạy thật | P2.T11 |
| B3 | `codestop` | Quy tắc CoDE-Stop với trần nâng lên như B2. Đọc bài gốc (arXiv 2604.04930) và ghi quy tắc vào ADR trước khi cài | Mô phỏng | Để sau (P5.T1) |
| B4 | `fifo_pool` | Quy tắc DEER, hạn mức gốc $B/N$ (A11); token dư của câu đã dừng vào quỹ; câu chạm hạn mức được cấp thêm theo thứ tự chạm trước được phục vụ trước, mỗi lần một đoạn | Mô phỏng | Để sau (P5.T1) |
| B5 | `sequential` | Xử lý lần lượt theo một hoán vị cố định; câu thứ $k$ được $B/N$ cộng phần dư chuyển từ các câu trước; quy tắc DEER | Mô phỏng | Để sau (P5.T1) |
| M1 | `price_fixed` | Tiếp tục nếu $\rho > \max(\mu, \mu_{\min})$ và $b < b_{\max}$, với $\mu$ cố định hiệu chỉnh theo A11 | Mô phỏng | Để sau (P5.T1) |
| M2 | `price_adaptive` | A7. **Phương pháp đề xuất** | Mô phỏng, chạy thật | P2.T11 |
| M3 | `price_adaptive` + `CurveRhoProvider` | Như M2, nhưng $\hat q$ thay bằng đường cong thực nghiệm (cố ý dùng thông tin từ các seed khác) | Mô phỏng | P2.T11 |
| O1 | `oracle` | A10 (cố ý biết trước kết quả) | Mô phỏng | Để sau (P5.T1) |

- **Quy tắc công bằng:** mọi phương pháp dùng cùng điểm quyết định, cùng lời nhắc gợi, cùng dạng câu trả lời, cùng bộ bảo vệ ngân sách và cùng số cấu hình khi chỉnh tham số. Token gợi được tính cho mọi phương pháp có gợi.
- **DEER:** chế độ tính độ tự tin theo code chính thức (`deer.conf_mode`: trung bình cộng với R1-Distill, trung bình nhân với Qwen3). Với Qwen3, thêm điều kiện nhánh gợi phải sinh `</think>` (`deer.require_think_end_after_probe`).
- **Các chính sách dùng điểm quyết định** (B1–B4, M1–M3): `initial_request` là tiếp tục với `min_tokens = b_min`, `max_tokens = b_min + l_max`; các vòng sau là `min_tokens = m`, `max_tokens = l_max`.

### 7.3. Thí nghiệm và độ đo

| Mã | Nội dung | Cấu hình | Phase |
|---|---|---|---|
| E1 | Chất lượng bộ ước lượng | P2.T6 trên `val`; với `test_main` chỉ chạy sau khi khóa | P2 (`val`), P3 (`test_main`) |
| E2 | So sánh chính | B2, M2, M3 × $\alpha \in \{0{,}2;\ 0{,}3;\ 0{,}4;\ 0{,}5;\ 0{,}7;\ 1{,}0\}$ × `test_main`; 200 lần rút. Các phương pháp của P5.T1 được thêm vào sau, trên cùng các lần rút | P3, P5 |
| E3 | Phân rã | Hiện tại đọc E2 theo thứ tự B2 → M2 → M3; sau P5.T1 đọc đầy đủ B1 → B2 → M1 → M2 → M3 → O1 | P3, P5 |
| E4 | Ablation | Chỉ một tầm $H = \{1\text{k}\}$; bỏ từng nhóm đặc trưng; bỏ hiệu chuẩn; bỏ $\mu_{\min}$; bỏ $b_{\max}$; ba cách trả lời khi dừng; logistic regression thay LightGBM; độ nhạy theo $L_{\max}$, $m$, $k$, $b_{\min}$. Các khóa config ở P3.T3 | P3 |
| E5 | Chạy thật | Qwen3-1.7B × MATH-500 × $\alpha = 0{,}5$ × {B2, M2} × 3 seed: 6 lần. Nếu con người yêu cầu mở rộng: thêm $\alpha \in \{0{,}3;\ 0{,}7\}$ và lô trộn MATH-500 với AIME, tối đa 36 lần. Thời gian báo cáo kèm tên GPU | P4 |
| E6 | Tổng quát hóa | `test_clean`; `test_ood`; thành phần lô (`balanced`, `easy`, `hard`, `u`, `mix`; P2.T14); $N \in \{32, 128, 500\}$; model thứ hai nếu làm (P5.T2) | P5 |

- $C_{\text{full}}$ của một tập là tổng trên các câu của trung bình qua các seed: `chain_len` cộng chi phí câu trả lời tự nhiên. Nếu chuỗi chạm $L_{\text{log}}$ thì dùng câu trả lời ép buộc tại điểm cuối. Không tính chi phí gợi.

| Độ đo | Định nghĩa |
|---|---|
| Độ chính xác tại $B$ | Tỉ lệ câu đúng; luôn kèm tổng token đã dùng |
| AUC | Diện tích dưới đường độ chính xác theo $\alpha$, chuẩn hóa về [0, 1] |
| Token tiết kiệm ở cùng độ chính xác | So với B2, nội suy ngược trên đường độ chính xác theo token |
| Tỉ lệ cắt nhầm | Tỉ lệ câu trả lời sai trong khi câu trả lời tự nhiên của cùng chuỗi là đúng |
| Độ chính xác theo độ khó | Theo mức của MATH, hoặc theo tỉ lệ đúng của câu trả lời tự nhiên qua các seed: [0; 0,25), [0,25; 0,75), [0,75; 1] |
| Tỉ trọng chi phí gợi | (`probe_prompt` + `probe_output`) / tổng token |
| Chi phí hệ thống | Các trường của `timing.parquet` |

### 7.4. Cổng quyết định

| Cổng | Dữ liệu | Tiêu chí đạt |
|---|---|---|
| G1 | Cuối P2: `val`, dữ liệu thăm dò | (a) AUROC của $\hat q(0)$ ≥ 0,75; ECE ≤ 0,05 với mọi $h$; tương quan Spearman của lợi ích ≥ 0,3. (b) M3 hơn B2 ít nhất 2 điểm phần trăm ở ít nhất một $\alpha \le 0{,}5$, với cận dưới của khoảng tin cậy hiệu chỉnh Bonferroni ($K = 4$, A12) lớn hơn 0. (c) Ít nhất 95% chuỗi qua `validate-logs` |
| G2 | Cuối P3: `test_main`, cấu hình đã khóa | M2 hơn B2 ở ít nhất một $\alpha \le 0{,}5$, với khoảng tin cậy hiệu chỉnh Bonferroni ($K = 4$) không chứa 0 |
| G3 | Cuối P4: lưới E5 | (a) Không lần chạy nào vượt $B$. (b) Với $K = 2$ cấu hình (lưới tối thiểu): cả hai có trung bình 3 seed nằm trong dải chấp nhận (A12). Với lưới mở rộng: số cấu hình nằm ngoài dải không quá $\max(1, \lfloor 0{,}1 K \rfloor)$. (c) Ở mọi cấu hình, trung bình chạy thật lệch trung bình mô phỏng không quá 3 điểm phần trăm |

Lệnh `gate` chỉ báo đạt hay không và số liệu. Agent không tự quyết định đi tiếp, kể cả khi cổng đạt.

### 7.5. Kỷ luật với tập kiểm tra

- Mọi lần chạy trên `test_*` đi qua `ttalloc eval-test` với config đã khóa và có người duyệt.
- Ngoại lệ duy nhất là tái lập baseline ở P1.T5: lệnh `baseline` chạy trên `test_main` với `--approved-by` và ghi một dòng vào `docs/test-log.md`. Không tham số nào được chọn từ kết quả này.
- Không chỉnh tham số sau khi xem kết quả kiểm tra. Nếu bắt buộc phải sửa lỗi, khóa lại với mã băm mới, ghi lý do vào `docs/test-log.md`, và báo cáo cả hai kết quả.

### 7.6. Sai khác đã biết giữa mô phỏng và chạy thật

1. Sự kiện `length` trong mô phỏng được làm tròn xuống điểm đã ghi gần nhất (chênh tối đa `log.l_max` token); chạy thật dừng đúng ở hạn mức.
2. Dữ liệu ghi dùng khoảng cách `log.m`, chạy thật dùng `dp.m`. Chạy thật có thể dừng ở một từ chuyển ý không có trong dữ liệu ghi; mô phỏng sẽ chọn từ chuyển ý đã ghi kế tiếp.
3. Chạy thật sinh chuỗi mới, nên kết quả khác do lấy mẫu ngẫu nhiên.
4. Thời gian chạy, thời gian chờ đồng bộ và hiệu ứng KV cache không mô phỏng được.

Báo cáo kết quả phải nêu bốn điểm này.

---

## 8. Review

### 8.1. Quy trình tại một điểm review

Reviewer là một agent có ngữ cảnh mới, không phải agent đã viết mã. Đầu vào gồm tài liệu này (mục 2, 3, 5, 8 và đặc tả của các task thuộc điểm review), dải commit cần review và kết quả test. Reviewer không sửa mã.

1. Chạy toàn bộ test CPU; chạy test GPU nếu có máy.
2. So mã với đặc tả: API, schema, config, tên lệnh.
3. Kiểm từng bất biến I1–I9 liên quan: có test không, test có thật sự kiểm điều cần kiểm không.
4. Lần vết bằng tay một lần chạy nhỏ (dữ liệu tổng hợp, $N = 3$, hai vòng): tự tính chi phí từng loại và quỹ còn lại, rồi so với kết quả của mã.
5. Tìm rò rỉ và nhìn trước: dữ liệu `test_*`, nhãn, seed đang mô phỏng, điểm tương lai.
6. Thử các trường hợp biên: $B$ rất nhỏ; $N = 1$; mọi câu tự kết thúc sớm; đáp án thử rỗng; không trích được đáp án; nhiều câu cùng $\rho$; chuỗi chạm $L_{\text{log}}$.
7. Xét hiệu năng: không có vòng giải mã bậc hai trên toàn chuỗi; mô phỏng dùng thao tác véc-tơ ở những chỗ chạy nhiều.
8. Viết báo cáo theo mẫu ở mục 8.3.

### 8.2. Checklist

- [ ] Mã khớp đặc tả; mọi chỗ lệch đều có ADR.
- [ ] Có đủ mọi test được liệt kê trong task, và test kiểm đúng điều cần kiểm, không chỉ chạy qua.
- [ ] Mỗi token được tính đúng một lần, vào đúng loại (mục 2.6).
- [ ] Không có tham số viết cứng; mọi tham số đến từ config và có trong `config_hash`.
- [ ] Mọi nguồn ngẫu nhiên có seed và được truyền vào.
- [ ] Module CPU không import `vllm`.
- [ ] Kết quả ghi vào `runs/` mới, không ghi đè.
- [ ] Không chạm `test_*` ngoài `eval-test` và lần tái lập baseline đã được duyệt (P1.T5).
- [ ] Mã cho mô phỏng và chạy thật không bị cài đặt hai lần (A2, A5, A7, A8, A9).
- [ ] Thông báo lỗi nêu rõ khóa câu hỏi, vị trí và tham số liên quan.

### 8.3. Mẫu báo cáo review

```markdown
# Review <điểm review, ví dụ P2b>
- Task: <ví dụ P2.T7–P2.T15>
- Dải commit: <base>..<head>
- Test: <số đạt>/<tổng> CPU; <số đạt>/<tổng> GPU (hoặc: không chạy, lý do)
- Lần vết bằng tay: <khớp / không khớp, chi tiết>

| # | Mức | Tệp:dòng | Mô tả | Đề xuất sửa |
|---|---|---|---|---|
| 1 | blocker | src/ttalloc/controller/budget.py:88 | ... | ... |

Kết luận: đạt / chưa đạt
```

**Mức độ:**
- **blocker:** vi phạm I1–I9, kết quả sai, rò rỉ dữ liệu, hoặc lỗi làm hỏng luồng chính.
- **major:** lệch đặc tả mà không có ADR, thiếu test bắt buộc, kết quả không tái lập, hoặc vấn đề hiệu năng đáng kể.
- **minor:** cách đặt tên, tài liệu, độ dễ đọc.

---

## 9. Phụ lục

### 9.1. Tham số mặc định

Nguồn gốc các giá trị ở [mục 10 của báo cáo phương pháp chi tiết](Bao-cao-Phuong-phap-chi-tiet.md#L286).

| Khóa | Mặc định | Ghi chú |
|---|---|---|
| `dp.markers` | Wait, Alternatively, Hmm, But, Let me check | Chuỗi dừng là `"\n\n"` + từ chuyển ý |
| `dp.m` | 256 | |
| `dp.l_max` | 1024 | |
| `dp.b_min` | 512 | |
| `log.m` | 128 | Không lớn hơn `dp.m` |
| `log.l_max` | 256 | Không lớn hơn `dp.l_max`; quyết định độ chênh ở mục 7.6 |
| `log.l_log` | 32768 | |
| `log.window` | 256 | Số chuỗi chạy đồng thời khi ghi |
| `probe.r_probe` | 20 | |
| `probe.think_end_lookahead` | 0 với R1-Distill, 3 với Qwen3 | Số token sinh thêm sau khi khung đóng để ghi `probe_has_think_end` (A1b) |
| `answer.mode` | chọn ở P1.T9 | `full` hoặc `short` |
| `answer.r_ans` | 256 (`full`), 32 (`short`) | |
| `answer.on_stop` | `forced` | `best_probe` và `vote` chỉ dùng cho ablation trong mô phỏng (P3.T3) |
| `features.k` | 4 | |
| `features.local_conf` | true | Tắt nếu P1.T4 cho thấy thông lượng giảm hơn 20% |
| `features.local_window` | 1024 | |
| `features.drop_groups` | rỗng | Ablation (P3.T3) |
| `estimator.h_train` | 256, 512, 1024, 2048, 4096, 8192, 16384 | |
| `estimator.kind` | `lightgbm` | Hoặc `logreg`; chọn ở P3, bước chạy 2 |
| `estimator.calibrate` | true | `false` chỉ dùng cho ablation |
| `policy.horizons` | 512, 1024, 2048, 4096, 8192 | Tập con của `h_train` |
| `policy.mu_min` | chỉnh trên `val` | Cùng đơn vị với `rho` trong mã (xác suất trên mỗi token) |
| `policy.b_max_mult` | 3, chỉnh trong {2, 3, 4} | Với M1–M3: $b_{\max}$ = min(`b_max_mult` × trung vị `chain_len` trên `val`, $L_{\text{log}} - c_{\text{ans,max}}$). Với các baseline: $b_{\max} = L_{\text{log}} - c_{\text{ans,max}}$, vì hạn mức của chúng đã đóng vai trò trần |
| `deer.lambda` | 0,95, chỉnh trên `val` | |
| `guard.min_segment` | 256 | Không nhỏ hơn `log.l_max` (A8) |
| `run_seed` | ghi trong config thí nghiệm | Dùng cho thứ tự nạp và phá thế hòa (`tie_key`) |
| `sim.draws` | 200 | |
| `study.curves_delta` | 0,2 | Ngưỡng của P2.T20 |
| `baseline.deer_commit` | ghi trong config | Commit của repo DEER chính thức dùng ở P1.T5 và P4.T5 |
| `online.max_active` | ước lượng từ dung lượng KV cache | |
| `engine.dtype` | `bfloat16` | Chốt ở P0, ghi ADR. Chỉ dùng `float16` nếu buộc phải ghi trên T4 (không hỗ trợ bf16). Mọi lần ghi và chạy thật dùng cùng giá trị; đưa vào `log_hash` (I9) |
| Lấy mẫu chuỗi chính | R1-Distill: temperature 0,6, top_p 0,95. Qwen3: temperature 0,6, top_p 0,95, top_k 20 | Gợi và trả lời luôn dùng temperature 0 |

### 9.2. Lời nhắc

| Tên | Nội dung |
|---|---|
| Hướng dẫn, toán | `Please reason step by step, and put your final answer within \boxed{}.` |
| Hướng dẫn, trắc nghiệm | Đề bài, các lựa chọn dạng `A) ...`, rồi `Please reason step by step, and put the letter of the correct option within \boxed{}.` |
| Gợi đáp án thử | Chép nguyên văn từ `vllm-deer.py` trong repo [DEER](https://github.com/iie-ycx/DEER), kèm mã commit nguồn. Tái hiện đúng vị trí chèn của DEER so với ngắt đoạn; ghi ADR. Quy tắc dừng và phạm vi token tính độ tự tin theo A1b |
| Trả lời `full` | Token `</think>` rồi `"\n\n"`; model viết tiếp tối đa `r_ans` token |
| Trả lời `short` | Token `</think>` rồi `"\n\nThe final answer is $\boxed{"`; đáp án là phần trong khung |

Token `</think>` được thêm bằng id, không tokenize từ chuỗi. Khi model đã tự sinh `</think>`, chỉ chèn phần còn lại của lời nhắc trả lời (A1c). Mọi lời nhắc nằm trong `configs/prompts/` và được đưa vào `log_hash`.

### 9.3. Ví dụ config thí nghiệm

```yaml
name: e2_main_qwen3_1_7b
extends: configs/base.yaml
model: qwen3_1_7b
splits: [test_main]
alphas: [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
methods:
  - {id: B2, policy: deer, params: {lambda: tuned, cap: calibrated}}
  - {id: M2, policy: price_adaptive, params: {mu_min: tuned}}
  - {id: M3, policy: price_adaptive, rho_provider: curve, params: {mu_min: tuned}}
  # P5.T1 thêm vào config mới, cùng sim.seed:
  # - {id: B0, policy: uniform}
  # - {id: B1, policy: deer, params: {lambda: tuned, cap: per_question_uniform}}
  # - {id: M1, policy: price_fixed, params: {mu: calibrated, mu_min: tuned}}
  # - {id: O1, policy: oracle}
tuned_from: runs/<thư mục chạy tune>
sim: {draws: 200, seed: 20261031}
locked: configs/locked/qwen3_1_7b_v1.yaml
```

### 9.4. Mẫu tệp theo dõi

`docs/progress.md`:

```markdown
| Task | Trạng thái | Commit | Ghi chú |
|---|---|---|---|
| P0.T1 | xong | a1b2c3d | |
| P0.T2 | đang làm | | |
```

`docs/decisions.md`:

```markdown
## ADR-007: Dạng câu trả lời khi dừng
- Task: P1.T9
- Bối cảnh: ...
- Quyết định: ...
- Hệ quả: ...
```

### 9.5. Câu giao việc mẫu cho agent

- **Implement:** "Đọc `docs/PLAN.md` và `docs/progress.md`. Làm task hoặc bước chạy kế tiếp chưa xong của phase P<k> theo quy trình ở mục 0.2 và thứ tự ở mục 6. Dừng khi gặp một điểm ở mục 0.4 hoặc khi tới một điểm review."
- **Review:** "Bạn là reviewer độc lập. Đọc `docs/PLAN.md` mục 2, 3, 5, 8 và đặc tả các task thuộc điểm review <tên, ví dụ P2b>. Review các commit `<base>..HEAD`, chạy test, kiểm từng mục ở checklist 8.2 và ghi báo cáo vào `docs/review/<tên>.md` theo mẫu 8.3. Không sửa mã."
- **Sửa sau review:** "Đọc `docs/review/<tên>.md`. Sửa mọi lỗi blocker và major, mỗi lỗi một commit có ghi số thứ tự lỗi, rồi cập nhật báo cáo review với trạng thái từng lỗi."
