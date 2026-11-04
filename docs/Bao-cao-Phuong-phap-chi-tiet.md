# Báo cáo Phương pháp chi tiết

**Sinh viên:** Dương Tuấn Minh
**MSSV:** 23021937
**Đề tài:** Phân bổ thích ứng tài nguyên suy luận cho mô hình suy luận lớn
*(Adaptive Test-Time Compute Allocation for Large Reasoning Models)*  
**Phiên bản thiết kế:** 1.0. Các giá trị tham số và một số lựa chọn thiết kế sẽ được điều chỉnh sau thí nghiệm thăm dò (mục 14).

---

## 1. Phạm vi tài liệu

Tài liệu này chỉ mô tả phương pháp: bài toán, các thành phần, thuật toán, tham số, cơ sở lý thuyết và các rủi ro kỹ thuật. Phần bối cảnh, bài báo nền tảng DEER và các công trình liên quan nằm trong báo cáo tổng quan của đề tài.

## 2. Bài toán

### 2.1. Phát biểu

Cho tập $N$ câu hỏi $Q = \{x_1, \dots, x_N\}$, một mô hình suy luận lớn (LRM) cố định, và tổng ngân sách $B$ token. Một **chính sách phân bổ** $\pi$ quyết định, tại các điểm quyết định trong chuỗi suy luận của từng câu, cho câu đó **nghĩ tiếp** hay **dừng để trả lời**.

Gọi $T_i^\pi$ là tổng token câu $i$ đã tiêu. Tổng này gồm token suy luận, token của lời nhắc và đáp án thử, và token của câu trả lời cuối. Bài toán:

$$\max_\pi \; \mathbb{E}\Big[\sum_{i=1}^{N} \mathbb{1}\big(\hat y_i^\pi = y_i\big)\Big] \quad \text{với} \quad \sum_{i=1}^{N} T_i^\pi \le B.$$

Kỳ vọng lấy theo tính ngẫu nhiên khi sinh token. Ràng buộc ngân sách phải thỏa ở **mọi lần chạy**, không chỉ trung bình.

### 2.2. Bối cảnh áp dụng

Bài toán dành cho các tình huống giải một tập câu hỏi với tổng chi phí cố định: chạy đánh giá model, sinh dữ liệu tổng hợp, sinh rollout cho huấn luyện RL. Mô hình suy luận không được huấn luyện lại.

### 2.3. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $b_i$ | Số token câu $i$ đã tiêu tới thời điểm đang xét |
| $a_{i,j}$, $c_{i,j}$ | Đáp án thử và độ tự tin ở lần gợi thứ $j$ của câu $i$ |
| $\mathbf{x}_i$ | Véc-tơ đặc trưng của câu $i$ tại điểm quyết định hiện tại |
| $H = \{h_1 < \dots < h_{\lvert H \rvert}\}$ | Tập tầm nhìn (số token nghĩ thêm được xét) |
| $\hat q_i(h)$ | Xác suất ước lượng trả lời đúng nếu câu $i$ nghĩ thêm $h$ token rồi dừng; $\hat q_i(0)$ là dừng ngay |
| $\rho_i$ | Lợi ích trên mỗi token của câu $i$ |
| $h_i^*$ | Tầm nhìn đạt $\rho_i$ |
| $\mu$, $\mu_{\min}$ | Giá token hiện tại và giá sàn |
| $R$ | Ngân sách còn lại trong quỹ chung |
| $A$ | Tập câu đang hoạt động (chưa dừng) |
| $r_{\text{ans}}$ | Phần token dự trữ cho câu trả lời cuối của mỗi câu |
| $b_{\min}$, $b_{\max}$ | Số token tối thiểu trước khi được xét dừng; số token tối đa của một câu |

Lưu ý: $\lambda$ được giữ cho ngưỡng độ tự tin của DEER. Giá token dùng ký hiệu $\mu$.

## 3. Tổng quan phương pháp

Phương pháp thực hiện nguyên tắc: **câu dễ nghĩ ít, câu khó còn cơ hội nghĩ nhiều, câu vượt quá khả năng thì cắt lỗ.** Có bốn thành phần:

1. **Bộ theo dõi** (mục 4): tại các điểm quyết định, gợi đáp án thử ở nhánh phụ và cập nhật đặc trưng.
2. **Bộ ước lượng** (mục 5): dự đoán $\hat q_i(h)$, rồi tính chỉ số lợi ích trên mỗi token $\rho_i$.
3. **Quy tắc quyết định** (mục 6): nghĩ tiếp khi $\rho_i > \max(\mu, \mu_{\min})$, ngược lại thì dừng.
4. **Bộ phân bổ** (mục 7): quản lý quỹ chung $R$ và cập nhật giá $\mu$. Các câu chạy song song và được xét đồng bộ theo vòng.

<figure>
<svg viewBox="0 0 640 300" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#1f1e1c">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#9a968f"/></marker>
  </defs>
  <g stroke="#c9c5bd" stroke-width="1.2" fill="#ffffff">
    <rect x="200" y="8" width="240" height="40" rx="8"/>
    <rect x="200" y="72" width="240" height="40" rx="8"/>
    <rect x="200" y="136" width="240" height="40" rx="8"/>
    <rect x="30" y="206" width="210" height="44" rx="8" fill="#e8f0fb" stroke="#4a7fd4"/>
    <rect x="400" y="206" width="210" height="44" rx="8"/>
    <rect x="400" y="262" width="210" height="32" rx="8" fill="#f1efea"/>
  </g>
  <g text-anchor="middle">
    <text x="320" y="25" font-weight="600">LRM suy luận tới điểm quyết định</text>
    <text x="320" y="41" font-size="11">“Wait”, … hoặc sau <tspan font-style="italic">L</tspan><tspan dy="3" font-size="8">max</tspan><tspan dy="-3"> token</tspan></text>
    <text x="320" y="89" font-weight="600">① Gợi đáp án thử (tách nhánh KV)</text>
    <text x="320" y="105" font-size="11">cập nhật đặc trưng <tspan font-weight="700">x</tspan><tspan dy="3" font-size="8">i</tspan></text>
    <text x="320" y="153" font-weight="600">② Ước lượng <tspan font-style="italic">q̂</tspan><tspan dy="3" font-size="9">i</tspan><tspan dy="-3">(</tspan><tspan font-style="italic">h</tspan>), tính <tspan font-style="italic">ρ</tspan><tspan dy="3" font-size="9">i</tspan></text>
    <text x="320" y="169" font-size="11">③ so với max(<tspan font-style="italic">μ</tspan>, <tspan font-style="italic">μ</tspan><tspan dy="3" font-size="8">min</tspan><tspan dy="-3">)</tspan></text>
    <text x="135" y="224" font-weight="600">Nghĩ tiếp</text>
    <text x="135" y="240" font-size="11">tới điểm quyết định kế tiếp</text>
    <text x="505" y="224" font-weight="600">Dừng, sinh câu trả lời</text>
    <text x="505" y="240" font-size="11">tối đa <tspan font-style="italic">r</tspan><tspan dy="3" font-size="8">ans</tspan><tspan dy="-3"> token</tspan></text>
    <text x="505" y="283" font-weight="600">④ Token dư → quỹ <tspan font-style="italic">R</tspan>, cập nhật <tspan font-style="italic">μ</tspan></text>
  </g>
  <g stroke="#9a968f" stroke-width="1.2" fill="none" marker-end="url(#ar)">
    <line x1="320" y1="48" x2="320" y2="70"/>
    <line x1="320" y1="112" x2="320" y2="134"/>
    <path d="M280,176 L280,190 L135,190 L135,204"/>
    <path d="M360,176 L360,190 L505,190 L505,204"/>
    <line x1="505" y1="250" x2="505" y2="260"/>
    <path d="M30,228 L14,228 L14,28 L198,28"/>
    <path d="M610,278 L628,278 L628,156 L442,156" stroke-dasharray="4 3"/>
  </g>
</svg>
<figcaption>Hình 1. Luồng xử lý của một câu. Số trong vòng tròn là thành phần tương ứng ở mục 3. Nét đứt: giá <i>μ</i> dùng chung cho mọi câu.</figcaption>
</figure>

## 4. Bộ theo dõi

### 4.1. Điểm quyết định

Một điểm quyết định xảy ra khi có một trong hai sự kiện:
- **Điểm chuyển ý.** Model sinh một từ đánh dấu chuyển hướng suy nghĩ ở đầu một đoạn: "Wait", "Alternatively", "Hmm", "But", "Let me check". Danh sách lấy theo DEER, có thể mở rộng theo từng model.
- **Ngưỡng độ dài.** Sau $L_{\max}$ token mà chưa gặp điểm chuyển ý, điểm quyết định được đặt tại chỗ xuống dòng giữa hai đoạn gần nhất.

Hai ràng buộc thêm:
- Hai lần gợi đáp án thử liên tiếp của một câu cách nhau ít nhất $m$ token, để giới hạn chi phí.
- Không xét dừng khi câu chưa tiêu đủ $b_{\min}$ token, vì đặc trưng ở đầu chuỗi chưa đủ tin cậy.

### 4.2. Gợi đáp án thử

Tại điểm quyết định, hệ thống **tách một nhánh** từ KV cache hiện tại (prefix caching của vLLM hoặc SGLang). Trên nhánh này, nó chèn một lời nhắc cố định yêu cầu model ghi ngay đáp án cuối cùng vào khung đáp án (dạng $\boxed{\;\cdots\;}$ mà model toán vẫn dùng), rồi sinh tối đa $r_{\text{probe}}$ token tới khi khung được đóng. Nhánh này bị bỏ sau khi đọc kết quả. Chuỗi suy luận chính không thay đổi.

- **Đáp án thử** $a_{i,j}$: nội dung trong khung đáp án, được **chuẩn hóa** (ví dụ bằng thư viện math-verify) để các cách viết khác nhau của cùng một giá trị, như $1/2$, $0{,}5$ và $\tfrac{1}{2}$, được coi là một.
- **Độ tự tin** $c_{i,j}$: trung bình nhân của xác suất lớn nhất tại từng token trong đáp án, như DEER.
- **Chi phí:** token lời nhắc và token đáp án thử được trừ vào $R$.

Mẫu lời nhắc phải cố định cho toàn bộ thí nghiệm, vì cách diễn đạt khác nhau làm đáp án thử thay đổi đáng kể.

### 4.3. Đặc trưng

Tại điểm quyết định thứ $j$ của câu $i$, véc-tơ đặc trưng $\mathbf{x}_i$ gồm:

| Nhóm | Đặc trưng | Định nghĩa |
|---|---|---|
| Đáp án thử | Độ tự tin hiện tại | $c_{i,j}$ |
| | Độ tự tin cao nhất | $\max_{l \le j} c_{i,l}$ |
| | Xu hướng | Hệ số góc hồi quy tuyến tính của $c$ theo **vị trí token**, trên $k$ lần gợi gần nhất, tính bằng đơn vị /1k token |
| | Độ nhất quán | Tỉ lệ trong $\min(k, j)$ lần gợi gần nhất có đáp án trùng $a_{i,j}$, kèm số lần gợi thực tế |
| | Số lần đổi đáp án | Số lần $a_{i,l} \ne a_{i,l-1}$ |
| Chuỗi suy luận | Số token đã dùng | $b_i$ |
| | Tần suất chuyển ý | Số điểm chuyển ý trên mỗi 1k token |
| | Độ tự tin cục bộ thấp nhất | Giá trị nhỏ nhất, trên các cửa sổ trượt $w$ token, của trung bình log-xác suất token đã sinh (theo DeepConf) |
| Dạng đáp án | Loại đáp án | Số nguyên, phân số hoặc biểu thức, lựa chọn |
| | Độ dài đáp án | Số token của đáp án thử |

Nhóm "dạng đáp án" cần thiết vì trung bình nhân phạt đáp án nhiều token. Nếu thiếu nhóm này, chỉ số của các câu có đáp án khác dạng sẽ không so sánh được với nhau.

## 5. Bộ ước lượng

### 5.1. Mô hình

Bộ ước lượng là một model nhỏ $f$ dự đoán

$$\hat q_i(h) = f(\mathbf{x}_i, h), \qquad h \in \{0\} \cup H.$$

Lựa chọn mặc định là gradient boosting (ví dụ LightGBM) với $h$ là một đặc trưng đầu vào. Phương án đơn giản hơn là logistic regression riêng cho từng $h$. Mô hình suy luận chính **không** được huấn luyện lại. Mỗi LRM cần một bộ ước lượng riêng.

### 5.2. Dữ liệu và nhãn

Dữ liệu huấn luyện lấy từ **các chuỗi suy luận đã ghi lại** trên một tập câu hỏi tách biệt hoàn toàn với tập kiểm tra:

1. Với mỗi câu và $S$ seed, chạy LRM tới khi tự kết thúc hoặc tới giới hạn $L_{\text{log}}$, với $L_{\text{log}} > b_{\max}$.
2. Gợi đáp án thử ở **mọi** điểm quyết định tiềm năng, và ghi lại vị trí token, đáp án, độ tự tin và đặc trưng.
3. Tại mỗi điểm, sinh thêm **câu trả lời cuối** sau khi ép model kết thúc phần suy nghĩ, rồi chấm đúng sai.

Nhãn cho mẫu tại điểm quyết định có vị trí $\tau$:
- $y(0)$: câu trả lời cuối khi dừng tại $\tau$ có đúng không.
- $y(h)$: câu trả lời cuối khi dừng tại điểm quyết định cuối cùng có vị trí $\le \tau + h$ có đúng không. Nếu chuỗi tự kết thúc trước $\tau + h$, dùng câu trả lời tự nhiên của chuỗi.

Tập huấn luyện và tập validation được chia **theo câu hỏi**, không theo mẫu, để tránh rò rỉ.

### 5.3. Hiệu chuẩn

Vì $\rho_i$ so sánh xác suất **giữa các câu**, $\hat q$ phải được hiệu chuẩn. Dùng isotonic regression trên tập validation, riêng cho từng $h$. Kiểm tra bằng reliability diagram và ECE.

### 5.4. Chỉ số lợi ích trên mỗi token

$$\rho_i = \max_{h \in H} \frac{\hat q_i(h) - \hat q_i(0)}{h + \bar p(h)}, \qquad h_i^* = \arg\max_{h \in H}(\cdot),$$

trong đó $\bar p(h)$ là số token gợi đáp án thử dự kiến trong $h$ token tới, ước lượng từ dữ liệu ghi lại.

**Vì sao xét nhiều tầm.** Đường cong xác suất đúng theo số token của một câu thường có dạng chữ S: phẳng cho tới khi model tìm ra hướng giải, rồi tăng nhanh. Lợi ích của một bước nhỏ ở đoạn phẳng gần bằng 0, nên quy tắc một bước sẽ cắt nhầm những câu này. Lấy max theo $h$ tương đương với độ dốc của bao lồi trên đường cong ước lượng: một câu sắp tăng vọt vẫn có $\rho_i$ dương.

Chỉ số $\rho_i$ có thể âm khi nghĩ thêm làm giảm xác suất đúng (overthinking). Khi đó câu luôn dừng.

## 6. Quy tắc quyết định

Tại mỗi điểm quyết định, sau khi câu $i$ đã tiêu ít nhất $b_{\min}$ token:

$$\text{nghĩ tiếp} \iff \rho_i > \max(\mu, \mu_{\min}) \;\text{ và }\; b_i < b_{\max} \;\text{ và quỹ còn đủ (mục 7.3)}.$$

Ngược lại, câu dừng. Ba trạng thái trong đề xuất ban đầu là ba trường hợp của quy tắc này:

| Trạng thái | $\hat q_i(0)$ | $\hat q_i(h) - \hat q_i(0)$ | Quyết định |
|---|---|---|---|
| Đã chắc chắn | cao | ≈ 0 hoặc âm | dừng, trả lời |
| Đang tiến bộ | bất kỳ | lớn ở một tầm $h$ nào đó | nghĩ tiếp |
| Vượt quá khả năng | thấp | ≈ 0 ở mọi $h$ | dừng (cắt lỗ) |

- **Giá sàn** $\mu_{\min}$ bảo đảm câu có lợi ích rất nhỏ luôn bị dừng, kể cả khi ngân sách dư. Nếu không có giá sàn, câu vô vọng sẽ tiếp tục tiêu token khi $\mu = 0$. $\mu_{\min}$ được chọn trên validation theo mức đánh đổi giữa chi phí và độ chính xác mong muốn.
- **Trần** $b_{\max}$ ngăn một câu dùng hết quỹ.

### 6.1. Sinh câu trả lời khi dừng

Mặc định giống DEER: ép model kết thúc phần suy nghĩ, chèn lời nhắc trả lời, sinh tối đa $r_{\text{ans}}$ token. Hai phương án sẽ so sánh trên validation:
- dùng đáp án thử có độ tự tin cao nhất;
- bỏ phiếu có trọng số trên các đáp án thử, trọng số tăng theo độ tự tin và độ mới.

Chọn đáp án thử có độ tự tin cao nhất có rủi ro bị thổi phồng do nhiễu, vì các lần gợi ở đầu chuỗi nhiễu nhất. Phương án này chỉ được dùng nếu validation cho thấy nó tốt hơn.

## 7. Bộ phân bổ và giá token

### 7.1. Ý nghĩa của $\mu$

$\mu$ là **giá của một token**: mức lợi ích (xác suất đúng tăng thêm) tối thiểu mà một token phải mang lại thì mới đáng dùng. Về mặt toán học, $\mu$ là nhân tử Lagrange của ràng buộc ngân sách. Khi gán giá $\mu$ cho mỗi token, bài toán chung tách thành $N$ bài toán dừng độc lập, mỗi câu chỉ cần so $\rho_i$ với $\mu$. Ngân sách eo hẹp thì $\mu$ cao, dư dả thì $\mu$ thấp.

### 7.2. Tính giá $\mu$: đồng bộ theo vòng

Các câu chạy song song trong batch. Mỗi vòng, mọi câu đang hoạt động chạy tới điểm quyết định kế tiếp của mình. Câu tới trước thì tạm dừng chờ. Khi tất cả đã tới:

1. Tính $\rho_i$ và $h_i^*$ cho mọi $i \in A$.
2. Quỹ dùng được: $R_{\text{avail}} = R - \lvert A \rvert \cdot r_{\text{ans}}$.
3. Sắp xếp $A$ theo $\rho_i$ giảm dần, cộng dồn $h_i^*$.
4. Gọi $i^\dagger$ là câu đầu tiên khiến tổng cộng dồn vượt $R_{\text{avail}}$ (câu ở ranh giới). Đặt $\mu = \rho_{i^\dagger}$. Nếu tổng không bao giờ vượt, đặt $\mu = 0$.
5. Câu có $\rho_i \ge \mu$, **gồm cả câu ở ranh giới**, được nghĩ tiếp nếu thỏa mục 6. Các câu còn lại dừng.

Câu được nghĩ tiếp chỉ chạy tới điểm quyết định kế tiếp, không được cam kết cả $h_i^*$ token. Kế hoạch được tính lại ở mỗi vòng; $h_i^*$ chỉ dùng để đặt giá. Câu ở ranh giới vẫn tiếp tục vì nó được cấp phần quỹ còn lại và sẽ được xét lại ngay vòng sau. Nếu loại câu ở ranh giới, phương pháp sẽ dừng nhầm đúng câu khó nhưng còn cơ hội.

Cách làm này được chọn vì mọi câu được so sánh ở cùng một thời điểm, nên giá $\mu$ phản ánh đúng tình trạng của cả lô. Đây cũng là kiểu chạy theo vòng mà các công trình phân bổ gần nhất sử dụng (Zuo & Zhu, UAB), giúp so sánh trực tiếp.

### 7.3. Bảo đảm ngân sách cứng

- Mọi token (suy luận, gợi đáp án thử, câu trả lời) đều trừ vào $R$ ngay khi sinh.
- Luôn giữ $r_{\text{ans}}$ token cho mỗi câu đang hoạt động. Câu trả lời cuối bị cắt ở $r_{\text{ans}}$ token.
- Nếu quỹ không đủ cho mọi câu được chọn nghĩ tiếp, chỉ những câu có $\rho_i$ cao nhất vừa với quỹ được chạy tiếp.
- Khi $R_{\text{avail}}$ nhỏ hơn độ dài một đoạn tối thiểu, mọi câu còn lại dừng và trả lời.

Nhờ vậy, $\sum_i T_i \le B$ ở mọi lần chạy.

### 7.4. Khi $N$ lớn hơn sức chứa của GPU

Nếu GPU chỉ chứa được $M < N$ chuỗi cùng lúc, hệ thống chạy theo cửa sổ trượt: luôn có tối đa $M$ câu hoạt động, câu nào dừng thì nạp câu mới. Giá $\mu$ được tính trên các câu đang hoạt động như mục 7.2, nhưng quỹ dùng được phải trừ thêm phần dự trữ $b_{\min} + r_{\text{ans}}$ cho mỗi câu chưa được nạp. Câu đang chờ trong vòng vẫn giữ KV cache. Nếu thiếu bộ nhớ, có thể đẩy KV cache ra bộ nhớ CPU (swap) hoặc tính lại.

### 7.5. Các mở rộng (ngoài phạm vi chính)

Các phần sau không thuộc phạm vi bắt buộc của đề tài. Chúng chỉ được làm nếu còn thời gian, hoặc nếu phần cài đặt hệ thống cần đến.

- **Chạy bất đồng bộ.** Để tránh thời gian chờ ở mỗi vòng, mỗi câu quyết định ngay khi tới điểm quyết định của mình, dùng $\mu$ hiện tại. $\mu$ được tính lại theo chu kỳ (mỗi $U$ token sinh ra trên toàn batch) như bước 3–4 ở mục 7.2, trên ảnh chụp $(\rho_i, h_i^*)$ gần nhất. Cách này hợp với continuous batching của vLLM, cần khi đo thời gian chạy thật.
- **Chạy lần lượt với điều chỉnh giá.** Khi các câu phải xử lý tuần tự, giá khởi đầu $\mu_0$ được chọn trên validation, rồi điều chỉnh sau mỗi câu (hoặc mỗi đợt) $j$ theo kiểu budget pacing:
$$\mu_{j+1} = \max\Big(\mu_{\min},\; \mu_j \cdot \exp\Big(\eta \cdot \frac{S_j - P_j}{P_j}\Big)\Big),$$
với $S_j$ là tổng token đã tiêu thực tế, $P_j = j \cdot B/N$ là mức chi tiêu theo kế hoạch, $\eta$ là tốc độ điều chỉnh. Tiêu nhanh hơn kế hoạch thì giá tăng, chậm hơn thì giá giảm.
- **Ngân sách khởi đầu theo độ khó dự đoán.** Trước khi suy luận, một bộ dự đoán nhẹ (từ nội dung câu hỏi, hoặc từ hidden state khi prefill như Predictive Scheduling) ước lượng độ khó $d_i$. Câu được đoán là khó nhận $b_{\min,i}$ lớn hơn. Quỹ chung sau đó sửa các dự đoán sai.

## 8. Thuật toán

<div class="algo">
<div class="l0"><b>Đầu vào:</b> câu hỏi $x_1, \dots, x_N$; LRM; bộ ước lượng $f$; ngân sách $B$; tham số (mục 10)</div>
<div class="l0">$R \leftarrow B$; &nbsp;$A \leftarrow \{1, \dots, N\}$</div>
<div class="l0"><b>for</b> $i \in A$ <b>do</b> suy luận câu $i$ tới điểm quyết định đầu tiên sau $b_{\min}$ token; gợi đáp án thử, cập nhật $\mathbf{x}_i$; trừ mọi token vào $R$</div>
<div class="l0"><b>loop</b></div>
<div class="l1"><b>for</b> $i \in A$ <b>do</b> $\hat q_i(h) \leftarrow f(\mathbf{x}_i, h)$ với $h \in \{0\} \cup H$; tính $\rho_i$, $h_i^*$</div>
<div class="l1">$R_{\text{avail}} \leftarrow R - \lvert A \rvert \cdot r_{\text{ans}}$</div>
<div class="l1">sắp $A$ theo $\rho_i$ giảm dần; cộng dồn $h_i^*$; $i^\dagger \leftarrow$ câu ở ranh giới</div>
<div class="l1">$\mu \leftarrow \rho_{i^\dagger}$ nếu có câu ở ranh giới, ngược lại $\mu \leftarrow 0$</div>
<div class="l1"><b>for</b> $i \in A$ <b>do</b></div>
<div class="l2"><b>if</b> $\rho_i \ge \mu$ <b>and</b> $\rho_i > \mu_{\min}$ <b>and</b> $b_i < b_{\max}$ <b>and</b> $R_{\text{avail}} \ge$ một đoạn tối thiểu <b>then</b> đánh dấu $i$ nghĩ tiếp</div>
<div class="l2"><b>else</b> dừng $i$: ép kết thúc phần suy nghĩ, sinh câu trả lời (tối đa $r_{\text{ans}}$ token), trừ vào $R$; $A \leftarrow A \setminus \{i\}$</div>
<div class="l1"><b>if</b> $A = \varnothing$ <b>then break</b></div>
<div class="l1">chạy song song các câu nghĩ tiếp tới điểm quyết định kế tiếp; câu nào tự kết thúc phần suy nghĩ thì ghi nhận câu trả lời và $A \leftarrow A \setminus \{i\}$</div>
<div class="l1">gợi đáp án thử cho các câu tới điểm quyết định, cập nhật $\mathbf{x}_i$, trừ token vào $R$</div>
<div class="l0"><b>Đầu ra:</b> câu trả lời của $N$ câu; tổng token đã dùng $\le B$</div>
</div>

## 9. Ví dụ minh họa

Bốn câu: câu 1 dễ, câu 2 trung bình, câu 3 khó nhưng giải được, câu 4 vượt khả năng. Tham số: $B = 10.000$ token, $b_{\min} = 1.000$, mỗi lần gợi đáp án thử tốn 30 token, $r_{\text{ans}} = 200$, $H = \{1\text{k}, 2\text{k}, 4\text{k}\}$, $\mu_{\min} = 0$. Đơn vị của $\rho$ và $\mu$: điểm phần trăm xác suất đúng trên mỗi 1.000 token.

**Vòng 1.** Đã tiêu 4.120 token, nên $R = 5.880$ và $R_{\text{avail}} = 5.080$.

| Câu | $\hat q(0)$ | +1k | +2k | +4k | $\rho$ | $h^*$ |
|---|---|---|---|---|---|---|
| 1 | 92% | 93% | 93% | 93% | 1 | 1k |
| 2 | 40% | 55% | 85% | 88% | 22,5 | 2k |
| 3 | 10% | 12% | 15% | 50% | 10 | 4k |
| 4 | 5% | 5% | 6% | 6% | 0,5 | 2k |

Cộng dồn theo $\rho$ giảm dần: câu 2 (2k), rồi câu 3 (tổng 6k > 5.080). Vậy câu 3 là câu ở ranh giới và $\mu = 10$. Câu 2 và câu 3 nghĩ tiếp. Câu 1 dừng vì đã chắc chắn, câu 4 dừng vì cắt lỗ. Nếu chỉ xét một bước +1k, câu 3 chỉ được 2 điểm và sẽ bị cắt nhầm.

**Vòng 2.** Câu 2 có $\hat q$ lần lượt là 88%, 88%, 87%, 86%, tức $\rho \le 0$, nên dừng. Câu 3 có $\rho = 20$; quỹ đủ cho phần còn lại nên $\mu = 0$, và câu 3 nghĩ tiếp.

**Vòng 3.** Câu 3 tự kết thúc.

Kết quả: câu 1 dùng 1.230 token, câu 4 dùng 1.230, câu 2 dùng 2.760, câu 3 dùng 4.210. Tổng 9.430 ≤ 10.000 token, kỳ vọng khoảng 2,6 câu đúng. DEER với giới hạn 2.500 token mỗi câu (cùng tổng) cắt câu 3 ở khoảng 20% xác suất đúng, kỳ vọng khoảng 2,0 câu đúng.

## 10. Tham số

| Tham số | Vai trò | Giá trị khởi đầu đề xuất |
|---|---|---|
| $b_{\min}$ | Token tối thiểu trước khi xét dừng | 512–1.024 |
| $b_{\max}$ | Trần token mỗi câu | 2–4 lần độ dài trung vị khi chạy thường |
| $H$ | Tập tầm nhìn | {512, 1k, 2k, 4k, 8k} |
| $L_{\max}$ | Độ dài tối đa giữa hai điểm quyết định | 1.024 |
| $m$ | Khoảng cách tối thiểu giữa hai lần gợi đáp án thử | 256 |
| $k$ | Số lần gợi gần nhất để tính xu hướng và độ nhất quán | 4 |
| $r_{\text{probe}}$, $r_{\text{ans}}$ | Độ dài tối đa của đáp án thử và của câu trả lời cuối | 20; 256 (toán) |
| $\mu_{\min}$ | Giá sàn | chọn trên validation |

Tất cả được chọn trên tập validation tách riêng. Dùng **một bộ tham số chung cho mọi mức $B$**, và báo cáo độ nhạy với từng tham số. So với đề xuất ban đầu, các ngưỡng trực tiếp quyết định đúng sai ($\lambda_{\text{cao}}$, $\lambda_{\text{thấp}}$, $s$) không còn. Phần lớn tham số còn lại chỉ ảnh hưởng tới chi phí và độ ổn định.

## 11. Cơ sở lý thuyết và giới hạn

- **Khi đường cong đã biết.** Nếu biết đường cong của mọi câu, xếp theo $\rho_i$ (độ dốc của bao lồi) rồi cấp tới khi hết ngân sách là lời giải tối ưu cho bản nới lỏng tuyến tính của bài toán knapsack nhiều lựa chọn [Sinha & Zoltners 1979]. Sai lệch so với lời giải nguyên tối đa khoảng một câu. Với đường cong lõm, quy tắc này trùng với phân bổ tham lam theo lợi ích biên, vốn tối ưu [Fox 1966].
- **Tách bài toán bằng giá.** Gán giá $\mu$ cho ràng buộc ngân sách là kỹ thuật nhân tử Lagrange [Everett 1963]. Với bài toán gồm nhiều quá trình quyết định chỉ liên kết qua ngân sách, kỹ thuật này cho một cận và một chính sách xấp xỉ [Adelman & Mersereau 2008]. Việc tách chỉ chính xác khi ràng buộc là tổng **kỳ vọng**. Khi $\mu$ được cập nhật trong lúc chạy để giữ ràng buộc cứng, chính sách là heuristic.
- **Không phải chỉ số Gittins.** $\rho_i$ dùng các tầm nhìn cố định chứ không tối ưu trên các thời điểm dừng. Nó có thể đánh giá thấp giá trị của việc nghĩ tiếp, nghĩa là hơi thiên về dừng sớm.
- **Giới hạn chung.** Đường cong chỉ được ước lượng, quan sát dần, và quyết định dừng không đảo ngược được. Chất lượng phương pháp phụ thuộc trực tiếp vào chất lượng hiệu chuẩn của $\hat q$.

## 12. Các trường hợp đặc biệt: baseline và ablation

Một số phương pháp có sẵn là trường hợp đặc biệt của khung này. Chúng được dùng làm baseline và ablation:

| Cấu hình | Tương ứng với |
|---|---|
| Không dừng sớm, $b_{\max} = B/N$ | Giới hạn đều |
| $\hat q(0)$ thay bằng độ tự tin $c$, dừng khi $c \ge 0{,}95$, không quỹ chung | DEER |
| Như trên, $b_{\max}$ nâng lên để tổng bằng $B$ | DEER với giới hạn nâng lên (baseline chính) |
| Quỹ chung, cấp theo thứ tự đến trước, không dùng $\rho$ | Tách giá trị của bộ ước lượng |
| $H$ chỉ có một tầm | Quy tắc một bước của đề xuất ban đầu |
| Bỏ đặc trưng độ nhất quán hoặc xu hướng | Kiểm tra giá trị của từng tín hiệu |

## 13. Rủi ro kỹ thuật và cách giảm

| Rủi ro | Biểu hiện | Cách giảm |
|---|---|---|
| Model tự tin quá mức vào đáp án sai | $\hat q(0)$ cao nhưng sai, dừng sớm sai | Hiệu chuẩn; đặc trưng độ tự tin cục bộ và độ nhất quán; kiểm tra theo từng model |
| Câu khó đến muộn bị cắt nhầm | Đáp án đúng xuất hiện sau tầm xa nhất của $H$ | Thêm tầm dài; $b_{\min}$; đo tỉ lệ cắt nhầm theo độ khó; mở rộng bằng kiểm soát rủi ro (Conformal Thinking) |
| Nhiễu do gợi đáp án thử | Đáp án thử dao động vì lời nhắc chứ không vì suy luận | Cố định mẫu lời nhắc; xu hướng tính bằng hồi quy trên $k$ điểm; đo mức nhiễu bằng gợi hai lần |
| Chỉ số không so sánh được giữa các câu | Câu có đáp án ngắn được ưu tiên sai | Đặc trưng dạng đáp án; hiệu chuẩn riêng |
| Một vài câu dùng hết quỹ | Câu khác bị bỏ đói | Trần $b_{\max}$; báo cáo độ chính xác theo nhóm độ khó |
| Thời gian chờ và bộ nhớ ở mỗi vòng | Thời gian hoàn thành cả batch tăng, KV cache bị chiếm | Cửa sổ trượt; swap KV; chạy bất đồng bộ (mở rộng, mục 7.5) |
| Thành phần batch khác validation | Giá sàn $\mu_{\min}$ không còn phù hợp | Chọn $\mu_{\min}$ trên validation có độ khó trộn đa dạng; báo cáo độ nhạy theo thành phần batch |
| Chi phí gợi đáp án thử cao | Tiết kiệm token bị bù trừ | Tách nhánh KV; khoảng cách $m$; tính đủ chi phí vào $B$ |

## 14. Các giả định cần kiểm tra trước khi cài đặt đầy đủ

Một thí nghiệm thăm dò trên dữ liệu ghi lại (mục 5.2) sẽ kiểm tra:

1. **Bộ ước lượng có tín hiệu.** $\hat q$ được hiệu chuẩn tốt và phân biệt được câu sẽ đúng khi nghĩ thêm với câu sẽ không đúng (AUROC, ECE, tương quan với lợi ích thực tế).
2. **Có dư địa để cải thiện.** Một bộ phân bổ lý tưởng (biết trước kết quả, giải bằng quy hoạch động trên dữ liệu ghi lại) hơn đáng kể so với DEER có giới hạn nâng lên, ở ít nhất một vùng $B/N$.
3. **Mô phỏng hợp lệ.** Vì chính sách chỉ quyết định dừng hay nghĩ tiếp trên một chuỗi chính không bị thay đổi, các chính sách có thể được mô phỏng trên dữ liệu ghi lại, với điều kiện dữ liệu đủ dài và có gợi đáp án thử ở mọi điểm quyết định tiềm năng. Thời gian chạy và hiệu ứng KV cache phải đo trên hệ thống thật.

Nếu giả định 2 không thỏa, phạm vi sẽ được thu hẹp về những vùng có dư địa (ngân sách eo hẹp, batch trộn độ khó) trước khi cài đặt bộ phân bổ đầy đủ.

## Tài liệu tham khảo

1. Yang et al. *Dynamic Early Exit in Reasoning Models (DEER).* arXiv:2504.15895, ICLR 2026.
2. Fu et al. *Deep Think with Confidence (DeepConf).* arXiv:2508.15260, ICLR 2026.
3. Zabounidis et al. *Re-FORC.* arXiv:2511.02130, ICML 2026.
4. Brown, Muppidi, Shahout. *Predictive Scheduling.* arXiv:2602.01237.
5. Wan et al. *The Shadow Price of Reasoning (CLEAR).* arXiv:2606.03092.
6. Zuo, Zhu. *Strategic Scaling of Test-Time Compute: A Bandit Learning Approach.* arXiv:2506.12721, ICLR 2026.
7. Hosseini et al. *CoDE-Stop.* arXiv:2604.04930.
8. Wang et al. *Conformal Thinking.* arXiv:2602.03814, ICML 2026.
9. Mo, Zhao, Geng. *Stable Answers, Unfinished Reasoning.* arXiv:2609.09989.
10. Dong, Qin, Shah. *When Does Learning to Stop Help?* arXiv:2606.30852.
11. Everett, H. *Generalized Lagrange Multiplier Method for Solving Problems of Optimum Allocation of Resources.* Operations Research 11(3), 1963.
12. Fox, B. *Discrete Optimization via Marginal Analysis.* Management Science 13(3), 1966.
13. Sinha, Zoltners. *The Multiple-Choice Knapsack Problem.* Operations Research, 1979.
14. Adelman, Mersereau. *Relaxations of Weakly Coupled Stochastic Dynamic Programs.* Operations Research 56(3), 2008.
15. Balseiro, Gur. *Learning in Repeated Auctions with Budgets: Regret Minimization and Equilibrium.* Management Science, 2019.
