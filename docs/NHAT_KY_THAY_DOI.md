# Nhật ký thay đổi code: IRLS (30/09/2026)

Phạm vi: mục 11.2 (IRLS float32 hay float64 trên GPU), rồi thêm phạt λ và đổi `tol` mặc định. Kết quả thí nghiệm ở [BAO_CAO_IRLS_DTYPE.md](BAO_CAO_IRLS_DTYPE.md).

Test: đầu phiên 26 → cuối phiên **32, tất cả đạt** (`python -m unittest discover -s tests -t .`).

## Tóm tắt

| File | Loại | Nội dung |
|---|---|---|
| `src/solvers.py` | sửa | `irls`: thêm `lam`, `pen`, `gram_dtype`, `callback`; `tol` mặc định 1e-12 → 1e-9; ngưỡng "J tăng" theo ε của kiểu số; bước giải dạng hiệu chỉnh |
| `tests/test_irls.py` | sửa | thêm 6 test (float32, pha độ chính xác, 4 test λ); ghim `tol=1e-12` cho 3 test đo độ chính xác |
| `scripts/irls_dtype_check.py` | mới | thí nghiệm 11.2: float32 / float64 / pha, đo giờ trên GPU |
| `results/irls_dtype/` | mới | 6 cặp JSON + NPY (số đo, quỹ đạo J và sai số test theo vòng, W) |
| `docs/BAO_CAO_IRLS_DTYPE.md` | mới | báo cáo 11.2 |
| `docs/NHAT_KY_THAY_DOI.md` | mới | file này |
| `README.md` | sửa | thêm thư mục `scripts/` và quy ước λ cho MAE/Huber |

Không đổi: `src/data.py`, `operators.py`, `metrics.py`, `models.py`, `sgd.py`, các notebook, `docs/CHECKLIST_CON_LAI.md`.

## 1. `src/solvers.py`, hàm `irls`

Chữ ký:

```python
# trước
irls(Xt, Y, delta, W0, constrained=False, max_iter=1000, tol=1e-12)
# sau
irls(Xt, Y, delta, W0, lam=0.0, pen=None, constrained=False, max_iter=1000, tol=1e-9,
     callback=None, gram_dtype=None)
```

Mọi lời gọi cũ vẫn chạy (repo không có chỗ nào truyền `constrained` theo vị trí). Kết quả số có thể khác chút do mục 1.2 và 1.6.

### 1.1. Ngưỡng "J tăng" theo ε của kiểu số

- **Lỗi:** chạy float32 là báo `RuntimeError: J tăng` ngay, cả trên ETTh1 lẫn dữ liệu giả của test.
  - Nguyên nhân: ngưỡng cố định `J_new > J_old·(1 + 1e-10)` được chỉnh cho float64. Ở float32, J tính lại lệch 1 ulp (khoảng 1e-7 tương đối) do làm tròn.
- **Sửa:** `rise_tol = max(1e-10, 100·ε)`.
  - float64 giữ nguyên hành vi, vì 100ε₆₄ = 2e-14 < 1e-10.
  - Nếu J tăng trong ngưỡng thì coi là đã chạm độ chính xác của kiểu số và dừng. Tăng vượt ngưỡng vẫn báo lỗi.

### 1.2. Bước giải dạng hiệu chỉnh

- **Trước:** `W = A⁻¹c`, với `c = Xtᵀ diag(w) y`.
- **Sau:** `W = W + A⁻¹g`, với `g = Xtᵀ diag(w) r` và r là phần dư của W hiện tại.
- **Lý do:**
  - Trong số học chính xác hai dạng bằng nhau, vì W + A⁻¹(c − AW) = A⁻¹c.
  - Điểm dừng g = 0 được tính ở độ chính xác của phần dư, nên A cho phép kém chính xác hơn. Đây là điều kiện để mục 1.3 hoạt động.
  - Dạng này cũng giúp float32 thuần: MAE trên ETTh1 đi được 62 vòng thay vì 38.
  - Nhánh NLinear (`constrained=True`) đổi theo: ŵ = W + A⁻¹g, rồi chiếu lên ràng buộc như cũ.

### 1.3. Tham số `gram_dtype` (pha độ chính xác)

- **Cách làm:** chỉ lập A_h bằng kiểu này, ví dụ `torch.float32` khi Xt là float64. Đây là phần tốn H·n·p² phép tính. Phần dư, gradient, J và phép giải vẫn dùng kiểu của Xt.
- **Kết quả trên ETTh1 H = 96:**
  - Nghiệm như float64: Huber lệch W 2,2e-10; MAE ở cùng số vòng lệch MSE/MAE test khoảng 1e-6.
  - Tốc độ: 1,49 s/vòng, so với 1,29 s (float32) và 15,2 s (float64).
- **Mặc định `None`**: giữ hành vi cũ.

### 1.4. Tham số `callback(it, W, J_new)`

Gọi mỗi vòng, trước phép kiểm J. Dùng để ghi quỹ đạo, và để lấy lại W khi `irls` báo lỗi. Không đổi hành vi.

### 1.5. Tham số `lam`, `pen` (phạt ridge)

**Hàm mục tiêu** (dạng tổng, như quy ước MSE `‖Y − XWᵀ‖² + λ‖W‖²` của README):

    F(W) = Σ ρ_δ(r)/δ + λ Σ_h w_hᵀ Pen w_h

- ρ_δ(r) = r²/2 nếu |r| ≤ δ, δ(|r| − δ/2) nếu ngược lại.
- w_h là L hệ số đầu của hàng h. **Không phạt bias.**
- `Pen = I` nếu `pen=None`.

**Hệ quả:**

| Trường hợp | F |
|---|---|
| MAE (δ = 1e-6 → 0) | Σ\|r\| + λ‖W‖² |
| Huber δ = 1 | Σρ₁(r) + λ‖W‖² |
| δ lớn | ‖r‖²/(2δ) + λ‖W‖², tức ridge MSE với λ_mse = 2δλ |

- **Vì sao chia cho δ:** nếu không chia, λ của MAE sẽ phải nhỏ cỡ δ mới có tác dụng.
- **Lưu ý khi chọn lưới λ:** với Huber, vùng bậc hai là r²/2, nên λ của Huber tương ứng khoảng λ_mse/2 khi phần dư nhỏ.

**Cài đặt:**

- Trọng số của IRLS là w = δ/max(|r|, δ), tức là 1/max(|r|, δ) đã nhân δ. Vì vậy phạt được nhân tương ứng:
  - A_h = Xtᵀ diag(w_h) Xt + 2λδ·Pen
  - g_h = Xtᵀ diag(w_h) r_h − 2λδ·Pen w_h
- Hàng và cột bias của Pen bằng 0.
- **`pen`** nhận ma trận [L, L] đối xứng nửa xác định dương (numpy hoặc tensor), tự chuyển sang dtype và device của Xt.
- **DLinear có weight decay** dùng `pen = N⁻¹`, giống `ridge_pen`. Tương đương này chỉ phụ thuộc W_eff nên đúng với mọi hàm mất mát. Nhờ đó DLinear λ > 0 chạy được ở p = L + 1 thay vì 2L + 1, rẻ hơn khoảng 4 lần.
- **J báo ra** (qua `callback`, dùng để dừng) là δ·F/(n·H). Khi λ = 0 thì J đúng bằng `huber_objective` như trước.

**Chạy thử trên GPU** (ETTh1 H = 96, Huber δ = 1, chế độ pha, `pen = N⁻¹` với k = 25, λ = 281 chưa chọn trên validation):

- 7 vòng, 10,6 s.
- MSE test 0,367131 (λ = 0 là 0,367800).
- ‖W‖ giảm từ 3,554 xuống 3,353.

### 1.6. `tol` mặc định 1e-12 → 1e-9

- **Lý do:** MAE trên ETTh1 H = 96 chạy 1000 vòng vẫn chưa đạt 1e-12.
- **Với 1e-9:**
  - Dừng ở khoảng vòng 189 (khoảng 280 s ở chế độ pha).
  - MSE/MAE test lệch nghiệm ở vòng 1000 khoảng 2e-6 / 5e-6 (bảng ở mục 5 của báo cáo).
  - Huber δ = 1 dừng ở vòng 7 với test y như `tol = 1e-12`.

### 1.7. Docstring của module

Bỏ câu "nhân mọi trọng số với cùng hằng số không đổi nghiệm", vì câu này sai khi λ > 0. Thay bằng quy ước λ cho MAE/Huber.

## 2. `tests/test_irls.py`

**Thêm 6 test:**

| Test | Kiểm gì | Sai số đo được → ngưỡng |
|---|---|---|
| `TestMAE.test_float32_no_false_alarm` | float32 không báo lỗi giả; MAE (tính lại bằng float64) kém float64 rất ít | 2e-6 → 1e-4 |
| `TestMAE.test_mixed_precision_matches_float64` | `gram_dtype=float32` cho cùng W với float64; W trả về là float64 | 7e-12 → 1e-8 |
| `TestPenalty.test_large_delta_equals_ridge` | δ lớn + λ = ridge MSE với 2δλ, bias không phạt, 2 vòng | 4e-16 → 1e-10 |
| `TestPenalty.test_large_delta_penalty_matrix` | như trên với `pen` là ma trận xác định dương ngẫu nhiên | 4e-16 → 1e-10 |
| `TestPenalty.test_constrained_large_delta_equals_kkt` | NLinear + λ = hệ KKT có phạt | → 1e-10 |
| `TestPenalty.test_mae_matches_qp` | MAE + λ + `pen`: so F = Σ\|r\| + λwᵀPen w với quy hoạch bậc hai giải bằng SLSQP (scipy), so hai chiều; và phạt làm nghiệm khác hẳn λ = 0 | 1e-8–3e-8 → 1e-6 |

- **Vì sao `test_mae_matches_qp` so hai chiều:** nếu `irls` dùng sai quy ước λ (ví dụ lệch 2 lần), F của nó kém hẳn. Nếu SLSQP dừng sớm, `irls` sẽ tốt hơn hẳn. Cả hai trường hợp đều bị bắt.
- **SLSQP kết thúc với "Positive directional derivative"**, tức không cải thiện được nữa ở giới hạn chính xác. Đây không phải lỗi.
- **Không dùng phép kiểm "gradient bằng 0"**, dù đã thử trước. Ở δ = 1e-6, clamp(r/δ) nhạy tới mức r lệch 1e-12 làm gradient lệch 1e-6, nên gradient không giảm đều theo số vòng.

**Sửa 3 test:** `test_multi_init_agree`, `test_float32_no_false_alarm`, `test_mixed_precision_matches_float64` giờ truyền `tol=1e-12` rõ ràng. Ba test này kiểm độ chính xác của nghiệm, không kiểm tiêu chí dừng. Với `tol` mặc định mới, chúng dừng sớm hơn mức mà ngưỡng của chúng yêu cầu.

**Import:** thêm `minimize` cạnh `linprog`. Nếu thiếu scipy thì hai test bị bỏ qua.

## 3. `scripts/irls_dtype_check.py` (mới)

- **Phạm vi:** chạy một ô (ETTh1, L = 336, H = 96, Linear, λ = 0, khởi tạo OLS). Mỗi lần một mục tiêu (`mae`, `huber`) ở một kiểu (`float32`, `mixed`, `float64`).
- **Đo và ghi:**
  - Thời gian thật trên GPU, có khởi động trước và `cuda.synchronize`.
  - Đỉnh bộ nhớ.
  - κ(A_h) tính bằng float64 qua trị riêng.
  - J train, MSE/MAE test: cả ở điểm dừng lẫn theo từng vòng, đều tính lại bằng float64 sau khi đã bấm giờ.
- `--summary` so float32 và mixed với float64, cả ở điểm dừng lẫn ở cùng số vòng.
- Ghim `TOL = 1e-12` để tái lập đúng số của báo cáo, dù mặc định của `irls` đã đổi.
- Tự chuyển stdout sang UTF-8, vì console Windows mặc định cp1252 không in được tiếng Việt.

## 4. Dữ liệu và tài liệu

- `results/irls_dtype/{huber,mae}_{float32,mixed,float64}.{json,npy}`. MAE float64 chỉ chạy 200 vòng (khoảng 51 phút); đủ 1000 vòng sẽ mất khoảng 4,2 giờ.
- `docs/BAO_CAO_IRLS_DTYPE.md`: báo cáo 11.2 (thời gian, κ, kết quả, ngưỡng dừng, ước lượng cho cả grid, câu hỏi về GPU). Mục "Việc tiếp theo" đã đánh dấu phần `tol` và λ là xong.
- `README.md`: thêm dòng `scripts/` vào sơ đồ, và một gạch đầu dòng quy ước λ cho MAE/Huber.

## 5. Còn để ngỏ

- `docs/CHECKLIST_CON_LAI.md` chưa cập nhật. Các mục có thể đánh dấu:
  - "Cài IRLS cho MAE / Huber" và hai test của outline: đã có từ trước phiên này.
  - "Chạy IRLS bằng PyTorch trên GPU và đo thời gian chạy thật": làm trong phiên này.
- Chưa có `runner.py`. Khi viết nên:
  - dùng `gram_dtype=torch.float32`;
  - chọn lưới λ riêng cho MAE và Huber theo quy ước ở mục 1.5;
  - chia khối theo H cho ETTm1 và H = 720, vì không vừa 4 GB.

---

# Nhật ký thay đổi code: đo trên RTX 5060 Ti (30/09/2026)

Phạm vi: `NHIEM_VU_5060TI.md`, đo tốc độ, bộ nhớ và warm start của IRLS trên máy mới. Kết quả ở [BAO_CAO_5060TI.md](BAO_CAO_5060TI.md).

Test: đầu phiên 33 test, 1 lỗi nạp (`test_sgd`, xem mục 6.3) → cuối phiên **35, tất cả đạt**.

## Tóm tắt

| File | Loại | Nội dung |
|---|---|---|
| `src/solvers.py` | sửa | `irls`: thêm tham số `chunk=8`, chuyển thẳng cho `weight_gram` |
| `src/sgd.py` | khôi phục | đưa lại `lr_factor` và tham số `lradj` như ở commit d4173f1 |
| `scripts/bench_5060ti.py` | mới | đo chunk, chạy đầy đủ, từng ô, warm start khi quét λ, ước lượng grid |
| `results/bench_5060ti/` | mới | `chunk`, `full_{mae,huber}` (+ `.npy`), 12 file `cell_*`, `warmstart_{mae,huber}`, `grid_estimate` |
| `requirements.txt` | sửa | ghim `torch==2.14.0`, hướng dẫn cài bản CUDA trước; thêm `scipy` |
| `docs/BAO_CAO_5060TI.md` | mới | báo cáo |

Không đổi: toán của `irls`, `MAE_DELTA`, quy ước λ, các ngưỡng test, `docs/CHECKLIST_CON_LAI.md`.

## 6.1. `irls(..., chunk=8)`

- Chữ ký mới: `irls(Xt, Y, delta, W0, lam=0.0, pen=None, constrained=False, max_iter=1000, tol=1e-9, callback=None, gram_dtype=None, chunk=8)`.
- Chỉ chuyển xuống `weight_gram(Xg, w, chunk)`. Mặc định 8 giữ hành vi cũ.
- Trên RTX 5060 Ti nên dùng `chunk=16` (báo cáo mục 3.1).

## 6.2. `scripts/bench_5060ti.py`

- **Các bước:** `--step chunk | full | ettm1 | cell --data D --H H | warmstart [--loss mae|huber]`, và `--summary`. `--chunk` ghi đè chunk; mặc định đọc `best_chunk` từ `chunk.json`.
- **Chế độ và thiết lập:**
  - Luôn chạy chế độ pha.
  - Tắt TF32 ngay lúc import.
  - Mọi JSON có khối `hardware`: GPU, torch, CUDA, cuDNN, driver, OS, trạng thái TF32.
- **Chọn chunk:** lấy chunk nhỏ nhất có thời gian một vòng trong khoảng 2% so với nhanh nhất. Chênh lệch giữa các chunk cỡ nhiễu đo, còn bộ nhớ tăng tuyến tính theo chunk.
- **`full`:** so MSE/MAE test với `results/irls_dtype/*_mixed.json` ở cùng số vòng, ghi `pass_1e-6`.
- **`cell`:** tự giảm chunk một nửa khi OOM.
- **`warmstart`:**
  - Khởi tạo lạnh bằng `fit_with_bias(ridge_pen(2δλ·N⁻¹))`, theo mục 1.5. Với MAE, cách này thực chất là OLS.
  - Tự mở rộng lưới tối đa 2 giá trị nếu λ* ở đầu mút trên.
- **`--summary`:** ghi `grid_estimate.json`, gồm nội suy s/vòng ≈ a + b·n·H trên 4 ô góc, và chi phí quét λ.

## 6.3. `src/sgd.py`: khôi phục `lr_factor`

- **Lỗi:** commit 86c012c ("add irls") đã thay `lr_factor` và `lradj` bằng `StepLR`, nhưng `tests/test_sgd.py` (từ d4173f1) vẫn import `lr_factor`. Test này không nạp được, trên cả venv CPU cũ.
- **Sửa:** theo yêu cầu người dùng, lấy lại `src/sgd.py` của d4173f1 (`git checkout 86c012c~1 -- src/sgd.py`).
- **Hệ quả:** `train_sgd` lại mặc định `lradj="type1"`, đúng lịch lr của LTSF-Linear. Lịch cũ của notebook là `lradj="step"`.

## 6.4. Môi trường

- **Venv mới `.venv-gpu`:** torch 2.14.0+cu130, có sm_120. Không commit, vì venv tự sinh `.gitignore`.
- **`requirements.txt`:** thêm ghi chú phải cài torch từ index CUDA trước. `torch==2.14.0` khớp bản `+cu130` (PEP 440 bỏ qua nhãn local), nên `pip install -r` không cài đè bằng bản CPU; đã kiểm bằng `--dry-run`.

---

# Nhật ký thay đổi code: nhiệm vụ 2, tiêu chí dừng, ba phép kiểm, `runner.py` (30/09/2026)

Phạm vi: `docs/NHIEM_VU_2.md`. Kết quả ở [BAO_CAO_TIEU_CHI_DUNG.md](BAO_CAO_TIEU_CHI_DUNG.md).

## 7.1. `irls(..., patience=1)`

- **Chữ ký mới:** thêm `patience=1` vào cuối (sau `chunk`).
- **Cách dừng:** chỉ dừng khi J giảm tương đối ít hơn `tol` trong `patience` vòng liên tiếp. Một vòng giảm ≥ `tol` đặt lại bộ đếm về 0.
- **Lý do:** khi warm start từ nghiệm của λ liền trước, vòng đầu J giảm rất ít dù chưa hội tụ. Với `patience = 1`, MAE dừng ngay sau 1 vòng ở λ = 1, 3, 10 (BAO_CAO_5060TI mục 5).
- **J tăng trong ngưỡng `rise_tol`** (chạm độ chính xác của kiểu số): vẫn dừng ngay, không chờ `patience`. Trước đây nhánh này rơi vào phép so `< tol` vì mức giảm âm; giờ tách thành phép kiểm `J_new > J_old` riêng.
- **Mặc định `patience = 1`** cho đúng hành vi cũ, kể cả W và số vòng (test `test_patience_one_is_default` so bằng `torch.equal`).
- Với `tol = 0`, `irls` chạy đến `max_iter`, trừ khi J tăng trong ngưỡng. Dùng cho nghiệm tham chiếu.

Test mới trong `tests/test_irls.py`, lớp `TestPatience`:

| Test | Kiểm gì | Đo được |
|---|---|---|
| `test_patience_one_is_default` | `patience=1` cho cùng W và cùng số vòng như khi không truyền | trùng từng bit |
| `test_warm_start_does_not_stop_after_one_iteration` | MAE, warm từ nghiệm λ = 1 sang λ = 1,1, `tol = 1e-6`: `patience = 5` chạy nhiều vòng hơn `patience = 1` và J không lớn hơn | 1 vòng → 5 vòng |

## 7.2. Test mới cho Phần B (`tests/test_irls.py`)

| Lớp / test | Kiểm gì | Đo được → ngưỡng |
|---|---|---|
| `TestDLinearEquivalence.test_huber` | DLinear giải trực tiếp trong 2L chiều (Z = [X Pᵀ, X (I − P)ᵀ] + cột 1, `pen = I` cỡ 2L) cho W_eff và bias như L chiều với `pen = N⁻¹`; L = 24, k = 5, n = 400, λ = 5 | W_eff 1e-15, bias 0 → 1e-8 |
| `TestDLinearEquivalence.test_mae` | như trên với MAE: hàm mục tiêu có phạt (L chiều) của hai nghiệm, so hai chiều; và với phạt 2L chiều của chính nghiệm 2L | 0 → 1e-6 |
| `TestNLinearMAE.test_matches_constrained_lp` | NLinear MAE (`constrained=True`) so với `linprog` có `A_eq`: MAE không kém quá 1e-6 tương đối, tổng hàng = 1 tới 1e-10, và nghiệm khác MAE không ràng buộc | đạt |

- Hai dãy lặp IRLS (2L chiều và N⁻¹) trùng nhau từng vòng trong số học chính xác, vì mỗi vòng là một bài ridge có trọng số. Vì vậy test này kiểm phép tương đương và cách cài `pen`, không kiểm hội tụ.
- Để test không đạt tầm thường, `check_penalty_matters` đòi nghiệm phạt N⁻¹ khác nghiệm phạt I hơn 1e-2 tương đối (đo được khoảng 0,11).
- Test: 37 → **40, tất cả đạt**.

## 7.3. Script mới

| File | Nội dung |
|---|---|
| `scripts/stopping_check.py` | A.2 (nghiệm tham chiếu, `tol = 0`), A.3 (quét λ lạnh/warm theo `tol`, `patience`), A.4 (ba tiêu chí → `acceptance.json`). Bỏ qua file đã có, `--force` để chạy lại |
| `scripts/checks_mae.py` | B.1 (ba điểm khởi tạo), B.2 (2L chiều so với N⁻¹ trên dữ liệu thật; nếu 2L lỗi thì ghi chẩn đoán trị riêng của A_0 và vài vòng với A_h float64) |
| `scripts/runner.py` | **soạn, chưa chạy**: đường λ của cả grid → `results/lambda_path.csv` và `results/weights/`; `STOP` để trống chờ quyết định Phần A |
| `scripts/select_lambda.py` | **soạn, chưa chạy**: `lambda_path.csv` → `results/results.csv` (hai cách chọn λ*) |
| `scripts/bench_5060ti.py` | thêm `git_commit()` và `save_json()`, ghi `git_commit` vào mọi JSON; `run_irls` nhận `patience`, `constrained`, `quiet`; thêm cột `converged` |
| `.gitignore` | thêm `results/weights/` (W của cả grid khoảng 1,9 GB) |

## 7.4. Sau khi người dùng quyết định

- **Phần A, phương án (a′):** `stopping_check.py --tol0` cho `tol` riêng ở λ = 0 (`EXTRA_MAE_CONFIGS`). Cấu hình `tol = 1e-10, patience = 1`, λ = 0: `3e-11` đạt A.4 và được điền vào `STOP` của `runner.py`.
- **B.2, MAE:** thêm `checks_mae.py --check dlinear_f64`, so 2L chiều và N⁻¹ ở cùng số vòng, mọi thứ float64.
- **NLinear trong `runner.py`:** đổi từ `pen = I` sang `pen = diag(1, …, 1, 0)`, tức weight decay thật của NLinear, giống notebook 01.
  - MSE của NLinear giờ là dạng đóng (ridge trên `nlinear_transform`), nên bỏ nhánh `irls` δ lớn.
  - Lý do: với `pen = I`, λ* của NLinear luôn bằng 0 (BAO_CAO_TIEU_CHI_DUNG mục 6).
- **README:** thêm mục "Chạy grid" và quy ước phạt của NLinear.
- **Test mới** `TestPenalty.test_nlinear_penalty_equals_transform_ridge`: NLinear ràng buộc với `pen = diag(1, …, 1, 0)`, δ lớn, bằng ridge trên `nlinear_transform`. Đo được khoảng 1e-15, ngưỡng 1e-10. Test cũng đòi nghiệm khác phạt I. Test: 40 → **41, tất cả đạt**.

## 7.5. Còn để ngỏ (trước khi quyết định; giữ lại để đối chiếu)

- ~~**Phần A không đạt A.4**~~ **Đã xử lý** bằng phương án (a′), mục 7.4.: chỉ trượt ở MSE val tại λ = 0, lệch 1,04e-5–1,1e-5 ở `tol = 1e-10`. Chờ người dùng chọn phương án (BAO_CAO_TIEU_CHI_DUNG mục 1), rồi điền `STOP` trong `runner.py`.
- ~~**B.2 với MAE trên dữ liệu thật**~~ **Đã xử lý** bằng phép so float64 ở cùng số vòng (`dlinear_f64`), mục 7.4.: cách giải 2L chiều lỗi `J tăng` ở chế độ pha (A_h suy biến, sai số float32 cỡ phạt 2λδ). Chưa chạy bản float64 vì mất khoảng 75 phút.

## 7.6. `irls(..., gram_fallback=True)`: fallback float64 theo từng h

- **Lỗi:** khi chạy grid, ETTh2 H = 720, Linear MAE λ = 0 báo `RuntimeError: J tăng ở vòng 16: 5,075604e-07 → 5,077229e-07`.
- **Chẩn đoán** (BAO_CAO_TIEU_CHI_DUNG mục 8):
  - Chỉ 1 trong 720 bước h hỏng: h = 3. Ở đó κ(A_3) = 8,3e7, nên κ·ε₃₂ ≈ 5.
  - A_3 lập bằng float32 không còn xác định dương, J_3 tăng 75%. Lập bằng float64 thì J giảm bình thường.
- **Sửa:** chỉ ở chế độ pha (`gram_dtype` khác kiểu của Xt). Sau mỗi bước, tính J_h theo từng h. Bước h nào có J_h tăng quá `rise_tol` thì lập lại A_h đó bằng kiểu của Xt (float64) và giải lại.
  - J tách theo h, nên các h khác giữ nguyên. Toán của IRLS không đổi.
  - Chi phí: khoảng 0,04 s cho mỗi h hỏng ở ETTh2 H = 720, so với 26,6 s nếu lập cả 720 A_h bằng float64.
- **Mặc định bật.** Khi không có bước h nào hỏng, kết quả trùng từng bit với `gram_fallback=False`, vì J tính theo đúng công thức cũ.
- **`irls.last_fallbacks`** ghi số bước h đã lập lại trong lần gọi gần nhất. `runner.py` in số này ra log, nhưng không thêm cột vào CSV để giữ header cũ.
- **Test mới (`TestGramFallback`):**
  - Hai cột gần cộng tuyến: không fallback thì báo J tăng ở vòng 48; có fallback thì MAE khớp float64 tới khoảng 3e-13.
  - Dữ liệu điều kiện tốt: kết quả trùng từng bit.
  - Test: 41 → **43, tất cả đạt**.
- **Lưu ý:** các dòng grid đã chạy trước khi sửa không bị lỗi, nhưng có thể đã có bước h đi sai rồi tự hồi phục mà không báo, vì tổng J vẫn giảm. Chưa kiểm.

## 7.7. Fallback lập A_h theo khối nhỏ hơn

- **Lỗi:** ETTm1 H = 720, Linear MAE λ = 0 chạy hơn 79 phút (ước lượng khoảng 37 phút).
  - Tiến trình dùng 14,5 GB VRAM cộng 1,3 GB bộ nhớ dùng chung, tức đã tràn sang RAM hệ thống.
  - Nguyên nhân: fallback float64 gọi `weight_gram` với cùng `chunk = 16`, nên tensor trung gian n·16·p·8 byte khoảng 10 GB ở ETTm1. Ở ETTh2 chỉ khoảng 2,3 GB nên không lộ ra.
- **Sửa:** fallback dùng `fb_chunk = chunk · (cỡ kiểu gram)/(cỡ kiểu Xt)`, tức 8 với float32/float64. Tensor trung gian của fallback vì vậy không lớn hơn của bước thường. Toán không đổi.
- **Đo trên ETTm1 H = 720:** bước float32 chunk 16 thêm 5,95 GiB; fallback float64 chunk 8 (40 bước h) thêm 4,75 GiB. Test 43/43 đạt.
- Đã dừng runner (mất phần đang chạy dở của Linear MAE λ = 0) và chạy tiếp.
