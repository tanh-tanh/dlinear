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
