# Mục 11.2: IRLS chạy float32 hay float64 trên GPU?

Ngày chạy: 30/09/2026. Script: [`scripts/irls_dtype_check.py`](../scripts/irls_dtype_check.py). Số liệu thô (số đo, quỹ đạo J và sai số test theo từng vòng, W): [`results/irls_dtype/`](../results/irls_dtype/).

## 1. Kết luận

- **Chọn cách pha độ chính xác** (`irls(..., gram_dtype=torch.float32)`): lập A_h bằng float32, còn lại tính bằng float64.
  - Cho đúng nghiệm float64: Huber lệch W 2,2e-10; MAE ở cùng số vòng lệch MSE/MAE test khoảng 1e-6.
  - Chi phí mỗi vòng chỉ hơn float32 thuần 15% (1,49 s so với 1,29 s), nhanh hơn float64 thuần 10 lần (15,2 s).
- **float32 thuần đủ cho Huber (δ = 1)** nhưng **không đủ cho MAE (δ = 1e-6)**.
  - Với MAE, float32 dừng ở vòng 62, vì J hết giảm được ở độ phân giải float32.
  - Lúc đó MSE/MAE test lệch nghiệm hội tụ khoảng 2,5e-5 / 3e-5 (tương đối khoảng 7e-5).
  - Mức lệch này nhỏ hơn khoảng 300 lần so với chênh lệch giữa mục tiêu MAE và MSE (8e-3 trên MAE test), nên không đổi kết luận RQ2. Nhưng nó cùng cỡ với chênh lệch giữa NLinear và Linear ở RQ1 (3,7e-5), nên không dùng được cho các so sánh tinh.
- **Nguồn lỗi của float32 không phải do A_h xấu điều kiện tới mức giải sai.**
  - κ(A_h) của MAE khoảng 6e5–1,7e6, lớn hơn XᵀX của MSE (7,5e3) khoảng 200 lần. Vậy κ·ε₃₂ ≈ 0,2 < 1: phép giải float32 vẫn có nghĩa.
  - Lỗi đến từ chỗ J và phần dư ở float32 chỉ phân giải được tới khoảng 1e-7 tương đối, nên IRLS dừng như thể `tol ≈ ε₃₂`.
- **float64 chậm hơn float32 khoảng 12 lần mỗi vòng**, không tới 30–60 lần như ước tính ban đầu.
  - float64 đã chạm trần phần cứng: 84 GFLOP/s, FP64 của GTX 1650 chỉ bằng 1/32 FP32.
  - float32 qua `einsum` mới đạt khoảng 1 TFLOP/s, tức khoảng 1/3 đỉnh.
- **Ngưỡng dừng `tol = 1e-12` quá chặt cho MAE**: sau 1000 vòng vẫn chưa đạt. Nên dùng khoảng `tol = 1e-9` đến `1e-10` (mục 5).

## 2. Thiết lập

| | |
|---|---|
| Ô | ETTh1, L = 336, H = 96, Linear có bias, λ = 0 |
| Dữ liệu | train n = 57 463, p = 337 (336 lag + cột 1); test 19 495 hàng |
| Khởi tạo | nghiệm OLS (`fit_with_bias(ols)`), float64 |
| Mục tiêu | MAE: Huber δ = `MAE_DELTA` = 1e-6. Huber: δ = 1,0 (mặc định của `torch.nn.HuberLoss`, dữ liệu đã chuẩn hóa) |
| Dừng | `tol = 1e-12` (J giảm tương đối), `max_iter = 1000`. MAE float64 chủ động giới hạn 200 vòng (xem mục 4) |
| Kiểu số | `float32`: mọi thứ float32. `float64`: mọi thứ float64. `mixed`: dữ liệu, W, phần dư, gradient, J và phép giải float64; chỉ lập A_h bằng float32 |
| Máy | GTX 1650 4 GB (driver WDDM), Ryzen 7 4800H, torch 2.6.0+cu124 |
| Chấm điểm | Mọi J train, MSE/MAE test và κ đều tính lại bằng float64, để ba kiểu được chấm như nhau |

## 3. Thời gian

### Một vòng IRLS (lập 96 ma trận A_h chiếm gần hết thời gian)

| | lập A_h | giải 96 hệ | GFLOP/s khi lập A_h |
|---|---|---|---|
| GPU float32, `chunk = 8` (mặc định) | 1,27 s | 0,02 s | 980 |
| GPU float32, `chunk = 16` | 0,87 s | 0,02 s | 1 440 |
| GPU float64 | 14,9 s | 0,10 s | 84 |
| CPU float64 | 17,7 s | 0,19 s | 71 |
| CPU float32 | 6,9 s | 0,05 s | 181 |

### Trong lần chạy thật (trung vị mỗi vòng)

| | s/vòng | so với float32 | đỉnh bộ nhớ GPU |
|---|---|---|---|
| float32 | 1,29 | 1× | 826 MB |
| mixed | 1,49 | 1,15× | 1 099 MB |
| float64 | 15,2 | 11,8× | 1 640 MB |

GPU float64 chỉ nhanh hơn CPU float64 khoảng 1,2 lần.

## 4. Kết quả

OLS (khởi tạo): MSE test 0,370235, MAE test 0,391538.

### Huber, δ = 1

| | vòng | thời gian | J train | MSE test | MAE test | ‖W − W₆₄‖/‖W₆₄‖ |
|---|---|---|---|---|---|---|
| float32 | 5 | 6,5 s | 0,14688852519 | 0,367800 | 0,387990 | 1,5e-4 |
| mixed | 10 | 15,0 s | 0,14688852475 | 0,367800 | 0,387989 | 2,2e-10 |
| float64 | 10 | 151,7 s | 0,14688852475 | 0,367800 | 0,387989 | — |

- κ(A_h) = 7,4e3–7,5e3, bằng κ(XᵀX). Trọng số nằm trong [0,21; 1].
- float32 lệch MSE test 1,6e-7 và MAE test 8,5e-7, không đáng kể.

### MAE, δ = 1e-6

| | vòng | thời gian | J train (×1e-7) | MSE test | MAE test | κ(A_h) min / trung vị / max |
|---|---|---|---|---|---|---|
| float32 | 62 (J hết giảm) | 79,9 s | 3,9239642819 | 0,365312 | 0,383350 | 6,5e5 / 1,2e6 / 1,6e6 |
| mixed | 1000 (chạm `max_iter`) | 1 476 s | 3,9239570911 | 0,365338 | 0,383320 | 5,9e5 / 1,2e6 / 1,6e6 |
| float64 | 200 (giới hạn) | 3 039 s | 3,9239573132 | 0,365335 | 0,383324 | 7,1e5 / 1,3e6 / 1,7e6 |

Trọng số nằm trong [2,0e-7; 1], trải khoảng 5e6 lần, đúng như dự kiến ở 11.2. Nhưng κ(A_h) chỉ tăng khoảng 200 lần so với MSE, không tăng nhiều bậc.

MAE float64 được giới hạn 200 vòng vì chạy đủ 1000 vòng mất khoảng 4,2 giờ. Để kiểm chế độ pha có trùng float64 không, chỉ cần so ở cùng số vòng.

### So ở cùng số vòng: tách lỗi do kiểu số khỏi lỗi do dừng sớm

| So sánh | J | MSE test | MAE test |
|---|---|---|---|
| float32 và float64, cùng vòng 62 | lệch ≤ 3,4e-6 tương đối (J của float32 tính bằng float32) | 0,365312 và 0,365310 | 0,383350 và 0,383351 |
| mixed và float64, cùng vòng 200 | mixed thấp hơn 3,8e-9 tương đối | 0,365336 và 0,365335 | 0,383325 và 0,383324 |

- Ở cùng số vòng, float32 gần như trùng float64. Sai khác cuối cùng của float32 đến từ việc **dừng ở vòng 62**, không phải từ từng bước đi sai.
- mixed có J thấp hơn float64 ở cả 200/200 vòng: chênh tối đa 4,9e-6 tương đối, co dần còn 3,8e-9 ở vòng 200. Tức sai số của A_h float32 không làm chậm hội tụ; nó còn nhanh hơn chút, lý do chưa kiểm.
- Không nên so bằng ‖W − W₆₄‖ ở MAE. Gần nghiệm, hàm mục tiêu rất phẳng: W của float32 lệch 1,7e-2 mà J chỉ kém 1,8e-6 tương đối.

## 5. Ngưỡng dừng cho MAE

Tính từ quỹ đạo của mixed. Mốc so là vòng 1000; bản thân mốc này cũng chưa hội tụ hẳn. Từ vòng 603 tới 1000, J còn giảm 1,6e-9 tương đối, còn MSE/MAE test đổi 4,8e-8 / 3,9e-7.

| `tol` | dừng ở vòng | thời gian (mixed) | MSE test lệch | MAE test lệch |
|---|---|---|---|---|
| 1e-6 | 38 | 57 s | −7,8e-5 | +7,4e-5 |
| 1e-7 | 62 | 92 s | −2,5e-5 | +3,1e-5 |
| 1e-8 | 106 | 158 s | −8,8e-6 | +1,3e-5 |
| 1e-9 | 189 | 281 s | −2,3e-6 | +4,8e-6 |
| 1e-10 | 336 | 499 s | −6,5e-7 | +1,5e-6 |
| 1e-11 | 603 | 893 s | −4,8e-8 | +3,9e-7 |

float32 thuần dừng đúng ở vòng 62, trùng với `tol = 1e-7` ≈ ε₃₂. Nó không thể đi xa hơn dòng thứ hai của bảng.

## 6. Có cần GPU tốt hơn không?

- **Với một ô ETTh1 H = 96: không.** MAE ở chế độ pha mất 5–8 phút (`tol` 1e-9 đến 1e-10), Huber mất 15 giây.
- **Với cả grid: máy hiện tại chạy được nhưng lâu.** Ước lượng thô:
  - Chi phí mỗi vòng tỷ lệ với n·H. Quy đổi ra "số ô ETTh1 H = 96": mỗi tổ hợp (mô hình, λ) tương đương 84 ô (ETTh1, ETTh2 mỗi dataset 13,3; ETTm1 57,5). 5 tổ hợp cần IRLS (bỏ DLinear λ = 0 vì ≡ Linear) cộng lại khoảng 420 ô.
  - Giả sử mọi ô cần số vòng như ô ETTh1 H = 96, MAE ở chế độ pha mất khoảng **33–58 giờ** với `tol` 1e-9 đến 1e-10. Huber mất khoảng 2 giờ.
  - Nếu dùng float64 thuần thì khoảng 340 giờ, không khả thi.
- **ETTm1 H = 720 không vừa 4 GB theo cách viết hiện tại.**
  - n = 234 535. Riêng Y, phần dư và trọng số ở float64 đã mỗi thứ 1,26 GiB, cộng Xt 0,59 GiB và khối `einsum` 2,36 GiB.
  - Cách khắc phục: chia H thành từng khối cột rồi gọi `irls` cho từng khối. 96 (hay 720) bài toán theo h độc lập với nhau, chỉ dùng chung tiêu chí dừng.
- Nếu đổi GPU, giờ thứ cần là **thông lượng FP32 và VRAM**, không phải FP64, vì phần nặng đã chạy bằng float32. Trước khi đổi GPU, nên thử các cách rẻ hơn:
  - `chunk = 16` (nhanh hơn 1,46 lần khi đo).
  - Nới `tol`.
  - Chia khối theo H.

## 7. Thay đổi code

`src/solvers.py`, hàm `irls`:

1. **Ngưỡng "J tăng" theo ε của kiểu số**: `rise_tol = max(1e-10, 100·ε)`.
   - Ngưỡng cũ 1e-10 báo `RuntimeError` giả ngay khi chạy float32: J "tăng" 1 ulp do làm tròn.
   - float64 giữ nguyên hành vi.
   - J tăng trong ngưỡng thì coi là đã chạm độ chính xác và dừng.
2. **Bước giải dạng hiệu chỉnh** ŵ = W + A⁻¹g, với g = Xtᵀ diag(w) r.
   - Trong số học chính xác, dạng này trùng với A⁻¹c. Nhưng điểm dừng g = 0 được tính ở độ chính xác của phần dư, nên cho phép A_h kém chính xác hơn.
   - Dạng này cũng giúp float32 thuần: MAE đi được 62 vòng thay vì 38.
3. **Tham số `gram_dtype`**: kiểu số để lập A_h. Mặc định là kiểu của Xt.
4. **Tham số `callback(it, W, J)`**: để ghi quỹ đạo. Không đổi hành vi.

`tests/test_irls.py` thêm hai test, hiện 28/28 test đạt:

- **float32 không báo lỗi giả**: MAE kém float64 dưới 1e-4 tương đối, đo được khoảng 2e-6.
- **Chế độ pha trùng float64**: W lệch dưới 1e-8, đo được 7e-12.

## 8. Việc tiếp theo

- ~~Chọn `tol` cho grid~~: **đã làm**. Mặc định của `irls` giờ là `tol = 1e-9`. Script của báo cáo này ghim `TOL = 1e-12` để tái lập đúng các số ở trên.
- ~~Thêm tham số phạt λ vào `irls`~~: **đã làm**. Thêm `lam` và `pen`, quy ước ghi ở [NHAT_KY_THAY_DOI.md](NHAT_KY_THAY_DOI.md).
- Dùng `gram_dtype=torch.float32` làm mặc định trong `runner.py`.
- Chia khối theo H để chạy được ETTm1 và H = 720 trên 4 GB.

## Tái lập

```bash
python scripts/irls_dtype_check.py --loss huber --dtype float32
python scripts/irls_dtype_check.py --loss huber --dtype mixed
python scripts/irls_dtype_check.py --loss huber --dtype float64
python scripts/irls_dtype_check.py --loss mae --dtype float32
python scripts/irls_dtype_check.py --loss mae --dtype mixed
python scripts/irls_dtype_check.py --loss mae --dtype float64 --max-iter 200
python scripts/irls_dtype_check.py --summary
```

Trên console Windows cần đặt `PYTHONIOENCODING=utf-8` hoặc chạy từ terminal UTF-8. Script đã tự chuyển stdout sang UTF-8, nhưng thông báo lỗi của Python thì không.
