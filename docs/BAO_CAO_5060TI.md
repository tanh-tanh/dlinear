# Đo IRLS trên RTX 5060 Ti 16 GB

Ngày chạy: 30/09/2026. Script: [`scripts/bench_5060ti.py`](../scripts/bench_5060ti.py). Số liệu thô (có thông tin phần cứng trong mọi file): [`results/bench_5060ti/`](../results/bench_5060ti/). Nhiệm vụ: `NHIEM_VU_5060TI.md`. So với máy cũ: [BAO_CAO_IRLS_DTYPE.md](BAO_CAO_IRLS_DTYPE.md).

## 1. Kết luận

- **Mỗi vòng IRLS ở ETTh1 H = 96, chế độ pha, mất 0,158 s** (trung vị trong lần chạy MAE đầy đủ). GTX 1650 mất 1,49 s, tức nhanh hơn **9,4 lần**.
  - Lập 96 ma trận A_h (`weight_gram`, float32) đạt 5,9 TFLOP/s, so với 1,0–1,4 TFLOP/s trên GTX 1650. Mức này khoảng 1/4 đỉnh FP32 lý thuyết của card.
  - Nghiệm khớp máy cũ: MSE/MAE test ở cùng số vòng lệch 3,3e-7 / 3,1e-8 (MAE) và 2,5e-11 / 1,8e-11 (Huber), đều dưới ngưỡng 1e-6.
- **Dùng `chunk = 16`.** Với chunk từ 8 tới 96, thời gian một vòng chỉ chênh nhau 0,170–0,183 s, cỡ nhiễu đo. Trong khi đó bộ nhớ tăng tuyến tính theo chunk: đỉnh 1,1 GB ở chunk 8, 7,7 GB ở chunk 96. Chunk 16 là chunk nhỏ nhất nằm trong khoảng 2% so với cấu hình nhanh nhất.
- **Không cần chia khối theo H.** Ô lớn nhất (ETTm1 H = 720, n = 234 535) chạy được với chunk 16: đỉnh bộ nhớ 11,9 GB trên 15,9 GB, 4,70 s/vòng.
- **Ước lượng thời gian grid** (5 tổ hợp (mô hình, λ) × 12 ô, chưa tính quét λ), giả định số vòng mọi ô như ETTh1 H = 96:

  | | ước lượng |
  |---|---|
  | MAE, `tol = 1e-8` (107 vòng) | 1,9 giờ |
  | MAE, `tol = 1e-9` (189 vòng) | 3,4 giờ |
  | Huber, `tol = 1e-9` (7 vòng) | 0,13 giờ |

  - GTX 1650 cần khoảng 33–58 giờ cho MAE, với `tol` 1e-9 đến 1e-10.
  - ETTm1 vẫn chiếm khoảng 67% chi phí.
- **Quét λ cho 3 mô hình × 12 ô:**

  | | khởi tạo lạnh | warm start |
  |---|---|---|
  | MAE, `tol = 1e-8` | khoảng 10 giờ | khoảng 5,1 giờ |
  | Huber | 0,7 giờ | 0,5 giờ |

- **Warm start tiết kiệm 51% số vòng cho MAE** (948 → 469 vòng cho cả lưới 10 λ) và **35% cho Huber** (68 → 44 vòng). λ* chọn trên validation giống nhau ở cả hai kiểu:
  - DLinear, ETTh1 H = 96: λ* = 100 theo chính mục tiêu huấn luyện, λ* = 300 theo MSE. Cả hai giống nhau cho MAE và Huber, và không nằm ở đầu mút lưới.
- **Cảnh báo cho MAE:** ảnh hưởng của λ lên validation nhỏ cỡ sai số dừng của IRLS ở `tol = 1e-8`.
  - MAE val chỉ đổi 1,3e-5 giữa λ = 0 và λ* = 100.
  - Nghiệm warm và nghiệm lạnh ở cùng λ có J lệch nhau chỉ khoảng 1e-8 tương đối, tức nằm trong `tol`. Nhưng MSE val của chúng lệch tới 7,5e-5 (mục 5).
  - Vì vậy λ* của MAE chọn ở `tol = 1e-8` chưa chắc chắn. Chi tiết ở mục 5 và 6.

## 2. Thiết lập

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti, 15,9 GiB, compute capability 12.0 (sm_120), driver 616.92 (WDDM, card đồng thời chạy màn hình) |
| CPU, OS | Intel Core i5-12400F, 32 GB RAM, Windows 11 Pro 10.0.26100 |
| Phần mềm | Python 3.13.5, torch 2.14.0+cu130 (CUDA 13.0, cuDNN 9.24), numpy 2.3.1, venv `.venv-gpu` |
| Kiểu số | Chế độ pha: Xt, Y, W0, phần dư, g, J và phép giải bằng float64; A_h lập bằng float32 (`gram_dtype=torch.float32`) |
| TF32 | Tắt: `allow_tf32 = False` (matmul, cuDNN), `float32_matmul_precision = 'highest'`. Ghi lại trong mọi JSON |
| Mô hình | Linear có bias, λ = 0, khởi tạo OLS (`fit_with_bias(ols)`). Bước 3 dùng DLinear (`pen = N⁻¹`, k = 25) |
| Đo giờ | Có khởi động trước (một lần gọi `irls` 2 vòng trên 2 000 hàng, cộng một lần gọi thử mỗi cấu hình), `cuda.synchronize()` trước mỗi lần đọc đồng hồ. Bộ nhớ: `reset_peak_memory_stats` rồi `max_memory_allocated` |
| Chấm điểm | J, MSE/MAE train, val, test tính lại bằng float64 trên CPU |

Máy cũ: GTX 1650 4 GB, Ryzen 7 4800H, torch 2.6.0+cu124 (không hỗ trợ sm_120).

## 3. Tốc độ và bộ nhớ

### 3.1. Quét `chunk`, ETTh1 H = 96 (n = 57 463, p = 337)

Mỗi cấu hình: 1 lần khởi động, 5 lần đo, lấy trung vị. Số phép tính của `weight_gram` là n·p²·H = 6,26e11. "Một vòng irls" là `irls(..., max_iter=1)` ở chế độ pha với trọng số MAE, gồm cả một lần tính J ban đầu.

| chunk | `weight_gram` (s) | GFLOP/s | bộ nhớ thêm khi lập A_h | một vòng irls (s) | đỉnh bộ nhớ cả vòng |
|---|---|---|---|---|---|
| 8 | 0,1176 | 5 326 | 636 MB | 0,1834 | 1 134 MB |
| **16** | **0,1053** | **5 948** | **1 231 MB** | **0,1704** | **1 729 MB** |
| 32 | 0,1133 | 5 531 | 2 420 MB | 0,1769 | 2 917 MB |
| 64 | 0,1055 | 5 939 | 4 798 MB | 0,1704 | 5 296 MB |
| 96 | 0,1015 | 6 170 | 7 176 MB | 0,1751 | 7 674 MB |

- Chênh lệch giữa các chunk rất nhỏ. Ở lần đo đầu tiên (đã ghi đè), chunk 64 nhanh nhất với 0,1716 s, còn chunk 16 là 0,1725 s. Vì vậy chunk được chọn theo quy tắc "nhỏ nhất trong 2% so với nhanh nhất", ghi trong `chunk.json`.
- `weight_gram` chiếm khoảng 62% một vòng. Phần còn lại, khoảng 0,065 s, gồm phần dư, g và J (các phép nhân float64 cỡ n·p·H), phép giải 96 hệ, và phép chuyển kiểu. Phần này chưa đo tách.

So với GTX 1650:

| | GTX 1650 | RTX 5060 Ti | nhanh hơn |
|---|---|---|---|
| `weight_gram` chunk 8 | 1,27 s (980 GFLOP/s) | 0,118 s (5 326) | 10,8× |
| `weight_gram` chunk 16 | 0,87 s (1 440 GFLOP/s) | 0,105 s (5 948) | 8,3× |
| s/vòng chế độ pha, lần chạy thật | 1,49 s (chunk 8) | 0,158 s (chunk 16) | 9,4× |
| đỉnh bộ nhớ, chế độ pha | 1 099 MB (chunk 8) | 1 718 MB (chunk 16) | |

### 3.2. Chạy đầy đủ, ETTh1 H = 96, chunk 16

| | vòng | tổng thời gian | s/vòng (tổng / vòng; trung vị) | J train | MSE test | MAE test |
|---|---|---|---|---|---|---|
| Huber δ = 1, `tol = 1e-9` | 7 | 1,3 s | 0,189; 0,155 | 0,14688852476 | 0,367800 | 0,387989 |
| MAE δ = 1e-6, `tol = 1e-8` | 107 | 17,0 s | 0,158; 0,158 | 3,9239584716e-7 | 0,365329 | 0,383333 |

- Vòng đầu của Huber mất 0,39 s, gồm một phần khởi động, nên tổng chia số vòng cao hơn trung vị.
- Số vòng khớp dự kiến trong báo cáo cũ: khoảng 106 vòng cho MAE ở `tol = 1e-8`, khoảng 7 vòng cho Huber.

**Kiểm chéo với GTX 1650** (`results/irls_dtype/*_mixed.json`, chế độ pha, cùng khởi tạo OLS), so ở cùng số vòng:

| | vòng | MSE test (1650 → 5060 Ti) | Δ | MAE test (1650 → 5060 Ti) | Δ | lệch lớn nhất qua mọi vòng |
|---|---|---|---|---|---|---|
| Huber | 7 | 0,3677997279 → 0,3677997278 | −2,5e-11 | 0,3879889211 → 0,3879889211 | −1,8e-11 | 2,1e-8 |
| MAE | 107 | 0,3653291232 → 0,3653287971 | −3,3e-7 | 0,3833327955 → 0,3833328267 | +3,1e-8 | 1,5e-5 (vòng 11) |

- **Đạt ngưỡng 1e-6 ở điểm dừng.** J của MAE lệch 1,4e-8 tương đối.
- Với MAE, hai quỹ đạo tách nhau nhiều nhất ở vòng 10–20 (lệch khoảng 1e-5), rồi co dần: 4e-7 ở vòng 100.
  - Lý do: A_h float32 làm tròn khác nhau trên hai phần cứng, nên các bước đầu đi hơi khác. Nhưng điểm dừng g = 0 tính bằng float64, nên hai quỹ đạo cùng về một nghiệm.
  - Đây đúng là hành vi đã thấy khi so chế độ pha với float64 trong báo cáo cũ.

### 3.3. Ô lớn nhất: ETTm1 H = 720

- Dữ liệu: L = 336, mốc chia nhân 4 (train 34 560 hàng). n = 234 535, p = 337.
- Kích thước tensor trên GPU:
  - Xt [234 535, 337], float64: 603 MiB.
  - Y [234 535, 720]: 1 288 MiB.
  - W [720, 337]: 1,9 MiB.
- Nạp dữ liệu và giải OLS trên CPU mất 2,8 s.

Chạy MAE, chế độ pha, `max_iter = 3`, chunk 16. Cảnh báo "chưa hội tụ" là bình thường ở đây.

| chunk | vòng 1 | vòng 2 | vòng 3 | s/vòng (trung vị vòng 2–3) | đỉnh bộ nhớ |
|---|---|---|---|---|---|
| 16 | 5,55 s | 4,70 s | 4,71 s | **4,70** | **11 876 MB** |

- Không OOM ở chunk 16, nên không phải giảm chunk.
- Đỉnh bộ nhớ này để lại khoảng 4 GB. Driver WDDM và màn hình đã chiếm khoảng 1,3 GB từ trước. Nếu cần thêm chỗ, chunk 8 bớt được khoảng 2,5 GB: bộ nhớ thêm của `weight_gram` là n·chunk·p·4 byte, tức khoảng 5,1 GB ở chunk 16.

### 3.4. Cả 12 ô, và ước lượng chi phí grid

Mỗi ô rẻ nên đo cả 12 ô, mỗi ô 3 vòng MAE (chunk 16). Phép nội suy s/vòng ≈ a + b·n·H khớp trên 4 ô góc (ETTh1 và ETTm1, H = 96 và 720): a = 0,027 s, b = 2,77e-8 s. Sau đó kiểm trên 8 ô còn lại.

| ô | n | n·H | s/vòng đo | nội suy | sai | đỉnh bộ nhớ |
|---|---|---|---|---|---|---|
| ETTh1 H = 96 | 57 463 | 5,5e6 | 0,161 | 0,180 | +11% | 1,7 GB |
| ETTh1 H = 192 | 56 791 | 1,1e7 | 0,308 | 0,329 | +7% | 2,0 GB |
| ETTh1 H = 336 | 55 783 | 1,9e7 | 0,544 | 0,547 | +0% | 2,3 GB |
| ETTh1 H = 720 | 53 095 | 3,8e7 | 1,172 | 1,087 | −7% | 3,3 GB |
| ETTh2 (mọi H) | như ETTh1 | | 0,161 / 0,307 / 0,541 / 1,165 | | | như ETTh1 |
| ETTm1 H = 96 | 238 903 | 2,3e7 | 0,608 | 0,663 | +9% | 6,6 GB |
| ETTm1 H = 192 | 238 231 | 4,6e7 | 1,216 | 1,296 | +7% | 7,3 GB |
| ETTm1 H = 336 | 237 223 | 8,0e7 | 2,185 | 2,238 | +2% | 8,4 GB |
| ETTm1 H = 720 | 234 535 | 1,7e8 | 4,702 | 4,713 | +0% | 11,9 GB |

- Nội suy theo n·H sai tới ±11%. Thực tế s/vòng gần tỷ lệ thuận với H trong từng dataset, và tăng chậm hơn n: ETTm1 / ETTh1 = 3,8 lần, trong khi n gấp 4,2 lần.
- Vì cả 12 ô đều đã đo, ước lượng bên dưới **dùng số đo, không dùng nội suy**.

Tổng s/vòng của 12 ô: 13,07 s. Ước lượng cho 5 tổ hợp (mô hình, λ) cần IRLS: Linear λ = 0, Linear λ*, DLinear λ*, NLinear λ = 0, NLinear λ*.

Giả định:

- **Số vòng mọi ô như ETTh1 H = 96**: MAE 107 vòng ở `tol = 1e-8`, 189 vòng ở `tol = 1e-9` (lấy từ quỹ đạo máy cũ); Huber 7 vòng.
- Chi phí mỗi vòng như Linear λ = 0. NLinear (ràng buộc) và DLinear (`pen`) có cùng p = 337, chỉ thêm một cột vế phải hoặc một ma trận phạt, nên phép giả định này là hợp lý nhưng chưa đo.
- Chưa tính nạp dữ liệu và OLS: khoảng 1–3 s mỗi ô.

| ô | MAE `tol` 1e-8 (phút) | MAE `tol` 1e-9 (phút) | Huber (phút) |
|---|---|---|---|
| ETTh1 H = 96 / 192 / 336 / 720 | 1,4 / 2,7 / 4,9 / 10,4 | 2,5 / 4,9 / 8,6 / 18,5 | 0,1 / 0,2 / 0,3 / 0,7 |
| ETTh2 H = 96 / 192 / 336 / 720 | 1,4 / 2,7 / 4,8 / 10,4 | 2,5 / 4,8 / 8,5 / 18,4 | 0,1 / 0,2 / 0,3 / 0,7 |
| ETTm1 H = 96 / 192 / 336 / 720 | 5,4 / 10,8 / 19,5 / 41,9 | 9,6 / 19,2 / 34,4 / 74,0 | 0,4 / 0,7 / 1,3 / 2,7 |
| **Tổng** | **1,9 giờ** | **3,4 giờ** | **0,13 giờ** |

- Theo dataset (MAE `tol = 1e-8` cộng Huber): ETTh1 0,35 giờ, ETTh2 0,34 giờ, ETTm1 1,38 giờ (67%).
- Lần chạy đơn lẻ dài nhất là ETTm1 H = 720 MAE ở `tol = 1e-9`: 189 × 4,7 s ≈ 15 phút, dưới ngưỡng 60 phút.

## 4. Warm start khi quét λ

- Ô: ETTh1 H = 96, **DLinear** (Linear có bias với `pen = N⁻¹`, k = 25), chế độ pha, chunk 16. λ tăng dần: {0, 1, 3, …, 10 000}.
- **Khởi tạo lạnh:** nghiệm ridge MSE dạng đóng ở cùng λ, quy đổi theo mục 1.5 của [NHAT_KY_THAY_DOI.md](NHAT_KY_THAY_DOI.md). Với δ lớn, F ≈ ‖r‖²/(2δ) + λwᵀPen w, nên dùng `fit_with_bias(ridge_pen(Pen = 2δλ·N⁻¹))`.
  - Với Huber δ = 1, đây là ridge với 2λ·N⁻¹.
  - Với MAE δ = 1e-6, 2δλ ≤ 0,02, nên **khởi tạo lạnh của MAE thực chất là OLS** ở mọi λ.
- **Warm start:** khởi tạo từ nghiệm IRLS của λ liền trước. λ = 0 dùng chung lần chạy lạnh (cùng khởi tạo OLS).
- λ* chọn trên validation. Test chỉ ghi để tham khảo.
- Không có lần chạy nào báo `RuntimeError: J tăng`.

### MAE (δ = 1e-6, `tol = 1e-8`)

| λ | vòng lạnh | vòng warm | s lạnh | s warm | MSE val lạnh | MSE val warm | MAE val lạnh | MAE val warm | MSE test warm | MAE test warm |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 107 | 107 | 16,9 | 16,9 | 0,666105 | 0,666105 | 0,539464 | 0,539464 | 0,365329 | 0,383333 |
| 1 | 107 | 1 | 16,9 | 0,2 | 0,666105 | 0,666106 | 0,539465 | 0,539464 | 0,365329 | 0,383333 |
| 3 | 107 | 1 | 17,0 | 0,2 | 0,666103 | 0,666107 | 0,539464 | 0,539464 | 0,365329 | 0,383332 |
| 10 | 107 | 1 | 17,0 | 0,2 | 0,666101 | 0,666108 | 0,539464 | 0,539464 | 0,365329 | 0,383332 |
| 30 | 106 | 31 | 16,8 | 4,9 | 0,666091 | 0,666129 | 0,539463 | 0,539464 | 0,365307 | 0,383316 |
| 100 | 103 | 71 | 16,3 | 11,3 | 0,666055 | 0,666116 | **0,539452** | **0,539451** | 0,365155 | 0,383244 |
| 300 | 96 | 77 | 15,3 | 12,2 | **0,666001** | **0,666076** | 0,539470 | 0,539469 | 0,364739 | 0,383092 |
| 1000 | 82 | 67 | 13,0 | 10,7 | 0,666015 | 0,666085 | 0,539668 | 0,539662 | 0,363680 | 0,382821 |
| 3000 | 71 | 59 | 11,3 | 9,4 | 0,667132 | 0,667186 | 0,540823 | 0,540812 | 0,362221 | 0,382941 |
| 10000 | 62 | 54 | 9,9 | 8,6 | 0,673261 | 0,673316 | 0,545554 | 0,545543 | 0,362000 | 0,385282 |
| **Tổng** | **948** | **469** | **150** | **75** | | | | | | |

### Huber (δ = 1, `tol = 1e-9`)

| λ | vòng lạnh | vòng warm | s lạnh | s warm | MSE val | Huber val | MSE test | MAE test |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 7 | 1,1 | 1,1 | 0,654767 | 0,254160 | 0,367800 | 0,387989 |
| 1 | 7 | 2 | 1,1 | 0,3 | 0,654766 | 0,254160 | 0,367797 | 0,387988 |
| 3 | 7 | 2 | 1,1 | 0,3 | 0,654764 | 0,254159 | 0,367791 | 0,387985 |
| 10 | 7 | 3 | 1,1 | 0,5 | 0,654757 | 0,254157 | 0,367769 | 0,387975 |
| 30 | 7 | 3 | 1,1 | 0,5 | 0,654739 | 0,254152 | 0,367710 | 0,387950 |
| 100 | 7 | 4 | 1,1 | 0,6 | 0,654695 | **0,254141** | 0,367521 | 0,387874 |
| 300 | 7 | 5 | 1,1 | 0,8 | **0,654681** | 0,254153 | 0,367096 | 0,387742 |
| 1000 | 7 | 6 | 1,1 | 1,0 | 0,655222 | 0,254419 | 0,366302 | 0,387740 |
| 3000 | 6 | 6 | 1,0 | 1,0 | 0,658032 | 0,255633 | 0,365941 | 0,388725 |
| 10000 | 6 | 6 | 1,0 | 1,0 | 0,668910 | 0,260041 | 0,368614 | 0,392833 |
| **Tổng** | **68** | **44** | **10,9** | **7,1** | | | | |

Với Huber, nghiệm lạnh và nghiệm warm trùng nhau tới 6 chữ số (lệch ≤ 5e-7), nên bảng chỉ ghi một cột.

### Tóm tắt

| | tổng vòng lạnh | tổng vòng warm | tiết kiệm | λ* theo mục tiêu (val) | λ* theo MSE val | ở đầu mút? |
|---|---|---|---|---|---|---|
| MAE | 948 | 469 | 51% | 100 (cả hai kiểu) | 300 (cả hai kiểu) | không |
| Huber | 68 | 44 | 35% | 100 (cả hai kiểu) | 300 (cả hai kiểu) | không |

- λ* không nằm ở đầu mút, nên không mở rộng lưới (script tự mở rộng tối đa 2 giá trị nếu cần).
- Với MAE, khởi tạo lạnh gần như là OLS ở mọi λ. Vì vậy số vòng lạnh chỉ giảm dần khi λ lớn: phạt làm hàm mục tiêu cong hơn.

## 5. Độ chính xác của λ* khi dùng MAE

Hai kiểu khởi tạo cho J khác nhau rất ít ở cùng λ: chênh tương đối lớn nhất 2,8e-8, và warm thấp hơn ở 7/9 giá trị λ > 0. Tức cả hai đều dừng hợp lệ theo `tol = 1e-8`. Nhưng vì mục tiêu MAE rất phẳng gần nghiệm, W của chúng khác nhau thấy rõ:

| λ | MSE val warm − lạnh | MAE val warm − lạnh | MSE test warm − lạnh |
|---|---|---|---|
| 10 | +7,4e-6 | +1,5e-7 | +2,1e-5 |
| 100 | +6,1e-5 | −8,5e-7 | +4,8e-5 |
| 300 | +7,5e-5 | −1,4e-6 | +3,6e-5 |
| 3000 | +5,4e-5 | −1,1e-5 | +1,7e-5 |

So với ảnh hưởng của λ:

- **MAE val** giữa λ = 0 và λ* = 100 chỉ chênh 1,3e-5. Mức này cùng cỡ với sai số dừng ở `tol = 1e-8` trong báo cáo cũ: 1,3e-5 trên MAE test.
- **MSE val** của nghiệm MAE giữa λ = 0 và λ = 300 chênh 1,0e-4 (lạnh) nhưng chỉ 2,9e-5 (warm).
- Ở λ nhỏ (1, 3, 10), warm start dừng ngay sau 1 vòng, vì J giảm ít hơn `tol`. Nghiệm vì thế gần như giữ nguyên nghiệm λ = 0. Nghiệm lạnh thì đi đủ 107 vòng.

**Hệ quả:** với MAE, lưới λ này chọn được λ* một cách nhất quán (100 và 300, giống nhau ở hai kiểu). Nhưng giá trị val tại λ* hơn λ = 0 chỉ một lượng cỡ sai số của bộ giải. Nói cách khác, trên ô này phạt λ hầu như không đổi MAE val. Cần một `tol` chặt hơn trước khi coi λ* của MAE là chắc chắn (mục 6).

Huber không có vấn đề này: lạnh và warm trùng nhau tới 5e-7, và Huber val giữa λ = 0 và λ* chênh 1,9e-5, lớn hơn mức lệch đó.

## 6. Đề xuất cho `runner.py`

- **`chunk = 16`** cho mọi ô. Đỉnh bộ nhớ lớn nhất là 11,9 GB, ở ETTm1 H = 720. Nếu máy đang chạy việc khác chiếm VRAM, giảm xuống 8.
- **Chế độ pha**: `gram_dtype=torch.float32`, dữ liệu và W bằng float64, TF32 tắt tường minh.
- **Không cần chia khối theo H** trên card 16 GB.
- **`tol`:**
  - Huber: `1e-9` (6–7 vòng, cả grid khoảng 8 phút).
  - MAE: `1e-8` là đủ để chạy nghiệm ở λ cố định (lệch nghiệm hội tụ khoảng 1e-5, báo cáo cũ mục 5). Cả grid ở mức này mất khoảng 1,9 giờ.
  - Khi **chọn λ cho MAE**, mức chênh val giữa các λ cùng cỡ sai số này (mục 5). Hai cách xử lý:
    - Quét λ ở `tol = 1e-9`. Tốn khoảng 1,8 lần so với 1e-8: quét warm cho cả grid khoảng 9 giờ thay vì 5 giờ, ước lượng theo tỷ lệ số vòng 189/107, chưa đo.
    - Hoặc chấp nhận rằng λ hầu như không ảnh hưởng MAE val, rồi dùng λ* chọn theo MSE val.

    Nên kiểm lại ô ETTh1 H = 96 ở `tol = 1e-9` trước khi chọn (một lần chạy `--step warmstart --loss mae` với `tol` đổi, khoảng 3 phút).
- **Lưới λ:** {0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10 000} đủ rộng cho DLinear ở ETTh1 H = 96: λ* = 100–300, không chạm đầu mút.
  - Các λ ≤ 10 gần như không đổi gì. Có thể bỏ 1 và 3 để tiết kiệm (chỉ có lợi cho khởi tạo lạnh).
  - Linear và NLinear dùng `pen = I` nên thang λ có thể khác. Giữ cơ chế tự mở rộng khi λ* ở đầu mút.
- **Kiểu khởi tạo:** warm start theo λ tăng dần, bắt đầu từ λ = 0 khởi tạo OLS. Warm start tiết kiệm 51% (MAE) và 35% (Huber), cho cùng λ*.
- **Ước lượng tổng** cho grid đầy đủ, với warm start và `tol` như trên:
  - IRLS ở λ cố định: khoảng 2 giờ.
  - Quét λ: khoảng 5,6 giờ (MAE 5,1 giờ, Huber 0,5 giờ).
  - Tổng khoảng 7–8 giờ. Nếu quét λ của MAE ở `tol = 1e-9` thì khoảng 12 giờ.

## 7. Môi trường và test

- **Môi trường mới `.venv-gpu`**: torch 2.14.0+cu130, cài từ `https://download.pytorch.org/whl/cu130`, rồi cài phần còn lại theo `requirements.txt`.
  - `get_arch_list()` có `sm_120`.
  - Venv cũ `.venv` là torch 2.6.0+cpu, giữ nguyên.
- **Test lần đầu: 33 test, 1 lỗi.**
  - `tests/test_sgd.py` không nạp được `lr_factor`: commit 86c012c đã xóa hàm này khỏi `src/sgd.py`. Venv CPU cũ cũng lỗi y hệt, nên lỗi không do máy mới.
  - Theo ý người dùng, đã khôi phục `src/sgd.py` như ở commit d4173f1.
  - Sau đó: **Ran 35 tests, OK**, cả trước lẫn sau khi thêm tham số `chunk`.

## Tái lập

```bash
# môi trường (Windows, Git Bash)
python -m venv .venv-gpu
.venv-gpu/Scripts/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv-gpu/Scripts/python -m pip install -r requirements.txt
export PYTHONIOENCODING=utf-8
.venv-gpu/Scripts/python -m unittest discover -s tests -t . -v

# đo
.venv-gpu/Scripts/python scripts/bench_5060ti.py --step chunk
.venv-gpu/Scripts/python scripts/bench_5060ti.py --step full
.venv-gpu/Scripts/python scripts/bench_5060ti.py --step ettm1
for d in ETTh1 ETTh2 ETTm1; do for h in 96 192 336 720; do
  .venv-gpu/Scripts/python scripts/bench_5060ti.py --step cell --data $d --H $h; done; done
.venv-gpu/Scripts/python scripts/bench_5060ti.py --step warmstart
.venv-gpu/Scripts/python scripts/bench_5060ti.py --summary     # in bảng, ghi grid_estimate.json
```

Vòng lặp `cell` chạy lại cả ETTm1 H = 720, ghi đè `cell_ETTm1_H720.json` bằng số đo cùng cách. Số trong báo cáo lấy từ lần chạy `--step ettm1`.
