# Tiêu chí dừng của IRLS khi quét λ cho MAE, và ba phép kiểm còn nợ

Ngày chạy: 30/09/2026. Nhiệm vụ: [NHIEM_VU_2.md](NHIEM_VU_2.md).

- Script: [`scripts/stopping_check.py`](../scripts/stopping_check.py) (Phần A), [`scripts/checks_mae.py`](../scripts/checks_mae.py) (Phần B).
- Số liệu: [`results/stopping/`](../results/stopping/), [`results/checks_mae/`](../results/checks_mae/). Mọi JSON có khối phần cứng và `git_commit`.
- Máy: RTX 5060 Ti, torch 2.14.0+cu130, như [BAO_CAO_5060TI.md](BAO_CAO_5060TI.md). Chế độ pha, chunk 16, TF32 tắt.

## 1. Kết luận

**Phần A: đạt, với cấu hình `tol = 1e-10`, `patience = 1`, và riêng λ = 0 dùng `tol = 3e-11`** (phương án a′, mục 4.4). Cấu hình này là cái `runner.py` dùng.

Diễn biến:

- **Bốn cấu hình của A.3 đều không đạt**, và đây là điểm dừng theo mục 6. Chỗ trượt là tiêu chí 3, và chỉ ở **một điểm: MSE val tại λ = 0**.
  - Lệch tham chiếu −2,8e-5 ở `tol = 1e-9`; −1,1e-5 (`patience = 1`) và −1,04e-5 (`patience = 5`) ở `tol = 1e-10`. Ngưỡng là 1e-5.
- **Điểm trượt không phải lỗi của warm start.** Ở λ = 0, hàng warm chính là lần chạy lạnh từ OLS; mặt MAE phẳng nhất ở λ = 0.
  - Theo quỹ đạo tham chiếu, MSE val chỉ vào trong 1e-5 từ khoảng vòng 400, khi J giảm khoảng 6,7e-11 tương đối mỗi vòng. Còn `tol = 1e-10` dừng ở vòng 344.
- **Người dùng chọn kiểm (a′):** λ = 0 ở `tol = 3e-11` (dừng ở vòng 466), các λ > 0 giữ `tol = 1e-10, patience = 1`.
  - Kết quả đạt cả ba tiêu chí. MSE val tại λ = 0 giờ lệch tham chiếu −6,2e-6; mọi độ lệch khác ≤ 1,4e-6.
  - Chi phí: 1 796 vòng warm cho cả lưới, so với 1 696 khi không có tol riêng cho λ = 0.

Kết quả theo từng tiêu chí:

- **Tiêu chí 1 (λ* ổn định):** mọi cấu hình đều đạt. λ* = 100 theo MAE val và λ* = 300 theo MSE val, giống nhau ở khởi tạo lạnh, warm start và nghiệm tham chiếu.
- **Tiêu chí 2 (phân biệt được λ* với λ tốt thứ hai):** chỉ `tol = 1e-10` đạt.
  - Ở `tol = 1e-9`, MSE val của nghiệm warm và nghiệm lạnh **tại chính λ* = 300** lệch nhau 2,7e-5–2,8e-5 (tại λ = 1000 là 2,6e-5).
  - Mức đó lớn hơn chênh lệch 1,3e-5–1,6e-5 giữa λ* và λ tốt thứ hai.
- **λ có ảnh hưởng thật lên val của MAE, nhưng nhỏ** (câu hỏi 1 của A.2):
  - Trên nghiệm tham chiếu, MAE val giữa λ liền kề chênh 6e-7 (0 → 30), 1,4e-5 (30 → 100), 2,0e-5 (100 → 300) và 2,0e-4 (300 → 1000).
  - Trong 500 vòng cuối, MAE val của tham chiếu chỉ còn đổi 1e-9 tới 2,5e-8.
  - Vậy chênh lệch quanh λ* lớn hơn sai số của nghiệm tham chiếu 500 tới 10 000 lần. Riêng 0 → 30 thì λ gần như không có tác dụng.
- **`patience` gần như không đổi gì với MAE** khi `tol` ≤ 1e-9: thêm 2–4% số vòng, λ* và sai số như cũ. Nó chỉ có tác dụng ở λ nhỏ ngay sau λ = 0 (λ = 1: 1 → 5 vòng).
  - Với Huber, `patience = 5` chỉ tốn thêm vòng (44 → 84) mà không đổi kết quả, nên Huber dùng `patience = 1`.
- **Chi phí:** grid đầy đủ ở cấu hình này khoảng **18–19 giờ** (mục 4.5), không còn chạy xong trong một đêm. Nên chạy từng dataset.

**Phần B: cả ba phép kiểm đạt.**

- **B.1 (tính duy nhất):** đạt ở mức hàm mục tiêu. Ba điểm khởi tạo cho J lệch nhau tối đa 2,0e-10 tương đối, dưới ngưỡng 1e-8.
  - W lệch nhau 3,3e-3–3,7e-3 tương đối, và MSE val lệch tới 1,0e-5.
  - Khác biệt W là do dừng sớm: mỗi nghiệm cũng cách nghiệm tham chiếu 2000 vòng khoảng 3,5e-3.
- **B.2 (DLinear ≡ phạt N⁻¹):** đạt.
  - **Huber**, chế độ pha: W_eff lệch 3,2e-11, hàm mục tiêu lệch 1,4e-16.
  - **MAE**: ở chế độ pha, cách giải 2L chiều báo `RuntimeError: J tăng` ở vòng 0. Đây là điểm dừng thứ hai theo mục 6.
    - Nguyên nhân: A_h suy biến, và sai số float32 cỡ phạt 2λδ. Toán không sai.
    - Người dùng chọn so hai cách ở cùng số vòng, bằng float64. Ba vòng đầu trùng tới 1e-11. Sau đó sai số làm tròn bị khuếch đại khoảng 3 lần mỗi vòng, nhưng J của hai cách vẫn lệch dưới 1e-8 tương đối trong 20 vòng, đổi dấu qua lại.
  - Test trên dữ liệu giả (float64) đạt cho cả MAE lẫn Huber.
- **B.3 (NLinear MAE so với quy hoạch tuyến tính có ràng buộc):** test trên dữ liệu giả đạt.

## 2. Thay đổi code

- `irls(..., patience=1)`: xem [NHAT_KY_THAY_DOI.md](NHAT_KY_THAY_DOI.md) mục 7.1.
- Test: 35 → **40, tất cả đạt**. Có 2 test `patience`, 2 test tương đương DLinear và 1 test NLinear MAE.

## 3. Nghiệm tham chiếu (A.2)

DLinear (`pen = N⁻¹`, k = 25), ETTh1 H = 96, MAE δ = 1e-6, `tol = 0`, `max_iter = 2000`, khởi tạo OLS. Val và test tính lại bằng float64.

| λ | vòng | giây | J train | MAE val | MSE val | Huber val | MSE test | MAE test | trôi 500 vòng cuối (MAE val / MSE val) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2000 | 315 | 3,9239570894e-7 | 0,53946431 | 0,66617613 | 0,25528057 | 0,365338 | 0,383320 | 1,1e-8 / 3,4e-7 |
| 30 | 2000 | 314 | 3,9239601079e-7 | 0,53946369 | 0,66616359 | 0,25528025 | 0,365278 | 0,383293 | 2,5e-8 / 3,9e-7 |
| **100** | 2000 | 316 | 3,9239885721e-7 | **0,53945016** | 0,66611733 | **0,25527116** | 0,365118 | 0,383228 | 1,2e-9 / 7,3e-8 |
| **300** | 1685 | 269 | 3,9242066141e-7 | 0,53947062 | **0,66607850** | 0,25528104 | 0,364709 | 0,383083 | 1,2e-9 / 1,7e-8 |
| 1000 | 1217 | 194 | 3,9258701687e-7 | 0,53967194 | 0,66609339 | 0,25538733 | 0,363661 | 0,382819 | 3,5e-9 / 2,9e-8 |

- **λ = 300 và 1000 dừng trước 2000 vòng.** J tăng trong ngưỡng `rise_tol`, tức đã chạm độ chính xác float64, nên `irls` dừng (hành vi có từ trước, không đổi). λ = 0, 30, 100 chạm `max_iter` và có cảnh báo; cảnh báo được ghi trong JSON.
- **λ* trên nghiệm tham chiếu** (câu hỏi 2): 100 theo MAE val (và theo Huber val), 300 theo MSE val.
- **Chênh MAE val giữa λ liền kề so với sai số bộ giải:**

  | cặp λ | chênh MAE val | so với trôi 500 vòng cuối |
  |---|---|---|
  | 0 → 30 | 6,2e-7 | lớn hơn khoảng 25 lần |
  | 30 → 100 | 1,35e-5 | lớn hơn khoảng 500 lần |
  | 100 → 300 | 2,05e-5 | lớn hơn khoảng 10 000 lần |

  Tức ảnh hưởng của λ quanh λ* là thật, nhưng chỉ cỡ 1e-5 trên MAE val khoảng 0,54 (tương đối 2e-5–4e-5).

## 4. Quét λ với tiêu chí dừng mới (A.3, A.4)

Cùng ô, lưới λ {0, 1, 3, …, 10 000} tăng dần. Khởi tạo lạnh: ridge MSE với 2δλ·N⁻¹, thực chất là OLS với MAE. Warm: nghiệm của λ liền trước. λ* không nằm ở đầu mút ở mọi cấu hình, nên không mở rộng lưới.

### 4.1. So các cấu hình MAE theo ba tiêu chí

| `tol` | `patience` | tổng vòng warm / lạnh | giây warm / lạnh | TC1 | TC2 MAE val (chênh vs \|warm − lạnh\|) | TC2 MSE val | TC3 (lệch tham chiếu lớn nhất, MAE val / MSE val) | đạt |
|---|---|---|---|---|---|---|---|---|
| 1e-9 | 1 | 895 / 1 588 | 142 / 252 | ✓ | ✓ 1,2e-5 vs 1,1e-6 | ✗ 1,3e-5 vs 2,8e-5 | 3,2e-6 / **2,8e-5** (λ = 0) | ✗ |
| 1e-9 | 5 | 930 / 1 628 | 148 / 260 | ✓ | ✓ 1,2e-5 vs 1,1e-6 | ✗ 1,3e-5 vs 2,7e-5 | 3,0e-6 / **2,8e-5** (λ = 0) | ✗ |
| 1e-10 | 1 | 1 696 / 2 659 | 268 / 421 | ✓ | ✓ 1,3e-5 vs 1,0e-6 | ✓ 1,5e-5 vs 9,7e-6 | 1,1e-6 / **1,1e-5** (λ = 0) | ✗ |
| 1e-10 | 5 | 1 730 / 2 699 | 275 / 430 | ✓ | ✓ 1,3e-5 vs 9,8e-7 | ✓ 1,5e-5 vs 9,2e-6 | 1,1e-6 / **1,04e-5** (λ = 0) | ✗ |
| **1e-10, λ = 0: 3e-11** (a′) | 1 | 1 796 / 2 785 | 290 / 452 | ✓ | ✓ 1,3e-5 vs 1,0e-6 | ✓ 1,5e-5 vs 9,7e-6 | 1,1e-6 / 6,2e-6 (λ = 0) | **✓** |

Cách đọc các cột:

- **TC2:** ghi cho nghiệm warm. Nghiệm lạnh cho kết luận như nhau; chi tiết ở `acceptance.json`.
- **TC3:** ở bốn cấu hình của A.3, mọi λ có tham chiếu đều trong 1e-5, trừ MSE val tại λ = 0.
- Dòng (a′) (`results/stopping/sweep_mae_tol1e-10_p1_lam0tol3e-11.json`) chạy sau, khi người dùng đồng ý. λ = 0 dừng ở vòng 466. Độ lệch warm − tham chiếu của MSE val tại λ = 0, 30, 100, 300, 1000 lần lượt là −6,2e-6, −1,4e-6, +4,2e-7, −5,7e-7, −8,3e-7.
- Ở `tol = 1e-10, patience = 5`, độ lệch warm − tham chiếu của MSE val tại λ = 0, 30, 100, 300, 1000 lần lượt là −1,0e-5, −1,5e-6, +4,2e-7, −5,8e-7, −6,1e-7.

### 4.2. Chi tiết `tol = 1e-10`, `patience = 5`

| λ | vòng lạnh | vòng warm | MAE val warm | MSE val warm | MAE val lạnh | MSE val lạnh | MSE test warm | MAE test warm |
|---|---|---|---|---|---|---|---|---|
| 0 | 344 | 344 | 0,53946392 | 0,66616572 | 0,53946392 | 0,66616572 | 0,365337 | 0,383322 |
| 1 | 345 | 5 | 0,53946393 | 0,66616595 | 0,53946375 | 0,66616526 | 0,365337 | 0,383321 |
| 3 | 340 | 33 | 0,53946392 | 0,66616714 | 0,53946349 | 0,66616436 | 0,365336 | 0,383321 |
| 10 | 345 | 187 | 0,53946408 | 0,66617114 | 0,53946381 | 0,66616148 | 0,365323 | 0,383314 |
| 30 | 334 | 267 | 0,53946263 | 0,66616214 | 0,53946347 | 0,66615294 | 0,365282 | 0,383295 |
| **100** | 296 | 269 | **0,53944998** | 0,66611775 | **0,53945096** | 0,66610980 | 0,365123 | 0,383229 |
| **300** | 239 | 216 | 0,53947022 | **0,66607793** | 0,53947047 | **0,66606869** | 0,364713 | 0,383084 |
| 1000 | 182 | 161 | 0,53967112 | 0,66609278 | 0,53967132 | 0,66608488 | 0,363663 | 0,382819 |
| 3000 | 149 | 134 | 0,54082899 | 0,66720706 | 0,54082987 | 0,66720193 | 0,362209 | 0,382944 |
| 10000 | 125 | 114 | 0,54556907 | 0,67335330 | 0,54557015 | 0,67334939 | 0,361994 | 0,385285 |

- **So với báo cáo 5060 Ti** (`tol = 1e-8, patience = 1`): warm ở λ = 1 và 3 không còn dừng sau 1 vòng.
  - λ = 1 dừng sau 5 vòng, đúng bằng `patience`, vì nghiệm của λ = 0 đã gần đúng.
  - λ = 3 đi 33 vòng, λ = 10 đi 187 vòng.
- **Warm start tiết kiệm 36% số vòng** (1 730 so với 2 699), thấp hơn mức 51% ở `tol = 1e-8`.

### 4.3. Huber (δ = 1, `tol = 1e-9`)

| `patience` | tổng vòng warm / lạnh | λ* (Huber val / MSE val), lạnh và warm |
|---|---|---|
| 1 | 44 / 68 | 100 / 300 |
| 5 | 84 / 108 | 100 / 300 |

`patience = 5` với Huber chỉ thêm 4 vòng mỗi λ, không đổi λ*. Nên giữ `patience = 1` cho Huber.

### 4.4. `tol` riêng cho λ = 0 (phương án a′)

Từ quỹ đạo tham chiếu của λ = 0 (sai số so với vòng 2000; quỹ đạo lấy mỗi 50 vòng):

| vòng | MSE val − cuối | MAE val − cuối |
|---|---|---|
| 300 | −1,31e-5 | −5,5e-7 |
| 350 | −1,01e-5 | −3,7e-7 |
| 400 | −8,1e-6 | −2,1e-7 |
| 500 | −5,4e-6 | −8,4e-8 |

- MSE val vào trong 1e-5 từ khoảng vòng 400. Quanh vòng 350–400, J giảm trung bình 6,7e-11 tương đối mỗi vòng. Vì vậy λ = 0 cần `tol` khoảng 5e-11; chọn 3e-11 để có biên.
- Các λ > 0 của tham chiếu vào trong 1e-5 sớm hơn (λ = 30: vòng 350; 100: vòng 300; 300: vòng 250; 1000: vòng 200).
- **Đã đo (a′):** λ = 0 ở `tol = 3e-11` dừng ở vòng 466. MSE val lệch tham chiếu −6,2e-6, MAE val −1,3e-7. Cả lưới đạt ba tiêu chí (mục 4.1).

### 4.5. Chi phí quét λ cho cả grid

Cách tính: 3 mô hình có λ* × 12 ô, tổng s/vòng của 12 ô là 13,07 s (báo cáo 5060 Ti mục 3.4). Giả định số vòng mỗi ô như ETTh1 H = 96.

| Cấu hình | MAE warm | MAE lạnh | Huber warm (`tol = 1e-9, patience = 1`) |
|---|---|---|---|
| `tol = 1e-8, patience = 1` (báo cáo 5060 Ti, λ* không tin được) | 5,1 giờ | 10,3 giờ | 0,5 giờ |
| `tol = 1e-9, patience = 1` | 9,7 giờ | 17,3 giờ | |
| `tol = 1e-10, patience = 1` | 18,5 giờ | 29,0 giờ | |
| `tol = 1e-10, patience = 5` | 18,8 giờ | 29,4 giờ | |
| **`tol = 1e-10, patience = 1`, λ = 0: `3e-11`** (a′, chọn) | **19,6 giờ** | 30,3 giờ | |

- `runner.py` không có lượt chạy λ cố định riêng: λ = 0 và mọi λ* đều là dòng trên chính các đường λ này. Không cộng thêm phần λ cố định của báo cáo 5060 Ti, vì như thế là tính hai lần.
- **Tổng grid ở cấu hình (a′), ước lượng:**

  | phần | giờ |
  |---|---|
  | MAE, 3 mô hình, warm | 19,6 |
  | trừ DLinear λ = 0 (chép từ Linear, khoảng 466 vòng × 13,07 s) | −1,7 |
  | Huber, `tol = 1e-9, patience = 1` | 0,5 |
  | NLinear MSE λ > 0 qua `irls` δ lớn (36 λ × khoảng 3 vòng × 13,07 s) | 0,4 |
  | Linear/DLinear MSE (dạng đóng) | không đáng kể |
  | **Tổng** | **khoảng 18–19** |

  Mức này không chạy xong trong một đêm; ETTm1 chiếm khoảng 67%.
- **Giả định chưa đo:** số vòng mỗi đường λ của Linear (`pen = I`) và NLinear (ràng buộc) bằng của DLinear.
- Có thể chạy từng dataset một lần; `runner.py` chạy tiếp được sau gián đoạn.

## 5. Ba phép kiểm (Phần B)

### 5.1. B.1: nghiệm MAE có duy nhất không (dữ liệu thật)

Linear, ETTh1 H = 96, λ = 0, MAE, `tol = 1e-10`, `patience = 5`. Ba điểm khởi tạo:

- OLS.
- W = 0.
- OLS cộng nhiễu Gauss, seed 0, độ lệch chuẩn mỗi phần tử bằng 0,1·‖hàng h của W_OLS‖. Điểm này cách OLS 1,83 lần chuẩn của W_OLS.

| khởi tạo | vòng | J train | MAE val | MSE val | MSE test | MAE test |
|---|---|---|---|---|---|---|
| OLS | 344 | 3,923957133346e-7 | 0,53946392 | 0,66616572 | 0,36533678 | 0,38332150 |
| 0 | 357 | 3,923957133026e-7 | 0,53946481 | 0,66617200 | 0,36533460 | 0,38331908 |
| OLS + nhiễu | 356 | 3,923957132551e-7 | 0,53946469 | 0,66617571 | 0,36533814 | 0,38332004 |

| cặp | J tương đối | ‖W_i − W_j‖/‖W_j‖ | ΔMAE val | ΔMSE val | ΔMSE test |
|---|---|---|---|---|---|
| OLS / 0 | +8,1e-11 | 3,3e-3 | −9,0e-7 | −6,3e-6 | +2,2e-6 |
| OLS / nhiễu | +2,0e-10 | 3,7e-3 | −7,8e-7 | −1,0e-5 | −1,4e-6 |
| 0 / nhiễu | +1,2e-10 | 3,7e-3 | +1,2e-7 | −3,7e-6 | −3,6e-6 |

- **Cùng giá trị hàm mục tiêu: có.** J lệch ≤ 2,0e-10 tương đối, dưới ngưỡng 1e-8. Không lần nào chạm `max_iter`, không có cảnh báo.
- **Cùng trọng số: không, ở `tol` này.** W lệch khoảng 3,5e-3 tương đối. Mặt MAE gần nghiệm rất phẳng, nên các W khác nhau cỡ đó cho cùng J tới 1e-10.
  - **Khác biệt này là do dừng sớm.** Cả ba nghiệm đều cách nghiệm tham chiếu λ = 0 (2000 vòng từ OLS, `results/stopping/reference_mae_lam0.npy`) một khoảng 3,5e-3–3,6e-3. Khoảng cách này bằng cỡ độ lệch giữa chúng, tức W vẫn đang dịch chuyển khi dừng ở `tol = 1e-10`.
  - Chính nghiệm tham chiếu cũng chưa hội tụ hẳn về W, nên chưa đo được độ lệch W ở nghiệm hội tụ thật. Trên dữ liệu giả (float64, `tol = 1e-12`), `test_multi_init_agree` cho W trùng tới 1e-14.
  - Với câu "về cùng một nghiệm" của outline, nên phát biểu là "cùng giá trị hàm mục tiêu tới 2e-10, cùng MAE val tới 1e-6, cùng MSE val tới 1e-5".
- Thời gian ở bảng này (0,17–0,34 s/vòng) không dùng làm số đo tốc độ: lần chạy bị chồng với tiến trình nền đang kết thúc.

### 5.2. B.2: DLinear + weight decay ≡ ridge với N⁻¹

**Dữ liệu giả** (`TestDLinearEquivalence`: L = 24, k = 5, H = 3, n = 400, λ = 5, float64, `tol = 1e-12`, `patience = 5`): đạt.

| | đo được | ngưỡng |
|---|---|---|
| Huber δ = 1: W_eff | 1,1e-15 | 1e-8 |
| Huber δ = 1: bias | 0 | 1e-8 |
| MAE: hàm mục tiêu có phạt (L chiều), hai chiều | 0 | 1e-6 |
| MAE: như trên, dùng phạt 2L chiều của chính nghiệm 2L | 0 | 1e-6 |
| MAE: W | 2,2e-15 | (không kiểm) |

- **Vì sao trùng tới sai số máy, kể cả với MAE:** mỗi vòng IRLS giải một bài ridge có trọng số. Ridge có trọng số trong 2L chiều với phạt I trùng về đại số với ridge trong L chiều với phạt N⁻¹, nên hai dãy lặp trùng nhau từng vòng.
- Như vậy test kiểm được cả phép tương đương lẫn cách cài `pen`, nhưng không độc lập về hội tụ. Phần hội tụ của `irls` có `pen` đã được `test_mae_matches_qp` kiểm (so với SLSQP).
- Test thêm một phép kiểm để không đạt một cách tầm thường: nghiệm phạt N⁻¹ khác nghiệm phạt I khoảng 0,11 tương đối, và khác λ = 0 khoảng 0,3–0,4.

**Dữ liệu thật** (ETTh1 H = 96, k = 25, λ = 100, chế độ pha, `patience = 5`). Khởi tạo L chiều là W_OLS; 2L chiều là W_t = W_s = W_OLS, cho cùng W_eff.

| | Huber δ = 1 (`tol = 1e-9`) | MAE (`tol = 1e-10`) |
|---|---|---|
| vòng: 2L chiều / N⁻¹ | 11 / 11 | **lỗi ở vòng 0** / (không chạy) |
| ‖ΔW_eff‖/‖W_eff‖ | 3,2e-11 | — |
| bias lệch lớn nhất | 3,4e-13 | — |
| hàm mục tiêu có phạt, tương đối | 1,4e-16 (cả khi dùng phạt 2L chiều) | — |
| MSE test (2L / N⁻¹) | 0,36752083 / 0,36752083 | — |
| MAE test (2L / N⁻¹) | 0,38787373 / 0,38787373 | — |

**MAE trong 2L chiều: `RuntimeError: J tăng ở vòng 0: 3,972517e-07 → 4,335719e-07`.** Chẩn đoán, lưu trong `results/checks_mae/dlinear.json`:

- **Z có hạng L, nên A_h = ZᵀWZ suy biến.** Trị riêng nhỏ nhất của A_0 (lập bằng float64) là −4,7e-13, tức bằng 0; ‖A_0‖ = 729.
- **Chỉ phần phạt làm A_h khả nghịch, và nó quá nhỏ so với sai số float32.**
  - Phạt trong hệ của IRLS là 2λδ = 2e-4, vì trọng số của MAE nhỏ cỡ δ.
  - A_0 lập bằng float32 lệch bản float64 1,8e-4 theo chuẩn phổ, và có trị riêng nhỏ nhất −9,9e-5.
  - Cộng phạt 2e-4 vào thì A_h float32 gần như suy biến ở L hướng, nên bước giải sai.
- **Lập A_h bằng float64 thì J giảm bình thường** (3,9725e-7 → 3,9580e-7 → 3,9498e-7). Tức toán không sai; chỉ chế độ pha không dùng được cho bài này.
- Huber không gặp vấn đề: trọng số của Huber cỡ 1, nên phạt trong hệ là 2λδ = 200, lớn hơn nhiều so với sai số float32.
- **Chạy đủ bằng float64 thì quá lâu:** 15,3 s/vòng ở p = 673. Với khoảng 300 vòng, mất khoảng 75 phút, vượt ngưỡng 60 phút của mục 6. Nên chưa chạy.

**So ở cùng số vòng, bằng float64** (người dùng chọn; `scripts/checks_mae.py --check dlinear_f64`, `results/checks_mae/dlinear_f64.json`). MAE, λ = 100, 20 vòng, `gram_dtype=None`, cùng khởi tạo tương ứng như trên. Thời gian: 2L chiều 307 s, N⁻¹ 78 s.

| vòng | J (2L − N⁻¹)/N⁻¹ | ‖ΔW_eff‖/‖W_eff‖ |
|---|---|---|
| 1 | 0 | 3,6e-13 |
| 3 | +4,3e-15 | 1,7e-11 |
| 5 | +5,5e-14 | 3,8e-10 |
| 10 | +3,4e-12 | 2,9e-7 |
| 15 | −3,7e-10 | 5,5e-5 |
| 20 | −9,2e-9 | 5,7e-4 |

- **Vài vòng đầu trùng tới sai số máy**, đúng như dự đoán: trong số học chính xác hai dãy lặp trùng nhau từng vòng.
- **Sau đó độ lệch W tăng khoảng 3 lần mỗi vòng.** Đây là sai số làm tròn float64 (thứ tự cộng khác nhau giữa 2L và L chiều) bị khuếch đại qua trọng số δ/max(|r|, δ) của MAE. Cùng hiện tượng với hai GPU khác nhau ở báo cáo 5060 Ti mục 3.2.
- **J vẫn lệch dưới 1e-8 tương đối trong cả 20 vòng**, đổi dấu qua lại, nên không cách nào tốt hơn cách kia một cách có hệ thống. MSE test ở vòng 20: 0,36486914 (2L) và 0,36486915 (N⁻¹).

**Kết luận B.2:**

- **Tương đương đúng với cả Huber lẫn MAE trên dữ liệu thật**, và với cả hai mục tiêu trên dữ liệu giả:
  - Huber: W_eff lệch 3,2e-11 ở nghiệm hội tụ.
  - MAE: hai dãy lặp trùng từng vòng tới sai số làm tròn đã khuếch đại, J lệch ≤ 9,2e-9 trong 20 vòng.
- **Với MAE, cách giải 2L chiều không chạy được ở chế độ pha**, vì A_h suy biến và sai số float32 cỡ phạt 2λδ.
- Việc này không ảnh hưởng tới grid, vì `runner.py` giải DLinear trong L chiều với `pen = N⁻¹` (A_h xác định dương). Nhưng nó là thêm một lý do để giải DLinear bằng N⁻¹ thay vì trong 2L chiều.

### 5.3. B.3: NLinear với MAE so với quy hoạch tuyến tính có ràng buộc (câu 11.6)

Test `TestNLinearMAE.test_matches_constrained_lp` trên dữ liệu `make_data` (n = 200, L = 4, H = 3). Với từng h, giải bằng `linprog` (HiGHS): min Σt, với −t ≤ y − Xt w ≤ t và aᵀw = 1, a = [1, …, 1, 0]. So với `irls(..., MAE_DELTA, constrained=True, tol=1e-12)`:

- MAE của `irls` không kém quy hoạch tuyến tính quá 1e-6 tương đối: **đạt**.
- Tổng hàng (L hệ số đầu) bằng 1 tới 1e-10: **đạt**.
- Ràng buộc có tác dụng thật: nghiệm MAE không ràng buộc có tổng hàng lệch 1 hơn 1e-3.

## 6. Phần C: chạy thử một ô (C.5)

`scripts/runner.py --data ETTh1 --H 96`, commit `bf57cff` (sạch). Kết quả: 170 dòng trong `results/lambda_path.csv` và W ở `results/weights/`. Tổng thời gian khoảng 19 phút.

Cấu hình:

- **MAE:** `tol = 1e-10`, `patience = 1`, λ = 0: `tol = 3e-11`.
- **Huber:** `tol = 1e-9`, `patience = 1`.
- **MSE:** Linear và DLinear dạng đóng, lưới {0} ∪ logspace(−2, 5, 36). NLinear λ > 0 giải bằng `irls` với δ = 1e6 (2–3 vòng mỗi λ).
- **NLinear:** `constrained=True, pen = I`, theo quyết định của người dùng.

| Đại lượng | Giá trị đã biết | Đo được | Đạt? |
|---|---|---|---|
| Linear, MSE, λ = 0: MSE test | 0,3702 | 0,370235 | ✓ |
| DLinear, MSE, λ* (`val_mse`) | λ* ≈ 562, MSE test 0,3697 | λ* = 398, MSE test 0,369864 | λ* cùng bậc ✓; MSE test lệch 1,6e-4 ✗ (ngưỡng 1e-4), giải thích bên dưới |
| NLinear, MSE, λ* (`val_mse`) | λ* ≈ 1780 | **λ* = 0** | **✗**, giải thích bên dưới |
| Linear, MAE, λ = 0: MSE / MAE test | 0,3653 / 0,3833 | 0,365337 / 0,383321 | ✓ |
| Linear, Huber, λ = 0: MSE / MAE test | 0,3678 / 0,3880 | 0,367800 / 0,387989 | ✓ |
| DLinear, MAE, λ* | 100 (MAE val) / 300 (MSE val) theo Phần A | 100 / 300 | ✓ |
| DLinear, Huber, λ* | 100 (Huber val) / 300 (MSE val) theo Phần A | 100 / 300 | ✓ |

**DLinear MSE: không phải lỗi, do lưới khác.**

- Ở λ = 1000, giá trị có mặt trong cả hai lưới, `runner.py` cho val / test = 0,651630 / 0,369494, trùng notebook tới 6 chữ số. Vậy nghiệm dạng đóng giống hệt.
- MSE val rất phẳng trên khoảng λ ∈ [316, 631]: 0,651555–0,651564, chênh dưới 1e-5. Lưới của notebook có 562 nên chọn 562; lưới mới có 398 và 631 nên chọn 398.
- MSE test thì dốc hơn theo λ, nên hai λ* cho MSE test lệch 1,2e-4. Tức ngưỡng 1e-4 trên MSE test không đạt được khi λ* không trùng.

**NLinear MSE: lệch vì hàm phạt khác, đúng như đã lường trước.**

- Với `pen = I` trên cả L hệ số của W_eff, MSE val tăng đơn điệu theo λ: 0,670366 ở λ = 0, 0,670655 ở λ = 631, 0,675665 ở λ = 6310. Vì vậy λ* = 0.
- Notebook 01 (`W_nlinear_wd`) phạt theo weight decay thật của NLinear, tức chỉ L − 1 hệ số đầu, không phạt hệ số lag cuối. Với cách đó λ* = 1778 và MSE val giảm từ 0,670366 xuống 0,670036.
- Hệ số lag cuối của NLinear lớn, vì các hàng cộng bằng 1 và lag cuối mang phần lớn trọng số. Phạt nó kéo dự báo về 0, nên luôn làm hỏng. Hiệu ứng tương tự thấy ở NLinear MAE và Huber: λ* ở đầu mút dưới, lưới tự mở rộng xuống 1/3 và 1/9.
- **Đây là điểm dừng theo mục 6** ("chạy thử ở C.5 lệch các giá trị đã biết quá ngưỡng"). Chưa sửa gì, chờ người dùng quyết định.

**Ghi chú khác:**

- **Warm start với `patience = 1` vẫn dừng sau 1–2 vòng ở λ = 1, 3.** Đúng như lúc kiểm Phần A với cấu hình này (vẫn đạt ba tiêu chí): vì λ nhỏ, nghiệm gần như không đổi so với λ = 0.
- **Mở rộng ở đầu mút dưới** (λ = 1/3, 1/9) chỉ chạy 1 vòng mỗi giá trị, vì các λ này gần như không khác λ = 0. Tốn không đáng kể.
- **Chưa chạy** `select_lambda.py` và `runner.py --dry-run` cho cả grid trong phiên này, vì công cụ chạy lệnh bị chặn tạm thời ở cuối phiên. Cũng chưa viết mục "Chạy grid" trong README (C.6), vì đang chờ quyết định về NLinear.

## 7. Ghi chú

- **Các JSON của Phần A và B ghi `git_commit = 86c012c82ac5-dirty`**, vì lúc chạy các thay đổi chưa được commit. Sau đó đã commit (`5888f61`, `bf57cff`). Lần chạy (a′) và `dlinear_f64` ghi `5888f61e85e4-dirty`; C.5 ghi `bf57cff` sạch.
- **Log của chuỗi chạy nền bị mất** khi phiên Claude Code khởi động lại. Mọi số trong báo cáo lấy từ các file JSON, không từ log.
- Hai phép kiểm B.1 và B.2 trên dữ liệu thật được chạy lại ở tiền cảnh sau đó.

## Tái lập

```bash
export PYTHONIOENCODING=utf-8
.venv-gpu/Scripts/python -m unittest discover -s tests -t . -v                      # 40 test
.venv-gpu/Scripts/python scripts/stopping_check.py --step reference                  # A.2, khoảng 24 phút
.venv-gpu/Scripts/python scripts/stopping_check.py --step sweep --all                # A.3, khoảng 45 phút
.venv-gpu/Scripts/python scripts/stopping_check.py --step analyze                    # A.4 → acceptance.json
.venv-gpu/Scripts/python scripts/checks_mae.py --check unique                        # B.1
.venv-gpu/Scripts/python scripts/stopping_check.py --step sweep --loss mae --tol 1e-10 --patience 1 --tol0 3e-11   # (a′)
.venv-gpu/Scripts/python scripts/checks_mae.py --check dlinear                       # B.2
.venv-gpu/Scripts/python scripts/checks_mae.py --check dlinear_f64                   # B.2, MAE, float64
.venv-gpu/Scripts/python scripts/runner.py --data ETTh1 --H 96                       # C.5, khoảng 19 phút
.venv-gpu/Scripts/python scripts/select_lambda.py
```
