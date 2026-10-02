# SGD trên ETTh2, kiểm giả thuyết về khoảng cách với nghiệm dạng đóng, và ba việc rẻ

Ngày chạy: 02–03/10/2026. Nhiệm vụ: [NHIEM_VU_3.md](NHIEM_VU_3.md).

- Script: [`scripts/sgd_etth2.py`](../scripts/sgd_etth2.py) (A), [`scripts/ridge_w0.py`](../scripts/ridge_w0.py) (B.2), [`scripts/recheck_paths.py`](../scripts/recheck_paths.py) (C.1), [`scripts/runner.py`](../scripts/runner.py) và [`scripts/select_lambda.py`](../scripts/select_lambda.py) (C.2), [`scripts/constant_windows.py`](../scripts/constant_windows.py) (C.3), [`scripts/report_etth2.py`](../scripts/report_etth2.py) (mọi bảng dưới đây).
- Số liệu: [`results/sgd_etth2/`](../results/sgd_etth2/), [`results/ridge_w0/`](../results/ridge_w0/), [`results/recheck/`](../results/recheck/), [`results/constant_windows/`](../results/constant_windows/), [`results/etth2_report/`](../results/etth2_report/) (bảng và `summary.json`). Mọi JSON có khối phần cứng và `git_commit`; không file nào mang nhãn `-dirty`.
- Máy: RTX 5060 Ti (IRLS ở C.1), CPU Intel (SGD, dạng đóng), torch 2.14.0+cu130, TF32 tắt. SGD chạy trên CPU.
- Thay đổi code: [NHAT_KY_THAY_DOI.md](NHAT_KY_THAY_DOI.md), mục 8.

## 1. Kết luận

**A. Khoảng cách SGD − dạng đóng trên ETTh2: không được xác nhận theo tiêu chí đã đặt, nhưng có cùng một hướng ở 7/8 ô.**

- **Tiêu chí** là SGD tốt hơn λ* ở ít nhất 3/4 ô, với mức lớn hơn 2 lần độ lệch chuẩn theo seed.
  - Linear đạt 1/4 ô (chỉ H = 720: −0,038 so với 2 sd = 0,036). DLinear đạt 0/4 ô.
- **Về hướng, SGD có trung bình thấp hơn λ*** ở Linear H ∈ {96, 336, 720} và DLinear ở cả bốn H. Mức thấp hơn từ 0,005 đến 0,043.
  - Khoảng cách có trên **cả val lẫn test** (Δ val từ −0,002 đến −0,017). Vậy nó không chỉ do dịch chuyển phân phối giữa val và test.
  - Ô ngược hướng duy nhất là Linear H = 192. Ở đó seed 2021 dừng ngay epoch 1 (val tăng ba epoch liền) và cho test 0,523.
- **Không xác nhận được chủ yếu vì độ lệch chuẩn theo seed lớn**, từ 0,007 đến 0,08 (MSE test). Lớn ngang hoặc hơn chính khoảng cách cần đo. Ba seed không đủ để tách hai thứ này.
- **SGD của repo tái lập số công bố** trong khoảng ±1,5% ở 5/8 ô Linear và DLinear, và +2,7% ở Linear H = 96. Có hai ngoại lệ lớn:
  - **DLinear H = 720: không tái lập được.** Repo cho 0,736 ± 0,018, công bố 0,605 (+21,6%). Ở ô này SGD của repo **ngang dạng đóng** (0,740).
  - Linear H = 192: +13,7%, do seed 2021 nói trên. Hai seed còn lại cho 0,385 và 0,378, công bố 0,377.
- **NLinear (đối chứng) hành xử như dự kiến.** Ở cả bốn H, SGD bằng hoặc kém dạng đóng (Δ test từ +0,004 đến +0,035), như trên các dataset khác. Pipeline SGD vì vậy không có dấu hiệu sai.
- **Hai lần đảo chiều ở H = 720** (mục 0 của nhiệm vụ):
  - **ETTh2:** chênh lệch DLinear − Linear trong bài (0,605 so với 0,698) **không tái lập được bằng SGD của repo**. Ở đây DLinear cho 0,736 ± 0,018, Linear cho 0,702 ± 0,018, tức Linear còn nhỉnh hơn. Ở nghiệm tối ưu, hai mô hình trùng nhau (0,7403 và 0,7404). Số 0,605 của DLinear trong bài nhiều khả năng là một lần chạy may mắn, hoặc có khác biệt cấu hình mà repo này không thấy.
  - **ETTh1:** không chạy lại SGD (ngoài phạm vi). Kết luận của báo cáo trước giữ nguyên: dạng đóng Linear 0,4701, DLinear 0,4697.

**B. Giả thuyết.**

- **"SGD ngầm co về một trọng số khác 0" không được ủng hộ.**
  - **B.1:** tổng hàng của W SGD không gần 1 hơn một cách nhất quán. Ở H = 96 thì có (trung vị 0,87–1,04 so với 0,88). Ở H = 720 thì không (0,60–0,69 so với 0,63). Chênh giữa các seed lớn hơn chênh giữa SGD và dạng đóng.
  - **B.2:** ridge co về `mean` hay `last` thay vì 0 đổi MSE test trên ETTh2 không quá 0,001, còn rất xa SGD. Kiểm `zero` khớp `lambda_path.csv` tới 2,2e-16. Riêng ở ETTh1 và ETTm1, `last` hạ MSE val nhưng làm tăng MSE test ở 14/16 tổ hợp. Xem mục 5.2.
- **B.3 không chạy**, vì điều kiện của nó (A xác nhận) không thỏa. Script [`gd_early_stop.py`](../scripts/gd_early_stop.py) đã sẵn sàng, chạy mất vài phút nếu cần.
- **Đoạn hằng của ETTh2 (C.3) là ứng viên mạnh nhất.** Bỏ khỏi train các cửa sổ hằng (2,5–6,3% số cửa sổ), λ* chọn trên val:
  - Linear/DLinear dạng đóng ở H = 720 giảm từ **0,740 xuống 0,646**. Mức này tốt hơn cả SGD của repo (0,702 / 0,736) và số công bố của Linear (0,698). Lưu ý: so sánh này đặt dạng đóng trên train đã lọc cạnh SGD trên train đủ. Nó ủng hộ nhận định "cửa sổ hằng làm hại bình phương tối thiểu nhiều hơn làm hại SGD", không cho biết SGD trên train đã lọc sẽ ra sao.
  - Ở H = 336 giảm từ 0,485 xuống 0,455, ngang SGD (0,454 / 0,442) và công bố (0,452 / 0,448).
  - Ở H = 192 chỉ đổi 0,001. Ở H = 96 thì **xấu đi** (0,301 → 0,315), có lẽ vì test ở H = 96 cũng có 234 cửa sổ đích hằng.
  - NLinear hầu như không đổi (≤ 0,0025). Một phần lý do: với cửa sổ có x hằng, x − x_L = 0, nên cửa sổ đó chỉ còn ảnh hưởng tới bias. Lập luận này không áp dụng cho các cửa sổ chỉ có y hằng, vốn chiếm phần lớn số cửa sổ bị bỏ ở H ≤ 192 (H = 96: 3135 cửa sổ y hằng, 1165 cửa sổ x hằng).
  - Cách diễn giải gợi ý (chưa kiểm trực tiếp): bình phương tối thiểu khớp đúng các đoạn cảm biến bị kẹt, còn SGD với dừng sớm thì không kịp khớp. Phép kiểm tiếp theo nên làm là so phần dư của SGD và của dạng đóng riêng trên các cửa sổ hằng của train.

**C. Ba việc rẻ: đều đạt.**

- **C.1:** ba đường MAE chạy lại bằng code hiện tại **trùng từng bit** với `lambda_path.csv`: lệch 0 ở mọi metric, cùng số vòng, cùng λ*, 0 bước fallback. Ba đường này vì vậy không bị ảnh hưởng. Đây là ba đường được chọn để kiểm, chưa phải bằng chứng cho mọi đường chạy trước khi có fallback.
- **C.2:** lưới MSE dày (141 λ) đổi λ* ở 25/36 đường, nhưng MSE test đổi nhiều nhất 4,4e-4 (NLinear ETTh1 H = 720), còn lại ≤ 1,5e-4.
  - Dòng C.5: λ* của DLinear ETTh1 H = 96 giờ là 446,7, không phải 562,34. Val gần như phẳng quanh đây (chênh 4e-6 giữa 446,7 và 562,34). Tại λ = 562,341, MSE test là 0,369745, **khớp 0,3697 tới 1e-4**. Tại λ* mới là 0,369827.
- **C.3:** xem mục B ở trên và mục 6.3.

**Bảng 36 ô** (mục 4): dạng đóng ở λ* tốt hơn số công bố quá 1% ở **19 ô**, trong ±1% ở **9 ô**, kém hơn quá 1% ở **8 ô**. Cả 8 ô kém đều là Linear và DLinear trên ETTh2, đúng như nghi vấn ban đầu (28/36 ô bằng hoặc tốt hơn).

## 2. Siêu tham số và khởi tạo của LTSF-Linear (A.1)

Repo gốc: `third_party/LTSF-Linear`, commit `0c11366` (28/01/2024).

| Tham số | Giá trị cho ETTh2 | Nguồn |
|---|---|---|
| learning rate | **0,05** (ETTh1: 0,005) | `scripts/EXP-LongForecasting/Linear/etth2.sh` dòng 24, 38, 52, 66 |
| batch size | 32 | như trên; mặc định `run_longExp.py` dòng 67 |
| số epoch | 10 | `run_longExp.py` dòng 66 (mặc định, script không đặt) |
| patience | 3 | `run_longExp.py` dòng 68 |
| `lradj` | type1: lr · 0,5^(epoch − 1) | `run_longExp.py` dòng 72; `utils/tools.py` dòng 12 |
| `individual` | False | `run_longExp.py` dòng 40 (script không bật `--individual`) |
| kernel size | 25, cố định trong code | `models/DLinear.py` dòng 48 (`--moving_avg` ở dòng 51 không được DLinear dùng) |
| seed | **2021**, đặt một lần đầu chương trình | `run_longExp.py` dòng 8–11 |
| optimizer, loss | Adam, MSE, không weight decay | `exp/exp_main.py` dòng 46, 50 |

**Một script cho cả ba mô hình.** `etth2.sh` chỉ đặt `model_name=DLinear` (dòng 10). README ghi "Linear, NLinear, and DLinear use the same scripts" (dòng 16) và "You can specify the name of the model in the script. (Linear, DLinear, NLinear)" (dòng 107). Vì vậy Linear và NLinear dùng cùng siêu tham số. Không có script nào khác cho ETTh2 đa biến: `Linear-I.sh` là bản `--individual`, còn `univariate/etth2.sh` là đơn biến. Không có chỗ mơ hồ, nên không dừng theo mục 6.

**Khởi tạo: mặc định của `nn.Linear`.** Mọi dòng đặt trọng số bằng 1/L đều bị comment:

- `models/DLinear.py` dòng 62–63 (nhánh `individual`) và dòng 69–70 (nhánh dùng chung);
- `models/Linear.py` dòng 16;
- `models/NLinear.py` dòng 15.

Vì vậy W₀,eff của B.2 không trùng khởi tạo của repo gốc. `mean` chỉ là một lựa chọn W₀, không phải điểm xuất phát của SGD.

**Khác biệt của `sgd_etth2.py` so với repo gốc:**

- **Train:** shuffle, `drop_last=True`, như gốc (`data_provider/data_factory.py` dòng 29–33).
- **Val dùng để dừng sớm:** ở đây là MSE trung bình toàn cục trên đủ cửa sổ. Repo gốc shuffle, `drop_last=True`, và lấy trung bình của trung bình từng batch (`exp_main.py` dòng 92–95). Cả hai cách đều chỉ dùng val.
- **Thứ tự lấy số ngẫu nhiên:** ở đây là `manual_seed(seed)` → tạo mô hình → các epoch. Repo gốc tạo dataset và các DataLoader trước khi train. Seed 2021 vì vậy không cho đúng cùng chuỗi ngẫu nhiên như repo gốc, chỉ cùng giá trị seed.
- **Test:** chấm trên W_eff, bias float64 với đúng ma trận cửa sổ của nghiệm dạng đóng (`load_cell`). Số cửa sổ được kiểm bằng `assert`: số cửa sổ × 7 = số hàng của dạng đóng, ở cả train, val, test và mọi H. Kiểm chéo thêm với forward float32 của mô hình (lệch < 1e-5).
- **Không cần thêm tham số vào `src/sgd.py`.** `train_sgd` đã có lr, epoch, patience, `lradj`; batch size và `drop_last` đặt ở DataLoader.

**A.2 (tái lập):** ETTh1 H = 96, seed 42, `lradj="type1"`, lr 0,005 cho MSE/MAE test **0,382463 / 0,404847**. Tham chiếu là 0,3825 / 0,4048, nên **đạt** trong 1e-3. File: [`results/sgd_etth2/a2_etth1_H96_seed42.json`](../results/sgd_etth2/a2_etth1_H96_seed42.json).

## 3. Bảng A.4: SGD so với nghiệm dạng đóng, ETTh2

SGD: trung bình ± độ lệch chuẩn (ddof = 1) theo 3 seed (2021, 2022, 2023), epoch tốt nhất chọn theo val MSE. Dạng đóng: mục tiêu MSE, λ* theo val MSE trên lưới dày (C.2). Mỗi lần chạy SGD mất 2–21 giây trên CPU.

### Linear

| H | | MSE test | MAE test | MSE val |
|---|---|---|---|---|
| 96 | SGD (3 seed) | 0,2959 ± 0,0137 | 0,3595 ± 0,0120 | 0,2163 ± 0,0043 |
| | dạng đóng, λ = 0 | 0,3019 | 0,3655 | 0,2219 |
| | dạng đóng, λ* = 1585 | 0,3009 | 0,3649 | 0,2212 |
| | công bố | 0,2880 | 0,3520 | không có |
| 192 | SGD (3 seed) | 0,4287 ± 0,0815 | 0,4425 ± 0,0473 | 0,3104 ± 0,0235 |
| | dạng đóng, λ = 0 | 0,3972 | 0,4276 | 0,3032 |
| | dạng đóng, λ* = 1122 | 0,3966 | 0,4273 | 0,3028 |
| | công bố | 0,3770 | 0,4130 | không có |
| 336 | SGD (3 seed) | 0,4544 ± 0,0340 | 0,4621 ± 0,0190 | 0,3948 ± 0,0062 |
| | dạng đóng, λ = 0 | 0,4858 | 0,4837 | 0,4090 |
| | dạng đóng, λ* = 1000 | 0,4853 | 0,4835 | 0,4087 |
| | công bố | 0,4520 | 0,4610 | không có |
| 720 | SGD (3 seed) | 0,7020 ± 0,0181 | 0,5947 ± 0,0068 | 0,6537 ± 0,0065 |
| | dạng đóng, λ = 0 | 0,7407 | 0,6116 | 0,6676 |
| | dạng đóng, λ* = 1000 | 0,7404 | 0,6115 | 0,6673 |
| | công bố | 0,6980 | 0,5950 | không có |

### DLinear

| H | | MSE test | MAE test | MSE val |
|---|---|---|---|---|
| 96 | SGD (3 seed) | 0,2916 ± 0,0073 | 0,3570 ± 0,0062 | 0,2184 ± 0,0017 |
| | dạng đóng, λ = 0 | 0,3019 | 0,3655 | 0,2219 |
| | dạng đóng, λ* = 1995 | 0,3008 | 0,3648 | 0,2213 |
| | công bố | 0,2890 | 0,3530 | không có |
| 192 | SGD (3 seed) | 0,3881 ± 0,0206 | 0,4205 ± 0,0154 | 0,2990 ± 0,0070 |
| | dạng đóng, λ = 0 | 0,3972 | 0,4276 | 0,3032 |
| | dạng đóng, λ* = 1259 | 0,3966 | 0,4273 | 0,3029 |
| | công bố | 0,3830 | 0,4180 | không có |
| 336 | SGD (3 seed) | 0,4423 ± 0,0277 | 0,4571 ± 0,0191 | 0,3913 ± 0,0100 |
| | dạng đóng, λ = 0 | 0,4858 | 0,4837 | 0,4090 |
| | dạng đóng, λ* = 1122 | 0,4853 | 0,4835 | 0,4087 |
| | công bố | 0,4480 | 0,4650 | không có |
| 720 | SGD (3 seed) | 0,7359 ± 0,0180 | 0,6100 ± 0,0067 | 0,6653 ± 0,0042 |
| | dạng đóng, λ = 0 | 0,7407 | 0,6116 | 0,6676 |
| | dạng đóng, λ* = 1122 | 0,7403 | 0,6115 | 0,6672 |
| | công bố | 0,6050 | 0,5510 | không có |

### NLinear (đối chứng)

| H | | MSE test | MAE test | MSE val |
|---|---|---|---|---|
| 96 | SGD (3 seed) | 0,2756 ± 0,0004 | 0,3373 ± 0,0005 | 0,2061 ± 0,0002 |
| | dạng đóng, λ = 0 | 0,2726 | 0,3352 | 0,2053 |
| | dạng đóng, λ* = 1778 | 0,2717 | 0,3341 | 0,2048 |
| | công bố | 0,2770 | 0,3380 | không có |
| 192 | SGD (3 seed) | 0,3464 ± 0,0059 | 0,3825 ± 0,0028 | 0,2726 ± 0,0004 |
| | dạng đóng, λ = 0 | 0,3343 | 0,3762 | 0,2717 |
| | dạng đóng, λ* = 1585 | 0,3336 | 0,3754 | 0,2713 |
| | công bố | 0,3440 | 0,3810 | không có |
| 336 | SGD (3 seed) | 0,3935 ± 0,0184 | 0,4188 ± 0,0098 | 0,3613 ± 0,0020 |
| | dạng đóng, λ = 0 | 0,3588 | 0,4004 | 0,3611 |
| | dạng đóng, λ* = 1585 | 0,3581 | 0,3997 | 0,3607 |
| | công bố | 0,3570 | 0,4000 | không có |
| 720 | SGD (3 seed) | 0,4040 ± 0,0139 | 0,4418 ± 0,0085 | 0,6106 ± 0,0070 |
| | dạng đóng, λ = 0 | 0,3930 | 0,4346 | 0,6041 |
| | dạng đóng, λ* = 1995 | 0,3924 | 0,4341 | 0,6035 |
| | công bố | 0,3940 | 0,4360 | không có |

### Khoảng cách SGD − λ* (MSE; âm là SGD tốt hơn)

| Mô hình | H | Δ test | 2·sd test | Δ val | SGD tốt hơn > 2 sd | SGD repo so với công bố | epoch tốt nhất theo seed |
|---|---|---|---|---|---|---|---|
| Linear | 96 | −0,0050 | 0,0273 | −0,0050 | không | +2,7 % | 6, 4, 9 |
| Linear | 192 | +0,0321 | 0,1630 | +0,0076 | không | **+13,7 %** | 1, 10, 8 |
| Linear | 336 | −0,0310 | 0,0680 | −0,0139 | không | +0,5 % | 8, 5, 6 |
| Linear | 720 | −0,0383 | 0,0361 | −0,0135 | **có** | +0,6 % | 6, 7, 9 |
| DLinear | 96 | −0,0092 | 0,0146 | −0,0029 | không | +0,9 % | 10, 6, 8 |
| DLinear | 192 | −0,0085 | 0,0412 | −0,0038 | không | +1,3 % | 9, 10, 5 |
| DLinear | 336 | −0,0430 | 0,0555 | −0,0174 | không | −1,3 % | 6, 7, 9 |
| DLinear | 720 | −0,0045 | 0,0361 | −0,0019 | không | **+21,6 %** | 10, 9, 9 |
| NLinear | 96 | +0,0040 | 0,0008 | +0,0013 | không | −0,5 % | 10, 9, 9 |
| NLinear | 192 | +0,0128 | 0,0118 | +0,0012 | không | +0,7 % | 6, 8, 10 |
| NLinear | 336 | +0,0354 | 0,0369 | +0,0006 | không | **+10,2 %** | 5, 6, 5 |
| NLinear | 720 | +0,0116 | 0,0278 | +0,0071 | không | +2,5 % | 10, 8, 4 |

Kết luận A theo tiêu chí: **Linear 1/4 ô, DLinear 0/4 ô, nên không xác nhận.**

Ghi chú:

- **Ô có SGD của repo lệch công bố quá 5%:** Linear H = 192 (một seed hỏng), DLinear H = 720, NLinear H = 336 (seed 2023 cho 0,414). So sánh với số công bố ở ba ô này không còn đáng tin.
- **Nhiều lần chạy chọn epoch 9 hoặc 10**, tức val vẫn đang giảm khi hết 10 epoch. SGD chưa hội tụ theo nghĩa của val; đây là cấu hình gốc và không đổi.
- **Ba seed của Linear H = 192:** test 0,5227 (dừng ở epoch 1/4), 0,3853, 0,3781. Bỏ seed 2021 thì trung bình 0,382, thấp hơn λ* (0,3966).

## 4. Bảng 36 ô: dạng đóng ở λ* so với công bố

MSE test, mục tiêu MSE, `val_objective`, lưới dày ở C.2. Chênh = (dạng đóng − công bố)/công bố. Ngưỡng 1%.

| Dataset | H | Mô hình | λ* | dạng đóng | công bố | chênh | |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | Linear | 0 | 0,3702 | 0,3750 | −1,3 % | dạng đóng tốt hơn |
| ETTh1 | 96 | NLinear | 1413 | 0,3695 | 0,3740 | −1,2 % | dạng đóng tốt hơn |
| ETTh1 | 96 | DLinear | 446,7 | 0,3698 | 0,3750 | −1,4 % | dạng đóng tốt hơn |
| ETTh1 | 192 | Linear | 223,9 | 0,4040 | 0,4180 | −3,4 % | dạng đóng tốt hơn |
| ETTh1 | 192 | NLinear | 2512 | 0,4027 | 0,4080 | −1,3 % | dạng đóng tốt hơn |
| ETTh1 | 192 | DLinear | 1000 | 0,4036 | 0,4050 | −0,4 % | |
| ETTh1 | 336 | Linear | 1995 | 0,4324 | 0,4790 | −9,7 % | dạng đóng tốt hơn |
| ETTh1 | 336 | NLinear | 8913 | 0,4261 | 0,4290 | −0,7 % | |
| ETTh1 | 336 | DLinear | 4467 | 0,4317 | 0,4390 | −1,7 % | dạng đóng tốt hơn |
| ETTh1 | 720 | Linear | 1,778e+04 | 0,4701 | 0,6240 | −24,7 % | dạng đóng tốt hơn |
| ETTh1 | 720 | NLinear | 1,995e+04 | 0,4329 | 0,4400 | −1,6 % | dạng đóng tốt hơn |
| ETTh1 | 720 | DLinear | 1,778e+04 | 0,4697 | 0,4720 | −0,5 % | |
| ETTh2 | 96 | Linear | 1585 | 0,3009 | 0,2880 | +4,5 % | **dạng đóng kém hơn** |
| ETTh2 | 96 | NLinear | 1778 | 0,2717 | 0,2770 | −1,9 % | dạng đóng tốt hơn |
| ETTh2 | 96 | DLinear | 1995 | 0,3008 | 0,2890 | +4,1 % | **dạng đóng kém hơn** |
| ETTh2 | 192 | Linear | 1122 | 0,3966 | 0,3770 | +5,2 % | **dạng đóng kém hơn** |
| ETTh2 | 192 | NLinear | 1585 | 0,3336 | 0,3440 | −3,0 % | dạng đóng tốt hơn |
| ETTh2 | 192 | DLinear | 1259 | 0,3966 | 0,3830 | +3,5 % | **dạng đóng kém hơn** |
| ETTh2 | 336 | Linear | 1000 | 0,4853 | 0,4520 | +7,4 % | **dạng đóng kém hơn** |
| ETTh2 | 336 | NLinear | 1585 | 0,3581 | 0,3570 | +0,3 % | |
| ETTh2 | 336 | DLinear | 1122 | 0,4853 | 0,4480 | +8,3 % | **dạng đóng kém hơn** |
| ETTh2 | 720 | Linear | 1000 | 0,7404 | 0,6980 | +6,1 % | **dạng đóng kém hơn** |
| ETTh2 | 720 | NLinear | 1995 | 0,3924 | 0,3940 | −0,4 % | |
| ETTh2 | 720 | DLinear | 1122 | 0,7403 | 0,6050 | +22,4 % | **dạng đóng kém hơn** |
| ETTm1 | 96 | Linear | 631 | 0,2992 | 0,3080 | −2,8 % | dạng đóng tốt hơn |
| ETTm1 | 96 | NLinear | 5012 | 0,3004 | 0,3060 | −1,8 % | dạng đóng tốt hơn |
| ETTm1 | 96 | DLinear | 1413 | 0,2992 | 0,2990 | +0,1 % | |
| ETTm1 | 192 | Linear | 354,8 | 0,3340 | 0,3400 | −1,8 % | dạng đóng tốt hơn |
| ETTm1 | 192 | NLinear | 5623 | 0,3356 | 0,3490 | −3,8 % | dạng đóng tốt hơn |
| ETTm1 | 192 | DLinear | 1122 | 0,3339 | 0,3350 | −0,3 % | |
| ETTm1 | 336 | Linear | 446,7 | 0,3683 | 0,3760 | −2,0 % | dạng đóng tốt hơn |
| ETTm1 | 336 | NLinear | 6310 | 0,3703 | 0,3750 | −1,3 % | dạng đóng tốt hơn |
| ETTm1 | 336 | DLinear | 1259 | 0,3683 | 0,3690 | −0,2 % | |
| ETTm1 | 720 | Linear | 1259 | 0,4232 | 0,4400 | −3,8 % | dạng đóng tốt hơn |
| ETTm1 | 720 | NLinear | 7079 | 0,4262 | 0,4330 | −1,6 % | dạng đóng tốt hơn |
| ETTm1 | 720 | DLinear | 2512 | 0,4232 | 0,4250 | −0,4 % | |

**Đếm:** tốt hơn quá 1% ở 19 ô, trong ±1% ở 9 ô, kém hơn quá 1% ở 8 ô.

- **8 ô kém** đúng là Linear và DLinear trên ETTh2, cả bốn H.
- **DLinear trên ETTh1 và ETTm1 nằm trong ±1% ở 6/8 ô.** Như vậy SGD của bài đã đến gần nghiệm tối ưu với DLinear. Với Linear thì không: chênh tới 24,7% ở ETTh1 H = 720.
- **Hai trường hợp H = 720 của mục 0:**
  - **ETTh1:** dạng đóng Linear 0,4701, DLinear 0,4697, trùng nhau. Chênh 0,624 so với 0,472 trong bài là do SGD của Linear, không phải do kiến trúc.
  - **ETTh2:** dạng đóng 0,7404 và 0,7403, kém cả hai số SGD công bố. Nhưng SGD của repo cũng cho DLinear 0,736, tức không tái lập được 0,605 (mục 3). Với Linear, SGD của repo (0,702) tái lập được số công bố (0,698).

## 5. Kiểm giả thuyết

### 5.1. B.1: tổng hàng của W (ETTh2)

Tổng hàng là w_hᵀ1 trên L hệ số đầu, với h = 1…H. P10/P90 là phân vị 10% và 90%. ‖b‖ là chuẩn Euclid của vector bias [H].

| H | Mô hình | Nghiệm | trung vị | P10 | P90 | ‖b‖ |
|---|---|---|---|---|---|---|
| 96 | Linear | SGD seed 2021 | 0,9752 | 0,8937 | 1,0328 | 0,1610 |
| 96 | Linear | SGD seed 2022 | 1,0449 | 0,9248 | 1,1228 | 0,1778 |
| 96 | Linear | SGD seed 2023 | 0,9063 | 0,8688 | 0,9514 | 0,0879 |
| 96 | Linear | dạng đóng λ = 0 | 0,8804 | 0,8336 | 0,9593 | 0,0883 |
| 96 | Linear | dạng đóng λ* = 1585 | 0,8793 | 0,8330 | 0,9578 | 0,0888 |
| 96 | DLinear | SGD seed 2021 | 0,8711 | 0,8520 | 0,9559 | 0,0859 |
| 96 | DLinear | SGD seed 2022 | 0,9794 | 0,8697 | 1,0292 | 0,1588 |
| 96 | DLinear | SGD seed 2023 | 0,9047 | 0,8796 | 0,9460 | 0,1305 |
| 96 | DLinear | dạng đóng λ = 0 | 0,8804 | 0,8336 | 0,9593 | 0,0883 |
| 96 | DLinear | dạng đóng λ* = 1995 | 0,8794 | 0,8329 | 0,9578 | 0,0888 |
| 96 | NLinear | dạng đóng λ* = 1778 | 1,0000 | 1,0000 | 1,0000 | 0,1030 |
| 192 | Linear | SGD seed 2021 | 0,8212 | 0,5884 | 1,0605 | 0,3894 |
| 192 | Linear | SGD seed 2022 | 0,8612 | 0,7499 | 0,9660 | 0,2320 |
| 192 | Linear | SGD seed 2023 | 0,8438 | 0,7807 | 0,9877 | 0,2585 |
| 192 | Linear | dạng đóng λ = 0 | 0,8246 | 0,7593 | 0,9403 | 0,2307 |
| 192 | Linear | dạng đóng λ* = 1122 | 0,8241 | 0,7589 | 0,9394 | 0,2312 |
| 192 | DLinear | SGD seed 2021 | 0,8011 | 0,7552 | 0,9316 | 0,2461 |
| 192 | DLinear | SGD seed 2022 | 0,8410 | 0,7887 | 0,9463 | 0,2411 |
| 192 | DLinear | SGD seed 2023 | 0,9110 | 0,8257 | 0,9976 | 0,3991 |
| 192 | DLinear | dạng đóng λ = 0 | 0,8246 | 0,7593 | 0,9403 | 0,2307 |
| 192 | DLinear | dạng đóng λ* = 1259 | 0,8242 | 0,7590 | 0,9395 | 0,2311 |
| 192 | NLinear | dạng đóng λ* = 1585 | 1,0000 | 1,0000 | 1,0000 | 0,2781 |
| 336 | Linear | SGD seed 2021 | 0,8019 | 0,7075 | 0,9356 | 0,5645 |
| 336 | Linear | SGD seed 2022 | 0,9141 | 0,7172 | 1,0063 | 0,6038 |
| 336 | Linear | SGD seed 2023 | 0,8108 | 0,6358 | 0,9793 | 0,5698 |
| 336 | Linear | dạng đóng λ = 0 | 0,7751 | 0,6468 | 0,9111 | 0,5760 |
| 336 | Linear | dạng đóng λ* = 1000 | 0,7747 | 0,6466 | 0,9103 | 0,5764 |
| 336 | DLinear | SGD seed 2021 | 0,8814 | 0,7259 | 0,9545 | 0,4765 |
| 336 | DLinear | SGD seed 2022 | 0,8332 | 0,6990 | 0,9744 | 0,6039 |
| 336 | DLinear | SGD seed 2023 | 0,7803 | 0,6817 | 0,8670 | 0,5771 |
| 336 | DLinear | dạng đóng λ = 0 | 0,7751 | 0,6468 | 0,9111 | 0,5760 |
| 336 | DLinear | dạng đóng λ* = 1122 | 0,7748 | 0,6466 | 0,9104 | 0,5763 |
| 336 | NLinear | dạng đóng λ* = 1585 | 1,0000 | 1,0000 | 1,0000 | 0,7071 |
| 720 | Linear | SGD seed 2021 | 0,6323 | 0,4613 | 0,8968 | 1,3438 |
| 720 | Linear | SGD seed 2022 | 0,6499 | 0,4200 | 0,8938 | 1,5664 |
| 720 | Linear | SGD seed 2023 | 0,6282 | 0,4150 | 0,9057 | 1,4609 |
| 720 | Linear | dạng đóng λ = 0 | 0,6331 | 0,3894 | 0,8674 | 1,4778 |
| 720 | Linear | dạng đóng λ* = 1000 | 0,6329 | 0,3891 | 0,8670 | 1,4783 |
| 720 | DLinear | SGD seed 2021 | 0,5982 | 0,3863 | 0,8576 | 1,4975 |
| 720 | DLinear | SGD seed 2022 | 0,6897 | 0,4105 | 0,8433 | 1,4568 |
| 720 | DLinear | SGD seed 2023 | 0,6818 | 0,3732 | 0,8670 | 1,5107 |
| 720 | DLinear | dạng đóng λ = 0 | 0,6331 | 0,3894 | 0,8674 | 1,4778 |
| 720 | DLinear | dạng đóng λ* = 1122 | 0,6330 | 0,3892 | 0,8672 | 1,4782 |
| 720 | NLinear | dạng đóng λ* = 1995 | 1,0000 | 1,0000 | 1,0000 | 2,0165 |

Nhận xét:

- **Ridge gần như không đổi W** ở λ* ≈ 1000–2000: trung vị tổng hàng lệch λ = 0 dưới 0,001. λ* trên ETTh2 vì vậy gần như không có tác dụng.
- **Trung vị tổng hàng của SGD lớn hơn của dạng đóng (λ = 0) ở 18/24 cặp** (seed, ô, mô hình), nhưng mức chênh biến động mạnh theo seed (−0,035 đến +0,164). Ở H = 720, ba trên sáu lần chạy cho trung vị dưới dạng đóng.
- **Kết luận B.1:** hướng dự đoán có xuất hiện (SGD gần 1 hơn) ở H nhỏ, nhưng yếu và không nhất quán. Không đủ để coi là nguồn của khoảng cách.

### 5.2. B.2: ridge co về W₀

Mục tiêu MSE, dạng đóng, λ* theo val MSE trên lưới 141 λ. Linear: Pen = I; DLinear: Pen = N⁻¹ (k = 25). Mỗi ô ghi λ* / MSE val / MSE test.

**Kiểm `zero` so với `lambda_path.csv`:** lệch lớn nhất **2,2e-16** trên 4 metric (MSE, MAE của val và test), 141 λ (cộng λ = 0), 2 mô hình, 12 ô; không thiếu dòng nào. **Đạt** 1e-10.

| Dataset | H | Mô hình | zero: λ* / val / test | mean: λ* / val / test | last: λ* / val / test |
|---|---|---|---|---|---|
| ETTh1 | 96 | Linear | 0 (đầu mút) / 0,6516 / 0,3702 | 0 (đầu mút) / 0,6516 / 0,3702 | 1413 / 0,6489 / 0,3751 |
| ETTh1 | 96 | DLinear | 446,7 / 0,6516 / 0,3698 | 398,1 / 0,6516 / 0,3699 | 3548 / 0,6484 / 0,3767 |
| ETTh1 | 192 | Linear | 223,9 / 0,8708 / 0,4040 | 177,8 / 0,8708 / 0,4040 | 1122 / 0,8685 / 0,4078 |
| ETTh1 | 192 | DLinear | 1000 / 0,8706 / 0,4036 | 891,3 / 0,8706 / 0,4036 | 3162 / 0,8675 / 0,4095 |
| ETTh1 | 336 | Linear | 1995 / 1,0631 / 0,4324 | 1585 / 1,0632 / 0,4325 | 501,2 / 1,0627 / 0,4345 |
| ETTh1 | 336 | DLinear | 4467 / 1,0626 / 0,4317 | 3162 / 1,0628 / 0,4319 | 1995 / 1,0618 / 0,4357 |
| ETTh1 | 720 | Linear | 1,778e+04 / 1,2162 / 0,4701 | 1e+04 / 1,2169 / 0,4694 | 223,9 / 1,2185 / 0,4717 |
| ETTh1 | 720 | DLinear | 1,778e+04 / 1,2156 / 0,4697 | 1,122e+04 / 1,2164 / 0,4691 | 1000 / 1,2179 / 0,4719 |
| ETTh2 | 96 | Linear | 1585 / 0,2212 / 0,3009 | 1778 / 0,2212 / 0,3008 | 28,18 / 0,2219 / 0,3019 |
| ETTh2 | 96 | DLinear | 1995 / 0,2213 / 0,3008 | 1995 / 0,2212 / 0,3007 | 316,2 / 0,2219 / 0,3017 |
| ETTh2 | 192 | Linear | 1122 / 0,3028 / 0,3966 | 1259 / 0,3028 / 0,3965 | 100 / 0,3032 / 0,3971 |
| ETTh2 | 192 | DLinear | 1259 / 0,3029 / 0,3966 | 1413 / 0,3028 / 0,3965 | 501,2 / 0,3031 / 0,3970 |
| ETTh2 | 336 | Linear | 1000 / 0,4087 / 0,4853 | 1122 / 0,4086 / 0,4852 | 177,8 / 0,4089 / 0,4858 |
| ETTh2 | 336 | DLinear | 1122 / 0,4087 / 0,4853 | 1259 / 0,4086 / 0,4852 | 707,9 / 0,4087 / 0,4857 |
| ETTh2 | 720 | Linear | 1000 / 0,6673 / 0,7404 | 1122 / 0,6672 / 0,7402 | 281,8 / 0,6673 / 0,7404 |
| ETTh2 | 720 | DLinear | 1122 / 0,6672 / 0,7403 | 1259 / 0,6672 / 0,7401 | 1259 / 0,6668 / 0,7400 |
| ETTm1 | 96 | Linear | 631 / 0,3822 / 0,2992 | 631 / 0,3822 / 0,2992 | 794,3 / 0,3820 / 0,2997 |
| ETTm1 | 96 | DLinear | 1413 / 0,3821 / 0,2992 | 1413 / 0,3821 / 0,2992 | 2818 / 0,3819 / 0,3002 |
| ETTm1 | 192 | Linear | 354,8 / 0,4899 / 0,3340 | 316,2 / 0,4899 / 0,3340 | 1122 / 0,4896 / 0,3347 |
| ETTm1 | 192 | DLinear | 1122 / 0,4899 / 0,3339 | 1000 / 0,4899 / 0,3339 | 4467 / 0,4892 / 0,3355 |
| ETTm1 | 336 | Linear | 446,7 / 0,6177 / 0,3683 | 398,1 / 0,6177 / 0,3683 | 1122 / 0,6173 / 0,3690 |
| ETTm1 | 336 | DLinear | 1259 / 0,6176 / 0,3683 | 1122 / 0,6176 / 0,3683 | 5012 / 0,6168 / 0,3700 |
| ETTm1 | 720 | Linear | 1259 / 0,8713 / 0,4232 | 1000 / 0,8713 / 0,4232 | 1000 / 0,8710 / 0,4238 |
| ETTm1 | 720 | DLinear | 2512 / 0,8713 / 0,4232 | 1995 / 0,8713 / 0,4232 | 4467 / 0,8705 / 0,4246 |

Kiểm theo tiêu chí của nhiệm vụ:

- **Trên ETTh2, `mean` và `last` không tốt hơn `zero` rõ rệt.**
  - `mean` đổi MSE test tối đa 0,0002 và MSE val tối đa 0,0001.
  - `last` không tốt hơn `zero` ở H ≤ 336, trên cả val và test (test kém hơn tới +0,001). Chỉ nhỉnh hơn ở DLinear H = 720 (test 0,7400 so với 0,7403).
  - Cả hai còn rất xa SGD của Phần A. Ví dụ DLinear H = 336: 0,485 so với 0,442; Linear H = 720: 0,740 so với 0,702.
- **Trên ETTh1 và ETTm1, `mean` gần như trùng `zero`.** `last` cho một hiện tượng riêng: ở 14/16 tổ hợp (ETTh1 H ≤ 336 và cả ETTm1), `last` có **MSE val thấp hơn** `zero` (tới −0,0032) nhưng **MSE test cao hơn** (tới +0,0069, ETTh1 H = 96 DLinear).
  - Nghĩa là co về persistence giúp trên val mà hại trên test. Nếu chọn W₀ theo val, ta sẽ chọn `last` và kém đi trên test.
  - Không tập nào trong 3 dataset cho thấy `last` hay `mean` giúp trên cả val lẫn test một cách đáng kể.
- **Kết luận B.2:** giả thuyết **không được ủng hộ**. Đổi điểm co của ridge không đưa nghiệm dạng đóng trên ETTh2 lại gần SGD. Điều này hợp với B.1: λ* ≈ 1000 hầu như không làm W thay đổi (mục 5.1), nên điểm co cũng gần như không quan trọng.

File: [`results/ridge_w0/summary.json`](../results/ridge_w0/summary.json), [`path.csv`](../results/ridge_w0/path.csv), từng ô ở [`results/ridge_w0/cells/`](../results/ridge_w0/cells/).

## 6. Ba việc rẻ

### 6.1. C.1: kiểm các đường chạy trước khi có fallback

Chạy lại bằng code hiện tại (commit `5481fa2`, có fallback float64 theo từng h). Với DLinear ETTh1 H = 720, λ = 0 được chạy lại qua Linear λ = 0 rồi chép sang như runner, nên λ = 0 cũng được kiểm. File chính không bị đụng.

| Đường | lệch lớn nhất (5 metric, mọi λ) | λ* (val MAE) cũ → mới | λ* (val MSE) cũ → mới | tổng vòng cũ → mới | fallback mới (bước h) | |
|---|---|---|---|---|---|---|
| ETTh2 H = 336 Linear MAE | 0 | 1000 → 1000 | 1000 → 1000 | 1770 → 1770 | 0 | đạt |
| ETTh2 H = 336 NLinear MAE | 0 | 1000 → 1000 | 3000 → 3000 | 1701 → 1701 | 0 | đạt |
| ETTh1 H = 720 DLinear MAE | 0 | 1e+04 → 1e+04 | 3e+04 → 3e+04 | 1939 → 1939 | 0 | đạt |

- **Trùng từng bit ở mọi λ**, kể cả hai λ mở rộng 30 000 và 90 000 của đường thứ ba, với cùng số vòng ở từng λ.
- **Không bước h nào cần fallback**, nên `gram_fallback` không hề chạy và kết quả buộc phải trùng code cũ.
- Số fallback của các lần chạy cũ không được ghi (code trước fallback). Từ lần này, runner ghi cột `n_fallback` vào các file CSV mới.
- File: [`results/recheck/compare.json`](../results/recheck/compare.json), [`lambda_path_recheck.csv`](../results/recheck/lambda_path_recheck.csv).

### 6.2. C.2: làm dày lưới λ của MSE

Lưới mới gồm {0} ∪ 10^{−2; −1,95; …; 5}, tổng cộng 141 giá trị dương. Runner chỉ thêm 105 λ mới mỗi đường; `lambda_path.csv` giờ có 5112 dòng MSE (36 đường × 142). λ* theo `val_objective` (với MSE là val MSE). Cột "cũ" lấy từ `results.csv` ở commit `0563109`.

| Dataset | H | Mô hình | λ* cũ | λ* mới | test cũ | test mới | chênh |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | Linear | 0 | 0 (đầu mút) | 0,3702 | 0,3702 | 0 |
| ETTh1 | 96 | NLinear | 1585 | 1413 | 0,3695 | 0,3695 | +3,7e-05 |
| ETTh1 | 96 | DLinear | 398,1 | 446,7 | 0,3699 | 0,3698 | −3,7e-05 |
| ETTh1 | 192 | Linear | 251,2 | 223,9 | 0,4040 | 0,4040 | +1,9e-05 |
| ETTh1 | 192 | NLinear | 2512 | 2512 | 0,4027 | 0,4027 | 0 |
| ETTh1 | 192 | DLinear | 1000 | 1000 | 0,4036 | 0,4036 | 0 |
| ETTh1 | 336 | Linear | 2512 | 1995 | 0,4323 | 0,4324 | +1,3e-04 |
| ETTh1 | 336 | NLinear | 1e+04 | 8913 | 0,4261 | 0,4261 | −1,1e-05 |
| ETTh1 | 336 | DLinear | 3981 | 4467 | 0,4318 | 0,4317 | −7,7e-05 |
| ETTh1 | 720 | Linear | 1,585e+04 | 1,778e+04 | 0,4699 | 0,4701 | +1,5e-04 |
| ETTh1 | 720 | NLinear | 2,512e+04 | 1,995e+04 | 0,4334 | 0,4329 | −4,4e-04 |
| ETTh1 | 720 | DLinear | 1,585e+04 | 1,778e+04 | 0,4695 | 0,4697 | +1,3e-04 |
| ETTh2 | 96 | Linear | 1585 | 1585 | 0,3009 | 0,3009 | 0 |
| ETTh2 | 96 | NLinear | 1585 | 1778 | 0,2717 | 0,2717 | −3,6e-05 |
| ETTh2 | 96 | DLinear | 1585 | 1995 | 0,3009 | 0,3008 | −7,7e-05 |
| ETTh2 | 192 | Linear | 1000 | 1122 | 0,3966 | 0,3966 | −1,9e-05 |
| ETTh2 | 192 | NLinear | 1585 | 1585 | 0,3336 | 0,3336 | 0 |
| ETTh2 | 192 | DLinear | 1585 | 1259 | 0,3965 | 0,3966 | +3,6e-05 |
| ETTh2 | 336 | Linear | 1000 | 1000 | 0,4853 | 0,4853 | 0 |
| ETTh2 | 336 | NLinear | 1585 | 1585 | 0,3581 | 0,3581 | 0 |
| ETTh2 | 336 | DLinear | 1000 | 1122 | 0,4854 | 0,4853 | −2,0e-05 |
| ETTh2 | 720 | Linear | 1000 | 1000 | 0,7404 | 0,7404 | 0 |
| ETTh2 | 720 | NLinear | 2512 | 1995 | 0,3924 | 0,3924 | +1,1e-05 |
| ETTh2 | 720 | DLinear | 1000 | 1122 | 0,7403 | 0,7403 | −2,1e-06 |
| ETTm1 | 96 | Linear | 631 | 631 | 0,2992 | 0,2992 | 0 |
| ETTm1 | 96 | NLinear | 6310 | 5012 | 0,3005 | 0,3004 | −2,2e-05 |
| ETTm1 | 96 | DLinear | 1585 | 1413 | 0,2992 | 0,2992 | +6,6e-06 |
| ETTm1 | 192 | Linear | 398,1 | 354,8 | 0,3340 | 0,3340 | +3,3e-06 |
| ETTm1 | 192 | NLinear | 6310 | 5623 | 0,3356 | 0,3356 | −9,7e-06 |
| ETTm1 | 192 | DLinear | 1000 | 1122 | 0,3339 | 0,3339 | −5,3e-06 |
| ETTm1 | 336 | Linear | 398,1 | 446,7 | 0,3683 | 0,3683 | −3,6e-06 |
| ETTm1 | 336 | NLinear | 6310 | 6310 | 0,3703 | 0,3703 | 0 |
| ETTm1 | 336 | DLinear | 1585 | 1259 | 0,3683 | 0,3683 | +1,2e-05 |
| ETTm1 | 720 | Linear | 1585 | 1259 | 0,4232 | 0,4232 | +1,5e-05 |
| ETTm1 | 720 | NLinear | 6310 | 7079 | 0,4262 | 0,4262 | +3,6e-06 |
| ETTm1 | 720 | DLinear | 2512 | 2512 | 0,4232 | 0,4232 | 0 |

- **λ* đổi ở 25/36 đường, nhưng MSE test chỉ đổi** tối đa 4,4e-4 (NLinear ETTh1 H = 720), còn lại ≤ 1,5e-4. Lưới cũ đã đủ dày cho mục đích so sánh.
- **Dòng C.5 (DLinear ETTh1 H = 96):**
  - λ* mới là 446,7 chứ không phải 562,34. Val MSE tại 446,7 / 501,2 / 562,3 lần lượt là 0,651554 / 0,651555 / 0,651558, gần như phẳng.
  - Tại λ = 562,341, MSE test là **0,369745**, khớp 0,3697 tới 1e-4. Tại λ* mới, MSE test là 0,369827.
- **ETTh1 H = 96 Linear có λ* = 0 ở đầu mút dưới**, như trước khi làm dày.
- `select_lambda.py` đã chạy lại, nên `results.csv` giờ dùng lưới dày.

### 6.3. C.3: đoạn hằng của ETTh2

Cửa sổ hằng là cửa sổ (một kênh, sau chuẩn hóa) có max − min của x **hoặc** của y nhỏ hơn 1e-12.

| H | tập | hằng / tổng | x hằng | y hằng | theo kênh (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) |
|---|---|---|---|---|---|
| 96 | train | 3612 / 57463 | 1165 | 3135 | 0, 0, 1026, 0, 342, 2244, 0 |
| 96 | val | 58 / 19495 | 0 | 58 | 0, 58, 0, 0, 0, 0, 0 |
| 96 | test | 234 / 19495 | 0 | 234 | 0, 203, 0, 0, 0, 31, 0 |
| 192 | train | 2922 / 56791 | 1165 | 2255 | 0, 0, 1026, 0, 230, 1666, 0 |
| 192 | val | 0 / 18823 | 0 | 0 | 0 |
| 192 | test | 35 / 18823 | 0 | 35 | 0, 35, 0, 0, 0, 0, 0 |
| 336 | train | 1976 / 55783 | 1165 | 1165 | 0, 0, 1026, 0, 86, 864, 0 |
| 336 | val, test | 0 | 0 | 0 | 0 |
| 720 | train | 1339 / 53095 | 1033 | 306 | 0, 0, 864, 0, 43, 432, 0 |
| 720 | val, test | 0 | 0 | 0 | 0 |

MSE dạng đóng, train bỏ các cửa sổ hằng, val và test giữ nguyên, λ* theo val MSE trên lưới dày. "Trước" là `lambda_path.csv` (cùng lưới, train đủ).

| H | Mô hình | số cửa sổ bỏ | λ* trước → sau | MSE test trước → sau |
|---|---|---|---|---|
| 96 | Linear | 3612 | 1585 → 1122 | 0,3009 → **0,3146** |
| 96 | DLinear | 3612 | 1995 → 1259 | 0,3008 → **0,3145** |
| 96 | NLinear | 3612 | 1778 → 1778 | 0,2717 → 0,2715 |
| 192 | Linear | 2922 | 1122 → 891,3 | 0,3966 → 0,3954 |
| 192 | DLinear | 2922 | 1259 → 1000 | 0,3966 → 0,3954 |
| 192 | NLinear | 2922 | 1585 → 1585 | 0,3336 → 0,3317 |
| 336 | Linear | 1976 | 1000 → 891,3 | 0,4853 → **0,4553** |
| 336 | DLinear | 1976 | 1122 → 1000 | 0,4853 → **0,4553** |
| 336 | NLinear | 1976 | 1585 → 1585 | 0,3581 → 0,3556 |
| 720 | Linear | 1339 | 1000 → 891,3 | 0,7404 → **0,6462** |
| 720 | DLinear | 1339 | 1122 → 1122 | 0,7403 → **0,6461** |
| 720 | NLinear | 1339 | 1995 → 1995 | 0,3924 → 0,3908 |

Khoảng cách NLinear − Linear (MSE test):

| H | trước | sau |
|---|---|---|
| 96 | −0,0292 | −0,0431 |
| 192 | −0,0630 | −0,0638 |
| 336 | −0,1272 | −0,0997 |
| 720 | −0,3480 | −0,2553 |

Nhận xét:

- **Đoạn hằng nằm ở ba kênh MUFL, LUFL, LULL trong train.** Val và test chỉ có đích hằng (kênh HULL, LULL) ở H ≤ 192.
- **Ở H ∈ {336, 720}, chỉ 2,5–3,5% số cửa sổ train làm Linear/DLinear kém đi 0,03–0,09 MSE test.** Bỏ chúng đưa dạng đóng về ngang hoặc tốt hơn cả SGD của repo (huấn luyện trên train đủ) lẫn số công bố.
- **Ở H = 96 thì ngược lại** (+0,014). Test ở H = 96 có 234 cửa sổ đích hằng. Có thể mô hình học từ các cửa sổ hằng của train lại dự báo các đoạn này tốt hơn; chưa kiểm.
- **Khoảng cách NLinear − Linear thu hẹp** ở H = 336 (−0,127 → −0,100) và H = 720 (−0,348 → −0,255), nhưng vẫn rất lớn. Đoạn hằng chỉ giải thích một phần ưu thế của NLinear trên ETTh2.
- **Không chạy MAE cho NLinear H = 720** (phần tùy chọn). Đây là đường cần 26 713 bước fallback; trong grid, cả đường mất khoảng 67 phút.

## 7. Tái lập

```bash
python -m unittest discover -s tests -t .              # 46 test, tất cả đạt
python scripts/sgd_etth2.py --step a2                   # A.2
python scripts/sgd_etth2.py --step run                  # A.3: 36 lần chạy, ~6 phút CPU
python scripts/recheck_paths.py                         # C.1: ~70 phút GPU
python scripts/runner.py --objective MSE                # C.2: thêm 105 λ mỗi đường MSE, chạy lại = chạy tiếp
python scripts/select_lambda.py                         # C.2: results.csv theo lưới dày
python scripts/constant_windows.py                      # C.3
python scripts/ridge_w0.py                              # B.2, chạy lại = chạy tiếp theo ô
python scripts/report_etth2.py                          # mọi bảng: results/etth2_report/tables.md
```

Ghi chú về các lần chạy:

- **SGD được chạy hai lần.** Lần đầu, vài JSON mang nhãn `-dirty` vì lúc đó có file script mới chưa track. Lần thứ hai chạy trên cây sạch (commit `72d5611`): W trùng từng bit với lần đầu (lệch lớn nhất 0), vì CPU cho kết quả tất định. Các file hiện có là của lần thứ hai.
- **C.2 bị dừng giữa chừng** do giới hạn thời gian của tiến trình nền, rồi chạy tiếp. Runner ghi từng dòng, nên không thiếu hay trùng dòng (đã kiểm khóa).
- **B.2 chạy lần đầu quá chậm**, vì gọi `Cell.evaluate` tính cả phần dư trên train. Lần chạy đó bị dừng, rồi sửa để chỉ tính val/test bằng đúng công thức và lưu theo từng ô (commit `2fdebb9`).
