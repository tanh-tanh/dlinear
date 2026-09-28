# Báo cáo tiến độ so với outline

Ngày lập: 29/09/2026. Căn cứ: code và output đã lưu trong các notebook, cộng các phép kiểm chạy lại khi tái cấu trúc (`python -m unittest discover -s tests -t .`, 19/19 đạt).

Ký hiệu: ✅ xong · 🟡 làm một phần · ❌ chưa làm

## 1. Tóm tắt

- **Tuần 1 xong cả 5 việc.** Có hai số trong outline đã cũ: MSE test của DLinear SGD là **0,3766** (không phải 0,3779) sau khi sửa cách tính trung bình, nên OLS (0,3702) tốt hơn SGD khoảng **1,7%** (không phải 2%).
- **Tuần 2 xong khoảng một nửa, chỉ trên một ô (ETTh1, H = 96).** Ba dự đoán của RQ1 đã kiểm xong và đều đúng. Code đã chuyển sang `src/` (lần tái cấu trúc này). Còn thiếu: IRLS cho MAE/Huber, `runner.py`, grid 216 ô, `results.csv`, bảng B3, notebook 02.
- **Tuần 3 và 4 chưa bắt đầu**, trừ một điểm dữ liệu cho RQ3 (ETTh1 H = 96) và README sơ bộ.
- **Grid thí nghiệm: 6/216 ô** (3 mô hình × MSE × {λ = 0, λ*} × ETTh1 × H = 96).

| Câu hỏi | Trạng thái | Đã có gì |
|---|---|---|
| RQ1 | 🟡 | Trả lời đầy đủ trên ETTh1 H = 96, sai khác ở mức sai số máy. Chưa chạy trên dataset và horizon khác |
| RQ2 | ❌ | Mới có mục tiêu MSE. Chưa có MAE, Huber, outlier |
| RQ3 | 🟡 | Một ô: SGD 0,3766 so với OLS 0,3702 và ridge DLinear ở λ* 0,3697 |

## 2. Tái cấu trúc đã làm

| Trước | Sau |
|---|---|
| `dlinearv2.ipynb` | `notebooks/00_dlinear_sgd_baseline.ipynb` |
| `Nlinear.ipynb` | `notebooks/00_nlinear_closed_form.ipynb` |
| `01_three_predictions_1.ipynb` | `notebooks/01_three_predictions.ipynb` |
| `dlinear.ipynb` | `notebooks/archive/00_dlinear_v1.ipynb` (bản đầu, thứ tự cell sai, không chạy được từ đầu) |
| `best.pt` | `checkpoints/dlinear_sgd_ETTh1_L336_H96.pt` |
| `outline.docx` | `docs/outline.docx` |
| hàm chép trong 3 notebook | `src/data.py`, `src/operators.py`, `src/solvers.py`, `src/metrics.py`, `src/models.py`, `src/sgd.py` |
| — | `tests/` (19 phép kiểm), `README.md`, `requirements.txt` |

Trong notebook chỉ sửa đường dẫn (`../data`, `../checkpoints`, `../results`, `../third_party/LTSF-Linear`); code và output giữ nguyên. Đã chạy lại `00_nlinear_closed_form` từ thư mục mới, ra đúng số cũ.

Các phép kiểm trong `tests/` xác nhận `src/` cho lại đúng số của notebook: MSE test OLS = 0,37023527067813916, MSE test NLinear = 0,3701986328402697, SSE train NLinear − OLS = 17002,829, và checkpoint SGD cho MSE/MAE test = 0,3766 / 0,3996.

## 3. Đối chiếu theo phần "Hướng triển khai"

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| MSE: OLS, ridge dạng đóng | ✅ | `src/solvers.py` |
| Quét λ nhanh bằng phân rã | ✅ | `ridge_path`: phân rã riêng `XᵀX` một lần (tương đương SVD). Notebook 01 quét 34 giá trị λ và khớp λ' trên lưới 441 điểm |
| MAE, Huber bằng IRLS (PyTorch, GPU) | ❌ | |
| NLinear: nghiệm có ràng buộc (Lagrange) | ✅ | `nlinear_constrained`; đã kiểm trên dữ liệu giả và trên ETTh1 |
| Kiểm dẫn xuất bằng module DLinear/NLinear của repo LTSF-Linear | ✅ | Notebook 01: nạp nghiệm vào module, gradient qua autograd ≈ 1e-15 lần lúc khởi tạo. Phần dẫn xuất toán ("Mục 1" của kế hoạch mà notebook nhắc tới) không có trong thư mục |
| Ma trận 216 ô | 🟡 | 6/216 |
| Quét kernel size cho DLinear | 🟡 | k = 5/15/25/49, nghiệm dạng đóng, chỉ ETTh1 H = 96. Chưa quét bằng SGD |
| Outlier nhân tạo 0/1/5/10% | ❌ | |
| Khoảng cách SGD so với nghiệm tối ưu, 3 dataset × 4 horizon | 🟡 | 1/12 ô |
| λ chỉ chọn trên validation | ✅ | Notebook 01, mục 11 |
| Cấu hình kênh và scaler giống baseline SGD | ✅ | Cả hai gộp 7 kênh dùng chung trọng số và chuẩn hóa bằng mean/std của train (ddof = 0). Cùng 2785 cửa sổ test (có test kiểm) |

## 4. Đối chiếu theo tuần

### Tuần 1: baseline, pipeline dữ liệu, solver dạng đóng

| Việc | Trạng thái | Bằng chứng |
|---|---|---|
| Tái lập DLinear bằng SGD | ✅ | `00_dlinear_sgd_baseline`: MSE **0,3766**, MAE 0,3996 (công bố 0,375 / 0,399, lệch 0,4%). Số 0,3779 trong outline là cách tính cũ, lấy trung bình theo batch |
| Ma trận trung bình trượt P, khớp `moving_avg` | ✅ | Sai lệch ≤ 2,2e-16 ở k = 5/15/25/49, kèm đối chứng âm (đệm 0 lệch 0,48). `tests/test_operators.py` |
| Pipeline ETTh1 (8640/2880/2880, lùi L hàng, chuẩn hóa theo train, gộp 7 kênh) | ✅ | `src/data.py`; train 57 463 × 336, test 19 495 × 336 |
| Solver OLS, ridge, NLinear có ràng buộc, xử lý bias; kiểm trên dữ liệu giả | ✅ | `00_nlinear_closed_form`: OLS khôi phục W_true tới 2,2e-16; `tests/test_solvers.py` |
| OLS dạng đóng ETTh1 H = 96 | ✅ | MSE test 0,3702, tốt hơn SGD khoảng **1,7%** (so với 0,3766) |

### Tuần 2: ba dự đoán và so sánh hàm mục tiêu

| Việc | Trạng thái | Bằng chứng / còn thiếu |
|---|---|---|
| Chuyển code sang `src/data.py`, `src/solvers.py`, `src/metrics.py` | ✅ | Làm trong lần tái cấu trúc này, thêm `operators.py`, `models.py`, `sgd.py`. Notebook cũ vẫn giữ bản chép hàm riêng |
| Tổng quát hóa loader cho ETTh2, ETTm1 (mốc ×4) | 🟡 | `load_ett(path, L, name)` đã có và mốc chia đã kiểm. Chưa chạy vì `data/` chưa có ETTh2.csv, ETTm1.csv |
| Dự đoán 1.1: DLinear ≡ Linear khi λ = 0 | ✅ | rank Z = 336/672; chênh dự báo test ≤ 2,2e-13; MSE trùng 10 chữ số ở cả 4 kernel |
| Dự đoán 1.2: DLinear ≠ Linear khi λ > 0 (quét k) | ✅ | Weight decay của DLinear = ridge với phạt λN⁻¹ (khớp ≤ 3,6e-13). Tách khỏi ridge thường 1e-4 → 0,5, nhìn chung lớn hơn khi k lớn. Chỉnh lại λ cũng không xóa được phần tách |
| Dự đoán 1.3: NLinear ≠ Linear ở mọi λ | ✅ | ‖V − W_ols‖ = ‖r‖‖u‖/s và SSE tăng đúng ‖r‖²/s, khớp 15 chữ số; mức tách tăng đơn điệu theo λ |
| SGD và OLS đánh giá trên cùng tập test (drop_last) | ✅ | Bản `main` của LTSF-Linear để `drop_last=False` cho test. Checkpoint SGD chấm trên đủ 2785 cửa sổ = 19 495 / 7 hàng của OLS (`tests/test_etth1.py`) |
| IRLS cho MAE và Huber | ❌ | |
| Kiểm Huber δ lớn → MSE; MAE nhiều điểm khởi tạo | ❌ | Phụ thuộc IRLS |
| Chọn λ* trên validation cho từng ô | 🟡 | Chỉ ETTh1 H = 96, mục tiêu MSE: Linear λ* = 0; DLinear k = 25 λ* = 562; NLinear λ* = 1780 |
| Chạy grid, ghi `results/results.csv` | ❌ | Chưa có `runner.py` |
| **Output:** operators.py ✅ · solvers.py 🟡 (thiếu IRLS) · runner.py ❌ · results.csv ❌ · B2 ✅ · B3 ❌ · notebook 01 ✅ · notebook 02 ❌ | | |

Notebook 01 còn có thêm những phần outline không yêu cầu: hình H1, CSV quét λ (204 dòng), diễn giải N⁻¹ phạt nhẹ lag cuối (0,48 ở k = 25) và phạt gấp đôi ở ngưỡng cắt của bộ lọc.

### Tuần 3: outlier và RQ3

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Outlier nhân tạo 0/1/5/10%, 3 mục tiêu | ❌ | Cần IRLS trước |
| SGD ở 3 dataset × 4 horizon, bảng B1 | 🟡 | 1/12 ô (ETTh1 H = 96), một seed |
| Khoảng cách SGD và nghiệm tối ưu, khoảng cách trọng số | 🟡 | Một ô, chỉ MSE test: SGD 0,3766, OLS 0,3702, DLinear ridge ở λ* 0,3697. Chưa so trọng số |
| **Output:** B1 ❌ · B4 ❌ · H2 ❌ · notebook 03 ❌ | | |

### Tuần 4: báo cáo và đóng gói

| Việc | Trạng thái |
|---|---|
| Báo cáo đầy đủ (vấn đề → thiết lập → kết quả → diễn giải → hạn chế → phụ lục toán) | ❌ |
| Đối chiếu Toner & Darlow (ICML 2024) | ❌ |
| README, `run_all.sh` chạy sạch trên môi trường trắng | 🟡 README sơ bộ; chưa có `run_all.sh` |
| Hạn chế và hướng phát triển | ❌ |

## 5. Vấn đề phát hiện khi rà code

1. **Số liệu trong outline đã cũ.** 0,3779 → 0,3766 và "khoảng 2%" → khoảng 1,7%. Checkpoint hiện tại cho đúng 0,3766 khi tính trung bình toàn cục.
2. **Lịch learning rate lệch LTSF-Linear một epoch.** Notebook dùng `StepLR`, giảm lr một nửa ngay sau epoch 1. LTSF-Linear (`lradj='type1'`, gọi `adjust_learning_rate(epoch + 1)`) giữ lr = 0,005 cho cả epoch 1 và 2. Nên sửa trước khi làm bảng B1 nếu cần tái lập sát.
3. **Output của notebook 01 được chạy ở môi trường khác** (đường dẫn `/home/claude/work/LTSF-Linear`). `results/three_predictions_sweep.csv`, `H1_three_predictions.png`, `B2_three_predictions.md` không có trên máy này, và LTSF-Linear chưa được clone. Muốn có lại thì clone vào `third_party/` rồi chạy lại notebook (khoảng 1,5 phút).
4. **SGD mới chạy một seed** (42). Chưa có sai số theo seed, trong khi RQ3 nhắc tới ảnh hưởng của seed.
5. **Cell dở dang:** cell cuối của `00_nlinear_closed_form` lỗi `NameError: pred_V`. Cell so dự báo DLinear/Linear của `00_dlinear_sgd_baseline` (mục 6) chưa in kết quả; phần này đã làm lại đầy đủ ở notebook 01.
6. **Thiếu dữ liệu** ETTh2 và ETTm1.

## 6. Việc nên làm tiếp (theo thứ tự)

1. Tải ETTh2.csv, ETTm1.csv vào `data/`; clone LTSF-Linear vào `third_party/`, chạy lại notebook 01 để có lại `results/`.
2. Cài IRLS (MAE, Huber) trong `src/solvers.py`, kèm hai phép kiểm của outline trong `tests/`.
3. Viết `src/runner.py`: vòng lặp mô hình × mục tiêu × λ ∈ {0, λ*} × dataset × horizon, chọn λ* trên validation, ghi `results/results.csv`.
4. Sửa lịch lr theo `type1`, chạy SGD cho 12 ô với vài seed, lập bảng B1 và B4.
5. Chuyển các notebook sang import từ `src/` để bỏ phần chép hàm.
