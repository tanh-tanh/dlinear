# Nhiệm vụ 2 cho Claude Code: tiêu chí dừng, ba phép kiểm còn nợ, `runner.py`

*Lập ngày 30/09/2026, sau `docs/BAO_CAO_5060TI.md`. Đặt file này ở gốc repo rồi yêu cầu Claude Code: "Đọc NHIEM_VU_2.md và làm theo".*

## 0. Bối cảnh

Đọc trước: `docs/BAO_CAO_5060TI.md` (mục 4–6), `docs/NHAT_KY_THAY_DOI.md` (mục 1.5, quy ước λ), `docs/BAO_CAO_IRLS_DTYPE.md` (mục 5, sai số theo `tol`).

Vấn đề cần sửa: `irls` dừng khi J giảm tương đối ít hơn `tol` **sau một vòng**. Khi warm start từ nghiệm của λ liền trước, J giảm rất ít ngay từ vòng đầu, nên thuật toán dừng sau 1 vòng dù chưa hội tụ. Báo cáo 5060 Ti ghi nhận điều này ở λ = 1, 3, 10 với MAE. Hệ quả: λ* của MAE chưa đáng tin, vì chênh lệch val giữa các λ cùng cỡ sai số của bộ giải.

Phiên này gồm ba phần, **làm theo thứ tự**:

- **Phần A:** sửa tiêu chí dừng, rồi chứng minh bằng số rằng λ* của MAE chọn được ổn định.
- **Phần B:** ba phép kiểm còn nợ.
- **Phần C:** viết `runner.py` và chạy thử một ô. **Chỉ làm C khi A đạt tiêu chí chấp nhận.**

Không chạy toàn bộ grid trong phiên này (xem mục 5).

## 1. Quy tắc bắt buộc

Giữ nguyên mọi quy tắc ở mục 1 của `NHIEM_VU_5060TI.md`: TF32 tắt tường minh, không nới ngưỡng test, không đổi toán của `irls`, mọi số liệu có file tái lập, ghi thông tin phần cứng vào JSON, đo giờ có khởi động và `synchronize`, `PYTHONIOENCODING=utf-8`.

Thêm:

- Dùng venv `.venv-gpu`. Sau mỗi thay đổi trong `src/`, chạy lại toàn bộ test. Kỳ vọng hiện tại: 35 test đạt, cộng các test mới thêm.
- Tập test **không bao giờ** được dùng để chọn λ, δ hay bất kỳ lựa chọn nào. Chỉ ghi lại để tham khảo.
- Ghi git commit hash vào mọi file JSON kết quả và mọi dòng của `results.csv`.

## 2. Phần A: tiêu chí dừng

### A.1. Thêm `patience` vào `irls`

Thêm tham số `patience=1`: chỉ dừng khi mức giảm tương đối của J nhỏ hơn `tol` trong **`patience` vòng liên tiếp**. Một vòng có mức giảm ≥ `tol` thì đặt lại bộ đếm về 0.

- Mặc định `patience=1` giữ nguyên hành vi cũ, nên các test hiện có (ví dụ δ lớn dừng đúng sau 2 vòng) không phải sửa.
- Nhánh "J tăng trong ngưỡng `rise_tol`" (coi là chạm độ chính xác của kiểu số) vẫn dừng ngay, không chờ `patience`.
- Số vòng trả về và cảnh báo khi chạm `max_iter` giữ nguyên.
- Thêm vào `tests/test_irls.py`:
  - `patience = 1` cho đúng kết quả cũ (so với một lần gọi không truyền `patience`).
  - Với MAE trên dữ liệu giả, khởi tạo warm từ nghiệm của một λ gần, `patience = 5` chạy nhiều vòng hơn `patience = 1`, và J không lớn hơn.

Ghi thay đổi vào `docs/NHAT_KY_THAY_DOI.md`.

### A.2. Nghiệm tham chiếu cho MAE trên ETTh1 H = 96

Ô: DLinear (`pen = N⁻¹`, k = 25), ETTh1 H = 96, chế độ pha, `chunk = 16`, MAE δ = `MAE_DELTA`.

Với λ ∈ {0, 30, 100, 300, 1000}, chạy **nghiệm tham chiếu**: `tol = 0`, `max_iter = 2000`, khởi tạo OLS. Mỗi lần khoảng 5 phút. Để tránh cảnh báo gây nhiễu, bắt cảnh báo và ghi lại thay vì in ra. Ghi J train, MSE/MAE/Huber val, MSE/MAE test, và quỹ đạo val theo vòng (mỗi 50 vòng).

Mục đích: biết giá trị val "thật" ở từng λ, để trả lời hai câu:

1. Chênh lệch MAE val giữa các λ có lớn hơn sai số của bộ giải không? So chênh lệch giữa λ liền kề với mức MAE val còn thay đổi trong 500 vòng cuối của nghiệm tham chiếu.
2. λ* theo MAE val và theo MSE val trên nghiệm tham chiếu là bao nhiêu?

### A.3. Quét λ lại với tiêu chí dừng mới

Lưới λ như cũ: {0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000}. Chạy khởi tạo lạnh và warm start (λ tăng dần) cho **bốn cấu hình**:

| `tol` | `patience` |
|---|---|
| 1e-9 | 1 |
| 1e-9 | 5 |
| 1e-10 | 1 |
| 1e-10 | 5 |

Với Huber (δ = 1), chỉ chạy `tol = 1e-9` với `patience` ∈ {1, 5}; báo cáo trước đã cho thấy Huber không có vấn đề.

Ghi mọi đại lượng như bước warm start trong `bench_5060ti.py`, thêm Huber val.

### A.4. Tiêu chí chấp nhận

Chọn cấu hình (`tol`, `patience`) **rẻ nhất** thỏa cả ba điều sau cho MAE:

1. **λ* ổn định:** λ* theo MAE val và λ* theo MSE val giống nhau giữa khởi tạo lạnh và warm start, và giống λ* của nghiệm tham chiếu (A.2) trên các λ có tham chiếu.
2. **Khoảng cách phân biệt được:** ở λ* của mỗi metric, chênh lệch val giữa λ* và λ tốt thứ hai lớn hơn chênh lệch warm − lạnh của metric đó ở hai λ này.
3. **Gần nghiệm tham chiếu:** ở mọi λ có tham chiếu, MAE val và MSE val của nghiệm warm lệch tham chiếu không quá 1e-5.

Nếu **không cấu hình nào đạt**: dừng lại và báo cáo (mục 6). Kèm kết luận rút ra từ A.2: với MAE trên ô này, λ có ảnh hưởng thật lên val hay không. Nếu không, đây là một kết quả nghiên cứu hợp lệ, và người dùng sẽ quyết định cách xử lý.

### A.5. Báo cáo Phần A

Viết `docs/BAO_CAO_TIEU_CHI_DUNG.md`, cùng văn phong các báo cáo trước:

- Kết luận ở đầu: cấu hình được chọn, hoặc lý do không cấu hình nào đạt.
- Bảng nghiệm tham chiếu.
- Bảng so bốn cấu hình theo ba tiêu chí.
- Chi phí quét λ cho cả grid với cấu hình được chọn, cập nhật ước lượng ở mục 6 của báo cáo 5060 Ti.

## 3. Phần B: ba phép kiểm còn nợ

Làm cả ba kể cả khi Phần A không đạt. Mỗi phép kiểm có một **test trên dữ liệu giả** trong `tests/` (nhanh, chạy được trên CPU) và một **lần chạy trên dữ liệu thật** trong script `scripts/checks_mae.py`, với kết quả ở `results/checks_mae/`.

### B.1. Tính duy nhất của nghiệm MAE trên dữ liệu thật

- Ô: Linear, ETTh1 H = 96, λ = 0, MAE, chế độ pha, `tol = 1e-10`, `patience = 5`.
- Ba điểm khởi tạo: OLS; W = 0; OLS cộng nhiễu Gauss với độ lệch chuẩn bằng 10% chuẩn của từng hàng (seed cố định).
- Báo cáo:
  - J train (chênh tương đối giữa các cặp);
  - MSE/MAE val và test;
  - ‖W_i − W_j‖ / ‖W_j‖.

Mục tiêu nghiên cứu mà outline yêu cầu là "về cùng một nghiệm". Vì hàm mục tiêu MAE rất phẳng gần nghiệm, báo cáo cần phân biệt rõ hai mức: cùng giá trị hàm mục tiêu (J lệch dưới 1e-8 tương đối) và cùng trọng số (W lệch bao nhiêu). Không có test trên dữ liệu giả cho mục này; `test_multi_init_agree` đã có.

### B.2. DLinear + weight decay ≡ ridge với N⁻¹, với MAE và Huber

Tương đương này mới được chứng minh trên giấy (`NHAT_KY_THAY_DOI.md` mục 1.5).

**Cách kiểm:**

1. Giải trực tiếp DLinear trong 2L chiều. Đầu vào Z = [X Pᵀ, X (I − P)ᵀ] cộng cột 1; phạt `pen = I` trên 2L hệ số đầu (weight decay trên W_t và W_s, không phạt bias). Ghép lại W_eff = W_t P + W_s (I − P).

   *Lưu ý:* kiểm quy ước nhân bằng hàm `P` trong `src/operators.py` và cách notebook 01 dựng Z, để trung bình trượt tác động đúng chiều. Nếu `irls` chỉ nhận `pen` có kích thước [L, L] theo L suy ra từ Xt, xác nhận rằng với đầu vào 2L + 1 cột thì `pen` phải là [2L, 2L].
2. Giải trong L chiều với `pen = N⁻¹`.
3. So hai nghiệm.

**Test trên dữ liệu giả** (`tests/test_irls.py`, lớp mới `TestDLinearEquivalence`): L = 24, k = 5, H = 3, n = 400, λ = 5. Dùng `tol = 1e-12`, `patience = 5`.

- Huber δ = 1: W_eff của cách 1 khớp W của cách 2 với sai số tương đối dưới 1e-8. Bias cũng phải khớp.
- MAE: vì mặt mục tiêu phẳng, so **giá trị hàm mục tiêu có phạt** của hai nghiệm, tính cùng một công thức trong không gian L chiều. Chênh tương đối dưới 1e-6, và bổ sung một phép so hai chiều như `test_mae_matches_qp`.

**Trên dữ liệu thật:** ETTh1 H = 96, k = 25, λ = 100, Huber và MAE. Báo cáo chênh lệch W_eff (với Huber), chênh lệch hàm mục tiêu, và MSE/MAE test của hai cách.

### B.3. NLinear với MAE so với quy hoạch tuyến tính có ràng buộc (câu 11.6)

**Test trên dữ liệu giả** (`tests/test_irls.py`): dùng dữ liệu của `make_data` hiện có. Với từng h, giải bằng `linprog`: min Σ t, với ràng buộc −t ≤ y − Xt w ≤ t, và ràng buộc đẳng thức aᵀw = 1 với a = [1, …, 1, 0] (`A_eq`, `b_eq`). So với `irls(..., MAE_DELTA, constrained=True, tol=1e-12)`:

- MAE của `irls` không kém MAE của quy hoạch tuyến tính quá 1e-6 tương đối;
- tổng hàng (L hệ số đầu) của nghiệm `irls` bằng 1 tới 1e-10.

Không cần chạy trên dữ liệu thật.

### B.4. Báo cáo Phần B

Thêm vào `docs/BAO_CAO_TIEU_CHI_DUNG.md` một mục "Ba phép kiểm", mỗi phép kiểm ghi kết quả đo và kết luận.

## 4. Phần C: `runner.py`

**Chỉ làm sau khi Phần A đạt tiêu chí chấp nhận.** Nếu A không đạt, dừng lại theo mục 6.

### C.1. Phạm vi

- **Mô hình:**
  - Linear: `pen = I`.
  - DLinear: `pen = N⁻¹`, k = 25.
  - NLinear: `constrained=True`, `pen = I`.
- **Mục tiêu:** MSE, MAE, Huber (δ = 1).
- **Dataset:** ETTh1, ETTh2, ETTm1. **Horizon:** 96, 192, 336, 720. L = 336.

### C.2. Cách giải

- **MSE:** dạng đóng. Linear và DLinear dùng `ridge_path` hoặc `ridge_pen` trên một lưới dày (log, khoảng 36 giá trị từ 1e-2 tới 1e5) vì rất rẻ. NLinear ở λ > 0 với MSE: nếu `src/` chưa có nghiệm dạng đóng có ràng buộc và có phạt, dùng `irls` với δ lớn (đã có test khớp hệ KKT có phạt, dừng sau 2 vòng). Ghi rõ đã dùng cách nào.
- **MAE, Huber:** `irls`, chế độ pha, `chunk = 16`, `tol` và `patience` theo kết quả Phần A. Lưới λ {0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000}, warm start theo λ tăng dần, λ = 0 khởi tạo từ nghiệm MSE λ = 0 của cùng mô hình. **Tự mở rộng** tối đa 2 giá trị (nhân hoặc chia 3) nếu λ* theo bất kỳ metric val nào nằm ở đầu mút.
- **DLinear ở λ = 0:** không giải; ghi lại dòng của Linear với cột `derived_from = Linear`, vì DLinear ≡ Linear khi λ = 0 với mọi mục tiêu.

### C.3. Bảng kết quả

**Không chọn λ* trong lúc chạy.** Lưu toàn bộ đường λ, để người dùng chọn theo metric sau khi hỏi TA.

`results/lambda_path.csv`: mỗi dòng là một bộ (dataset, H, mô hình, mục tiêu, λ). Các cột:

- **Định danh:** dataset, H, model, objective, lam, lam_convention (ghi quy ước λ của mục tiêu đó), delta, k (với DLinear).
- **Kết quả:** train_obj, val_mse, val_mae, val_huber, test_mse, test_mae.
- **Bộ giải:** n_iter, converged (dừng theo `tol` hay chạm `max_iter`), seconds, init (`ols`, `warm` hoặc `closed_form`), derived_from.
- **Môi trường:** tol, patience, chunk, gram_dtype, gpu, torch_version, git_commit, timestamp.

`scripts/select_lambda.py`: đọc `lambda_path.csv` và sinh `results/results.csv`, gồm **216 ô** (3 mô hình × 3 mục tiêu × 2 mức λ × 3 dataset × 4 horizon). Mức λ* có hai phiên bản, theo cột `selection` ∈ {`val_objective`, `val_mse`}:

- `val_objective`: λ* tối thiểu hóa metric val ứng với chính mục tiêu huấn luyện (MSE → val_mse, MAE → val_mae, Huber → val_huber).
- `val_mse`: λ* tối thiểu hóa val_mse với mọi mục tiêu.

Cột `lam_at_edge` báo λ* có nằm ở đầu mút lưới sau khi mở rộng hay không.

### C.4. Chạy tiếp được sau gián đoạn

- Khóa của một dòng là (dataset, H, model, objective, lam). Khi khởi động, đọc `lambda_path.csv`, bỏ qua mọi khóa đã có.
- Warm start cần nghiệm của λ liền trước. Lưu W của mỗi dòng vào `results/weights/{dataset}_H{H}_{model}_{objective}_lam{lam}.npy` (float64) để chạy tiếp đúng từ giữa một đường λ.
- Ghi từng dòng ngay khi xong, `flush` sau mỗi dòng. Không giữ kết quả trong bộ nhớ đến cuối.
- Thứ tự chạy: ETTh1, rồi ETTh2, rồi ETTm1; trong mỗi dataset, H tăng dần; trong mỗi ô, MSE trước. Nhờ vậy, kết quả dừng giữa chừng vẫn dùng được.
- Giải phóng bộ nhớ GPU giữa các ô (`del` và `torch.cuda.empty_cache()`).
- Tham số dòng lệnh: `--data`, `--H`, `--model`, `--objective` (lọc, mặc định là tất cả), `--dry-run` (in danh sách việc và ước lượng thời gian, không chạy).

### C.5. Chạy thử và tiêu chí chấp nhận

Chạy `runner.py --data ETTh1 --H 96` (mọi mô hình và mục tiêu), rồi `select_lambda.py`. Kiểm:

| Đại lượng | Giá trị đã biết | Nguồn | Sai lệch cho phép |
|---|---|---|---|
| Linear, MSE, λ = 0: MSE test | 0,3702 | `tests/test_etth1.py` | 1e-4 |
| DLinear, MSE, λ* (`val_objective`) | λ* ≈ 562, MSE test 0,3697 | notebook 01 | λ* cùng bậc độ lớn; MSE test 1e-4 |
| NLinear, MSE, λ* (`val_objective`) | λ* ≈ 1780 | notebook 01 | cùng bậc độ lớn |
| Linear, MAE, λ = 0: MSE/MAE test | 0,3653 / 0,3833 | báo cáo 5060 Ti | 1e-4 |
| Linear, Huber, λ = 0: MSE/MAE test | 0,3678 / 0,3880 | báo cáo 5060 Ti | 1e-4 |
| DLinear, MAE/Huber, λ* | khớp kết quả Phần A | Phần A | trùng |

Lưới MSE mới khác lưới của notebook 01, nên λ* chỉ cần cùng bậc độ lớn. Nếu lệch hơn, báo cáo và giải thích.

Sau đó chạy `runner.py --dry-run` cho toàn bộ grid, in ước lượng thời gian dựa trên số đo của báo cáo 5060 Ti.

### C.6. Báo cáo Phần C

Thêm vào `README.md` một mục "Chạy grid", gồm lệnh chạy toàn bộ, lệnh chạy tiếp sau gián đoạn, và lệnh sinh `results.csv`. Ghi thay đổi vào `docs/NHAT_KY_THAY_DOI.md`.

## 5. Không làm trong phiên này

- Không chạy toàn bộ grid. Người dùng sẽ tự chạy qua đêm bằng lệnh trong README.
- Không làm thí nghiệm outlier và không chạy SGD.
- Không đổi `MAE_DELTA`, δ của Huber, k của DLinear, quy ước λ hay ngưỡng của các test cũ.
- Không sửa file kế hoạch hay `docs/CHECKLIST_CON_LAI.md`.

## 6. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Có test không đạt sau khi thêm `patience` hoặc các test mới.
- Không cấu hình nào ở A.3 đạt tiêu chí A.4.
- B.2 cho thấy tương đương DLinear **không** đúng với MAE hoặc Huber, vượt ngưỡng.
- B.1 cho thấy J khác nhau quá 1e-8 tương đối giữa các điểm khởi tạo.
- `irls` báo `RuntimeError: J tăng` trong bất kỳ lần chạy nào.
- Chạy thử ở C.5 lệch các giá trị đã biết quá ngưỡng.
- Một lần chạy đơn lẻ dự kiến vượt 60 phút.
