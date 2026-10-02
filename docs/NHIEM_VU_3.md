# Nhiệm vụ 3 cho Claude Code: SGD trên ETTh2, kiểm giả thuyết, ba việc rẻ

*Lập ngày 01/10/2026, sau báo cáo tổng hợp grid (commit `0563109`); cập nhật cùng ngày với số công bố từ Bảng 2 của bài gốc. Đặt file ở gốc repo rồi yêu cầu Claude Code: "Đọc NHIEM_VU_3.md và làm theo".*

## 0. Bối cảnh

Grid 216 ô đã chạy xong. Kết quả ở `results/lambda_path.csv` và `results/results.csv`.

**Một nghi vấn cần kiểm.** Người dùng đã đối chiếu Bảng 2 của bài LTSF-Linear (mục 0.1). So với số công bố (SGD, một lần chạy), nghiệm dạng đóng ở λ* (mục tiêu MSE) **bằng hoặc tốt hơn ở 28/36 ô**. Tám ô ngoại lệ đều là **Linear và DLinear trên ETTh2, ở cả bốn horizon**: dạng đóng cho 0,3009 / 0,3965 / 0,4854 / 0,7403, trong khi DLinear SGD công bố 0,289 / 0,383 / 0,448 / 0,605 và Linear SGD công bố 0,288 / 0,377 / 0,452 / 0,698. NLinear trên ETTh2 thì không bị.

Nếu đúng, RQ3 có câu trả lời khác hẳn dự kiến: SGD có một thiên lệch ngầm có ích trên dữ liệu có dịch chuyển phân phối, mà ridge (co về 0) không có. Hiện tượng không riêng DLinear, nên Phần A phải chạy cả Linear.

**Hai lần đảo chiều đáng chú ý ở H = 720.** Bài gốc cho thấy DLinear vượt xa Linear ở H = 720 trên cả ETTh1 lẫn ETTh2. Tại nghiệm tối ưu, Linear và DLinear trùng nhau ở cả hai dataset, nhưng theo hai hướng ngược nhau:

- **ETTh1:** công bố DLinear 0,472, Linear 0,624; dạng đóng 0,4695 và 0,4699. Chênh lệch trong bài hoàn toàn do SGD của Linear không đến được nghiệm tối ưu.
- **ETTh2:** công bố DLinear 0,605, Linear 0,698; dạng đóng 0,7403 và 0,7404, kém cả hai bản SGD.

Phiên này gồm bốn phần:

- **Phần A:** chạy SGD của chính repo trên ETTh2, để xác nhận hoặc bác bỏ khoảng cách.
- **Phần B:** kiểm giả thuyết về nguồn gốc khoảng cách (rẻ, chạy kể cả khi A bác bỏ).
- **Phần C:** ba việc rẻ còn nợ từ báo cáo tổng hợp.
- **Phần D:** báo cáo.

### 0.1. Số công bố (Bảng 2, Zeng et al., AAAI 2023; L = 336; MSE/MAE)

Người dùng đã chép từ bài gốc. **Không tự tra cứu thêm.** Dùng đúng các số dưới đây.

| Dataset | H | Linear | NLinear | DLinear |
|---|---|---|---|---|
| ETTh1 | 96 | 0,375 / 0,397 | 0,374 / 0,394 | 0,375 / 0,399 |
| ETTh1 | 192 | 0,418 / 0,429 | 0,408 / 0,415 | 0,405 / 0,416 |
| ETTh1 | 336 | 0,479 / 0,476 | 0,429 / 0,427 | 0,439 / 0,443 |
| ETTh1 | 720 | 0,624 / 0,592 | 0,440 / 0,453 | 0,472 / 0,490 |
| ETTh2 | 96 | 0,288 / 0,352 | 0,277 / 0,338 | 0,289 / 0,353 |
| ETTh2 | 192 | 0,377 / 0,413 | 0,344 / 0,381 | 0,383 / 0,418 |
| ETTh2 | 336 | 0,452 / 0,461 | 0,357 / 0,400 | 0,448 / 0,465 |
| ETTh2 | 720 | 0,698 / 0,595 | 0,394 / 0,436 | 0,605 / 0,551 |
| ETTm1 | 96 | 0,308 / 0,352 | 0,306 / 0,348 | 0,299 / 0,343 |
| ETTm1 | 192 | 0,340 / 0,369 | 0,349 / 0,375 | 0,335 / 0,365 |
| ETTm1 | 336 | 0,376 / 0,393 | 0,375 / 0,388 | 0,369 / 0,386 |
| ETTm1 | 720 | 0,440 / 0,435 | 0,433 / 0,422 | 0,425 / 0,421 |

Số công bố là **một lần chạy**, có thể dùng cách lấy trung bình theo batch và xử lý cửa sổ test hơi khác (repo này đã gặp: 0,3779 so với 0,3766). Vì vậy **chênh lệch dưới khoảng 1% không diễn giải**.

## 1. Quy tắc bắt buộc

Giữ nguyên mọi quy tắc của `NHIEM_VU_5060TI.md` và `NHIEM_VU_2.md`:

- TF32 tắt tường minh; không nới ngưỡng test; không đổi toán của `irls`.
- Mọi số liệu có file tái lập; ghi phần cứng và git commit vào mọi JSON.
- **Tập test không bao giờ được dùng để chọn bất cứ thứ gì** (λ, epoch, seed, cấu hình). Với SGD, epoch tốt nhất chọn theo val loss như LTSF-Linear.

Thêm:

- Không tra cứu số công bố ngoài mục 0.1.
- Không ghi đè `results/lambda_path.csv` hay `results/results.csv`, trừ ở C.2 (chỉ **thêm** dòng mới, theo cơ chế bỏ qua dòng đã có của runner).
- Commit sạch trước mỗi lần chạy dài, để JSON không mang nhãn `-dirty`.
- Máy phải cắm điện, vì khi chạy pin máy vẫn ngủ sau 10 phút.

## 2. Phần A: SGD trên ETTh2

### A.1. Lấy đúng siêu tham số của LTSF-Linear

Đọc script chính thức trong `third_party/LTSF-Linear/scripts/` cho DLinear trên ETTh2 (thư mục `EXP-LongForecasting/Linear/` hoặc tương đương) và ghi lại chính xác cho từng H:

- learning rate, batch size, số epoch, patience của dừng sớm;
- `lradj`, `individual`, kernel size;
- seed mặc định của repo.

**Không dùng giá trị của ETTh1 cho ETTh2**, vì các script đặt learning rate khác nhau theo dataset. Nếu `src/sgd.py` không hỗ trợ một tham số nào trong số này, báo cáo và thêm tham số đó với mặc định giữ hành vi cũ, kèm test.

Đọc thêm `models/DLinear.py`, `models/Linear.py`, `models/NLinear.py` của repo gốc và ghi lại **cách khởi tạo trọng số**: mặc định của `nn.Linear`, hay có dòng đặt trọng số bằng 1/L. Ghi rõ dòng nào đang bật, dòng nào bị comment, kèm số dòng trong file. Thông tin này dùng cho B.1.

### A.2. Kiểm tái lập trước

Chạy `train_sgd` trên ETTh1 H = 96, seed 42, `lradj="type1"`, với siêu tham số ETTh1 như các lần trước. Kết quả phải ra MSE/MAE test **0,3825 / 0,4048** (checklist mục 0), lệch không quá 1e-3. Nếu lệch hơn, dừng lại theo mục 6.

### A.3. Chạy

- **Mô hình:** DLinear và Linear, **cả hai bắt buộc** (hiện tượng xảy ra với cả hai). NLinear làm đối chứng nếu thời gian cho phép: theo mục 0.1, NLinear SGD và dạng đóng ngang nhau trên ETTh2, nên SGD của repo cũng phải cho kết quả ngang. Nếu không, đó là dấu hiệu pipeline SGD có vấn đề.
- **Ô:** ETTh2, H ∈ {96, 192, 336, 720}, L = 336.
- **Seed:** seed mặc định của repo gốc, cộng thêm 2 seed (mặc định + 1, mặc định + 2).
- **Đánh giá:** cùng pipeline với nghiệm dạng đóng (`src/data.py`, cùng scaler, cùng các cửa sổ test). Kiểm số cửa sổ test bằng số hàng test của nghiệm dạng đóng chia 7, như đã làm với ETTh1.
- **Ghi cho mỗi lần chạy:** MSE/MAE val và test ở epoch tốt nhất (chọn theo val), epoch tốt nhất, thời gian, và **W_eff, bias** của mô hình (với DLinear, W_eff = W_t P + W_s (I − P)). Lưu W vào `results/sgd_etth2/`.

### A.4. So với nghiệm dạng đóng

Với mỗi ô và mô hình, lập bảng:

| | MSE test | MAE test | MSE val |
|---|---|---|---|
| SGD (trung bình ± độ lệch chuẩn theo seed) | | | |
| Dạng đóng, λ = 0 | | | |
| Dạng đóng, λ* (`val_objective`) | | | |
| Công bố (mục 0.1) | | | không có |

Thêm cột **MSE val**: nếu SGD thắng trên test **và** trên val, thì khoảng cách có trên cả hai tập. Nếu SGD chỉ thắng trên test, khoảng cách chủ yếu đến từ dịch chuyển phân phối giữa val và test.

**Kết luận A:** khoảng cách SGD − dạng đóng (theo MSE test) ở từng ô, kèm độ lệch chuẩn theo seed. Khoảng cách được coi là **xác nhận** nếu SGD tốt hơn λ* ở ít nhất 3/4 ô, với mức lớn hơn 2 lần độ lệch chuẩn theo seed. Xét riêng cho Linear và cho DLinear.

Ghi thêm SGD của repo so với số công bố: nếu SGD của repo lệch công bố quá khoảng 5% ở một ô, ghi rõ, vì khi đó so sánh với số công bố ở ô đó không còn đáng tin.

## 3. Phần B: kiểm giả thuyết (chỉ MSE, dạng đóng, rất rẻ)

Chạy Phần B **kể cả khi A bác bỏ khoảng cách**. Kết quả vẫn có giá trị cho báo cáo.

### B.1. Tổng hàng của W

Với mỗi ô ETTh2, so phân bố tổng hàng (L hệ số đầu, tức w_hᵀ1, với h = 1…H) và bias của:

- W của SGD (từng seed);
- W dạng đóng ở λ = 0 và ở λ*;
- W của NLinear dạng đóng (tổng hàng bằng 1 theo ràng buộc, làm mốc).

Báo cáo trung vị, khoảng phân vị 10–90% của tổng hàng, và ‖bias‖. Giả thuyết dự đoán tổng hàng của SGD gần 1 hơn so với dạng đóng.

### B.2. Ridge co về một trọng số khác 0

Ridge thường co W về 0. Thay bằng co về một ma trận W₀:

  min ‖Y − X Wᵀ − b‖² + λ · tr((W − W₀) Pen (W − W₀)ᵀ)

Nghiệm dạng đóng: đặt V = W − W₀, Y' = Y − X W₀ᵀ, giải ridge với Pen như cũ trên (X, Y'), rồi W = V + W₀. Bias không phạt, xử lý như `fit_with_bias`.

Ba lựa chọn W₀:

| Tên | W₀ | Ý nghĩa |
|---|---|---|
| `zero` | 0 | ridge hiện tại; phải trùng `lambda_path.csv` (kiểm tới 1e-10) |
| `mean` | mọi phần tử 1/L | dự báo bằng trung bình cửa sổ |
| `last` | 1 ở cột lag cuối, 0 ở chỗ khác | dự báo bằng giá trị cuối (persistence) |

- **Mô hình:** Linear (Pen = I) và DLinear (Pen = N⁻¹, k = 25). Với DLinear, phạt weight decay co về (W_t0, W_s0) quy về phạt N⁻¹ trên W − W₀,eff theo cùng lập luận ở 1.2 của kế hoạch. Nếu A.1 cho thấy repo gốc khởi tạo W_t = W_s = 1/L, thì W₀,eff = (1/L)·1ᵀ, trùng `mean`.
- **Ô:** cả **12 ô** (3 dataset × 4 H), để biết hiệu ứng có riêng ở ETTh2 không.
- **λ:** lưới 20 giá trị mỗi bậc như C.2, chọn λ* trên MSE val.
- **Báo cáo:** MSE test và MSE val ở λ* cho từng W₀, cùng λ* tương ứng.

Giả thuyết được ủng hộ nếu, trên ETTh2, `mean` hoặc `last` cho MSE val và MSE test tốt hơn `zero` rõ rệt và tiến gần mức của SGD (Phần A), trong khi trên ETTh1 và ETTm1 khác biệt nhỏ.

### B.3. (Nếu A xác nhận và B.2 không giải thích được) Dừng sớm của gradient descent

Chạy gradient descent đầy đủ (full batch, lr cố định) trên bài MSE của Linear, ETTh2 H = 720, xuất phát từ khởi tạo của repo gốc (theo A.1). Ghi MSE val và MSE test theo số bước, rồi chọn số bước theo val. So với ridge λ*. Mục đích: tách tác dụng của dừng sớm khỏi tác dụng của nhiễu mini-batch. Chỉ làm nếu thời gian cho phép, giới hạn 30 phút.

## 4. Phần C: ba việc rẻ

### C.1. Kiểm các đường chạy trước khi có fallback

Chạy lại bằng code hiện tại (có fallback float64 theo từng h) **ba đường λ**, ghi vào file riêng `results/recheck/lambda_path_recheck.csv`, không đụng file chính:

1. ETTh2 H = 336, Linear, MAE
2. ETTh2 H = 336, NLinear, MAE
3. ETTh1 H = 720, DLinear, MAE

So từng λ với dòng tương ứng trong `lambda_path.csv`: MSE/MAE val và test, số vòng, và số bước fallback (nếu runner ghi).

- **Đạt:** mọi metric lệch không quá 1e-5, và λ* (cả hai cách chọn) giống nhau.
- **Không đạt:** dừng lại theo mục 6, kèm danh sách λ lệch và mức lệch. Đừng tự chạy lại phần grid cũ.

### C.2. Làm dày lưới λ của MSE

Đổi lưới MSE trong `runner.py` thành {0} ∪ 10^{−2, −1,95, …, 5} (20 giá trị mỗi bậc, 141 giá trị). Lưới cũ (bước 0,2) là tập con của lưới mới, và lưới mới chứa 10^2,75 ≈ 562,34 và 10^3,25 ≈ 1778,3 của notebook 01.

- Chạy runner chỉ cho mục tiêu MSE trên cả 12 ô. Runner chỉ thêm các λ mới, giữ nguyên các dòng cũ.
- Chạy lại `select_lambda.py`.
- **Báo cáo:** với mỗi (ô, mô hình), λ* và MSE test trước và sau khi làm dày, và độ chênh. Kiểm lại dòng C.5 của DLinear trên ETTh1 H = 96: MSE test phải khớp 0,3697 tới 1e-4 nếu λ* = 562,34.

### C.3. Độ nhạy với đoạn hằng của ETTh2

- Định nghĩa **cửa sổ hằng**: cửa sổ (theo từng kênh, sau chuẩn hóa) có max − min của đầu vào x **hoặc** của đích y nhỏ hơn 1e-12.
- Đếm số cửa sổ hằng trong train, val, test của ETTh2, theo từng H và kênh.
- Chạy MSE dạng đóng (Linear, DLinear, NLinear; lưới mới ở C.2; λ* chọn trên val) với **train bỏ các cửa sổ hằng**. Val và test giữ nguyên.
- **Báo cáo:** MSE test và λ* trước và sau khi bỏ, và khoảng cách NLinear − Linear trước và sau.
- Nếu thời gian cho phép, làm thêm MAE cho NLinear ở H = 720, vì đây là chỗ cần tới 26 713 bước fallback.

## 5. Phần D: báo cáo

Viết `docs/BAO_CAO_ETTH2_SGD.md`, cùng văn phong các báo cáo trước:

1. **Kết luận** ở đầu: khoảng cách SGD với dạng đóng trên ETTh2 có được xác nhận không; giả thuyết nào được ủng hộ; kết quả của ba việc rẻ.
2. **Siêu tham số và khởi tạo** của LTSF-Linear (A.1), kèm đường dẫn file và số dòng.
3. **Bảng A.4** cho 4 ô, cột "công bố" lấy từ mục 0.1.
4. **Bảng so 36 ô:** MSE test của nghiệm dạng đóng ở λ* (mục tiêu MSE, `val_objective`, lưới đã làm dày ở C.2) với số công bố, cho Linear, DLinear, NLinear trên 3 dataset × 4 H. Thêm cột chênh tương đối, đánh dấu các ô chênh quá 1% theo từng hướng. Kết luận nêu số ô dạng đóng tốt hơn, ngang, kém; và hai trường hợp H = 720 ở mục 0.
5. **Bảng B.1, B.2** (và B.3 nếu có).
6. **Bảng C.1, C.2, C.3.**
7. **Tái lập:** các lệnh đã chạy.

Ghi thay đổi code vào `docs/NHAT_KY_THAY_DOI.md`.

## 6. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Có test không đạt.
- A.2 không tái lập được 0,3825 / 0,4048 trong khoảng 1e-3.
- Siêu tham số trong script gốc mơ hồ, ví dụ có nhiều script cho cùng cấu hình với giá trị khác nhau.
- `zero` ở B.2 không trùng `lambda_path.csv` tới 1e-10.
- C.1 không đạt.
- Một lần chạy đơn lẻ dự kiến vượt 60 phút.

## 7. Không làm trong phiên này

- Không chạy SGD trên ETTh1 (trừ A.2) hay ETTm1. Phần đó thuộc Tuần 3, sau khi biết kết quả ETTh2.
- Không làm thí nghiệm outlier.
- Không sửa kế hoạch, checklist, hay các báo cáo cũ.
- Không tra cứu số công bố ngoài mục 0.1.
