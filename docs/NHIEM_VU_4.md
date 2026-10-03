# Nhiệm vụ 4 cho Claude Code: chốt các câu hỏi về ETTh2

*Lập ngày 03/10/2026, sau `docs/BAO_CAO_ETTH2_SGD.md`. Đặt file ở gốc repo rồi yêu cầu Claude Code: "Đọc NHIEM_VU_4.md và làm theo".*

**Giới hạn thời gian: khoảng nửa ngày, cả phần chạy lẫn viết.** Đây là nhiệm vụ đóng phần ETTh2, không mở hướng mới. Nếu một phần vượt giới hạn của nó, dừng phần đó, ghi lại những gì đã có, và chuyển sang phần sau.

## 0. Bối cảnh

Từ `BAO_CAO_ETTH2_SGD.md`:

- Trên ETTh2, SGD (trung bình 3 seed) tốt hơn nghiệm dạng đóng ở λ* tại 7/8 ô Linear/DLinear, với khoảng cách từ 0,005 đến 0,043. Nhưng độ lệch chuẩn theo seed (0,007–0,08) lớn ngang khoảng cách, nên **chưa xác nhận được**.
- Giả thuyết "SGD ngầm co về một trọng số khác 0" **bị bác** (B.1, B.2).
- **Đoạn cảm biến bị kẹt là ứng viên mạnh nhất.** Bỏ 2,5–6,3% cửa sổ hằng khỏi train làm nghiệm dạng đóng của Linear/DLinear ở H = 720 giảm từ 0,740 xuống 0,646, tốt hơn cả SGD (0,702 / 0,736). Nhưng phép so này đặt dạng đóng trên train **đã lọc** cạnh SGD trên train **đủ**.
- **SGD của repo không tái lập được 0,605** của DLinear ở H = 720 (cho 0,736 ± 0,018). Pipeline SGD của repo khác repo gốc ở thứ tự lấy số ngẫu nhiên và cách tính val khi dừng sớm.

Nhiệm vụ này trả lời bốn câu:

| Phần | Câu hỏi | Giới hạn |
|---|---|---|
| A | Khoảng cách SGD với dạng đóng trên ETTh2 có thật không? | 60 phút |
| B | Đoạn hằng có giải thích khoảng cách không? | 60 phút |
| C | Mức lợi của MAE trên ETTh2 có đến từ đoạn hằng không? | 60 phút |
| D | Số 0,605 là do bài báo hay do pipeline của repo? | 60 phút |

Làm theo thứ tự A → B → C → D. Phần D độc lập với ba phần trước. Nếu GPU rảnh trong lúc A và B chạy trên CPU, có thể chạy C hoặc D song song.

## 1. Quy tắc bắt buộc

Giữ nguyên mọi quy tắc của các nhiệm vụ trước:

- TF32 tắt tường minh; không nới ngưỡng test; không đổi toán của `irls`.
- Mọi số liệu có file tái lập; ghi phần cứng và git commit vào mọi JSON; commit sạch trước mỗi lần chạy dài.
- **Tập test không bao giờ được dùng để chọn bất cứ thứ gì.** Dừng sớm của SGD chọn theo val như cũ.
- Không ghi đè `results/lambda_path.csv`, `results/results.csv`, hay kết quả của nhiệm vụ 3.
- Không sửa file trong `third_party/LTSF-Linear` tại chỗ (xem D.1).

**Định nghĩa "train đã lọc"** dùng chung cho B và C: đúng như C.3 của nhiệm vụ 3. Cửa sổ hằng là cửa sổ (một kênh, sau chuẩn hóa) có max − min của x **hoặc** của y nhỏ hơn 1e-12. Chỉ lọc train; val và test giữ nguyên. Dùng lại đúng hàm lọc của `scripts/constant_windows.py`, không viết lại.

## 2. Phần A: thêm seed cho SGD trên ETTh2

### A.1. Chạy

- **Mô hình:** Linear, DLinear. NLinear làm đối chứng với 10 seed nếu còn thời gian.
- **Ô:** ETTh2, H ∈ {96, 192, 336, 720}.
- **Seed:** 2021 đến 2040 (20 seed). Ba seed 2021–2023 đã có từ nhiệm vụ 3: chạy lại để kiểm tính tất định (phải trùng từng bit) rồi dùng kết quả đó.
- Siêu tham số, pipeline, cách chấm điểm giữ nguyên `scripts/sgd_etth2.py`.

### A.2. Báo cáo

Với mỗi (mô hình, H), ghi:

- trung bình, độ lệch chuẩn, trung vị của MSE test và MSE val theo 20 seed;
- Δ = trung bình SGD − λ* (MSE test và MSE val), kèm **khoảng tin cậy 95% của Δ** (phân phối t, 19 bậc tự do);
- tỷ lệ seed có MSE test tốt hơn λ*;
- số seed "hỏng" (dừng ở epoch 1 hoặc MSE test lớn hơn trung vị cộng 3 lần MAD). Báo cáo kết quả **cả khi giữ lẫn khi bỏ** các seed này, nhưng kết luận chính dùng bản giữ.

### A.3. Tiêu chí

Câu hỏi ở đây là "trung bình SGD có tốt hơn nghiệm dạng đóng không", không phải "mọi lần chạy SGD có tốt hơn không". Vì vậy tiêu chí chuyển từ 2 lần độ lệch chuẩn sang khoảng tin cậy của trung bình.

- **Xác nhận** cho một mô hình nếu khoảng tin cậy 95% của Δ (MSE test) nằm hẳn dưới 0 ở ít nhất 3/4 ô.
- Ghi thêm kết quả theo tiêu chí cũ (2 lần độ lệch chuẩn), và nói rõ trong báo cáo rằng tiêu chí đã đổi và vì sao.

## 3. Phần B: SGD trên train đã lọc

### B.1. Chạy

- Linear, DLinear; ETTh2, 4 H; seed 2021–2040; siêu tham số như A. **Chỉ khác ở chỗ train đã lọc.**
- Nghiệm dạng đóng trên train đã lọc đã có ở C.3 của nhiệm vụ 3. Dùng lại, kiểm số dòng và λ*.

### B.2. Bảng chính

Với mỗi (mô hình, H), bảng 2 × 2 MSE test (SGD là trung bình 20 seed ± khoảng tin cậy 95%):

| | train đủ | train đã lọc | Δ do lọc |
|---|---|---|---|
| dạng đóng, λ* | | | Δ_CF |
| SGD | | | Δ_SGD (± CI) |
| SGD − dạng đóng | | | |

Thêm cùng bảng cho MSE val.

### B.3. Phân tích phần dư trên cửa sổ hằng

Trên chính các cửa sổ hằng của **train** (những cửa sổ bị lọc), tính MSE phần dư của:

- nghiệm dạng đóng (λ*, train đủ);
- SGD (train đủ; trung bình theo seed).

Tách riêng ba nhóm: x hằng, chỉ y hằng, cả hai. Nếu dạng đóng khớp các cửa sổ này tốt hơn SGD rõ rệt, điều đó ủng hộ cơ chế "bình phương tối thiểu khớp đúng đoạn kẹt, còn SGD với dừng sớm thì không kịp".

### B.4. Cách đọc kết quả

- **Cơ chế được ủng hộ** nếu ở H ∈ {336, 720}, cho cả Linear và DLinear: Δ_CF âm và lớn hơn hẳn Δ_SGD về độ lớn (khoảng tin cậy của Δ_SGD − Δ_CF nằm hẳn trên 0), **và** trên train đã lọc khoảng cách SGD − dạng đóng không còn âm (khoảng tin cậy chứa 0 hoặc nằm trên 0).
- **Cơ chế không đủ** nếu SGD cũng cải thiện tương đương khi lọc, tức khoảng cách SGD − dạng đóng vẫn âm trên train đã lọc.
- H ∈ {96, 192} báo cáo đủ nhưng không dùng để kết luận, vì test ở hai horizon này cũng chứa đích hằng (C.3).

## 4. Phần C: MAE trên train đã lọc

### C.1. Chạy

- **Mô hình:** Linear (bắt buộc), NLinear (nếu kịp; bỏ H = 720 nếu ước lượng vượt 30 phút, vì đường này từng cần 26 713 bước fallback).
- **Mục tiêu:** MAE, **chỉ λ = 0**, cấu hình dừng như grid: `tol = 3e-11`, chế độ pha, `chunk = 16`, khởi tạo từ nghiệm MSE λ = 0 trên **cùng train đã lọc**.
- **Ô:** ETTh2, 4 H.
- Ghi số vòng, số bước fallback, thời gian.

### C.2. Bảng và cách đọc

Với mỗi (mô hình, H), MSE test và MAE test:

| | MSE λ = 0 | MAE λ = 0 | lợi của MAE (MSE test) |
|---|---|---|---|
| train đủ (từ `lambda_path.csv`) | | | |
| train đã lọc | | | |

- Nếu **lợi của MAE co lại mạnh** khi lọc (ví dụ còn dưới một nửa ở H ∈ {336, 720}), thì phần lớn lợi của MAE trên ETTh2 đến từ việc MAE ít bị đoạn kẹt kéo đi.
- Nếu lợi gần như giữ nguyên, thì MAE đang sửa một vấn đề khác (ví dụ dịch chuyển mức).

So thêm với Linear dạng đóng MSE trên train đã lọc (0,646 ở H = 720): MAE trên train đủ (0,589) có còn tốt hơn MSE trên train đã lọc không.

## 5. Phần D: chạy code gốc của LTSF-Linear

### D.1. Chuẩn bị

- **Không sửa `third_party/LTSF-Linear` tại chỗ.** Chép sang `scratch/ltsf_official/` (thêm vào `.gitignore`). Mọi thay đổi trên bản chép ghi thành một file diff trong `results/official_ltsf/patch.diff`.
- Chỉ được sửa những gì cần để chạy: đường dẫn dữ liệu; tương thích với torch 2.14 nếu bị lỗi; và **cho phép truyền seed qua dòng lệnh** (repo gốc đặt cứng `fix_seed = 2021` ở `run_longExp.py` dòng 8–11). Không sửa gì khác trong mô hình, vòng huấn luyện hay cách tính metric.
- Dùng đúng siêu tham số của `scripts/EXP-LongForecasting/Linear/etth2.sh` (đã ghi ở mục 2 của `BAO_CAO_ETTH2_SGD.md`).

### D.2. Chạy

- DLinear và Linear, ETTh2, **H = 720** (bắt buộc), thêm H = 336 nếu kịp.
- Seed 2021 (đúng cấu hình gốc), rồi 2022–2025.
- Ghi MSE/MAE test **do code gốc in ra** (cách tính của repo gốc), cùng epoch dừng.

### D.3. Chấm chéo

Nạp checkpoint do code gốc lưu vào pipeline chấm điểm của repo (`load_cell`, W_eff và bias float64), rồi tính MSE/MAE test. So với số code gốc in ra. Chênh lệch giữa hai cách chấm cho biết phần khác biệt đến từ **cách tính metric**, tách khỏi phần đến từ **quá trình huấn luyện**.

### D.4. Cách đọc

- Code gốc, seed 2021, DLinear H = 720 cho **0,605 ± 2%**: pipeline SGD của repo có khác biệt chưa tìm ra. **Dừng lại theo mục 7** và báo cáo các khác biệt đã biết, trước khi lập bảng B1.
- Code gốc cho mức như repo (khoảng 0,72–0,75): số công bố không tái lập được với code gốc hiện tại. Ghi rõ trong báo cáo, kèm commit của repo gốc (`0c11366`).
- Mức ở giữa: báo cáo đủ số liệu, kèm phân tách khác biệt theo D.3.

## 6. Phần E: báo cáo

Viết `docs/BAO_CAO_ETTH2_2.md`, cùng văn phong các báo cáo trước:

1. **Kết luận** ở đầu: trả lời bốn câu hỏi ở mục 0, mỗi câu một đoạn ngắn, kèm con số chính.
2. Bảng của A, B, C, D.
3. **Một đoạn nháp khoảng 150–250 từ** cho mục "ETTh2 và các đoạn cảm biến bị kẹt" của báo cáo cuối. Đoạn nháp viết theo hướng: kết quả chính dùng dữ liệu gốc; đoạn hằng là phân tích độ nhạy; chỉ nêu những điều đã được kiểm trong nhiệm vụ 3 và 4.
4. **Tái lập:** các lệnh đã chạy.

Ghi thay đổi code vào `docs/NHAT_KY_THAY_DOI.md`.

## 7. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Có test không đạt.
- Ba seed 2021–2023 ở A không trùng từng bit với nhiệm vụ 3.
- Code gốc ở D cho 0,605 ± 2% (DLinear, H = 720, seed 2021).
- Code gốc không chạy được mà cần sửa ngoài phạm vi cho phép ở D.1.
- Một phần vượt giới hạn thời gian ở mục 0 mà chưa có kết quả dùng được.

## 8. Không làm trong phiên này

- Không chạy SGD trên ETTh1 hay ETTm1. Phần đó thuộc bảng B1 và B4, làm ở nhiệm vụ sau.
- Không làm thí nghiệm outlier.
- Không chạy lại grid trên dữ liệu đã lọc, ngoài các phép chạy nêu trên.
- Không sửa kế hoạch, checklist, hay các báo cáo cũ.
