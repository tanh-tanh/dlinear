# Nhiệm vụ 5 cho Claude Code: trình tối ưu trên ETTh2, rồi SGD 20 seed cho bảng B1 và B4

*Lập ngày 03/10/2026, sau `docs/BAO_CAO_ETTH2_2.md`. Đặt file ở gốc repo rồi yêu cầu Claude Code: "Đọc NHIEM_VU_5.md và làm theo".*

## 0. Bối cảnh

Từ `BAO_CAO_ETTH2_2.md`:

- Trên ETTh2, "SGD" của LTSF-Linear (thực ra là **Adam mini-batch**, dừng sớm theo val) tốt hơn nghiệm ridge ở λ* với Linear và DLinear, trên **cả val lẫn test**.
- Đã loại trừ hai cơ chế: co về một trọng số khác 0, và các đoạn cảm biến bị kẹt.
- Ứng viên còn lại là chính trình tối ưu. Lý thuyết: gradient descent toàn batch xuất phát từ 0 rồi dừng sớm cho nghiệm rất gần ridge (Ali, Kolter & Tibshirani, 2019).
- Bài gốc báo cáo một lần chạy; độ lệch chuẩn theo seed trên ETTh2 là 0,02–0,07, lớn hơn nhiều khác biệt mà bài dùng để xếp hạng mô hình.

Phiên này gồm ba phần:

| Phần | Nội dung | Giới hạn |
|---|---|---|
| A | Khoảng cách trên ETTh2 đến từ đâu: dừng sớm, cách Adam co giãn bước, hay nhiễu mini-batch? | 45 phút |
| B | SGD 20 seed cho ETTh1 và ETTm1 (3 mô hình × 4 H), gộp với ETTh2 đã có | xem B.2 |
| C | Bảng B1, B4 và báo cáo | |

Sau Phần A, **dừng hẳn việc điều tra ETTh2**. Dù kết quả A thế nào cũng không mở thêm thí nghiệm về cơ chế.

## 1. Quy tắc bắt buộc

Giữ nguyên mọi quy tắc của các nhiệm vụ trước:

- TF32 tắt tường minh; không nới ngưỡng test; không đổi toán của `irls`.
- Mọi số liệu có file tái lập; ghi phần cứng và git commit vào mọi JSON; commit sạch trước mỗi lần chạy dài.
- **Tập test không bao giờ được dùng để chọn bất cứ thứ gì.** Mọi điểm dừng, learning rate, số bước đều chọn theo val.
- Không ghi đè kết quả của các nhiệm vụ trước.
- Kết luận chính luôn dùng **đủ seed** (giữ cả seed hỏng). Các hàng bỏ seed hỏng chỉ để tham khảo, vì tiêu chí "hỏng" dựa trên test.

## 2. Phần A: trình tối ưu trên ETTh2 (giới hạn 45 phút)

### A.1. Cấu hình

Ô: **Linear, ETTh2, H = 720**. Dữ liệu, scaler, hàm mất mát (MSE trung bình trên mọi phần tử, có bias) giống hệt `scripts/sgd_etth2.py`. Train đủ, không lọc.

Ba trình tối ưu mới, so với Adam mini-batch đã có (20 seed, `results/sgd_etth2_20seed/`):

| Tên | Batch | Trình tối ưu | Khởi tạo | Seed |
|---|---|---|---|---|
| `gd_zero` | toàn bộ train | GD, lr cố định | W = 0, b = 0 | không cần (tất định) |
| `gd_init` | toàn bộ train | GD, lr cố định | như LTSF-Linear (mặc định của `nn.Linear`) | 2021–2030 |
| `adam_full` | toàn bộ train | Adam, lr cố định | như LTSF-Linear | 2021–2030 |

- **lr của GD:** đặt bằng 1 / λ_max, với λ_max là trị riêng lớn nhất của Hessian của hàm mất mát (tính bằng power iteration trên ma trận Gram của đầu vào có cột 1, chia cho n·H theo đúng cách tính MSE trung bình). Ghi λ_max và lr.
- **lr của Adam toàn batch:** thử {0,05 (như script gốc), 0,005, 0,0005} với seed 2021, chọn lr có **MSE val tốt nhất** (theo bước tốt nhất), rồi chạy 10 seed với lr đó.
- **Số bước và dừng sớm:** đánh giá MSE val mỗi `k` bước (k sao cho có ít nhất 200 điểm đánh giá trên cả quá trình), lưu W tại bước có val tốt nhất. Chạy tối đa 20 000 bước với GD và 5 000 bước với Adam, hoặc dừng khi val không cải thiện trong 25% số bước tối đa. Ghi đủ quỹ đạo MSE val và MSE test theo bước (test chỉ ghi, không dùng để chọn).
- Chạy bằng float32 trên GPU (TF32 tắt) hoặc float64 trên CPU, miễn mỗi lần dưới 5 phút. Ghi lựa chọn.

### A.2. Kiểm tính đúng

`gd_zero` có một mốc lý thuyết: quỹ đạo của nó phải đi qua vùng của đường ridge. Báo cáo:

- MSE val và test tốt nhất của `gd_zero`, so với ridge λ* (`lambda_path.csv`);
- ở bước tốt nhất, ‖W_gd − W_ridge(λ)‖ / ‖W_ridge(λ)‖ nhỏ nhất theo λ trên lưới 141 giá trị, và λ đạt mức đó.

Nếu `gd_zero` cho val **tốt hơn** ridge λ* quá 0,005, kiểm lại code (lr, cách tính gradient, bias) trước khi đi tiếp. Lý thuyết dự đoán hai bên gần nhau, không phải GD thắng rõ.

### A.3. Bảng và cách đọc

| Cách | MSE val (TB ± sd) | MSE test (TB ± sd) | Δ val so với λ* [CI] | Δ test so với λ* [CI] | bước tốt nhất |
|---|---|---|---|---|---|
| ridge λ* | | | — | — | — |
| `gd_zero` | | | | | |
| `gd_init` | | | | | |
| `adam_full` | | | | | |
| Adam mini-batch (20 seed, đã có) | | | | | |

Áp **quy tắc đọc ba trường hợp** (mục 3.4) cho từng dòng. Rồi kết luận theo bảng sau:

| Quan sát | Kết luận |
|---|---|
| `gd_zero`, `gd_init` ≈ ridge; `adam_full` thắng như Adam mini-batch | khoảng cách đến từ **cách Adam co giãn bước theo từng hệ số** |
| `gd_*` ≈ ridge; `adam_full` ≈ ridge; chỉ Adam mini-batch thắng | khoảng cách đến từ **nhiễu mini-batch** |
| `gd_init` thắng, `gd_zero` không | khoảng cách đến từ **khởi tạo ngẫu nhiên** kết hợp dừng sớm |
| `gd_zero` cũng thắng | dừng sớm tự nó đã vượt ridge trên ô này; ghi rõ là trái với kỳ vọng lý thuyết |
| không khớp mẫu nào | báo cáo số liệu, không tự giải thích thêm |

## 3. Phần B: SGD 20 seed cho ETTh1 và ETTm1

### B.1. Cấu hình

- **Mô hình:** Linear, DLinear, NLinear. **Dataset:** ETTh1, ETTm1. **H:** 96, 192, 336, 720. **Seed:** 2021–2040.
- **Siêu tham số:** đọc từ script chính thức trong `third_party/LTSF-Linear/scripts/EXP-LongForecasting/Linear/` cho từng dataset, như đã làm với ETTh2 (lr, batch, số epoch, patience, `lradj`). **Không dùng giá trị của dataset khác.** Ghi bảng siêu tham số vào báo cáo kèm đường dẫn và số dòng.
- Dùng `scripts/sgd_etth2.py` (tổng quát hóa cho mọi dataset nếu cần, không đổi hành vi với ETTh2: kiểm 3 seed ETTh2 vẫn trùng từng bit). Pipeline chấm điểm giống hệt dạng đóng.
- Lưu W_eff và bias của mọi lần chạy.

### B.2. Thời gian

ETTm1 có nhiều cửa sổ hơn ETTh1 khoảng 4 lần, và script gốc có thể dùng batch nhỏ, nên một lần chạy có thể lâu hơn nhiều.

1. **Đo trước:** chạy 1 seed cho mỗi (dataset, H) với DLinear, ghi thời gian, rồi ước lượng tổng.
2. Được phép chạy song song nhiều tiến trình CPU, và dùng GPU nếu nhanh hơn. Nếu dùng GPU, đặt các cờ tất định của torch và ghi rõ; kiểm 2 lần chạy cùng seed cho cùng kết quả.
3. **Ngân sách tổng cho Phần B: 10 giờ.** Nếu ước lượng vượt:
   - giảm ETTm1 xuống 10 seed (2021–2030);
   - nếu vẫn vượt, giảm ETTm1 xuống 5 seed và ghi rõ trong báo cáo.

   ETTh1 luôn giữ 20 seed.
4. Chạy theo thứ tự ETTh1 trước, rồi ETTm1, và ghi kết quả ngay sau mỗi lần chạy để kết quả dở dang vẫn dùng được.

### B.3. Kiểm tái lập

Trước khi chạy hàng loạt: DLinear, ETTh1, H = 96, seed 42, `lradj="type1"`, siêu tham số ETTh1 phải cho lại **0,3825 / 0,4048** (lệch không quá 1e-3).

### B.4. Quy tắc đọc ba trường hợp

Dùng cho mọi so sánh "SGD so với dạng đóng" ở Phần A và Phần C. Với Δ = trung bình SGD − dạng đóng λ*, và CI 95% (phân phối t):

- **thắng** nếu CI nằm hẳn dưới 0; **thua** nếu nằm hẳn trên 0; **hòa** nếu chứa 0.

| Val | Test | Nhãn |
|---|---|---|
| thắng | thắng | **SGD tốt hơn thật** |
| thua | thua | **dạng đóng tốt hơn thật** |
| hòa | thắng hoặc thua | **khác biệt do val khác test** |
| thắng | thua hoặc hòa | **SGD có dấu hiệu khớp val quá mức** |
| thua | thắng hoặc hòa | **dạng đóng có dấu hiệu khớp val quá mức** |
| hòa | hòa | **không phân biệt được** |

## 4. Phần C: bảng và báo cáo

### C.1. Bảng B1: SGD so với số công bố (36 ô)

Gộp ETTh1, ETTm1 (Phần B) với ETTh2 (`results/sgd_etth2_20seed/`). Số công bố lấy đúng như mục 0.1 của `NHIEM_VU_3.md`, không tra cứu thêm.

| Dataset | H | Mô hình | SGD: TB ± sd (n) | trung vị | công bố | công bố ở phân vị thứ mấy của 20 seed | công bố nằm trong [min, max] của các seed? |

Kết luận: số ô mà số công bố nằm trong phân phối theo seed, và các ô nằm ngoài. Riêng **Linear, ETTh1, H = 720** (công bố 0,624, dạng đóng 0,4699), ghi rõ SGD của repo cho bao nhiêu.

### C.2. Bảng B4: SGD so với nghiệm tối ưu (36 ô)

| Dataset | H | Mô hình | dạng đóng λ*: val / test | SGD: val / test (TB) | Δ val [CI] | Δ test [CI] | nhãn (B.4) |

Thêm một bảng đếm: số ô theo từng nhãn, tách theo dataset và theo mô hình.

### C.3. DLinear so với Linear khi huấn luyện bằng SGD

Với mỗi (dataset, H): trung bình SGD của DLinear − Linear, CI 95% Welch, trên cả val và test. Đặt cạnh chênh lệch tương ứng của dạng đóng (luôn rất nhỏ theo RQ1) và của số công bố. Kết luận: ở bao nhiêu ô DLinear tốt hơn Linear một cách có ý nghĩa khi huấn luyện bằng SGD.

### C.4. Một hình

`results/sgd_b1_b4/H3_sgd_vs_optimum.png`: với mỗi (dataset, H), một ô nhỏ; trong mỗi ô, các điểm MSE test theo seed của ba mô hình (SGD), một vạch ngang cho dạng đóng λ*, một dấu cho số công bố. Ghi đủ nhãn trục, đơn vị, chú thích.

### C.5. Báo cáo

Viết `docs/BAO_CAO_SGD_B1_B4.md`, cùng văn phong các báo cáo trước:

1. **Kết luận** ở đầu:
   - kết quả Phần A, theo bảng ở A.3;
   - B1: số công bố có nằm trong nhiễu theo seed không;
   - B4: số ô theo từng nhãn; có ô nào ngoài ETTh2 mà SGD tốt hơn thật;
   - C.3: DLinear có tốt hơn Linear khi huấn luyện bằng SGD không.
2. Bảng siêu tham số (B.1), thời gian chạy, số seed thực tế cho từng dataset.
3. Bảng A.3, B1, B4, C.3, và hình.
4. **Một đoạn nháp khoảng 200 từ** cho mục RQ3 của báo cáo cuối, chỉ nêu những gì đã kiểm.
5. **Tái lập:** các lệnh đã chạy.

Ghi thay đổi code vào `docs/NHAT_KY_THAY_DOI.md`.

## 5. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Có test không đạt.
- B.3 không tái lập được 0,3825 / 0,4048.
- Ba seed ETTh2 không còn trùng từng bit sau khi tổng quát hóa script.
- `gd_zero` thắng ridge λ* trên val quá 0,005 mà không tìm ra lỗi.
- Siêu tham số trong script gốc mơ hồ.
- Ước lượng Phần B vượt 10 giờ kể cả sau khi giảm ETTm1 xuống 5 seed.

## 6. Không làm trong phiên này

- Không mở thêm thí nghiệm về cơ chế ETTh2 sau Phần A.
- Không làm thí nghiệm outlier.
- Không chạy lại grid hay IRLS.
- Không sửa kế hoạch, checklist, hay các báo cáo cũ.
