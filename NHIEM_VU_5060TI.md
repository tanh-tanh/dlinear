# Nhiệm vụ cho Claude Code: đo IRLS trên RTX 5060 Ti 16 GB

*Lập ngày 30/09/2026. Đặt file này ở gốc repo `dlinear-objective-lab` rồi yêu cầu Claude Code: "Đọc NHIEM_VU_5060TI.md và làm theo".*

## 0. Bối cảnh (đọc trước khi làm)

Repo này so DLinear, NLinear và Linear tại nghiệm tối ưu thay vì bằng SGD. Với MSE dùng nghiệm dạng đóng. Với MAE và Huber dùng IRLS trong `src/solvers.py`.

Trên máy cũ (GTX 1650 4 GB) đã đo xong câu hỏi float32/float64. Chi tiết ở `docs/BAO_CAO_IRLS_DTYPE.md`, nhật ký sửa code ở `docs/NHAT_KY_THAY_DOI.md`. Kết luận chính:

- Dùng **chế độ pha**: `irls(..., gram_dtype=torch.float32)`, với Xt, Y, W0 ở float64. Chế độ này cho nghiệm như float64 thuần mà chỉ chậm hơn float32 thuần khoảng 15%.
- Lập 96 ma trận A_h (`weight_gram`, dùng `einsum`) chiếm gần hết thời gian mỗi vòng.
- Mức đo trên GTX 1650, ETTh1 H = 96, chế độ pha: **1,49 s/vòng**. MAE (δ = 1e-6) cần khoảng 106 vòng ở `tol = 1e-8` và khoảng 189 vòng ở `tol = 1e-9`. Huber (δ = 1) cần khoảng 7 đến 10 vòng.
- Grid cần khoảng 420 "ô tương đương ETTh1 H = 96", chưa tính quét λ. Riêng ETTm1 chiếm khoảng 68% chi phí.

**Mục tiêu của phiên này:** thay các ước lượng bằng số đo trên RTX 5060 Ti. **Không chạy toàn bộ grid** trong phiên này.

API hiện tại (không đổi chữ ký trừ khi mục 3.1 yêu cầu):

```python
irls(Xt, Y, delta, W0, lam=0.0, pen=None, constrained=False, max_iter=1000, tol=1e-9,
     callback=None, gram_dtype=None)   # trả (W [H, p], số vòng)
weight_gram(Xt, w, chunk=8)            # A [H, p, p]
MAE_DELTA = 1e-6
```

Quy ước: Xt [n, p] có cột 1 ở cuối cho bias (p = L + 1 = 337). Hàm phạt λ tuân theo quy ước chia cho δ, ghi ở `NHAT_KY_THAY_DOI.md` mục 1.5.

## 1. Quy tắc bắt buộc

1. **Không bật TF32.** Không dùng `torch.set_float32_matmul_precision('high'/'medium')`, `torch.backends.cuda.matmul.allow_tf32 = True` hay `torch.backends.cudnn.allow_tf32 = True`. Ở đầu mọi script đo, đặt tường minh:
   ```python
   torch.backends.cuda.matmul.allow_tf32 = False
   torch.backends.cudnn.allow_tf32 = False
   torch.set_float32_matmul_precision('highest')
   ```
   TF32 chỉ giữ 10 bit phần định trị, nên sẽ làm sai kết luận của báo cáo float32/float64.
2. **Không nới ngưỡng test để test đạt.** Nếu có test không đạt trên máy mới, dừng lại và báo cáo test nào, sai số đo được bao nhiêu.
3. **Không đổi toán của `irls`.** Chỉ được phép thêm tham số chuyển tiếp như ở mục 3.1.
4. **Mọi con số đưa vào báo cáo phải có file kết quả tái lập được** trong `results/`.
5. **Ghi thông tin phần cứng vào mọi file JSON kết quả:** `torch.cuda.get_device_name()`, phiên bản torch, CUDA, driver, hệ điều hành.
6. Đo thời gian trên GPU phải **khởi động trước** (ít nhất một vòng không tính) và gọi `torch.cuda.synchronize()` trước mỗi lần đọc đồng hồ. Đỉnh bộ nhớ đo bằng `torch.cuda.reset_peak_memory_stats()` rồi `torch.cuda.max_memory_allocated()`.
7. Trên Windows, đặt `PYTHONIOENCODING=utf-8` trước khi chạy.

## 2. Bước 1: môi trường

1. Kiểm tra repo có đủ: `data/ETTh1.csv`, `data/ETTh2.csv`, `data/ETTm1.csv`, `src/`, `tests/`, `scripts/irls_dtype_check.py`, `results/irls_dtype/` (để so với máy cũ).
2. Tạo môi trường ảo mới, cài PyTorch **bản dựng với CUDA 12.8 trở lên** (card Blackwell, sm_120; bản `2.6.0+cu124` của máy cũ không hỗ trợ card này). Lấy lệnh cài đúng từ trang cài đặt chính thức của PyTorch, rồi cài phần còn lại theo `requirements.txt` (không để `requirements.txt` cài đè torch bằng bản CPU hoặc bản CUDA cũ).
3. Kiểm tra:
   ```python
   import torch
   print(torch.__version__, torch.version.cuda)
   print(torch.cuda.is_available(), torch.cuda.get_device_name())
   print(torch.cuda.get_arch_list())          # phải có 'sm_120'
   print(torch.cuda.get_device_properties(0).total_memory / 2**30)   # ~16
   ```
4. Chạy toàn bộ test từ thư mục gốc:
   ```
   python -m unittest discover -s tests -t . -v
   ```
   Kỳ vọng: **Ran 32 tests, OK**. Nếu không đạt, dừng lại và báo cáo (quy tắc 2).
5. Cập nhật `requirements.txt` nếu cần, kèm ghi chú về bản torch cho card Blackwell.

## 3. Bước 2: đo tốc độ và bộ nhớ ở hai đầu của grid

Viết script mới `scripts/bench_5060ti.py`. Mọi lần chạy dùng Linear, λ = 0, khởi tạo W0 từ nghiệm OLS (`fit_with_bias(ols, X, Y)`, ghép bias vào cột cuối), chế độ pha (Xt, Y, W0 ở float64 trên GPU, `gram_dtype=torch.float32`).

### 3.1. Cho phép truyền `chunk` qua `irls`

`irls` hiện gọi `weight_gram` với `chunk` mặc định. Thêm tham số `chunk=8` vào `irls` và chuyển thẳng xuống `weight_gram`. Không đổi gì khác. Chạy lại test, phải vẫn 32/32.

### 3.2. ETTh1 H = 96: quét `chunk` và so với máy cũ

- Với `chunk` ∈ {8, 16, 32, 64, 96}, đo **riêng** thời gian một lần `weight_gram` (float32) và thời gian một vòng `irls` đầy đủ (chế độ pha). Mỗi cấu hình: 1 lần khởi động và 5 lần đo; báo trung vị. Ghi thêm đỉnh bộ nhớ và GFLOP/s của `weight_gram`, với số phép tính = n·p²·H.
- Với `chunk` tốt nhất, chạy **đầy đủ** hai lần ở chế độ pha:
  - Huber δ = 1, `tol = 1e-9`.
  - MAE δ = `MAE_DELTA`, `tol = 1e-8`.

  Với mỗi lần, ghi: số vòng, tổng thời gian, s/vòng, J train, MSE/MAE test (tính lại bằng float64).
- **Kiểm chéo độ đúng:** MSE/MAE test ở hai lần chạy trên phải khớp với số của máy cũ trong `results/irls_dtype/` ở cùng số vòng, sai lệch không quá 1e-6. Nếu lệch hơn, dừng lại và báo cáo.

### 3.3. ETTm1 H = 720: ô lớn nhất

- Nạp ETTm1 với L = 336, H = 720, các mốc chia nhân 4 như `src/data.py` đã làm. Ghi n và kích thước các tensor.
- Chạy **3 vòng** `irls` (MAE, chế độ pha, `chunk` tốt nhất ở 3.2; nếu thiếu bộ nhớ thì giảm `chunk`), với `max_iter=3`. Cảnh báo "chưa hội tụ" là bình thường ở đây.
- Ghi: s/vòng (trung vị vòng 2 và 3), **đỉnh bộ nhớ thật**, `chunk` đã dùng.
- Nếu hết bộ nhớ (OOM) ở mọi `chunk`: không tự viết cơ chế chia khối theo H. Dừng lại và báo cáo đỉnh bộ nhớ ở cấu hình nhỏ nhất đã thử.

### 3.4. Ước lượng lại chi phí grid

Từ s/vòng đo được ở 3.2 và 3.3, ước lượng thời gian mỗi vòng cho 12 ô (dataset × horizon), bằng cách nội suy theo n·H. Đo thêm ETTh1 H = 720 và ETTm1 H = 96 (mỗi ô 3 vòng) nếu cần để kiểm phép nội suy. Nhân với số vòng đo được ở 3.2 để ra thời gian cho:

- 5 tổ hợp (mô hình, λ) cần IRLS: Linear λ = 0, Linear λ*, DLinear λ*, NLinear λ = 0, NLinear λ*. DLinear λ = 0 trùng Linear nên không chạy.
- × 2 mục tiêu (MAE, Huber).
- Chưa tính quét λ; phần đó dùng số đo ở Bước 3.

Ghi rõ đây là ước lượng từ số đo, kèm giả định "số vòng như ETTh1 H = 96".

## 4. Bước 3: warm start khi quét λ

Mục đích: biết quét λ cho MAE tốn bao nhiêu, để chọn cách chọn λ*.

- Ô: ETTh1 H = 96, **DLinear** (Linear với `pen = N⁻¹`, k = 25, dùng `src/operators.py`), chế độ pha, `chunk` tốt nhất.
- Mục tiêu: MAE (δ = `MAE_DELTA`, `tol = 1e-8`) và Huber (δ = 1, `tol = 1e-9`).
- Lưới λ, **tăng dần**: {0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000}. Lưới này là tạm thời; nếu MSE/MAE validation còn giảm ở đầu mút thì mở rộng thêm 2 giá trị theo hướng đó.
- Chạy hai kiểu:
  - **Khởi tạo lạnh:** mỗi λ khởi tạo từ nghiệm ridge MSE dạng đóng ở cùng λ (quy đổi theo quy ước ở `NHAT_KY_THAY_DOI.md` mục 1.5) hoặc từ OLS nếu quy đổi phức tạp; ghi rõ đã dùng cách nào.
  - **Warm start:** mỗi λ khởi tạo từ nghiệm IRLS của λ liền trước.
- Với mỗi (kiểu, mục tiêu, λ), ghi: số vòng, thời gian, J train, **MSE và MAE trên validation**, MSE và MAE test.
- Báo cáo: tổng số vòng của cả lưới theo từng kiểu, λ tốt nhất trên validation theo từng metric (theo chính mục tiêu huấn luyện, và theo MSE), và λ* đó có nằm ở đầu mút lưới không.

**Lưu ý:** tập test chỉ được ghi lại để tham khảo, không được dùng để chọn λ.

## 5. Kết quả phải nộp

1. `scripts/bench_5060ti.py`, chạy lại được bằng tham số dòng lệnh cho từng bước (ví dụ `--step chunk`, `--step full`, `--step ettm1`, `--step warmstart`, `--summary`).
2. `results/bench_5060ti/*.json` (và `.npy` nếu lưu W), mỗi file có thông tin phần cứng theo quy tắc 5.
3. `docs/BAO_CAO_5060TI.md`, viết cùng văn phong với `docs/BAO_CAO_IRLS_DTYPE.md`:
   - **Kết luận** ở đầu: s/vòng, `chunk` nên dùng, có cần chia khối theo H không, ước lượng thời gian grid, warm start tiết kiệm bao nhiêu.
   - **Thiết lập**: phần cứng, phiên bản, cấu hình.
   - **Bảng số** cho từng bước ở mục 3 và 4, kèm so sánh với GTX 1650.
   - **Đề xuất** cho `runner.py`: `chunk`, `tol` cho MAE và Huber, lưới λ, kiểu khởi tạo.
   - **Tái lập**: các lệnh đã chạy.
4. Thêm một mục mới vào `docs/NHAT_KY_THAY_DOI.md` cho các thay đổi code (tham số `chunk`, script mới, `requirements.txt`).

## 6. Không làm trong phiên này

- Không viết `runner.py` và không chạy grid.
- Không viết cơ chế chia khối theo H (chỉ báo cáo nếu cần).
- Không đổi `MAE_DELTA`, quy ước λ hay các ngưỡng test.
- Không sửa `docs/CHECKLIST_CON_LAI.md` hay file kế hoạch; người dùng sẽ tự cập nhật từ báo cáo.

## 7. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Có test không đạt sau khi cài môi trường mới.
- Kết quả ở 3.2 lệch máy cũ quá 1e-6.
- `irls` báo `RuntimeError: J tăng` trong bất kỳ lần chạy nào.
- ETTm1 H = 720 bị OOM ở mọi `chunk`.
- Một lần chạy đơn lẻ dự kiến vượt 60 phút.
