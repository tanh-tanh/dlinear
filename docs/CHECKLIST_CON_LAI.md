# Checklist việc còn lại

Lập từ [BAO_CAO_TIEN_DO.md](BAO_CAO_TIEN_DO.md), đối chiếu với [outline.docx](outline.docx). Đánh dấu `[x]` khi xong.

## 0. Chuẩn bị và dọn dẹp

- [ ] Tải `ETTh2.csv`, `ETTm1.csv` vào `data/`
- [ ] Chạy thử `load_ett` trên ETTh2, ETTm1 và kiểm kích thước các tập
- [ ] Clone LTSF-Linear vào `third_party/LTSF-Linear`
- [ ] Chạy lại `notebooks/01_three_predictions.ipynb` để sinh lại `results/three_predictions_sweep.csv`, `results/H1_three_predictions.png`, `results/B2_three_predictions.md`
- [ ] Sửa lịch lr của SGD theo `lradj='type1'` của LTSF-Linear (giữ lr = 0,005 cho epoch 1 và 2)
- [ ] Cập nhật số trong outline: 0,3779 → 0,3766; "tốt hơn khoảng 2%" → khoảng 1,7%
- [ ] Xóa hoặc sửa cell lỗi `NameError: pred_V` ở cuối `00_nlinear_closed_form.ipynb`
- [ ] Mục 6 của `00_dlinear_sgd_baseline.ipynb`: chạy lại cell so dự báo DLinear/Linear, hoặc bỏ vì đã làm ở notebook 01
- [ ] Chuyển các notebook sang import từ `src/`, bỏ phần chép hàm

## 1. Tuần 2: hàm mục tiêu MSE, MAE, Huber

**IRLS**
- [ ] Cài IRLS cho MAE trong `src/solvers.py`
- [ ] Cài IRLS cho Huber
- [ ] Chạy IRLS bằng PyTorch trên GPU và đo thời gian chạy thật
- [ ] Test: Huber với δ lớn phải về đúng nghiệm MSE
- [ ] Test: MAE chạy từ 2–3 điểm khởi tạo phải về cùng một nghiệm

**Grid 216 ô** (3 mô hình × 3 mục tiêu × 2 mức λ × 3 dataset × 4 horizon; đã có 6/216)
- [ ] Viết `src/runner.py` chạy toàn bộ grid
- [ ] Chọn λ* trên validation cho từng ô (đã có: ETTh1, H = 96, mục tiêu MSE)
- [ ] Ghi `results/results.csv`, mỗi ô có cả MSE và MAE test
- [ ] Lập bảng B3
- [ ] Notebook `02_loss_comparison.ipynb`
- [ ] (Nên làm) Kiểm lại ba dự đoán của RQ1 ở các dataset và horizon khác (đã có: ETTh1, H = 96)
- [ ] (Nên làm) Mở rộng quét kernel size 5/15/25/49 sang các ô khác (đã có: ETTh1, H = 96)

## 2. Tuần 3: outlier và RQ3

**Outlier (RQ2)**
- [ ] Hàm chèn outlier nhân tạo vào train ở tỷ lệ 0/1/5/10%, seed cố định
- [ ] Đo MSE/MAE test của ba mục tiêu theo từng tỷ lệ outlier

**SGD so với nghiệm tối ưu (RQ3)**
- [ ] Chạy DLinear SGD ở 11 ô còn thiếu (3 dataset × 4 horizon; đã có ETTh1, H = 96)
- [ ] Chạy SGD với nhiều seed để có sai số theo seed (hiện chỉ có seed 42)
- [ ] Lập bảng B1 so với số công bố
- [ ] Đo khoảng cách MSE/MAE test giữa SGD và OLS, và giữa SGD và ridge ở λ*, trên 12 ô (đã có 1 ô)
- [ ] Đo khoảng cách trọng số giữa nghiệm SGD và nghiệm tối ưu
- [ ] Lập bảng B4
- [ ] Vẽ hình H2
- [ ] Notebook `03_outliers_and_sgd.ipynb`

## 3. Tuần 4: báo cáo và đóng gói

**Báo cáo**
- [ ] Viết báo cáo: vấn đề → thiết lập → kết quả → diễn giải → hạn chế → phụ lục toán
- [ ] Đưa phần dẫn xuất toán ("Mục 1" của kế hoạch) vào phụ lục
- [ ] Đối chiếu Toner & Darlow (ICML 2024) ở phần công trình liên quan
- [ ] Nêu hạn chế (3 dataset, chỉ mô hình tuyến tính, outlier nhân tạo) và hướng phát triển

**Toolkit**
- [ ] Viết `run_all.sh` tái lập toàn bộ kết quả bằng một lệnh
- [ ] Kiểm `run_all.sh` chạy sạch trên môi trường trắng
- [ ] Hoàn thiện `README.md` (hiện là bản sơ bộ)
- [ ] Công khai repo `dlinear-objective-lab`
