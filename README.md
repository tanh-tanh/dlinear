# dlinear-objective-lab

So sánh Linear / DLinear / NLinear (LTSF-Linear, Zeng et al., 2023) tại **nghiệm tối ưu tất định** thay vì huấn luyện SGD. Kế hoạch đầy đủ: [docs/outline.docx](docs/outline.docx). Tiến độ so với kế hoạch: [docs/BAO_CAO_TIEN_DO.md](docs/BAO_CAO_TIEN_DO.md). Việc còn lại: [docs/CHECKLIST_CON_LAI.md](docs/CHECKLIST_CON_LAI.md).

## Cấu trúc

```
├── data/                 ETTh1.csv (ETTh2, ETTm1 chưa có)
├── src/                  code dùng chung, chỉ numpy trừ models.py và sgd.py
│   ├── data.py           load_ett / load_etth1, make_windows, make_dataset (gộp 7 kênh)
│   ├── operators.py      build_P, build_N, make_Z, nlinear_transform, W_eff của DLinear/NLinear
│   ├── solvers.py        ols, ridge, ridge_pen, lstsq_solver, nlinear_constrained, fit_with_bias, ridge_path
│   ├── metrics.py        predict, sse, mse, mae (trung bình toàn cục)
│   ├── models.py         DLinear bằng PyTorch (moving_avg, series_decomp)
│   └── sgd.py            ETTDataset, evaluate, train_sgd (baseline SGD)
├── scripts/              irls_dtype_check.py: IRLS float32 / float64 / pha trên GPU (docs/BAO_CAO_IRLS_DTYPE.md)
├── tests/                kiểm src/ trên dữ liệu giả và kiểm hồi quy các số trong notebook
├── notebooks/
│   ├── 00_dlinear_sgd_baseline.ipynb   (trước là dlinearv2.ipynb)  SGD, ma trận P, dự đoán 1.1–1.3 bản nháp
│   ├── 00_nlinear_closed_form.ipynb    (trước là Nlinear.ipynb)    pipeline dữ liệu, solver dạng đóng
│   ├── 01_three_predictions.ipynb      (trước là 01_three_predictions_1.ipynb)  kiểm RQ1 qua module của repo
│   └── archive/00_dlinear_v1.ipynb     (trước là dlinear.ipynb) bản đầu, thứ tự cell sai, chỉ để tham khảo
├── checkpoints/          dlinear_sgd_ETTh1_L336_H96.pt (trước là best.pt)
├── results/              CSV, bảng, hình do notebook sinh ra
├── third_party/          (tự tạo) nơi clone LTSF-Linear cho notebook 01
└── docs/                 outline.docx, báo cáo tiến độ
```

Các notebook giữ nguyên code và output đã chạy, chỉ sửa đường dẫn cho vị trí mới (chạy với thư mục làm việc là `notebooks/`). Notebook mới nên import từ `src/` thay vì chép hàm:

```python
import sys; sys.path.insert(0, "..")
from src.data import load_ett, make_dataset
from src.solvers import fit_with_bias, ols, ridge
```

## Chạy

```bash
pip install -r requirements.txt
```

```bash
python -m unittest discover -s tests -t . -v
```

Notebook 01 cần mã nguồn LTSF-Linear để nạp đúng module `DLinear` / `NLinear` của repo:

```bash
git clone https://github.com/cure-lab/LTSF-Linear third_party/LTSF-Linear
```

## Quy ước

- λ phạt trên **tổng** bình phương lỗi: `‖Y − XWᵀ‖² + λ‖W‖²`, không phạt bias. Tương ứng `weight_decay ≈ 2λ/(n·H)` của PyTorch khi loss là MSE trung bình.
- MAE/Huber (`irls`) dùng cùng kiểu tổng: MAE là `Σ|r| + λ‖W‖²`, Huber là `Σ ρ_δ(r)/δ + λ‖W‖²` (ρ_δ = r²/2 trong ngưỡng δ). Không phạt bias. DLinear có weight decay dùng `pen = N⁻¹`, giống `ridge_pen`.
- Mọi mô hình dùng chung trọng số cho 7 kênh (`individual=False`), chuẩn hóa từng kênh bằng mean/std của train, giống baseline SGD.
- λ chỉ chọn trên validation.
