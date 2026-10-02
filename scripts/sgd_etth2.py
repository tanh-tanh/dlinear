"""SGD (Adam) của repo trên ETTh2, theo siêu tham số của LTSF-Linear (docs/NHIEM_VU_3.md, Phần A).

    python scripts/sgd_etth2.py --step a2                          # A.2: tái lập ETTh1 H = 96, seed 42
    python scripts/sgd_etth2.py --step run                         # A.3: 3 mô hình × 4 H × 3 seed
    python scripts/sgd_etth2.py --step run --model DLinear --H 96 --seed 2021

Siêu tham số (scripts/EXP-LongForecasting/Linear/etth2.sh và run_longExp.py của LTSF-Linear; README dòng 16
và 107: Linear, NLinear, DLinear dùng chung script): lr 0,05, batch 32, 10 epoch, patience 3, lradj type1,
individual=False, k = 25, seed mặc định 2021. Khởi tạo mặc định của nn.Linear.

Khác repo gốc (ghi trong docs/BAO_CAO_ETTH2_SGD.md):
    - train: shuffle, drop_last=True như gốc;
    - val để dừng sớm: MSE trung bình toàn cục trên đủ cửa sổ (gốc: shuffle, drop_last, trung bình theo batch);
    - thứ tự lấy số ngẫu nhiên: manual_seed(seed) → tạo mô hình → các epoch (gốc tạo mô hình sau vài bước khác).
Chạy trên CPU (như A.2). Đánh giá: W_eff, bias float64 trên cùng ma trận cửa sổ với nghiệm dạng đóng
(load_cell), nên cùng scaler, cùng cửa sổ test.

Kết quả: results/sgd_etth2/{model}_H{H}_seed{seed}.json và .npy (W [H, L + 1], bias ở cột cuối).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_5060ti import load_cell, save_json                 # noqa: E402  (tắt TF32 khi import)
from src.data import load_ett, load_etth1                      # noqa: E402
from src.models import MODELS, DLinear, effective_weights      # noqa: E402
from src.sgd import ETTDataset, evaluate, train_sgd            # noqa: E402

L = 336
HORIZONS = [96, 192, 336, 720]
SEED0 = 2021                         # run_longExp.py dòng 8
SEEDS = [SEED0, SEED0 + 1, SEED0 + 2]
HP = {"lr": 0.05, "batch_size": 32, "epochs": 10, "patience": 3, "lradj": "type1"}   # etth2.sh
OUT = ROOT / "results" / "sgd_etth2"
DEVICE = "cpu"


def metrics(W, b, X, Y):
    R = Y - (X @ W.T + b)
    return float((R ** 2).mean()), float(np.abs(R).mean())


def step_a2():
    """A.2: ETTh1 H = 96, seed 42, lr 0,005, như 00_dlinear_sgd_baseline (seed trước loader và trước mô hình)."""
    H = 96
    tr, va, te = load_etth1(ROOT / "data" / "ETTh1.csv", L)
    torch.manual_seed(42)
    trl = DataLoader(ETTDataset(tr, L, H), batch_size=32, shuffle=True)
    val = DataLoader(ETTDataset(va, L, H), batch_size=32, shuffle=False)
    tel = DataLoader(ETTDataset(te, L, H), batch_size=32, shuffle=False)
    torch.manual_seed(42)
    model = DLinear(L, H)
    t0 = time.perf_counter()
    hist = train_sgd(model, trl, val, OUT / "a2_ckpt.pt", lr=5e-3, epochs=10, patience=3, lradj="type1")
    sec = time.perf_counter() - t0
    mse, mae = evaluate(model, tel, "mse"), evaluate(model, tel, "mae")
    ref = (0.3825, 0.4048)
    ok = abs(mse - ref[0]) <= 1e-3 and abs(mae - ref[1]) <= 1e-3
    print(f"A.2: test MSE {mse:.6f} MAE {mae:.6f} (tham chiếu {ref}), {'đạt' if ok else 'KHÔNG ĐẠT'}")
    save_json(OUT / "a2_etth1_H96_seed42.json", {
        "device": DEVICE, "dataset": "ETTh1", "H": H, "seed": 42, "lr": 5e-3, "batch_size": 32,
        "lradj": "type1", "test_mse": mse, "test_mae": mae, "reference": ref, "pass_1e-3": ok,
        "history": hist, "seconds": sec})
    (OUT / "a2_ckpt.pt").unlink()
    return ok


def run_one(name, H, seed, cell, splits):
    tr, va, _ = splits
    train_ds = ETTDataset(tr, L, H)
    # cùng cửa sổ với nghiệm dạng đóng: số hàng của make_dataset = số cửa sổ × 7 kênh
    assert len(train_ds) * 7 == len(cell["X"])
    assert len(ETTDataset(splits[2], L, H)) * 7 == len(cell["Y_te"])
    assert len(ETTDataset(va, L, H)) * 7 == len(cell["Y_va"])
    torch.manual_seed(seed)
    model = MODELS[name](L, H)
    trl = DataLoader(train_ds, batch_size=HP["batch_size"], shuffle=True, drop_last=True)
    val = DataLoader(ETTDataset(va, L, H), batch_size=256, shuffle=False)
    ckpt = OUT / f"{name}_H{H}_seed{seed}.pt"
    t0 = time.perf_counter()
    hist = train_sgd(model, trl, val, ckpt, lr=HP["lr"], epochs=HP["epochs"], patience=HP["patience"],
                     lradj=HP["lradj"])
    sec = time.perf_counter() - t0
    ckpt.unlink()
    W, b = effective_weights(model)
    best = min(range(len(hist)), key=lambda i: hist[i][2])           # epoch có val nhỏ nhất (đã nạp lại)
    res = {"device": DEVICE, "dataset": "ETTh2", "H": H, "model": name, "seed": seed, "L": L,
           "k": 25 if name == "DLinear" else None, **HP, "drop_last_train": True,
           "init": "nn.Linear mặc định", "best_epoch": hist[best][0], "n_epochs_run": len(hist),
           "history": [{"epoch": e, "train_loss": t, "val_mse_torch": v, "lr": lr} for e, t, v, lr in hist],
           "seconds": sec, "n_train_batches": len(trl)}
    for tag, split in (("train", ""), ("val", "_va"), ("test", "_te")):
        X = cell["X"] if tag == "train" else cell[f"Xt{split}"][:, :-1]
        Y = cell["Y"] if tag == "train" else cell[f"Y{split}"]
        res[f"{tag}_mse"], res[f"{tag}_mae"] = metrics(W, b, X, Y)
    # kiểm chéo: forward float32 của mô hình trên cùng tập test (lệch cỡ làm tròn float32)
    tel = DataLoader(ETTDataset(splits[2], L, H), batch_size=256, shuffle=False)
    res["test_mse_torch"], res["test_mae_torch"] = evaluate(model, tel, "mse"), evaluate(model, tel, "mae")
    assert abs(res["test_mse_torch"] - res["test_mse"]) < 1e-5 * max(1, res["test_mse"])
    res["row_sum"] = W.sum(1).tolist()
    res["bias"] = b.tolist()
    np.save(OUT / f"{name}_H{H}_seed{seed}.npy", np.concatenate([W, b[:, None]], axis=1))
    save_json(OUT / f"{name}_H{H}_seed{seed}.json", res)
    print(f"  {name} H={H} seed {seed}: epoch tốt nhất {res['best_epoch']}/{len(hist)}, val MSE {res['val_mse']:.6f}, "
          f"test MSE {res['test_mse']:.6f} MAE {res['test_mae']:.6f}, {sec:.0f} s", flush=True)


def step_run(models, horizons, seeds, force):
    splits = load_ett(ROOT / "data" / "ETTh2.csv", L, "ETTh2")
    for H in horizons:
        cell = load_cell("ETTh2", H)
        for name in models:
            for seed in seeds:
                if (OUT / f"{name}_H{H}_seed{seed}.json").exists() and not force:
                    print(f"  bỏ qua {name} H={H} seed {seed} (đã có)")
                    continue
                run_one(name, H, seed, cell, splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["a2", "run"], required=True)
    ap.add_argument("--model", nargs="+", default=["DLinear", "Linear", "NLinear"], choices=list(MODELS))
    ap.add_argument("--H", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step == "a2":
        sys.exit(0 if step_a2() else 1)
    step_run(args.model, args.H, args.seed, args.force)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
