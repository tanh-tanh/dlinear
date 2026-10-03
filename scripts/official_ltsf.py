"""Chạy code gốc của LTSF-Linear trên ETTh2 và chấm chéo bằng pipeline của repo (docs/NHIEM_VU_4.md, Phần D).

    python scripts/official_ltsf.py --prepare                 # chép third_party/LTSF-Linear → scratch/ltsf_official, áp patch
    python scripts/official_ltsf.py --model DLinear Linear --H 720 --seed 2021 2022 2023 2024 2025

Bản chép không sửa gì ngoài results/official_ltsf/patch.diff:
    - run_longExp.py: thêm --seed (mặc định 2021 = hành vi gốc; gốc đặt cứng fix_seed = 2021 ở dòng 8–11);
    - utils/tools.py: np.Inf → np.inf (numpy 2.x đã bỏ np.Inf).
Dữ liệu chép vào scratch/ltsf_official/dataset/ (đường dẫn mặc định của script gốc).

Lệnh chạy đúng scripts/EXP-LongForecasting/Linear/etth2.sh (lr 0,05, batch 32, các tham số khác mặc định), thêm:
    --model <Linear|DLinear>, --seed, --num_workers 0 (trên Windows, worker spawn chạy lại toàn bộ run_longExp.py
    vì file không có `if __name__ == "__main__"`; thứ tự xáo do tiến trình chính lấy nên không đổi).

Mỗi lần chạy: log ở results/official_ltsf/logs/, MSE/MAE test do code gốc in ra, epoch tốt nhất theo val (dòng
"Validation loss decreased"), rồi nạp checkpoint vào lớp mô hình của repo (cùng tên tham số) và chấm lại bằng
W_eff, bias float64 trên load_cell (D.3). Kết quả: results/official_ltsf/{model}_H{H}_seed{seed}.json.
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_5060ti import load_cell, save_json              # noqa: E402
from src.models import MODELS, effective_weights          # noqa: E402

SRC = ROOT / "third_party" / "LTSF-Linear"
OFF = ROOT / "scratch" / "ltsf_official"
OUT = ROOT / "results" / "official_ltsf"
PATCH = OUT / "patch.diff"
CKPT = ROOT / "scratch" / "official_ckpt"
L = 336


def prepare():
    if OFF.exists():
        shutil.rmtree(OFF)
    shutil.copytree(SRC, OFF, ignore=shutil.ignore_patterns(".git"))
    # chạy từ gốc repo với --directory: chạy trong scratch/ (thư mục con của repo này) thì git apply hiểu đường dẫn
    # theo gốc repo và lặng lẽ bỏ qua patch
    subprocess.run(["git", "apply", f"--directory={OFF.relative_to(ROOT).as_posix()}", str(PATCH)], cwd=ROOT, check=True)
    assert "args.seed" in (OFF / "run_longExp.py").read_text(encoding="utf-8"), "patch chưa được áp"
    assert "np.Inf" not in (OFF / "utils" / "tools.py").read_text(encoding="utf-8"), "patch chưa được áp"
    (OFF / "dataset").mkdir()
    shutil.copyfile(ROOT / "data" / "ETTh2.csv", OFF / "dataset" / "ETTh2.csv")
    print(f"→ {OFF.relative_to(ROOT)} (đã áp {PATCH.relative_to(ROOT)})")


def parse_log(text):
    epochs = [(int(e), float(v), float(t)) for e, v, t in
              re.findall(r"Epoch: (\d+), Steps: \d+ \| Train Loss: [\d.]+ Vali Loss: ([\d.]+) Test Loss: ([\d.]+)", text)]
    best, cur = None, None
    for line in text.splitlines():
        m = re.match(r"Epoch: (\d+), Steps", line)
        if m:
            cur = int(m.group(1))
        if line.startswith("Validation loss decreased"):
            best = cur
    mse, mae = map(float, re.search(r"mse:([\d.eE+-]+), mae:([\d.eE+-]+)", text).groups())
    return {"epochs": [{"epoch": e, "vali_loss": v, "test_loss": t} for e, v, t in epochs],
            "best_epoch": best, "n_epochs_run": len(epochs), "official_test_mse": mse, "official_test_mae": mae}


def run_one(model, H, seed):
    f = OUT / f"{model}_H{H}_seed{seed}.json"
    if f.exists():
        print(f"  bỏ qua {model} H={H} seed {seed} (đã có)")
        return
    args = ["--is_training", "1", "--root_path", "./dataset/", "--data_path", "ETTh2.csv",
            "--model_id", f"ETTh2_{L}_{H}", "--model", model, "--data", "ETTh2", "--features", "M",
            "--seq_len", str(L), "--pred_len", str(H), "--enc_in", "7", "--des", "Exp", "--itr", "1",
            "--batch_size", "32", "--learning_rate", "0.05", "--num_workers", "0", "--seed", str(seed)]
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", "run_longExp.py", *args], cwd=OFF, capture_output=True, text=True)
    sec = time.perf_counter() - t0
    log = p.stdout + p.stderr
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / f"{model}_H{H}_seed{seed}.log").write_text(log, encoding="utf-8")
    if p.returncode:
        sys.exit(f"code gốc lỗi ({model} H={H} seed {seed}), xem log")
    res = parse_log(log)
    setting = (f"ETTh2_{L}_{H}_{model}_ETTh2_ftM_sl{L}_ll48_pl{H}_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0")
    ckpt = OFF / "checkpoints" / setting / "checkpoint.pth"
    CKPT.mkdir(parents=True, exist_ok=True)
    keep = CKPT / f"{model}_H{H}_seed{seed}.pth"
    shutil.copyfile(ckpt, keep)                      # thư mục setting không chứa seed: lần sau ghi đè
    # D.3: chấm chéo bằng pipeline của repo
    net = MODELS[model](L, H)
    net.load_state_dict(torch.load(keep, map_location="cpu"))
    W, b = effective_weights(net)
    cell = load_cell("ETTh2", H)
    for tag, split in (("val", "va"), ("test", "te")):
        R = cell[f"Y_{split}"] - (cell[f"Xt_{split}"][:, :-1] @ W.T + b)
        res[f"repo_{tag}_mse"], res[f"repo_{tag}_mae"] = float((R ** 2).mean()), float(np.abs(R).mean())
    res.update(model=model, H=H, seed=seed, seconds=sec, command=["python", "-u", "run_longExp.py", *args],
               ltsf_commit="0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6", patch=PATCH.read_text(encoding="utf-8"),
               device="cuda" if torch.cuda.is_available() else "cpu")
    save_json(f, res)
    print(f"  {model} H={H} seed {seed}: code gốc test MSE {res['official_test_mse']:.4f} MAE "
          f"{res['official_test_mae']:.4f}, chấm lại {res['repo_test_mse']:.4f} / {res['repo_test_mae']:.4f}, "
          f"epoch tốt nhất {res['best_epoch']}/{res['n_epochs_run']}, {sec:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--model", nargs="+", default=["DLinear", "Linear"], choices=["DLinear", "Linear"])
    ap.add_argument("--H", nargs="+", type=int, default=[720])
    ap.add_argument("--seed", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    args = ap.parse_args()
    if args.prepare:
        prepare()
        return
    for H in args.H:
        for model in args.model:
            for seed in args.seed:
                run_one(model, H, seed)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
