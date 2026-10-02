"""Lập các bảng của docs/BAO_CAO_ETTH2_SGD.md từ kết quả đã chạy (docs/NHIEM_VU_3.md, Phần D).

    python scripts/report_etth2.py

Đọc results/sgd_etth2/, results/results.csv (lưới đã làm dày), results/weights/, results/ridge_w0/,
results/recheck/, results/constant_windows/, và results.csv của commit 0563109 (trước khi làm dày).
Ghi results/etth2_report/tables.md và summary.json. Không chọn gì trên test.
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_5060ti import save_json              # noqa: E402

OUT = ROOT / "results" / "etth2_report"
SGD = ROOT / "results" / "sgd_etth2"
OLD_COMMIT = "0563109"
L = 336
HORIZONS = [96, 192, 336, 720]
DATASETS = ["ETTh1", "ETTh2", "ETTm1"]
MODELS = ["Linear", "NLinear", "DLinear"]
SEEDS = [2021, 2022, 2023]
# Bảng 2 của Zeng et al. (AAAI 2023), L = 336, (MSE, MAE); chép nguyên từ docs/NHIEM_VU_3.md mục 0.1
PUB = {
    ("ETTh1", 96): {"Linear": (0.375, 0.397), "NLinear": (0.374, 0.394), "DLinear": (0.375, 0.399)},
    ("ETTh1", 192): {"Linear": (0.418, 0.429), "NLinear": (0.408, 0.415), "DLinear": (0.405, 0.416)},
    ("ETTh1", 336): {"Linear": (0.479, 0.476), "NLinear": (0.429, 0.427), "DLinear": (0.439, 0.443)},
    ("ETTh1", 720): {"Linear": (0.624, 0.592), "NLinear": (0.440, 0.453), "DLinear": (0.472, 0.490)},
    ("ETTh2", 96): {"Linear": (0.288, 0.352), "NLinear": (0.277, 0.338), "DLinear": (0.289, 0.353)},
    ("ETTh2", 192): {"Linear": (0.377, 0.413), "NLinear": (0.344, 0.381), "DLinear": (0.383, 0.418)},
    ("ETTh2", 336): {"Linear": (0.452, 0.461), "NLinear": (0.357, 0.400), "DLinear": (0.448, 0.465)},
    ("ETTh2", 720): {"Linear": (0.698, 0.595), "NLinear": (0.394, 0.436), "DLinear": (0.605, 0.551)},
    ("ETTm1", 96): {"Linear": (0.308, 0.352), "NLinear": (0.306, 0.348), "DLinear": (0.299, 0.343)},
    ("ETTm1", 192): {"Linear": (0.340, 0.369), "NLinear": (0.349, 0.375), "DLinear": (0.335, 0.365)},
    ("ETTm1", 336): {"Linear": (0.376, 0.393), "NLinear": (0.375, 0.388), "DLinear": (0.369, 0.386)},
    ("ETTm1", 720): {"Linear": (0.440, 0.435), "NLinear": (0.433, 0.422), "DLinear": (0.425, 0.421)},
}
REL_TOL = 0.01


def f4(x):
    return f"{x:.4f}".replace(".", ",")


def pct(x):
    return f"{100 * x:+.1f} %".replace(".", ",")


def lamf(x):
    return f"{x:.4g}".replace(".", ",")


def read_csv(text_or_path):
    if isinstance(text_or_path, Path):
        text_or_path = text_or_path.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text_or_path)))


def mse_star(rows):
    """{(ds, H, model): {"lam0": row, "lam_star": row}} cho mục tiêu MSE, chọn val_objective (= val_mse)."""
    out = {}
    for r in rows:
        if r["objective"] == "MSE" and r["selection"] == "val_objective":
            out.setdefault((r["dataset"], int(r["H"]), r["model"]), {})[r["level"]] = r
    return out


def load_sgd():
    runs = {}
    for f in sorted(SGD.glob("*Linear_H*_seed*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        runs.setdefault((r["model"], r["H"]), []).append(r)
    return runs


def weights(ds, H, model, lam):
    W = np.load(ROOT / "results" / "weights" / f"{ds}_H{H}_{model}_MSE_lam{float(lam):.6g}.npy")
    return W[:, :-1], W[:, -1]


def row_stats(W, b):
    s = W.sum(1)
    return {"median": float(np.median(s)), "p10": float(np.percentile(s, 10)), "p90": float(np.percentile(s, 90)),
            "mean": float(s.mean()), "bias_norm": float(np.linalg.norm(b))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md, summ = [], {}
    new = mse_star(read_csv(ROOT / "results" / "results.csv"))
    old_text = subprocess.run(["git", "show", f"{OLD_COMMIT}:results/results.csv"], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8", check=True).stdout
    old = mse_star(read_csv(old_text))
    runs = load_sgd()

    # ---------------- A.4 ----------------
    md.append("## A.4. SGD so với nghiệm dạng đóng, ETTh2\n")
    a4 = []
    for model in ["Linear", "DLinear", "NLinear"]:
        md.append(f"### {model}\n")
        md.append("| H | | MSE test | MAE test | MSE val |\n|---|---|---|---|---|")
        for H in HORIZONS:
            rs = runs.get((model, H), [])
            if len(rs) < 2:
                continue
            arr = {k: np.array([r[k] for r in rs]) for k in ("test_mse", "test_mae", "val_mse")}
            m = {k: (float(v.mean()), float(v.std(ddof=1))) for k, v in arr.items()}
            cf = new[("ETTh2", H, model)]
            pub = PUB[("ETTh2", H)][model]
            star = cf["lam_star"]
            gap = m["test_mse"][0] - float(star["test_mse"])
            gap_val = m["val_mse"][0] - float(star["val_mse"])
            std = m["test_mse"][1]
            dev_pub = (m["test_mse"][0] - pub[0]) / pub[0]
            a4.append({"model": model, "H": H, "n_seeds": len(rs), "sgd": m,
                       "epochs": [r["best_epoch"] for r in rs], "n_epochs_run": [r["n_epochs_run"] for r in rs],
                       "cf_lam0": {k: float(cf["lam0"][k]) for k in ("test_mse", "test_mae", "val_mse")},
                       "cf_star": {k: float(star[k]) for k in ("test_mse", "test_mae", "val_mse", "lam")},
                       "pub": pub, "gap_test": gap, "gap_val": gap_val, "std_test": std,
                       "sgd_better_2std": gap < 0 and -gap > 2 * std, "sgd_vs_pub": dev_pub})
            pm = lambda t: f"{f4(t[0])} ± {f4(t[1])}"     # noqa: E731
            md.append(f"| {H} | SGD ({len(rs)} seed) | {pm(m['test_mse'])} | {pm(m['test_mae'])} | {pm(m['val_mse'])} |")
            md.append(f"| | dạng đóng, λ = 0 | {f4(float(cf['lam0']['test_mse']))} | {f4(float(cf['lam0']['test_mae']))} | "
                      f"{f4(float(cf['lam0']['val_mse']))} |")
            md.append(f"| | dạng đóng, λ* = {lamf(float(star['lam']))} | {f4(float(star['test_mse']))} | "
                      f"{f4(float(star['test_mae']))} | {f4(float(star['val_mse']))} |")
            md.append(f"| | công bố | {f4(pub[0])} | {f4(pub[1])} | không có |")
        md.append("")
    md.append("Khoảng cách SGD − λ* (MSE; âm là SGD tốt hơn):\n")
    md.append("| Mô hình | H | Δ test | 2·sd test | Δ val | SGD tốt hơn > 2 sd | SGD so với công bố | epoch tốt nhất |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in a4:
        md.append(f"| {r['model']} | {r['H']} | {f4(r['gap_test'])} | {f4(2 * r['std_test'])} | {f4(r['gap_val'])} | "
                  f"{'có' if r['sgd_better_2std'] else 'không'} | {pct(r['sgd_vs_pub'])} | {r['epochs']} |")
    verdict = {}
    for model in ["Linear", "DLinear", "NLinear"]:
        rs = [r for r in a4 if r["model"] == model]
        n = sum(r["sgd_better_2std"] for r in rs)
        verdict[model] = {"n_cells_sgd_better": n, "n_cells": len(rs), "confirmed": n >= 3}
    md.append("\n" + "; ".join(f"{m}: {v['n_cells_sgd_better']}/{v['n_cells']} ô" for m, v in verdict.items()) + "\n")
    summ["A4"] = {"cells": a4, "verdict": verdict}

    # ---------------- bảng 36 ô ----------------
    md.append("## So 36 ô: dạng đóng ở λ* (MSE, val_objective, lưới dày) với công bố\n")
    md.append("| Dataset | H | Mô hình | λ* | dạng đóng | công bố | chênh | |\n|---|---|---|---|---|---|---|---|")
    t36, cnt = [], {"better": 0, "tie": 0, "worse": 0}
    for ds in DATASETS:
        for H in HORIZONS:
            for model in MODELS:
                r = new[(ds, H, model)]["lam_star"]
                cf, pub = float(r["test_mse"]), PUB[(ds, H)][model][0]
                rel = (cf - pub) / pub
                tag = "better" if rel < -REL_TOL else "worse" if rel > REL_TOL else "tie"
                cnt[tag] += 1
                t36.append({"dataset": ds, "H": H, "model": model, "lam_star": float(r["lam"]), "cf": cf, "pub": pub,
                            "rel": rel, "tag": tag})
                mark = {"better": "dạng đóng tốt hơn > 1 %", "worse": "**dạng đóng kém hơn > 1 %**", "tie": ""}[tag]
                md.append(f"| {ds} | {H} | {model} | {lamf(float(r['lam']))} | {f4(cf)} | {f4(pub)} | {pct(rel)} | {mark} |")
    md.append(f"\nTốt hơn quá 1 %: {cnt['better']}; trong ±1 %: {cnt['tie']}; kém hơn quá 1 %: {cnt['worse']}.\n")
    summ["table36"] = {"rows": t36, "counts": cnt}

    # ---------------- B.1 ----------------
    md.append("## B.1. Tổng hàng của W (ETTh2)\n")
    md.append("| H | Mô hình | Nghiệm | trung vị | P10 | P90 | ‖b‖ |\n|---|---|---|---|---|---|---|")
    b1 = []
    for H in HORIZONS:
        for model in ["Linear", "DLinear"]:
            items = []
            for r in sorted(runs.get((model, H), []), key=lambda r: r["seed"]):
                Wb = np.load(SGD / f"{model}_H{H}_seed{r['seed']}.npy")
                items.append((f"SGD seed {r['seed']}", Wb[:, :-1], Wb[:, -1]))
            cf = new[("ETTh2", H, model)]
            items.append(("dạng đóng λ = 0", *weights("ETTh2", H, model, 0)))
            items.append((f"dạng đóng λ* = {lamf(float(cf['lam_star']['lam']))}",
                          *weights("ETTh2", H, model, cf["lam_star"]["lam"])))
            for name, W, b in items:
                s = row_stats(W, b)
                b1.append({"H": H, "model": model, "solution": name, **s})
                md.append(f"| {H} | {model} | {name} | {f4(s['median'])} | {f4(s['p10'])} | {f4(s['p90'])} | "
                          f"{f4(s['bias_norm'])} |")
        nl = new[("ETTh2", H, "NLinear")]["lam_star"]
        s = row_stats(*weights("ETTh2", H, "NLinear", nl["lam"]))
        b1.append({"H": H, "model": "NLinear", "solution": f"dạng đóng λ* = {nl['lam']}", **s})
        md.append(f"| {H} | NLinear | dạng đóng λ* = {lamf(float(nl['lam']))} | {f4(s['median'])} | {f4(s['p10'])} | "
                  f"{f4(s['p90'])} | {f4(s['bias_norm'])} |")
    summ["B1"] = b1
    md.append("")

    # ---------------- B.2 ----------------
    f = ROOT / "results" / "ridge_w0" / "summary.json"
    if f.exists():
        b2 = json.loads(f.read_text(encoding="utf-8"))
        md.append("## B.2. Ridge co về W₀ (λ* theo val MSE)\n")
        zc = b2["zero_check"]
        md.append(f"Kiểm `zero` so với lambda_path.csv: lệch lớn nhất {zc['max_abs_dev']:.1e} "
                  f"({'đạt' if zc['pass'] else 'KHÔNG ĐẠT'} 1e-10).\n")
        md.append("| Dataset | H | Mô hình | zero: λ* / val / test | mean: λ* / val / test | last: λ* / val / test |")
        md.append("|---|---|---|---|---|---|")
        by = {}
        for c in b2["cells"]:
            by.setdefault((c["dataset"], c["H"], c["model"]), {})[c["W0"]] = c
        for (ds, H, model), v in by.items():
            md.append(f"| {ds} | {H} | {model} | " + " | ".join(
                f"{lamf(v[w]['lam_star'])}{' (mút)' if v[w]['lam_at_edge'] else ''} / {f4(v[w]['val_mse'])} / "
                f"{f4(v[w]['test_mse'])}" for w in ("zero", "mean", "last")) + " |")
        md.append("")

    # ---------------- C.1 ----------------
    f = ROOT / "results" / "recheck" / "compare.json"
    if f.exists():
        c1 = json.loads(f.read_text(encoding="utf-8"))
        md.append("## C.1. Kiểm lại ba đường λ chạy trước khi có fallback\n")
        md.append("| Đường | lệch lớn nhất | λ* (val MAE) cũ → mới | λ* (val MSE) cũ → mới | vòng cũ → mới (tổng) | "
                  "fallback mới (bước h) | |\n|---|---|---|---|---|---|---|")
        for p in c1["paths"]:
            rows = [r for r in p["rows"] if "only_in" not in r]
            st = p["lam_star"]
            md.append(f"| {p['dataset']} H={p['H']} {p['model']} {p['objective']} | {p['max_abs_dev']:.1e} | "
                      f"{lamf(st['val_mae']['old'])} → {lamf(st['val_mae']['new'])} | "
                      f"{lamf(st['val_mse']['old'])} → {lamf(st['val_mse']['new'])} | "
                      f"{sum(r['n_iter_old'] for r in rows)} → {sum(r['n_iter_new'] for r in rows)} | "
                      f"{sum(r['n_fallback_new'] for r in rows)} | {'đạt' if p['pass'] else 'KHÔNG ĐẠT'} |")
        md.append("")

    # ---------------- C.2 ----------------
    md.append("## C.2. λ* và MSE test trước và sau khi làm dày lưới MSE\n")
    md.append("| Dataset | H | Mô hình | λ* cũ | λ* mới | test cũ | test mới | chênh |\n|---|---|---|---|---|---|---|---|")
    c2 = []
    for ds in DATASETS:
        for H in HORIZONS:
            for model in MODELS:
                o, n = old[(ds, H, model)]["lam_star"], new[(ds, H, model)]["lam_star"]
                d = float(n["test_mse"]) - float(o["test_mse"])
                c2.append({"dataset": ds, "H": H, "model": model, "lam_old": float(o["lam"]), "lam_new": float(n["lam"]),
                           "test_old": float(o["test_mse"]), "test_new": float(n["test_mse"]), "diff": d,
                           "val_old": float(o["val_mse"]), "val_new": float(n["val_mse"]),
                           "edge_new": n["lam_at_edge"]})
                md.append(f"| {ds} | {H} | {model} | {lamf(float(o['lam']))} | {lamf(float(n['lam']))}"
                          f"{' (mút)' if n['lam_at_edge'] == 'True' else ''} | {f4(float(o['test_mse']))} | "
                          f"{f4(float(n['test_mse']))} | {d:+.1e} |")
    c5 = new[("ETTh1", 96, "DLinear")]["lam_star"]
    md.append(f"\nC.5 (DLinear ETTh1 H = 96): λ* = {lamf(float(c5['lam']))}, MSE test {float(c5['test_mse']):.6f} "
              f"(cần 0,3697 ± 1e-4 nếu λ* = 562,34).\n")
    summ["C2"] = {"rows": c2, "c5": {"lam": float(c5["lam"]), "test_mse": float(c5["test_mse"])}}

    # ---------------- C.3 ----------------
    f = ROOT / "results" / "constant_windows" / "summary.json"
    if f.exists():
        c3 = json.loads(f.read_text(encoding="utf-8"))
        md.append("## C.3. Cửa sổ hằng của ETTh2\n")
        md.append("| H | tập | hằng / tổng | theo kênh (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) |\n|---|---|---|---|")
        for c in c3["counts"]:
            md.append(f"| {c['H']} | {c['split']} | {c['n_constant']} / {c['n_windows']} | {c['per_channel']} |")
        md.append("\n| H | Mô hình | bỏ (train) | λ* trước → sau | test trước → sau |\n|---|---|---|---|---|")
        for c in c3["cells"]:
            b, a = c["before"], c["after"]
            md.append(f"| {c['H']} | {c['model']} | {c['n_train_dropped']} | {lamf(b['lam_star'])} → {lamf(a['lam_star'])} | "
                      f"{f4(b['test_mse'])} → {f4(a['test_mse'])} |")
        md.append("\n| H | NLinear − Linear trước | sau |\n|---|---|---|")
        by = {(c["H"], c["model"]): c for c in c3["cells"]}
        for H in HORIZONS:
            nl, li = by[(H, "NLinear")], by[(H, "Linear")]
            md.append(f"| {H} | {f4(nl['before']['test_mse'] - li['before']['test_mse'])} | "
                      f"{f4(nl['after']['test_mse'] - li['after']['test_mse'])} |")
        md.append("")

    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    save_json(OUT / "summary.json", summ)
    print(f"→ {(OUT / 'tables.md').relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
