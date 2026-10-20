"""
train_test_cross_multienv.py - Cross-environment evaluation of the 10 KAN methods on MultiEnv.

For a training environment A and test environments B, C:
    train      : 80% of A  (same stratified split as train_test.py, seed 42)
    validation : the other 20% of A - used only to pick the best epoch
    test       : 20% of B and 20% of C - never used for training or model selection
Inputs are standardized with the mean/std of the training data only.

Usage:
    py run.py   (option 4)               # {train 1 | test 2,3}, {train 2 | test 1,3}, {train 3 | test 1,2}
    from train_test_cross_multienv import train_cross_all; train_cross_all()
    from train_test_cross_multienv import train_cross
    results = train_cross(train_env=1, test_envs=(2, 3), models=["wavkan"], epochs=10)
"""

import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from kan_model import count_parameters, get_model
from read_data import ENVS, load_env
from train_test import KAN_METHODS

CROSS_DIR = os.path.join("results", "MultiEnv_cross")


def _split_env(env, time_pool, val_size, seed):
    """Load one environment, pool over time, flatten, and return its stratified 80/20 split."""
    X, y = load_env(env)
    n, t, c = X.shape
    t_new = t // time_pool
    X = X[:, : t_new * time_pool].reshape(n, t_new, time_pool, c).mean(axis=2).reshape(n, -1)
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)


def train_cross(train_env, test_envs, models=KAN_METHODS, n_hidden=64, time_pool=10, val_size=0.2,
                epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4, class_weighted=True, seed=42,
                out_dir=None, device=None):
    """Train on 80% of `train_env`, pick the best epoch on its other 20%, test on 20% of each test env.

    Returns {model_name: metrics}. Writes to `out_dir` (default results/MultiEnv_cross/train<env>):
    results.json, log.txt and <model_name>.pt (weights of the selected epoch).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = out_dir or os.path.join(CROSS_DIR, f"train{train_env}")
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    # ---- data ----
    X_tr, X_va, y_tr, y_va = _split_env(train_env, time_pool, val_size, seed)
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-6
    X_tr, X_va = (X_tr - mean) / std, (X_va - mean) / std
    test_sets = {}
    for env in test_envs:
        _, X_te, _, y_te = _split_env(env, time_pool, val_size, seed)  # the 20% part of that env
        test_sets[f"env{env}"] = ((X_te - mean) / std, y_te)

    n_input, n_output = X_tr.shape[1], int(y_tr.max()) + 1
    log(f"train=env{train_env} {X_tr.shape}  val=env{train_env} {X_va.shape}  "
        + "  ".join(f"test={k} {v[0].shape}" for k, v in test_sets.items())
        + f"  device={device}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr)),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    eval_sets = {"val": (X_va, y_va), **test_sets}
    eval_tensors = {k: torch.from_numpy(X).float() for k, (X, _) in eval_sets.items()}

    weight = None
    if class_weighted:
        counts = np.bincount(y_tr, minlength=n_output)
        weight = torch.tensor(len(y_tr) / (n_output * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    def evaluate(model):
        model.eval()
        scores = {}
        with torch.no_grad():
            for k, X in eval_tensors.items():
                preds = torch.cat([model(xb.to(device)).argmax(1).cpu() for xb in X.split(256)]).numpy()
                y_true = eval_sets[k][1]
                scores[k] = (float((preds == y_true).mean()), float(f1_score(y_true, preds, average="macro")))
        return scores

    results = {}
    for name in models:
        torch.manual_seed(seed)
        model = get_model(name, [n_input, n_hidden, n_output]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best = {"val_acc": -1.0}
        log(f"\n===== {name}  ({count_parameters(model):,} params) =====")
        start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * len(yb)
                tr_correct += (out.argmax(1) == yb).sum().item()
                tr_total += len(yb)
            scheduler.step()

            scores = evaluate(model)
            log(f"epoch {epoch:3d}/{epochs}  train_loss={tr_loss / tr_total:.4f}  "
                f"train_acc={tr_correct / tr_total:.4f}  "
                + "  ".join(f"{k}_acc={a:.4f}" for k, (a, _) in scores.items()))

            if scores["val"][0] > best["val_acc"]:  # selection uses the training environment only
                best = {"epoch": epoch, "val_acc": scores["val"][0], "val_f1": scores["val"][1]}
                for k in test_sets:
                    best[f"test_{k}_acc"], best[f"test_{k}_f1"] = scores[k]
                torch.save(model.state_dict(), os.path.join(out_dir, f"{name}.pt"))

        best.update(params=count_parameters(model), train_time_s=round(time.time() - start, 1))
        results[name] = best
        log(f"best {name}: {best}")
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log_file.close()
    return results


def write_table(all_results, out_dir=CROSS_DIR):
    """Write summary.csv and summary.md: in-environment validation vs. cross-environment test accuracy."""
    settings = []  # (train_env, test_key) in column order
    for tr in sorted(all_results):
        settings += [(tr, k[len("test_"):-len("_acc")]) for k in all_results[tr][KAN_METHODS[0]]
                     if k.startswith("test_") and k.endswith("_acc")]

    rows = []
    for name in KAN_METHODS:
        same = [all_results[tr][name]["val_acc"] for tr in sorted(all_results)]
        cross = [all_results[tr][name][f"test_{te}_acc"] for tr, te in settings]
        cross_f1 = [all_results[tr][name][f"test_{te}_f1"] for tr, te in settings]
        rows.append({"model": name, "same": same, "cross": cross, "cross_f1": cross_f1,
                     "mean_same": np.mean(same), "mean_cross": np.mean(cross),
                     "mean_cross_f1": np.mean(cross_f1)})
    rows.sort(key=lambda r: -r["mean_cross"])

    trains = sorted(all_results)
    same_cols = [f"{tr}->{tr}" for tr in trains]
    cross_cols = [f"{tr}->{te}" for tr, te in settings]
    header = (["model"] + same_cols + ["mean_same_env"] + cross_cols
              + ["mean_cross_env", "drop", "mean_cross_env_f1"])

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["model"]] + [f"{v:.4f}" for v in r["same"]] + [f"{r['mean_same']:.4f}"]
                       + [f"{v:.4f}" for v in r["cross"]]
                       + [f"{r['mean_cross']:.4f}", f"{r['mean_same'] - r['mean_cross']:.4f}",
                          f"{r['mean_cross_f1']:.4f}"])

    def pct(v):
        return f"{100 * v:.1f}"

    lines = [
        "# MultiEnv cross-environment results",
        "",
        "Accuracy (%). `A->A`: validation on the held-out 20% of the training environment "
        "(used to select the epoch). `A->B`: test on 20% of environment B, never seen in training.",
        "",
        "| " + " | ".join(h.replace("_", " ") for h in header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    for r in rows:
        cells = ([r["model"]] + [pct(v) for v in r["same"]] + [pct(r["mean_same"])]
                 + [pct(v) for v in r["cross"]]
                 + [f"**{pct(r['mean_cross'])}**", pct(r["mean_same"] - r["mean_cross"]),
                    pct(r["mean_cross_f1"])])
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {out_dir}/summary.csv, summary.md and summary.json")


def train_cross_all():
    """Run the three cross-environment settings and write results/MultiEnv_cross/summary.{json,md,csv}."""
    all_results = {}
    for train_env in ENVS:
        test_envs = tuple(e for e in ENVS if e != train_env)
        print(f"\n########## train on env{train_env}, test on {test_envs} ##########")
        all_results[f"env{train_env}"] = train_cross(train_env, test_envs)
    os.makedirs(CROSS_DIR, exist_ok=True)
    with open(os.path.join(CROSS_DIR, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    write_table(all_results)
    return all_results


# if __name__ == "__main__":
#     train_cross_all()
