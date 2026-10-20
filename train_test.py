"""
train_test.py - Train and validate the 10 KAN methods from kan_model.py on the MultiEnv
Wi-Fi CSI dataset, separately for each of its 3 environments.

Usage:
    py run.py   (option 1)                            # all 3 environments -> results/MultiEnv/
    from train_test import train_multienv; train_multienv()
    from train_test import train
    results = train(env=2, models=["fastkan", "chebykan"], epochs=10)
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

KAN_METHODS = ["efficientkan", "fastkan", "fasterkan", "bsrbf_kan", "chebykan",
               "jacobikan", "fourierkan", "wavkan", "relukan", "rationalkan"]
MULTIENV_DIR = os.path.join("results", "MultiEnv")


def train(models=KAN_METHODS, env=1, n_hidden=64, time_pool=10, val_size=0.2, epochs=30,
          batch_size=64, lr=1e-3, weight_decay=1e-4, class_weighted=True, seed=42,
          out_dir=None, device=None, load_fn=None):
    """Train every model in `models` on the same split, validate each epoch, keep the best epoch.

    env       : MultiEnv environment (1, 2 or 3) to train and validate on; ignored if load_fn is given.
    time_pool : average every `time_pool` time steps (850 -> 850 // time_pool) to shrink n_input.
    load_fn   : optional function returning (X, y), X of shape (samples, time_steps, subcarriers).
    Returns a dict {model_name: metrics}. Writes to `out_dir` (default results/MultiEnv/env<env>):
    results.json (best-epoch metrics), log.txt (per-epoch log) and <model_name>.pt (best weights).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = out_dir or os.path.join(MULTIENV_DIR, f"env{env}")
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    # ---- data: load, pool over time, flatten, stratified split, standardize on train stats ----
    X, y = load_fn() if load_fn is not None else load_env(env)
    n, t, c = X.shape
    t_new = t // time_pool
    X = X[:, : t_new * time_pool].reshape(n, t_new, time_pool, c).mean(axis=2).reshape(n, -1)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
    del X
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-6
    X_tr, X_va = (X_tr - mean) / std, (X_va - mean) / std

    n_input, n_output = X_tr.shape[1], int(y.max()) + 1
    source = "custom load_fn" if load_fn is not None else f"MultiEnv environment {env}"
    log(f"data={source}  device={device}  train={X_tr.shape}  val={X_va.shape}  "
        f"n_input={n_input}  classes={n_output}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr)),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va)),
                            batch_size=batch_size)

    weight = None
    if class_weighted:
        counts = np.bincount(y_tr, minlength=n_output)
        weight = torch.tensor(len(y_tr) / (n_output * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

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
            # -- train --
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

            # -- validate --
            model.eval()
            va_loss, preds = 0.0, []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    va_loss += criterion(out, yb).item() * len(yb)
                    preds.append(out.argmax(1).cpu())
            preds = torch.cat(preds).numpy()
            va_acc = float((preds == y_va).mean())
            va_f1 = float(f1_score(y_va, preds, average="macro"))

            log(f"epoch {epoch:3d}/{epochs}  train_loss={tr_loss / tr_total:.4f}  "
                f"train_acc={tr_correct / tr_total:.4f}  val_loss={va_loss / len(y_va):.4f}  "
                f"val_acc={va_acc:.4f}  val_f1={va_f1:.4f}")

            if va_acc > best["val_acc"]:
                best = {"val_acc": va_acc, "val_f1": va_f1, "epoch": epoch}
                torch.save(model.state_dict(), os.path.join(out_dir, f"{name}.pt"))

        best.update(params=count_parameters(model), train_time_s=round(time.time() - start, 1))
        results[name] = best
        log(f"best {name}: {best}")
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- summary ----
    log(f"\n{'model':<14}{'params':>12}{'best_ep':>9}{'val_acc':>9}{'val_f1':>9}{'time(s)':>9}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["val_acc"]):
        log(f"{name:<14}{r['params']:>12,}{r['epoch']:>9}{r['val_acc']:>9.4f}{r['val_f1']:>9.4f}"
            f"{r['train_time_s']:>9}")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log_file.close()
    return results


def train_multienv():
    """Train the 10 KANs on each MultiEnv environment and write results/MultiEnv/summary.{json,csv}."""
    all_results = {f"env{env}": train(env=env) for env in ENVS}

    # ---- comparison across environments: results/MultiEnv/summary.{json,csv} ----
    with open(os.path.join(MULTIENV_DIR, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    envs = list(all_results)
    rows = []
    for name in KAN_METHODS:
        accs = [all_results[e][name]["val_acc"] for e in envs]
        f1s = [all_results[e][name]["val_f1"] for e in envs]
        rows.append((name, accs, f1s, all_results[envs[0]][name]["params"]))
    rows.sort(key=lambda r: -np.mean(r[1]))

    header = (["model", "params"] + [f"{e}_acc" for e in envs] + ["mean_acc", "std_acc"]
              + [f"{e}_f1" for e in envs] + ["mean_f1"])
    with open(os.path.join(MULTIENV_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name, accs, f1s, params in rows:
            w.writerow([name, params] + [f"{a:.4f}" for a in accs] + [f"{np.mean(accs):.4f}",
                       f"{np.std(accs):.4f}"] + [f"{v:.4f}" for v in f1s] + [f"{np.mean(f1s):.4f}"])

    print(f"\n===== Validation accuracy per environment =====")
    print(f"{'model':<14}" + "".join(f"{e:>9}" for e in envs) + f"{'mean':>9}{'std':>8}{'mean_f1':>9}")
    for name, accs, f1s, _ in rows:
        print(f"{name:<14}" + "".join(f"{a:>9.4f}" for a in accs)
              + f"{np.mean(accs):>9.4f}{np.std(accs):>8.4f}{np.mean(f1s):>9.4f}")
    print(f"\nsaved {MULTIENV_DIR}/summary.csv and summary.json")
    return all_results


# if __name__ == "__main__":
#     train_multienv()
