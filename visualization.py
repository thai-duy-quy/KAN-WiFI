"""
visualization.py - Visualize and compare the 10 KAN methods trained by train_test.py on the
3 environments of the MultiEnv dataset.

Inputs (written by train_test.py), for each environment N = 1, 2, 3:
    results/MultiEnv/envN/results.json   best-epoch metrics per model
    results/MultiEnv/envN/log.txt        per-epoch log
    results/MultiEnv/envN/<model>.pt     best weights (only needed for confusion matrices)
    results/MultiEnv/summary.csv         per-environment accuracy table

Outputs:
    results/MultiEnv/envN/figures/
        1_scores.png           val accuracy and macro-F1 per model
        2_accuracy_curves.png  train / val accuracy per epoch, one panel per model
        3_loss_curves.png      train / val loss per epoch, one panel per model
        4_efficiency.png       accuracy vs. parameter count and vs. training time
        5_confusion.png        row-normalized confusion matrix per model (validation set)
        summary.csv            the comparison table
    results/MultiEnv/figures/
        environments.png       val accuracy of every model in every environment

Usage:
    from visualization import visualize_multienv
    visualize_multienv()                 # all environments
    visualize_multienv(env=2)            # one environment
    visualize_multienv(confusion=False)  # skip the figure that needs the data + GPU
"""

import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

MULTIENV_DIR = os.path.join("results", "MultiEnv")
ENVS = (1, 2, 3)
# Defaults point at environment 1; run() overrides them per environment.
RESULTS_DIR = os.path.join(MULTIENV_DIR, "env1")
LOG_FILE = os.path.join(RESULTS_DIR, "log.txt")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# Colors: categorical slots (validated CVD-safe, all pairs) + recessive ink and grid.
SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue = validation, orange = training
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 9, "lines.linewidth": 2,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_results(path=None):
    with open(path or os.path.join(RESULTS_DIR, "results.json")) as f:
        return json.load(f)


def parse_log(path=None):
    path = path or LOG_FILE
    """Return {model: {"epoch": [...], "train_loss": [...], "train_acc": [...], ...}}."""
    with open(path, "rb") as f:
        raw = f.read()
    # PowerShell `>` may write UTF-16; plain shells write UTF-8.
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")

    history, current = {}, None
    header = re.compile(r"^===== (\S+)")
    row = re.compile(r"epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+train_acc=([\d.]+)\s+"
                     r"val_loss=([\d.]+)\s+val_acc=([\d.]+)\s+val_f1=([\d.]+)")
    keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]
    for line in text.splitlines():
        m = header.match(line.strip())
        if m:
            current = m.group(1)
            history[current] = {k: [] for k in keys}
            continue
        m = row.search(line)
        if m and current:
            for k, v in zip(keys, m.groups()):
                history[current][k].append(int(v) if k == "epoch" else float(v))
    return history


def _order(results):
    """Model names sorted by val accuracy (best first) - the same order in every figure."""
    return sorted(results, key=lambda n: -results[n]["val_acc"])


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_scores(results):
    """Dot plot: accuracy and macro-F1 per model. Dots (not bars) so the axis may zoom in."""
    names = _order(results)[::-1]  # best at the top
    acc = np.array([results[n]["val_acc"] for n in names])
    f1 = np.array([results[n]["val_f1"] for n in names])
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hlines(y, np.minimum(acc, f1), np.maximum(acc, f1), color=GRID, linewidth=2, zorder=1)
    ax.scatter(f1, y, s=64, color=SERIES_2, edgecolor=SURFACE, linewidth=2, zorder=3, label="Macro-F1")
    ax.scatter(acc, y, s=64, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3, label="Accuracy")
    for yi, a in zip(y, acc):
        ax.annotate(f"{a:.3f}", (a, yi), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)

    lo = min(acc.min(), f1.min())
    ax.set_xlim(np.floor(lo * 100 - 1) / 100, 1.0)
    ax.set_yticks(y, names, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Validation score (best epoch)")
    ax.set_title("Validation accuracy and macro-F1 by KAN method", loc="left")
    ax.legend(loc="lower right", ncol=2)
    _save(fig, "1_scores.png")


def _small_multiples(history, results, train_key, val_key, ylabel, title, fname, best_marker):
    names = [n for n in _order(results) if n in history]
    ncols = 5
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        h = history[name]
        ax.plot(h["epoch"], h[train_key], color=SERIES_2, label="Train")
        ax.plot(h["epoch"], h[val_key], color=SERIES_1, label="Validation")
        if best_marker:
            ep = results[name]["epoch"]
            val = h[val_key][h["epoch"].index(ep)]
            ax.scatter([ep], [val], s=64, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{val:.3f} @ ep {ep}", (ep, val), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK_2)
        ax.set_title(name, loc="left", fontsize=10)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("Epoch")
    for ax in axes[::ncols]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, fname)


def plot_accuracy_curves(history, results):
    _small_multiples(history, results, "train_acc", "val_acc", "Accuracy",
                     "Training vs. validation accuracy per epoch", "2_accuracy_curves.png", True)


def plot_loss_curves(history, results):
    _small_multiples(history, results, "train_loss", "val_loss", "Cross-entropy loss",
                     "Training vs. validation loss per epoch", "3_loss_curves.png", False)


def _label_points(ax, xs, ys, labels):
    """Direct-label scatter points, trying a few offsets so labels don't overlap each other."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    candidates = [((6, 3), "left"), ((6, -11), "left"), ((-6, 3), "right"), ((-6, -11), "right")]
    placed = []
    for x, y, text in sorted(zip(xs, ys, labels), key=lambda p: -p[1]):
        for offset, ha in candidates:
            ann = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha,
                              fontsize=8, color=INK_2)
            box = ann.get_window_extent(renderer)
            if not any(box.overlaps(b) for b in placed):
                break
            ann.remove()
        else:
            ann = ax.annotate(text, (x, y), xytext=candidates[0][0], textcoords="offset points",
                              fontsize=8, color=INK_2)
            box = ann.get_window_extent(renderer)
        placed.append(box)


def plot_efficiency(results):
    """Accuracy against model size and against training time - one measure per panel."""
    names = _order(results)
    acc = np.array([results[n]["val_acc"] for n in names])
    params = np.array([results[n]["params"] for n in names]) / 1e6
    times = np.array([results[n]["train_time_s"] for n in names])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, x, xlabel, title in [
        (axes[0], params, "Parameters (millions)", "Accuracy vs. model size"),
        (axes[1], times, "Total training time (s)", "Accuracy vs. training time"),
    ]:
        ax.scatter(x, acc, s=64, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.margins(x=0.18, y=0.15)
    axes[0].set_ylabel("Validation accuracy (best epoch)")
    fig.tight_layout()
    # Label after layout is final, so overlap checks use the real positions.
    _label_points(axes[0], params, acc, names)
    _label_points(axes[1], times, acc, names)
    _save(fig, "4_efficiency.png")


def plot_confusion(results, n_hidden=64, time_pool=10, val_size=0.2, seed=42, load_fn=None):
    """Rebuild the exact validation split used by train_test.train() and evaluate saved weights."""
    import torch
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    from kan_model import get_model

    if load_fn is None:  # default: MultiEnv environment 1, the data the original results used
        from read_data import load_env

        def load_fn():
            return load_env(1)

    # Same preprocessing as train_test.train() - keep the two in sync.
    X, y = load_fn()
    n, t, c = X.shape
    t_new = t // time_pool
    X = X[:, : t_new * time_pool].reshape(n, t_new, time_pool, c).mean(axis=2).reshape(n, -1)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-6
    X_va = torch.from_numpy((X_va - mean) / std).float()
    del X, X_tr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(y.max()) + 1
    names = [m for m in _order(results) if os.path.exists(os.path.join(RESULTS_DIR, f"{m}.pt"))]

    ncols = 5
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        model = get_model(name, [X_va.shape[1], n_hidden, n_classes])
        model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"{name}.pt"), map_location="cpu"))
        model.to(device).eval()
        with torch.no_grad():
            preds = torch.cat([model(xb.to(device)).argmax(1).cpu() for xb in X_va.split(256)]).numpy()
        cm = confusion_matrix(y_va, preds, labels=range(n_classes), normalize="true")

        ax.imshow(cm, cmap=SEQ_BLUE, vmin=0, vmax=1)
        for i in range(n_classes):
            for j in range(n_classes):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                            color=SURFACE if cm[i, j] > 0.55 else INK_2)
        ax.set_title(f"{name}  ({results[name]['val_acc']:.3f})", loc="left", fontsize=10)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        del model
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("Predicted class")
    for ax in axes[::ncols]:
        ax.set_ylabel("True class")
    fig.suptitle("Confusion matrices on the validation set (% of each true class)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "5_confusion.png")


def plot_environments(summary_path=os.path.join(MULTIENV_DIR, "summary.csv"),
                      fig_dir=os.path.join(MULTIENV_DIR, "figures")):
    """Dot plot: validation accuracy of each model in each environment, sorted by the mean."""
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    envs = [k[: -len("_acc")] for k in rows[0] if k.startswith("env") and k.endswith("_acc")]
    rows.sort(key=lambda r: float(r["mean_acc"]))  # best at the top
    names = [r["model"] for r in rows]
    y = np.arange(len(names))
    accs = np.array([[float(r[f"{e}_acc"]) for e in envs] for r in rows])
    means = np.array([float(r["mean_acc"]) for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    ax.hlines(y, accs.min(axis=1), accs.max(axis=1), color=GRID, linewidth=2, zorder=1)
    # Color + marker shape per environment, so identity never relies on color alone.
    # Small vertical offsets keep equal scores from hiding each other.
    offsets = np.linspace(-0.2, 0.2, len(envs))
    for i, (e, color, marker) in enumerate(zip(envs, [SERIES_1, SERIES_2, SERIES_3], ["o", "s", "D"])):
        ax.scatter(accs[:, i], y + offsets[i], s=48, color=color, marker=marker, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=f"Environment {e[3:]}")
    ax.scatter(means, y, s=90, marker="|", color=INK, linewidth=2, zorder=4, label="Mean")
    for yi, hi, m in zip(y, accs.max(axis=1), means):
        ax.annotate(f"mean {m:.3f}", (hi, yi), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)

    ax.set_xlim(np.floor(accs.min() * 100 - 1) / 100, 1.0)
    ax.set_yticks(y, names, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Validation accuracy (best epoch)")
    ax.set_title("Validation accuracy per MultiEnv environment", loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2,
              handletextpad=0.3, columnspacing=1.2)
    global FIG_DIR
    FIG_DIR = fig_dir
    _save(fig, "environments.png")


def write_summary(results, history):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "summary.csv")
    fields = ["rank", "model", "val_acc", "val_f1", "best_epoch", "params", "train_time_s",
              "final_train_acc", "final_val_acc"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for rank, name in enumerate(_order(results), 1):
            r, h = results[name], history.get(name, {})
            w.writerow([rank, name, f"{r['val_acc']:.4f}", f"{r['val_f1']:.4f}", r["epoch"], r["params"],
                        r["train_time_s"],
                        f"{h['train_acc'][-1]:.4f}" if h.get("train_acc") else "",
                        f"{h['val_acc'][-1]:.4f}" if h.get("val_acc") else ""])
    print(f"saved {path}")


def run(results_dir=RESULTS_DIR, log_file=LOG_FILE, fig_dir=FIG_DIR, load_fn=None, time_pool=10,
        confusion=True):
    """Make every figure for one training run (results_dir + log_file) and save them in fig_dir."""
    global RESULTS_DIR, LOG_FILE, FIG_DIR
    RESULTS_DIR, LOG_FILE, FIG_DIR = results_dir, log_file, fig_dir

    results = load_results()
    history = parse_log() if os.path.exists(LOG_FILE) else {}

    plot_scores(results)
    plot_efficiency(results)
    if history:
        plot_accuracy_curves(history, results)
        plot_loss_curves(history, results)
    else:
        print(f"[warning] {LOG_FILE} not found - skipping learning curves")
    if confusion:
        plot_confusion(results, time_pool=time_pool, load_fn=load_fn)
    write_summary(results, history)


def visualize_multienv(env=None, confusion=True):
    """Draw the figures of one MultiEnv environment (env = 1, 2, 3) or of all of them (env=None),
    plus the comparison across environments. confusion=False skips the step that needs data + GPU."""
    from functools import partial

    from read_data import load_env

    for e in [env] if env else ENVS:
        env_dir = os.path.join(MULTIENV_DIR, f"env{e}")
        print(f"\n=== MultiEnv environment {e} ===")
        run(results_dir=env_dir, log_file=os.path.join(env_dir, "log.txt"),
            fig_dir=os.path.join(env_dir, "figures"), load_fn=partial(load_env, e),
            time_pool=10, confusion=confusion)
    if os.path.exists(os.path.join(MULTIENV_DIR, "summary.csv")):
        print("\n=== Comparison across environments ===")
        plot_environments()


# if __name__ == "__main__":
#     visualize_multienv()
