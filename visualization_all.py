"""
visualization_all.py - Compare the 10 KAN methods across the three datasets.

Inputs:
    results/MultiEnv/summary.json    per-environment results (train_test.py); averaged over env1-3
    results_our_data/results.json    our dataset (train_test_our_data.py)
    results_stan_wifi/results.json   StanWiFi (train_test_stan_wifi.py)

Outputs (results/figures_all/):
    1_scores.png       accuracy and macro-F1 of every model on every dataset
    2_ranks.png        rank of every model on every dataset (1 = best)
    3_efficiency.png   accuracy vs. parameters, one panel per dataset
    summary.csv        the comparison table
    summary.md         the comparison table in Markdown

Caveat: StanWiFi samples are overlapping windows of the same recordings, so its random
train/validation split leaks; its scores are optimistic and are marked as such everywhere.

Usage:
    from visualization_all import visualize; visualize()   (also run by run.py)
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Importing visualization applies the shared matplotlib style.
from visualization import (GRID, INK, INK_2, SEQ_BLUE, SERIES_1, SERIES_2, SERIES_3, SURFACE,
                           _label_points)

KAN_METHODS = ["efficientkan", "fastkan", "fasterkan", "bsrbf_kan", "chebykan",
               "jacobikan", "fourierkan", "wavkan", "relukan", "rationalkan"]
FIG_DIR = os.path.join("results", "figures_all")
LEAKY = "StanWiFi"  # dataset whose validation split overlaps the training split

# name -> (color, marker); fixed order so each dataset keeps its color in every figure.
DATASETS = {
    "MultiEnv (avg 3 envs)": (SERIES_1, "o"),
    "MINE lab (ours)": (SERIES_2, "s"),
    "StanWiFi": (SERIES_3, "D"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all():
    """Return {dataset: {model: {"acc", "f1", "params", "time"}}}."""
    data = {}

    with open(os.path.join("results", "MultiEnv", "summary.json")) as f:
        multienv = json.load(f)  # {env: {model: metrics}}
    data["MultiEnv (avg 3 envs)"] = {
        m: {"acc": np.mean([multienv[e][m]["val_acc"] for e in multienv]),
            "f1": np.mean([multienv[e][m]["val_f1"] for e in multienv]),
            "params": multienv[next(iter(multienv))][m]["params"],
            "time": np.mean([multienv[e][m]["train_time_s"] for e in multienv])}
        for m in KAN_METHODS}

    for name, path in [("MINE lab (ours)", "results_our_data"), ("StanWiFi", "results_stan_wifi")]:
        with open(os.path.join(path, "results.json")) as f:
            r = json.load(f)
        data[name] = {m: {"acc": r[m]["val_acc"], "f1": r[m]["val_f1"], "params": r[m]["params"],
                          "time": r[m]["train_time_s"]} for m in KAN_METHODS}
    return data


def ranks(data):
    """{dataset: {model: rank}}; rank 1 = highest accuracy, ties share the best rank."""
    out = {}
    for ds, res in data.items():
        accs = {m: round(res[m]["acc"], 6) for m in KAN_METHODS}
        out[ds] = {m: 1 + sum(a > accs[m] for a in accs.values()) for m in KAN_METHODS}
    return out


def model_order(data, rank):
    """Models sorted by mean rank over the leak-free datasets (best first), then by mean accuracy."""
    clean = [ds for ds in data if ds != LEAKY]
    return sorted(KAN_METHODS, key=lambda m: (np.mean([rank[ds][m] for ds in clean]),
                                              -np.mean([data[ds][m]["acc"] for ds in clean])))


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _ds_label(ds):
    return f"{ds} (leaky split)" if ds == LEAKY else ds


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_scores(data, order):
    """Two panels (accuracy | macro-F1); one row per model, one marker per dataset."""
    names = order[::-1]  # best at the top
    y = np.arange(len(names))
    offsets = np.linspace(-0.22, 0.22, len(data))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, key, title in [(axes[0], "acc", "Validation accuracy"), (axes[1], "f1", "Validation macro-F1")]:
        vals = np.array([[data[ds][m][key] for ds in data] for m in names])
        ax.hlines(y, vals.min(axis=1), vals.max(axis=1), color=GRID, linewidth=2, zorder=1)
        for i, ds in enumerate(data):
            color, marker = DATASETS[ds]
            ax.scatter(vals[:, i], y + offsets[i], s=46, color=color, marker=marker,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, label=_ds_label(ds))
        ax.set_xlim(np.floor(vals.min() * 20) / 20, 1.005)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Best-epoch score")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3)
    fig.suptitle("10 KAN methods on three Wi-Fi CSI datasets", x=0.01, y=0.995, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    _save(fig, "1_scores.png")


def plot_ranks(data, rank, order):
    """Heatmap of ranks (1 = best) with accuracy in each cell."""
    datasets = list(data)
    mat = np.array([[rank[ds][m] for ds in datasets] for m in order])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    # Reverse the scale so rank 1 is the darkest cell.
    ax.imshow(11 - mat, cmap=SEQ_BLUE, vmin=0, vmax=10, aspect="auto")
    for i, m in enumerate(order):
        for j, ds in enumerate(datasets):
            r = mat[i, j]
            ax.text(j, i, f"#{r}\n{100 * data[ds][m]['acc']:.1f}%", ha="center", va="center",
                    fontsize=8, color=SURFACE if r <= 4 else INK_2)
    ax.set_xticks(range(len(datasets)), [_ds_label(ds).replace(" (", "\n(") for ds in datasets], color=INK)
    ax.set_yticks(range(len(order)), order, color=INK)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Rank by validation accuracy (1 = best)", loc="left")
    _save(fig, "2_ranks.png")


def plot_efficiency(data):
    """Accuracy vs. parameters, one panel per dataset (each has its own input size)."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 4.2))
    for ax, ds in zip(axes, data):
        color, marker = DATASETS[ds]
        params = np.array([data[ds][m]["params"] for m in KAN_METHODS]) / 1e6
        acc = np.array([data[ds][m]["acc"] for m in KAN_METHODS])
        ax.scatter(params, acc, s=56, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_title(_ds_label(ds), loc="left")
        ax.set_xlabel("Parameters (millions)")
        ax.margins(x=0.2, y=0.2)
    axes[0].set_ylabel("Validation accuracy (best epoch)")
    fig.tight_layout()
    for ax, ds in zip(axes, data):
        params = np.array([data[ds][m]["params"] for m in KAN_METHODS]) / 1e6
        acc = np.array([data[ds][m]["acc"] for m in KAN_METHODS])
        _label_points(ax, params, acc, KAN_METHODS)
    _save(fig, "3_efficiency.png")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_tables(data, rank, order):
    datasets = list(data)
    clean = [ds for ds in datasets if ds != LEAKY]
    os.makedirs(FIG_DIR, exist_ok=True)

    header = ["model"]
    for ds in datasets:
        header += [f"{ds} acc", f"{ds} f1", f"{ds} rank"]
    header += ["mean acc (leak-free)", "mean rank (leak-free)"]
    with open(os.path.join(FIG_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in order:
            row = [m]
            for ds in datasets:
                row += [f"{data[ds][m]['acc']:.4f}", f"{data[ds][m]['f1']:.4f}", rank[ds][m]]
            row += [f"{np.mean([data[ds][m]['acc'] for ds in clean]):.4f}",
                    f"{np.mean([rank[ds][m] for ds in clean]):.1f}"]
            w.writerow(row)

    md_header = (["Model"] + [f"{_ds_label(ds)} Acc / F1 (rank)" for ds in datasets]
                 + ["Mean acc, leak-free", "Mean rank, leak-free"])
    lines = ["# 10 KAN methods on three Wi-Fi CSI datasets", "",
             "Validation accuracy / macro-F1 in %, rank by accuracy in parentheses. MultiEnv is the "
             "mean over its three environments. Means and sorting use MultiEnv and our dataset only: "
             "StanWiFi's random split puts overlapping windows of the same recordings in both "
             "training and validation, so its scores are optimistic.", "",
             "| " + " | ".join(md_header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(md_header) - 1)) + "|"]
    for m in order:
        cells = [m] + [f"{100 * data[ds][m]['acc']:.1f} / {100 * data[ds][m]['f1']:.1f} ({rank[ds][m]})"
                       for ds in datasets]
        cells += [f"**{100 * np.mean([data[ds][m]['acc'] for ds in clean]):.1f}**",
                  f"{np.mean([rank[ds][m] for ds in clean]):.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(FIG_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {FIG_DIR}/summary.csv and summary.md")


def visualize():
    """Draw the cross-dataset comparison figures and tables."""
    data = load_all()
    rank = ranks(data)
    order = model_order(data, rank)
    plot_scores(data, order)
    plot_ranks(data, rank, order)
    plot_efficiency(data)
    write_tables(data, rank, order)


# if __name__ == "__main__":
#     visualize()
