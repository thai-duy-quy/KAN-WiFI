"""
visualization_cross.py - Compare the 10 KAN methods in the cross-environment setting of MultiEnv
(train_test_cross_multienv.py: train on 80% of one environment, test on 20% of each other one).

Inputs:
    results/MultiEnv_cross/summary.json      selected-epoch metrics for every (train env, model)
    results/MultiEnv_cross/train<N>/log.txt  per-epoch accuracy on validation and test environments

Outputs (results/MultiEnv_cross/figures/):
    1_same_vs_cross.png    same-environment vs. cross-environment accuracy and macro-F1 per model
    2_settings.png         accuracy of every model in every train -> test setting
    3_transfer_matrix.png  3x3 train-env x test-env accuracy matrix, one panel per model
    4_curves.png           same-env and cross-env accuracy per epoch, averaged over training envs

Usage:
    from visualization_cross import visualize; visualize()   (also run by run.py, option 4)
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

# Importing visualization applies the shared matplotlib style.
from visualization import GRID, INK, INK_2, INK_MUTED, SEQ_BLUE, SERIES_1, SERIES_2, SURFACE

CROSS_DIR = os.path.join("results", "MultiEnv_cross")
FIG_DIR = os.path.join(CROSS_DIR, "figures")
KAN_METHODS = ["efficientkan", "fastkan", "fasterkan", "bsrbf_kan", "chebykan",
               "jacobikan", "fourierkan", "wavkan", "relukan", "rationalkan"]
MAJORITY_ACC = 0.40     # always predicting class 0 (40% of every MultiEnv environment)
MAJORITY_F1 = 0.0952    # macro-F1 of that trivial predictor: F1(class 0) = 0.571, others 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_summary():
    """{train_env: {model: metrics}} with keys like 'env1'."""
    with open(os.path.join(CROSS_DIR, "summary.json")) as f:
        return json.load(f)


def settings(summary):
    """[(train_env, test_env), ...] for the cross-environment settings, e.g. ('env1', 'env2')."""
    out = []
    for tr in sorted(summary):
        out += [(tr, k[len("test_"):-len("_acc")]) for k in summary[tr][KAN_METHODS[0]]
                if k.startswith("test_") and k.endswith("_acc")]
    return out


def same_cross(summary, model, metric="acc"):
    """(mean same-env score, mean cross-env score) of one model."""
    same = np.mean([summary[tr][model][f"val_{metric}"] for tr in summary])
    cross = np.mean([summary[tr][model][f"test_{te}_{metric}"] for tr, te in settings(summary)])
    return same, cross


def parse_logs(summary):
    """{model: {"same": [per-epoch acc averaged over train envs], "cross": [...]}}."""
    curves = {m: {"same": [], "cross": []} for m in KAN_METHODS}
    per_env = {m: [] for m in KAN_METHODS}
    for tr in sorted(summary):
        model = None
        with open(os.path.join(CROSS_DIR, f"train{tr[3:]}", "log.txt"), encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^===== (\S+)", line.strip())
                if m:
                    model = m.group(1)
                    per_env[model].append({"same": [], "cross": []})
                    continue
                if model and line.startswith("epoch"):
                    vals = dict(re.findall(r"(\w+)_acc=([\d.]+)", line))
                    per_env[model][-1]["same"].append(float(vals["val"]))
                    per_env[model][-1]["cross"].append(
                        np.mean([float(v) for k, v in vals.items() if k.startswith("env")]))
    for m in KAN_METHODS:
        curves[m]["same"] = np.mean([run["same"] for run in per_env[m]], axis=0)
        curves[m]["cross"] = np.mean([run["cross"] for run in per_env[m]], axis=0)
    return curves


def _order(summary):
    """Models sorted by mean cross-environment accuracy, best first."""
    return sorted(KAN_METHODS, key=lambda m: -same_cross(summary, m)[1])


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_same_vs_cross(summary):
    """Dumbbells: same-env (blue) vs. cross-env (orange), accuracy and macro-F1 panels."""
    names = _order(summary)[::-1]  # best cross-env at the top
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, metric, title, trivial in [(axes[0], "acc", "Accuracy", MAJORITY_ACC),
                                       (axes[1], "f1", "Macro-F1", MAJORITY_F1)]:
        same, cross = np.array([same_cross(summary, m, metric) for m in names]).T
        ax.hlines(y, cross, same, color=GRID, linewidth=2, zorder=1)
        ax.scatter(same, y, s=56, color=SERIES_1, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label="Same environment (validation)")
        ax.scatter(cross, y, s=56, marker="s", color=SERIES_2, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label="Other environments (test)")
        for yi, s_, c_ in zip(y, same, cross):
            ax.annotate(f"{100 * c_:.1f}", (c_, yi), xytext=(-8, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8, color=INK_2)
            ax.annotate(f"{100 * s_:.1f}", (s_, yi), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.axvline(trivial, color=INK_MUTED, linewidth=1, linestyle="--", zorder=0)
        ax.annotate("always predict class 0", (trivial, len(names) - 0.4), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK_MUTED, va="center")
        ax.set_xlim(0, 1.05)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mean over the 3 training environments")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=2)
    fig.suptitle("Same-environment vs. cross-environment performance (MultiEnv)", x=0.01, y=0.995,
                 ha="left", fontsize=12, fontweight="bold", color=INK)
    _save(fig, "1_same_vs_cross.png")


def plot_settings(summary):
    """Heatmap: model x (train -> test) setting, plus the mean, accuracy in each cell."""
    names = _order(summary)
    sets = settings(summary)
    cols = [f"{tr[3:]}→{te[3:]}" for tr, te in sets] + ["mean"]
    mat = np.array([[summary[tr][m][f"test_{te}_acc"] for tr, te in sets]
                    + [same_cross(summary, m)[1]] for m in names])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.imshow(mat, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{100 * mat[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if j == mat.shape[1] - 1 else "normal",
                    color=SURFACE if mat[i, j] > 0.55 else INK_2)
    ax.axvline(len(sets) - 0.5, color=SURFACE, linewidth=4)
    ax.set_xticks(range(len(cols)), cols, color=INK)
    ax.set_yticks(range(len(names)), names, color=INK)
    ax.set_xlabel("Training env → test env")
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cross-environment test accuracy (%)", loc="left")
    _save(fig, "2_settings.png")


def plot_transfer_matrices(summary):
    """One 3x3 matrix per model: rows = training env, columns = test env, diagonal = same env."""
    names = _order(summary)
    envs = sorted(summary)
    ncols = 5
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, names):
        mat = np.array([[summary[tr][m]["val_acc"] if tr == te else summary[tr][m][f"test_{te}_acc"]
                         for te in envs] for tr in envs])
        ax.imshow(mat, cmap=SEQ_BLUE, vmin=0, vmax=1)
        for i in range(len(envs)):
            for j in range(len(envs)):
                ax.text(j, i, f"{100 * mat[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        fontweight="bold" if i == j else "normal",
                        color=SURFACE if mat[i, j] > 0.55 else INK_2)
        ax.set_title(f"{m}  ({100 * same_cross(summary, m)[1]:.1f})", loc="left", fontsize=10)
        ax.set_xticks(range(len(envs)), [e[3:] for e in envs])
        ax.set_yticks(range(len(envs)), [e[3:] for e in envs])
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("Test environment")
    for ax in axes[::ncols]:
        ax.set_ylabel("Training environment")
    fig.suptitle("Accuracy (%) per training/test environment; diagonal = same environment, "
                 "title = mean cross-env accuracy", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "3_transfer_matrix.png")


def plot_curves(summary):
    """Per-epoch same-env vs. cross-env accuracy, averaged over the 3 training environments."""
    curves = parse_logs(summary)
    names = _order(summary)
    ncols = 5
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, names):
        epochs = np.arange(1, len(curves[m]["same"]) + 1)
        ax.plot(epochs, curves[m]["same"], color=SERIES_1, label="Same environment (validation)")
        ax.plot(epochs, curves[m]["cross"], color=SERIES_2, label="Other environments (test)")
        ax.axhline(MAJORITY_ACC, color=INK_MUTED, linewidth=1, linestyle="--", zorder=0)
        peak = int(np.argmax(curves[m]["cross"]))
        ax.scatter([epochs[peak]], [curves[m]["cross"][peak]], s=40, color=SERIES_2,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"peak {100 * curves[m]['cross'][peak]:.1f} @ ep {epochs[peak]}",
                    (epochs[peak], curves[m]["cross"][peak]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK_2)
        ax.set_title(m, loc="left", fontsize=10)
        ax.set_ylim(0, 1.0)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("Epoch")
    for ax in axes[::ncols]:
        ax.set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("Accuracy per epoch, mean over training environments (dashed: always class 0)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "4_curves.png")


def visualize():
    """Draw the cross-environment comparison figures."""
    summary = load_summary()
    plot_same_vs_cross(summary)
    plot_settings(summary)
    plot_transfer_matrices(summary)
    plot_curves(summary)


# if __name__ == "__main__":
#     visualize()
