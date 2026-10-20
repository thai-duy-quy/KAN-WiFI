"""
visualization_our.py - Visualize and compare the 10 KAN methods trained on our dataset
by train_test_our_data.py. Same figures as visualization.py.

Inputs:
    results_our_data/results.json   best-epoch metrics per model
    results_our_data_log.txt        per-epoch log (`py train_test_our_data.py > results_our_data_log.txt`)
    results_our_data/<model>.pt     best weights (only needed for confusion matrices)

Outputs (results_our_data/figures_our/):
    1_scores.png, 2_accuracy_curves.png, 3_loss_curves.png, 4_efficiency.png, 5_confusion.png, summary.csv

Usage:
    from visualization_our import visualize
    visualize()                  # also run by run.py, option 2
    visualize(confusion=False)
"""

import os

from read_data_our_dataset import load_data
from visualization import run

RESULTS_DIR = "results_our_data"
LOG_FILE = "results_our_data_log.txt"
FIG_DIR = os.path.join(RESULTS_DIR, "figures_our")
TIME_POOL = 20  # must match train_test_our_data.train()

def visualize(confusion=True):
    """Draw every figure for our dataset. confusion=False skips the step that needs data + GPU."""
    # Runs since train_test.py writes its own log use results_our_data/log.txt; older runs used LOG_FILE.
    log_file = os.path.join(RESULTS_DIR, "log.txt")
    if not os.path.exists(log_file):
        log_file = LOG_FILE
    run(results_dir=RESULTS_DIR, log_file=log_file, fig_dir=FIG_DIR, load_fn=load_data,
        time_pool=TIME_POOL, confusion=confusion)


# if __name__ == "__main__":
#     visualize()
