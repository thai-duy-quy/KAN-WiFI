"""
visualization_stan_wifi.py - Visualize and compare the 10 KAN methods trained on StanWiFi
by train_test_stan_wifi.py. Same figures as visualization.py.

Inputs:
    results_stan_wifi/results.json   best-epoch metrics per model
    results_stan_wifi/log.txt        per-epoch log
    results_stan_wifi/<model>.pt     best weights (only needed for confusion matrices)

Outputs (results_stan_wifi/figures_stan_wifi/):
    1_scores.png, 2_accuracy_curves.png, 3_loss_curves.png, 4_efficiency.png, 5_confusion.png, summary.csv

Usage:
    from visualization_stan_wifi import visualize
    visualize()                  # also run by run.py, option 3
    visualize(confusion=False)
"""

import os

from read_data_stan_wifi import load_data
from visualization import run

RESULTS_DIR = "results_stan_wifi"
LOG_FILE = os.path.join(RESULTS_DIR, "log.txt")
FIG_DIR = os.path.join(RESULTS_DIR, "figures_stan_wifi")
TIME_POOL = 20  # must match train_test_stan_wifi.train()

def visualize(confusion=True):
    """Draw every figure for StanWiFi. confusion=False skips the step that needs data + GPU."""
    run(results_dir=RESULTS_DIR, log_file=LOG_FILE, fig_dir=FIG_DIR, load_fn=load_data,
        time_pool=TIME_POOL, confusion=confusion)


# if __name__ == "__main__":
#     visualize()
