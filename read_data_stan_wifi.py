"""
read_data_stan_wifi.py - Load the StanWiFi Wi-Fi CSI dataset from the `data` folder.

Files:
    data/data_amp_2000.npy  data  : (num_samples, time_steps, subcarriers)   e.g. (2385, 2000, 90)
    data/label_2000.npy     labels: (num_samples,)                           integer class ids

Usage:
    from read_data_stan_wifi import show_dataset
    show_dataset()                            # print shapes / class distribution
    from read_data_stan_wifi import load_data
    X, y = load_data()
"""

import os

import numpy as np

from read_data import DATA_DIR, describe, load_pair

DATA_FILE = os.path.join(DATA_DIR, "data_amp_2000.npy")
LABEL_FILE = os.path.join(DATA_DIR, "label_2000.npy")


def load_data(data_path=DATA_FILE, label_path=LABEL_FILE, dtype=np.float32):
    """Load StanWiFi into memory. Returns (X, y)."""
    return load_pair(data_path, label_path, dtype=dtype)


def show_dataset():
    """Print shapes and class distribution of StanWiFi."""
    X, y = load_pair(DATA_FILE, LABEL_FILE, mmap=True)  # fast: header + labels only
    describe(X, y, name="StanWiFi")


# if __name__ == "__main__":
#     show_dataset()
