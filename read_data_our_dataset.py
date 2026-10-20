"""
read_data_our_dataset.py - Load our own Wi-Fi CSI dataset from the `data` folder.

Files:
    data/our_data.npy        data  : (num_samples, time_steps, subcarriers)   e.g. (1017, 1000, 270)
    data/our_data_label.npy  labels: (num_samples,)                           integer class ids

Usage:
    from read_data_our_dataset import show_dataset
    show_dataset()                              # print shapes / class distribution
    from read_data_our_dataset import load_data
    X, y = load_data()
"""

import os

import numpy as np

from read_data import DATA_DIR, describe, load_pair

DATA_FILE = os.path.join(DATA_DIR, "our_data.npy")
LABEL_FILE = os.path.join(DATA_DIR, "our_data_label.npy")


def load_data(data_path=DATA_FILE, label_path=LABEL_FILE, dtype=np.float32):
    """Load our dataset into memory. Returns (X, y)."""
    return load_pair(data_path, label_path, dtype=dtype)


def show_dataset():
    """Print shapes and class distribution of our (MINE lab) dataset."""
    X, y = load_pair(DATA_FILE, LABEL_FILE, mmap=True)  # fast: header + labels only
    describe(X, y, name="our_data")


# if __name__ == "__main__":
#     show_dataset()
