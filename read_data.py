"""
read_data.py - Load the MultiEnv Wi-Fi CSI dataset (3 environments) from the `data` folder.

Files are expected in pairs, one pair per environment:
    data/data_6c_<env>.npy  <->  data/label_6c_<env>.npy        env = 1, 2, 3
    data  : (num_samples, time_steps, subcarriers)   e.g. (3000, 850, 90)
    labels: (num_samples,)                           integer class ids

Usage:
    from read_data import show_dataset
    show_dataset()                       # print shapes / class distribution of every environment
    from read_data import load_env, load_data
    X, y = load_env(2)                   # one environment
    X, y = load_data(envs=[1, 2])        # several environments concatenated
    X, y = load_data()                   # all environments concatenated
"""

import glob
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ENVS = (1, 2, 3)


def env_paths(env, data_dir=DATA_DIR):
    """(data_path, label_path) of one MultiEnv environment."""
    return (os.path.join(data_dir, f"data_6c_{env}.npy"),
            os.path.join(data_dir, f"label_6c_{env}.npy"))


def find_pairs(data_dir=DATA_DIR):
    """Return [(tag, data_path, label_path), ...] for every matching data/label pair."""
    pairs = []
    for data_path in sorted(glob.glob(os.path.join(data_dir, "data_*.npy"))):
        tag = os.path.basename(data_path)[len("data_"):-len(".npy")]
        label_path = os.path.join(data_dir, f"label_{tag}.npy")
        if os.path.exists(label_path):
            pairs.append((tag, data_path, label_path))
        else:
            print(f"[warning] no label file for {data_path}, skipped")
    return pairs


def load_pair(data_path, label_path, mmap=False, dtype=np.float32):
    """Load one data/label pair. With mmap=True the data is not read into RAM."""
    X = np.load(data_path, mmap_mode="r" if mmap else None)
    y = np.load(label_path).astype(np.int64)
    if not mmap and dtype is not None:
        X = X.astype(dtype, copy=False)
    if len(X) != len(y):
        raise ValueError(f"{data_path}: {len(X)} samples but {len(y)} labels")
    return X, y


def load_env(env, data_dir=DATA_DIR, dtype=np.float32):
    """Load one environment (1, 2 or 3). Returns (X, y)."""
    data_path, label_path = env_paths(env, data_dir)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Environment {env} not found: {data_path}")
    return load_pair(data_path, label_path, dtype=dtype)


def load_data(envs=ENVS, data_dir=DATA_DIR, dtype=np.float32):
    """Load and concatenate the given environments (default: all three). Returns (X, y)."""
    loaded = [load_env(e, data_dir, dtype) for e in envs]
    if len(loaded) == 1:
        return loaded[0]
    return np.concatenate([X for X, _ in loaded]), np.concatenate([y for _, y in loaded])


def describe(X, y, name=""):
    print(f"--- {name} ---" if name else "---")
    print(f"data   shape: {X.shape}  dtype: {X.dtype}")
    print(f"labels shape: {y.shape}  dtype: {y.dtype}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"num classes : {len(classes)}")
    for c, n in zip(classes, counts):
        print(f"  class {c}: {n:5d} samples ({100 * n / len(y):.1f}%)")
    sample = np.asarray(X[0])
    print(f"one sample  : {sample.shape}  min={sample.min():.3f}  max={sample.max():.3f}  "
          f"mean={sample.mean():.3f}")
    print(f"flattened n_input for KAN: {int(np.prod(X.shape[1:]))}")


def show_dataset():
    """Print shapes and class distribution of every MultiEnv environment."""
    total = 0
    for env in ENVS:
        data_path, label_path = env_paths(env)
        if not os.path.exists(data_path):
            print(f"[warning] environment {env} missing: {data_path}")
            continue
        X, y = load_pair(data_path, label_path, mmap=True)  # fast: header + labels only
        describe(X, y, name=f"MultiEnv environment {env}")
        total += len(y)
    print(f"\nTotal samples over all environments: {total}")


# if __name__ == "__main__":
#     show_dataset()
