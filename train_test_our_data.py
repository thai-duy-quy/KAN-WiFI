"""
train_test_our_data.py - Train and validate the 10 KAN methods from kan_model.py on our dataset
(data/our_data.npy, data/our_data_label.npy).

Same procedure as train_test.py; only the data source, pooling and output folder differ.
Samples are (1000, 270), so time_pool=20 gives n_input = 50 * 270 = 13,500.

Usage:
    py run.py   (option 2)
    from train_test_our_data import train
    results = train(models=["wavkan", "fasterkan"], epochs=10)
"""

from read_data_our_dataset import load_data
from train_test import KAN_METHODS
from train_test import train as _train

OUT_DIR = "results_our_data"


def train(models=KAN_METHODS, time_pool=20, out_dir=OUT_DIR, **kwargs):
    """Train every model in `models` on our dataset. Other kwargs go to train_test.train()."""
    return _train(models=models, time_pool=time_pool, out_dir=out_dir, load_fn=load_data, **kwargs)


# if __name__ == "__main__":
#     train()
