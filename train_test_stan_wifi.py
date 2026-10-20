"""
train_test_stan_wifi.py - Train and validate the 10 KAN methods from kan_model.py on the StanWiFi
dataset (data/data_amp_2000.npy, data/label_2000.npy).

Same procedure as train_test.py; only the data source, pooling and output folder differ.
Samples are (2000, 90), so time_pool=20 gives n_input = 100 * 90 = 9,000.

Usage:
    py run.py   (option 3)
    from train_test_stan_wifi import train
    results = train(models=["wavkan", "fasterkan"], epochs=10)
"""

from read_data_stan_wifi import load_data
from train_test import KAN_METHODS
from train_test import train as _train

OUT_DIR = "results_stan_wifi"


def train(models=KAN_METHODS, time_pool=20, out_dir=OUT_DIR, **kwargs):
    """Train every model in `models` on StanWiFi. Other kwargs go to train_test.train()."""
    return _train(models=models, time_pool=time_pool, out_dir=out_dir, load_fn=load_data, **kwargs)


# if __name__ == "__main__":
#     train()
