# KAN-WiFi: Benchmarking Kolmogorov–Arnold Networks for Wi-Fi CSI-Based Human Activity Recognition

## Introduction
KAN-WiFi is a benchmark library for Wi-Fi CSI human activity recognition (HAR) with **Kolmogorov–Arnold Networks (KANs)**. It implements ten KAN variants that differ only in the basis of their learnable edge functions — B-splines, radial basis functions, Chebyshev and Jacobi polynomials, Fourier series, wavelets, squared-ReLU bumps and rational functions — and trains them under one identical pipeline on CSI amplitude from three datasets (MultiEnv, StanWiFi and our MINE lab dataset). Besides within-dataset accuracy, it provides a **leak-free cross-environment evaluation** on MultiEnv and an analysis of train/validation overlap in sliding-window datasets. It is implemented in PyTorch. This is the code of our paper *KAN-WiFi: Benchmarking Kolmogorov–Arnold Networks for Wi-Fi CSI-Based Human Activity Recognition*.

![KAN-WiFi pipeline](images/pipeline.png)

```
@misc{thai2026kanwifi,
  title  = {KAN-WiFi: Benchmarking Kolmogorov--Arnold Networks for Wi-Fi CSI-Based Human Activity Recognition},
  author = {Duy-Quy Thai and Hoang-Thang Ta and Ervin Yohannes},
  year   = {2026},
  url    = {https://github.com/thai-duy-quy/KAN-WiFI}
}
```

### The ten KAN methods
| Method | Basis of the edge function | Reference |
|---|---|---|
| EfficientKAN | Cubic B-splines + SiLU | [Liu et al., 2024](https://arxiv.org/abs/2404.19756), [efficient-kan](https://github.com/Blealtan/efficient-kan) |
| FastKAN | Gaussian radial basis functions | [Li, 2024](https://arxiv.org/abs/2405.06721) |
| FasterKAN | Reflectional switch (sech²) | [faster-kan](https://github.com/AthanasiosDelis/faster-kan) |
| BSRBF-KAN | B-splines + Gaussian RBF | [Ta, 2024](https://arxiv.org/abs/2406.11173) |
| ChebyKAN | Chebyshev polynomials | [SS et al., 2024](https://arxiv.org/abs/2405.07200) |
| JacobiKAN | Jacobi polynomials | [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN) |
| FourierKAN | Fourier series | [FourierKAN](https://github.com/GistNoesis/FourierKAN) |
| WavKAN | Mexican-hat wavelet | [Bozorgasl & Chen, 2024](https://arxiv.org/abs/2405.12832) |
| ReLU-KAN | Squared ReLU bumps | [Qiu et al., 2024](https://arxiv.org/abs/2406.02075) |
| RationalKAN | Safe Padé (rational) functions | [Aghaei, 2024](https://arxiv.org/abs/2406.14495) |

An MLP baseline is included for comparison. All models share one interface in `kan_model.py`:
```python
from kan_model import get_model
model = get_model("wavkan", [n_input, 64, 6])   # logits = model(x), x: (batch, n_input)
```

## Requirements

1. Install `pytorch` and CUDA (we use `torch=2.1.2` and `cu121`, Python 3.9).
2. `pip install -r requirements.txt`

All paths are built with `os.path`, so the code runs on Windows (where it was developed, with an NVIDIA GeForce RTX 3060 12 GB) as well as on Linux.

## Run
### Download Processed Data
Original data can be downloaded at:
1. [MultiEnv](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments) (also on [Mendeley Data](https://data.mendeley.com/datasets/v38wjmz6f6/1)): collected in three scenarios — line-of-sight (LOS) in an office (E1) and a hall (E2), and non-line-of-sight (NLOS, E3).
2. [StanWiFi](https://github.com/ermongroup/Wifi_Activity_Recognition): continuous CSI data for six activities without precise segmentation timestamps for each sample.
3. MINE lab: collected by our team; available from the authors on request.

We use the CSI amplitude processed as in [PA-CSI](https://github.com/thai-duy-quy/PA-CSI-HAR) (time alignment and sliding windows). Processed data can be downloaded at: [PA-CSI dataset](https://drive.google.com/drive/folders/1fiJBDWDC3WKkD5pLYRwpLCWMXPVzCQ-i?usp=sharing)

Put the files in the `data/` folder:

| Dataset | Files | Shape (samples, packets, streams) |
|---|---|---|
| MultiEnv E1 / E2 / E3 | `data_6c_1.npy`, `data_6c_2.npy`, `data_6c_3.npy` + `label_6c_{1,2,3}.npy` | (3000, 850, 90) each |
| MINE lab | `our_data.npy` + `our_data_label.npy` | (1017, 1000, 270) |
| StanWiFi | `data_amp_2000.npy` + `label_2000.npy` | (2385, 2000, 90) |

### Run the benchmark
```
py run.py          # or: python run.py
```
Then select an option:
```
  1. MultiEnv  - train on each of the 3 environments   (~8 min)
  2. MINE lab  - train on our dataset                  (~3 min)
  3. StanWiFi  - train on StanWiFi                     (~3 min)
  4. Cross-domain - MultiEnv: train on one environment,
                    test on the other two              (~9 min)
  5. Exit
```
Each option trains the ten KANs (AdamW, lr 1e-3, cosine schedule, 30 epochs, batch size 64, stratified 80/20 split, seed 42), saves the best weights and per-epoch logs, and draws the figures (scores, learning curves, efficiency and confusion matrices). Times are for an RTX 3060.

| Option | Results | Figures |
|---|---|---|
| 1 | `results/MultiEnv/env{1,2,3}/`, `results/MultiEnv/summary.csv` | `results/MultiEnv/env*/figures/`, `results/MultiEnv/figures/` |
| 2 | `results_our_data/` | `results_our_data/figures_our/` |
| 3 | `results_stan_wifi/` | `results_stan_wifi/figures_stan_wifi/` |
| 4 | `results/MultiEnv_cross/train{1,2,3}/`, `results/MultiEnv_cross/summary.md` | `results/MultiEnv_cross/figures/` |

After options 1–3 have all been run, the comparison across the three datasets is saved in `results/figures_all/`. Do not run two copies of the same option at the same time: they write to the same log files.

### Cross-domain protocol
For a training environment E<sub>a</sub>, the model is trained on 80% of E<sub>a</sub>, the best epoch is selected on the other 20% of E<sub>a</sub>, and it is tested on 20% of each other environment E<sub>b</sub>. The test environments are never used for training, model selection or normalization statistics.

## Results
Validation accuracy (%) of the best epoch. MultiEnv: mean over E1–E3. Cross-domain: mean test accuracy over the six E<sub>a</sub>→E<sub>b</sub> settings.

| Model | MultiEnv | MINE lab† | StanWiFi† | Cross-domain |
|---|---:|---:|---:|---:|
| EfficientKAN | 92.50 | 83.33 | 99.79 | 33.7 |
| FastKAN | 92.28 | 79.90 | 100.00 | 31.8 |
| FasterKAN | 93.33 | 85.29 | 100.00 | **33.8** |
| BSRBF-KAN | 92.33 | 84.80 | 100.00 | 32.6 |
| ChebyKAN | 91.06 | 69.12 | 99.37 | 28.0 |
| JacobiKAN | 91.39 | 83.82 | 99.79 | 29.8 |
| FourierKAN | 90.28 | 69.12 | 100.00 | 28.7 |
| WavKAN | **93.61** | **86.76** | 100.00 | 29.1 |
| ReLU-KAN | 92.33 | 86.27 | 99.58 | 33.4 |
| RationalKAN | 91.39 | 80.39 | 99.37 | 33.1 |
| MLP (baseline) | 89.44 | 84.31 | 98.11 | 37.4 |

† MINE lab and StanWiFi samples are overlapping sliding windows, so a random window-level split places most of each validation window in the training set (86.8% and 91.5% of its packets on average); their scores are optimistic. MultiEnv samples are separate trials and have no overlap.

![Scores on the three datasets](images/scores_all_datasets.png)

Within an environment, WavKAN is the most accurate and smallest KAN. Across environments, every model collapses to 28–37% accuracy — below the 40% of always predicting the majority class — and the MLP transfers best:

![Same-environment vs. cross-environment performance](images/same_vs_cross.png)

## Project structure
```
kan_model.py                   10 KAN variants + MLP, get_model()
read_data*.py                  loaders for MultiEnv, MINE lab and StanWiFi
train_test.py                  train() and train_multienv()          (option 1)
train_test_our_data.py         MINE lab                              (option 2)
train_test_stan_wifi.py        StanWiFi                              (option 3)
train_test_cross_multienv.py   train_cross() and train_cross_all()   (option 4)
visualization*.py              figures for each option and the cross-dataset comparison
run.py                         menu entry point
paper/                         LaTeX source of the paper
```
