"""
run.py - Entry point of KAN-WiFi: train and evaluate the 10 KAN methods from a menu.

    py run.py

Options
    1  MultiEnv  : train + validate on each of the 3 environments -> results/MultiEnv/
    2  MINE lab  : train + validate on our dataset                 -> results_our_data/
    3  StanWiFi  : train + validate on StanWiFi                    -> results_stan_wifi/
    4  Cross-domain (MultiEnv): train on one environment, test on the other two
                                                                   -> results/MultiEnv_cross/
    5  Exit

Each option also draws its figures. After options 1-3 the comparison across the three datasets
(results/figures_all/) is refreshed once all three have results.

Do not run two copies at the same time on the same option: both write to the same log files.
"""

import os
import time
import traceback

KAN_METHODS_TEXT = ("EfficientKAN, FastKAN, FasterKAN, BSRBF-KAN, ChebyKAN, JacobiKAN, "
                    "FourierKAN, WavKAN, ReLU-KAN, RationalKAN")

MENU = """
================== KAN-WiFi ==================
 10 KAN methods: {methods}

  1. MultiEnv  - train on each of the 3 environments   (~8 min)
  2. MINE lab  - train on our dataset                  (~3 min)
  3. StanWiFi  - train on StanWiFi                     (~3 min)
  4. Cross-domain - MultiEnv: train on one environment,
                    test on the other two              (~9 min)
  5. Exit
==============================================="""

RESULT_FILES = {  # results each dataset option produces; used to refresh the cross-dataset figures
    "MultiEnv": os.path.join("results", "MultiEnv", "summary.json"),
    "MINE lab": os.path.join("results_our_data", "results.json"),
    "StanWiFi": os.path.join("results_stan_wifi", "results.json"),
}


def run_multienv():
    from train_test import train_multienv
    from visualization import visualize_multienv
    train_multienv()
    visualize_multienv()


def run_mine_lab():
    from train_test_our_data import train
    from visualization_our import visualize
    train()
    visualize()


def run_stan_wifi():
    from train_test_stan_wifi import train
    from visualization_stan_wifi import visualize
    print("Note: StanWiFi windows overlap, so its random train/validation split is optimistic.")
    train()
    visualize()


def run_cross_domain():
    from train_test_cross_multienv import train_cross_all
    from visualization_cross import visualize
    train_cross_all()
    visualize()


def refresh_comparison():
    """Redraw results/figures_all/ when all three datasets have results."""
    missing = [name for name, path in RESULT_FILES.items() if not os.path.exists(path)]
    if missing:
        print(f"\n(Cross-dataset comparison skipped: no results yet for {', '.join(missing)}.)")
        return
    from visualization_all import visualize
    print("\n=== Comparison across the three datasets ===")
    visualize()


OPTIONS = {
    "1": ("MultiEnv", run_multienv, True),
    "2": ("MINE lab", run_mine_lab, True),
    "3": ("StanWiFi", run_stan_wifi, True),
    "4": ("Cross-domain (MultiEnv)", run_cross_domain, False),
}


def main():
    while True:
        print(MENU.format(methods=KAN_METHODS_TEXT))
        try:
            choice = input("Select an option [1-5]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if choice == "5":
            print("Bye.")
            return
        if choice not in OPTIONS:
            print(f"'{choice}' is not a valid option. Please enter a number from 1 to 5.")
            continue

        name, action, is_dataset = OPTIONS[choice]
        print(f"\n>>> {name}: running the 10 KAN methods ...")
        start = time.time()
        try:
            action()
            if is_dataset:
                refresh_comparison()
            print(f"\n>>> {name}: done in {(time.time() - start) / 60:.1f} min.")
        except KeyboardInterrupt:
            print(f"\n>>> {name}: interrupted. Results of this run may be incomplete.")
        except FileNotFoundError as e:
            print(f"\n>>> {name}: data file not found - {e}\n    Put the .npy files in the data/ folder.")
        except Exception:
            traceback.print_exc()
            print(f"\n>>> {name}: failed (see the error above).")


if __name__ == "__main__":
    main()
