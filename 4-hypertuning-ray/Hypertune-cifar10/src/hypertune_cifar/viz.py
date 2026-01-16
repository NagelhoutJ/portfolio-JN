from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ResultGrid


def results_to_df(results: ResultGrid) -> pd.DataFrame:
    df = results.get_dataframe()
    # Houd het schoon: alleen nuttige kolommen
    keep = [c for c in df.columns if c.startswith("config/")] + ["val_acc", "val_loss", "train_loss", "epoch"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def scatter_lr_vs_acc(df: pd.DataFrame) -> None:
    x = df["config/lr"]
    y = df["val_acc"]
    plt.figure()
    plt.xscale("log")
    plt.scatter(x, y)
    plt.xlabel("learning rate (log)")
    plt.ylabel("val_acc")
    plt.title("LR vs validation accuracy")
    plt.show()
