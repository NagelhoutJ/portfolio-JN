from __future__ import annotations

from pathlib import Path
import tomllib

import os

os.environ["RAY_TMPDIR"] = r"C:\raytmp"
os.environ["TMP"] = r"C:\raytmp"
os.environ["TEMP"] = r"C:\raytmp"


from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch

from .trainable import train_tune


def load_base_config(path: str) -> dict:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = {}
    cfg.update(raw.get("dataset", {}))
    cfg.update(raw.get("training", {}))
    cfg.update(raw.get("model", {}))
    return cfg


def short_trial_dirname(trial: "tune.ExperimentTrial") -> str:
    # Super korte naam, uniek per trial
    # trial.trial_id is al kort (bv. "a1b2c3d4")
    return f"t_{trial.trial_id}"


def main() -> None:
    base = load_base_config("config/base.toml")

    # Zet dit ook kort en buiten je diepe projectpad:
    out_dir = r"C:\ray_results"

    search_space = {
        **base,
        "epochs": 6,
        "lr": tune.loguniform(1e-4, 3e-2),
        "optimizer": tune.choice(["adam", "sgd"]),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "num_blocks": tune.choice([2, 3, 4]),
        "base_channels": tune.choice([16, 32, 48]),
        "kernel_size": tune.choice([3, 5]),
        "use_batchnorm": tune.choice([True, False]),
        "dropout": tune.uniform(0.0, 0.5),
        "mlp_hidden": tune.choice([128, 256, 384, 512]),
    }

    reporter = CLIReporter(metric_columns=["val_acc", "val_loss", "epoch"])

    tuner = tune.Tuner(
        train_tune,
        tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            num_samples=25,
            search_alg=HyperOptSearch(),
            trial_dirname_creator=short_trial_dirname,   # <<< dit is de fix
        ),
        run_config=tune.RunConfig(
            name="c10p1",                 # <<< kort!
            storage_path=out_dir,         # <<< kort pad
            progress_reporter=reporter,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best = results.get_best_result(metric="val_acc", mode="max")
    print("BEST CONFIG:", best.config)
    print("BEST val_acc:", best.metrics.get("val_acc"))


if __name__ == "__main__":
    main()