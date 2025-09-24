from pathlib import Path
import pandas as pd
import math

# --- TOML reader: Python 3.11+ has tomllib; else fall back to tomli ---
try:
    import tomllib  # Py 3.11+
    def load_toml(fp): return tomllib.load(fp)
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli for Py<3.11
    def load_toml(fp): return tomllib.load(fp)

logdir = Path("2-hypertuning-mlflow\modellogs")

# Helper: flatten nested dicts using dot-notation keys
def flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten(v, key))
        else:
            # stringify lists to keep columns simple (feel free to change)
            if isinstance(v, list):
                flat[key] = ",".join(map(str, v))
            else:
                flat[key] = v
    return flat

# Try to extract a numeric "final metric" from files in the run dir.
# Looks for common names like Accuracy / val_accuracy / valid_Accuracy / val_acc.
POSSIBLE_METRIC_COLUMNS = [
    "Accuracy", "accuracy", "val_accuracy", "valid_accuracy",
    "val_Accuracy", "valid_Accuracy", "val_acc", "ACC", "acc"
]

def try_read_final_metric(rundir: Path):
    # 1) Look for CSVs first (e.g., history.csv, metrics.csv, trainer logs)
    for csv in rundir.glob("*.csv"):
        try:
            df = pd.read_csv(csv)
            # prefer "val_*" columns if present
            cols = [c for c in df.columns if any(c.lower() == m.lower() for m in POSSIBLE_METRIC_COLUMNS)]
            if not cols:
                # heuristic: any column containing both "val" and "acc"
                cols = [c for c in df.columns if "val" in c.lower() and "acc" in c.lower()]
            if not cols:
                # fallback: any column with "acc" or "accuracy"
                cols = [c for c in df.columns if "acc" in c.lower() or "accuracy" in c.lower()]
            if cols:
                # choose the last row (final epoch); take the first matching column
                val = df[cols[0]].iloc[-1]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    return float(val), cols[0], csv.name
        except Exception:
            pass

    # 2) Look for metrics in any TOML files (other than the two config TOMLs)
    for toml_file in rundir.glob("*.toml"):
        if toml_file.name in {"model.toml", "settings.toml"}:
            continue
        try:
            with open(toml_file, "rb") as f:
                data = load_toml(f)
            flat = flatten(data)
            # same column heuristics as above
            for k, v in flat.items():
                if any(k.lower().endswith(m.lower()) for m in POSSIBLE_METRIC_COLUMNS) or \
                   "acc" in k.lower() or "accuracy" in k.lower():
                    try:
                        fv = float(v)
                        if not math.isnan(fv):
                            return fv, k, toml_file.name
                    except Exception:
                        continue
        except Exception:
            pass

    return None, None, None  # no metric found

records = []

for rundir in sorted([p for p in logdir.iterdir() if p.is_dir()]):
    row = {"run": rundir.name}

    model_toml = rundir / "model.toml"
    settings_toml = rundir / "settings.toml"

    if model_toml.exists():
        with open(model_toml, "rb") as f:
            model_cfg = load_toml(f)
        row.update(flatten(model_cfg, "model"))

    if settings_toml.exists():
        with open(settings_toml, "rb") as f:
            settings_cfg = load_toml(f)
        row.update(flatten(settings_cfg, "settings"))

    # optional: pull a numeric metric if available
    metric_val, metric_col, metric_file = try_read_final_metric(rundir)
    row["final_metric"] = metric_val
    row["final_metric_source"] = metric_col
    row["final_metric_file"] = metric_file

    records.append(row)

df = pd.DataFrame.from_records(records).sort_values(["final_metric"], ascending=[False], na_position="last")

# save and show a quick preview
out_path = "modellogs_summary.csv"
df.to_csv(out_path, index=False)

print(f"\nSaved summary with {len(df)} runs to {out_path}\n")
print(df.head(10).to_string(index=False))

# If we found any metric at all, print the best run succinctly
if df["final_metric"].notna().any():
    best = df.iloc[0]
    print("\nBest run:")
    print(f"  run: {best['run']}")
    print(f"  final_metric: {best['final_metric']} (from {best['final_metric_source']} in {best['final_metric_file']})")
    # a few handy hyperparams to display (adapt to your columns)
    for col in [
        "model.conv_blocks", "model.base_filters", "model.filters_growth",
        "model.norm", "model.hidden_units", "model.dropout_p",
        "settings.model.optimizer_kwargs.lr", "settings.model.optimizer_kwargs.weight_decay",
        "settings.model.epochs"
    ]:
        if col in best and pd.notna(best[col]):
            print(f"  {col}: {best[col]}")
else:
    print("\nNo numeric metric found in run folders. You can still sort/filter by hyperparams in the CSV.")
