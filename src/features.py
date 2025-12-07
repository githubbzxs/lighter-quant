from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import joblib
from src.config import Config


def _compute_lags(df: pd.DataFrame, cols, lags):
    for col in cols:
        for l in lags:
            df[f"{col}_lag_{l}"] = df[col].shift(l)
    return df


def _compute_roll(df: pd.DataFrame, cols, windows):
    for col in cols:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
    return df

def build_dataset(config: Config, logger) -> Tuple[np.ndarray, np.ndarray]:
    input_paths = [Path(p) for p in config.dataset.input_paths]
    frames = []
    for p in input_paths:
        if not p.exists():
            logger.warning("Input path %s not found, skipping", p)
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("No dataset inputs found")
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("local_time", inplace=True)

    df["spread"] = df["best_ask"] - df["best_bid"]
    df["rel_spread"] = df["spread"] / df["mid"]
    for depth in config.dataset.agg_depths:
        df[f"imbalance_{depth}"] = (df[f"bid_vol_top_{depth}"] - df[f"ask_vol_top_{depth}"]) / (
            df[f"bid_vol_top_{depth}"] + df[f"ask_vol_top_{depth}"] + 1e-9
        )

    df["ret_1"] = df["mid"].pct_change(1)
    df["ret_5"] = df["mid"].pct_change(5)
    df["ret_10"] = df["mid"].pct_change(10)
    df["ret_50"] = df["mid"].pct_change(50)

    df = _compute_lags(df, ["ret_1", "ret_5", "ret_10", "ret_50", "spread", "rel_spread"], config.dataset.lag_steps)
    df = _compute_roll(df, ["ret_1", "spread"], [5, 10, 20])

    future_h = config.dataset.future_horizon
    df["future_ret"] = df["mid"].shift(-future_h) / df["mid"] - 1
    if config.dataset.label_mode == "binary":
        df["label"] = (df["future_ret"] > config.dataset.up_threshold).astype(int)
    else:
        def label_func(x):
            if x > config.dataset.up_threshold:
                return 1
            if x < config.dataset.down_threshold:
                return -1
            return 0
        df["label"] = df["future_ret"].apply(label_func)

    df.dropna(inplace=True)
    feature_cols = [c for c in df.columns if c not in {"label", "future_ret"} and not c.startswith("ask_") and not c.startswith("bid_")]
    X = df[feature_cols].values
    y = df["label"].values

    Path(config.paths.cache_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(config.dataset.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X": X, "y": y, "features": feature_cols}, out_path)
    logger.info("Built dataset: %s samples, %s features -> %s", X.shape[0], X.shape[1], out_path)
    return X, y
