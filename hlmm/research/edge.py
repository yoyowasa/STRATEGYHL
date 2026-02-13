from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")  # 非GUI環境での描画


def _to_array(data: Iterable[object]) -> np.ndarray:
    arr = np.array(list(data), dtype=float)
    arr = arr.astype(float)
    return arr


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def _newey_west_se(series: np.ndarray, lag: int = 5) -> float:
    """Newey-West 標準誤差（自己相関対策）。"""
    n = len(series)
    if n < 2:
        return math.nan
    series = series - series.mean()
    var0 = np.dot(series, series) / n
    cov_sum = 0.0
    for lag_i in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - lag_i / (lag + 1)
        cov = np.dot(series[lag_i:], series[:-lag_i]) / n
        cov_sum += weight * cov
    var = var0 + 2 * cov_sum
    if var < 0:
        var = 0.0
    return math.sqrt(var / n)


def _ic_with_ci(x: np.ndarray, y: np.ndarray, lag: int = 5) -> Tuple[float, Tuple[float, float]]:
    mask = _finite_mask(x, y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan, (math.nan, math.nan)
    x = x.astype(float)
    y = y.astype(float)
    x_std = (x - x.mean()) / (x.std() if x.std() else 1.0)
    y_std = (y - y.mean()) / (y.std() if y.std() else 1.0)
    z = x_std * y_std
    ic = float(z.mean())
    se = _newey_west_se(z, lag=lag)
    if math.isnan(se):
        return ic, (math.nan, math.nan)
    ci = (ic - 1.96 * se, ic + 1.96 * se)
    return ic, ci


def _quantile_effect(x: np.ndarray, y: np.ndarray, q: float = 0.2) -> Tuple[float, float]:
    mask = _finite_mask(x, y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return math.nan, math.nan
    lo = np.quantile(x, q)
    hi = np.quantile(x, 1 - q)
    bot_mask = x <= lo
    top_mask = x >= hi
    if bot_mask.sum() == 0 or top_mask.sum() == 0:
        return math.nan, math.nan
    bot_mean = float(y[bot_mask].mean())
    top_mean = float(y[top_mask].mean())
    diff = top_mean - bot_mean
    effect_bps = diff * 10_000
    return effect_bps, diff


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)


def _plot_feature(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
) -> None:
    mask = _finite_mask(x, y)
    x = x[mask]
    y = y[mask]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # 分位リターン
    if len(x):
        lo = np.quantile(x, 0.2)
        hi = np.quantile(x, 0.8)
        bot = y[x <= lo]
        mid = y[(x > lo) & (x < hi)]
        top = y[x >= hi]
        axes[0].bar([0, 1, 2], [bot.mean() if len(bot) else np.nan, mid.mean() if len(mid) else np.nan, top.mean() if len(top) else np.nan])
        axes[0].set_xticks([0, 1, 2], ["Q20", "Mid", "Q80"])
    axes[0].set_title(f"{name} quantile return")

    # IC 推移（累積平均）
    if len(x):
        x_std = (x - x.mean()) / (x.std() if x.std() else 1.0)
        y_std = (y - y.mean()) / (y.std() if y.std() else 1.0)
        z = x_std * y_std
        roll = np.cumsum(z) / np.arange(1, len(z) + 1)
        axes[1].plot(roll)
    axes[1].set_title("IC (cum avg)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_edge_screen(
    dataset_path: str | Path,
    splits_path: str | Path,
    out_dir: str | Path = "edge_output",
    target: str | None = None,
    ic_threshold: float = 0.01,
    nw_lag: int = 5,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "edge_plots"
    plots_dir.mkdir(exist_ok=True)

    table = pq.read_table(dataset_path)
    data = table.to_pydict()
    columns = list(data.keys())
    if target is None:
        target_candidates = [c for c in columns if c.startswith("y_")]
        if not target_candidates:
            raise ValueError("ターゲット列(y_*)が見つかりません")
        target = target_candidates[0]
    if target not in columns:
        raise ValueError(f"ターゲット列 {target} が存在しません")

    split_col = data.get("split")
    if split_col:
        split_mask = np.array([s == "train" for s in split_col], dtype=bool)
        if split_mask.any():
            use_indices = np.where(split_mask)[0]
        else:
            use_indices = np.arange(len(split_col))
    else:
        use_indices = np.arange(len(next(iter(data.values()))))

    y = _to_array([data[target][i] for i in use_indices])

    numeric_features = []
    for c in columns:
        if c == target or c == "split":
            continue
        sample = data[c][0] if len(data[c]) else None
        if isinstance(sample, (bool, str, dict, list)):
            continue
        numeric_features.append(c)
    numeric_features = sorted(numeric_features)

    report_feats: List[Dict] = []
    for feat in numeric_features:
        x = _to_array([data[feat][i] for i in use_indices])
        ic, ci = _ic_with_ci(x, y, lag=nw_lag)
        effect_bps, quantile_diff = _quantile_effect(x, y)
        keep = bool(np.isfinite(ic) and abs(ic) >= ic_threshold)
        report_feats.append(
            {
                "name": feat,
                "ic": ic,
                "ic_ci": ci,
                "effect_bps": effect_bps,
                "quantile_diff": quantile_diff,
                "keep": keep,
                "ic_threshold": ic_threshold,
            }
        )
        plot_path = plots_dir / f"{_sanitize_name(feat)}.png"
        _plot_feature(feat, x, y, plot_path)

    with open(splits_path, "r", encoding="utf-8") as fh:
        splits = json.load(fh)

    report = {
        "target": target,
        "ic_threshold": ic_threshold,
        "features": report_feats,
        "splits": splits,
    }
    report_path = out_dir / "edge_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
