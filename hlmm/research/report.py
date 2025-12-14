from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.use("Agg")


def _to_array(series: Iterable[object]) -> np.ndarray:
    return np.array(list(series), dtype=float)


def _fill_rate(trades: pa.Table, blocks: pa.Table) -> float:
    if trades.num_rows == 0:
        return 0.0
    submitted = trades.num_rows  # 約定記録数＝成立注文数
    return submitted / submitted


def _realized_spread(trades: pa.Table, features: Optional[pa.Table], horizon_ms: int) -> float:
    if features is None or trades.num_rows == 0:
        return float("nan")
    trades_py = trades.to_pydict()
    features_py = features.to_pydict()
    markouts = []
    mid_series = features_py.get("mid")
    ts_series = features_py.get("block_ts_ms")
    if mid_series is None or ts_series is None:
        return float("nan")
    for px, side, ts in zip(trades_py["price"], trades_py["side"], trades_py["block_ts_ms"]):
        future_ts = ts + horizon_ms
        # 最初に future_ts 以上となる mid を探す
        fut_mid = None
        for m_ts, m_mid in zip(ts_series, mid_series):
            if m_ts is None or m_mid is None:
                continue
            if m_ts >= future_ts:
                fut_mid = m_mid
                break
        if fut_mid is None:
            continue
        if side == "buy":
            spread = fut_mid - px
        else:
            spread = px - fut_mid
        markouts.append(spread)
    if not markouts:
        return float("nan")
    return float(np.mean(markouts))


def _inventory_stats(ledger: pa.Table) -> dict:
    if ledger.num_rows == 0:
        return {"mean": 0.0, "p10": 0.0, "p90": 0.0}
    pos = _to_array(ledger.to_pydict()["position"])
    return {
        "mean": float(np.nanmean(pos)),
        "p10": float(np.nanpercentile(pos, 10)),
        "p90": float(np.nanpercentile(pos, 90)),
    }


def _fee_breakdown(ledger: pa.Table) -> dict:
    py = ledger.to_pydict()
    return {
        "fees": float(py.get("fees", [0.0])[-1] if py.get("fees") else 0.0),
        "rebates": float(py.get("rebates", [0.0])[-1] if py.get("rebates") else 0.0),
        "funding": float(py.get("funding", [0.0])[-1] if py.get("funding") else 0.0),
    }


def compute_metrics(
    ledger_path: str | Path,
    trades_path: str | Path,
    features_path: str | Path | None = None,
) -> dict:
    ledger = pq.read_table(ledger_path)
    trades = pq.read_table(trades_path)
    features = pq.read_table(features_path) if features_path else None

    metrics = {
        "fill_rate": _fill_rate(trades, ledger),
        "pnl": float(ledger.to_pydict().get("total_pnl", [0.0])[-1] if ledger.num_rows else 0.0),
        "realized_spread_1s": _realized_spread(trades, features, 1000),
        "realized_spread_5s": _realized_spread(trades, features, 5000),
        "realized_spread_15s": _realized_spread(trades, features, 15000),
        "inventory": _inventory_stats(ledger),
        "fee_breakdown": _fee_breakdown(ledger),
    }
    return metrics


def _plot_equity(ledger: pa.Table, out_path: Path) -> None:
    py = ledger.to_pydict()
    ts = py.get("block_ts_ms", [])
    pnl = py.get("total_pnl", [])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ts, pnl)
    ax.set_title("Equity Curve")
    ax.set_xlabel("ts_ms")
    ax.set_ylabel("total_pnl")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_inventory(ledger: pa.Table, out_path: Path) -> None:
    py = ledger.to_pydict()
    ts = py.get("block_ts_ms", [])
    pos = py.get("position", [])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ts, pos)
    ax.set_title("Inventory")
    ax.set_xlabel("ts_ms")
    ax.set_ylabel("position")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_report(
    run_id: str,
    ledger_path: str | Path,
    trades_path: str | Path,
    features_path: str | Path | None = None,
    reports_dir: str | Path = "reports",
) -> Path:
    reports_dir = Path(reports_dir)
    out_dir = reports_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    metrics = compute_metrics(ledger_path, trades_path, features_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger = pq.read_table(ledger_path)
    _plot_equity(ledger, plots_dir / "equity.png")
    _plot_inventory(ledger, plots_dir / "inventory.png")

    return out_dir
