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
    if blocks.num_rows == 0:
        return 0.0
    # 近似: 1 block あたり bid/ask の2注文を出す前提
    submitted = blocks.num_rows * 2
    if submitted <= 0:
        return 0.0
    return float(min(1.0, trades.num_rows / submitted))


def _realized_spread(
    trades: pa.Table, features: Optional[pa.Table], horizon_ms: int
) -> Optional[float]:
    if features is None or trades.num_rows == 0:
        return None
    trades_py = trades.to_pydict()
    features_py = features.to_pydict()
    mid_series = features_py.get("mid")
    ts_series = features_py.get("block_ts_ms")
    if mid_series is None or ts_series is None:
        return None

    pairs = []
    for ts, mid in zip(ts_series, mid_series):
        if ts is None or mid is None:
            continue
        try:
            pairs.append((int(ts), float(mid)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    ts_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    mid_arr = np.array([p[1] for p in pairs], dtype=float)

    out = []
    for px, side, ts in zip(trades_py.get("price", []), trades_py.get("side", []), trades_py.get("block_ts_ms", [])):
        if ts is None or px is None or side is None:
            continue
        try:
            ts_i = int(ts)
            px_f = float(px)
        except (TypeError, ValueError):
            continue
        future_ts = ts_i + int(horizon_ms)
        idx = int(np.searchsorted(ts_arr, future_ts, side="left"))
        if idx < 0 or idx >= ts_arr.size:
            continue
        fut_mid = float(mid_arr[idx])
        if str(side) == "buy":
            out.append(fut_mid - px_f)
        else:
            out.append(px_f - fut_mid)
    if not out:
        return None
    return float(np.mean(out))


def _markout(trades: pa.Table, features: Optional[pa.Table], horizon_ms: int) -> Optional[float]:
    """約定後のmid変化（逆選択）: buyは mid_future-mid_now、sellは mid_now-mid_future。"""
    if features is None or trades.num_rows == 0:
        return None
    trades_py = trades.to_pydict()
    features_py = features.to_pydict()
    mid_series = features_py.get("mid")
    ts_series = features_py.get("block_ts_ms")
    if mid_series is None or ts_series is None:
        return None

    pairs = []
    for ts, mid in zip(ts_series, mid_series):
        if ts is None or mid is None:
            continue
        try:
            pairs.append((int(ts), float(mid)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    ts_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    mid_arr = np.array([p[1] for p in pairs], dtype=float)

    out = []
    for side, ts in zip(trades_py.get("side", []), trades_py.get("block_ts_ms", [])):
        if ts is None or side is None:
            continue
        try:
            ts_i = int(ts)
        except (TypeError, ValueError):
            continue
        # mid_now: lookahead防止のため「最後の <= ts」
        idx_now = int(np.searchsorted(ts_arr, ts_i, side="right") - 1)
        if idx_now < 0 or idx_now >= ts_arr.size:
            continue
        mid_now = float(mid_arr[idx_now])

        future_ts = ts_i + int(horizon_ms)
        idx_fut = int(np.searchsorted(ts_arr, future_ts, side="left"))
        if idx_fut < 0 or idx_fut >= ts_arr.size:
            continue
        mid_fut = float(mid_arr[idx_fut])

        if str(side) == "buy":
            out.append(mid_fut - mid_now)
        else:
            out.append(mid_now - mid_fut)
    if not out:
        return None
    return float(np.mean(out))


def _notional_traded(trades: pa.Table) -> float:
    if trades.num_rows == 0:
        return 0.0
    py = trades.to_pydict()
    if "price" in py and "size" in py:
        prices = py.get("price") or []
        sizes = py.get("size") or []
    elif "px" in py and "sz" in py:
        prices = py.get("px") or []
        sizes = py.get("sz") or []
    else:
        return 0.0
    total = 0.0
    for p, s in zip(prices, sizes):
        if p is None or s is None:
            continue
        try:
            total += abs(float(p)) * abs(float(s))
        except (TypeError, ValueError):
            continue
    return float(total)


def _inventory_stats(ledger: pa.Table) -> dict:
    if ledger.num_rows == 0:
        return {"mean": 0.0, "p10": 0.0, "p90": 0.0, "p95": 0.0, "max_abs": 0.0}
    pos = _to_array(ledger.to_pydict()["position"])
    return {
        "mean": float(np.nanmean(pos)),
        "p10": float(np.nanpercentile(pos, 10)),
        "p90": float(np.nanpercentile(pos, 90)),
        "p95": float(np.nanpercentile(pos, 95)),
        "max_abs": float(np.nanmax(np.abs(pos))) if pos.size else 0.0,
    }


def _max_drawdown(ledger: pa.Table) -> float:
    if ledger.num_rows == 0:
        return 0.0
    py = ledger.to_pydict()
    pnl = _to_array(py.get("total_pnl", []))
    if pnl.size == 0:
        return 0.0
    peak = np.maximum.accumulate(pnl)
    drawdown = pnl - peak
    return float(np.min(drawdown)) if drawdown.size else 0.0


def _fee_breakdown(ledger: pa.Table) -> dict:
    py = ledger.to_pydict()
    return {
        "fees": float(py.get("fees", [0.0])[-1] if py.get("fees") else 0.0),
        "rebates": float(py.get("rebates", [0.0])[-1] if py.get("rebates") else 0.0),
        "funding": float(py.get("funding", [0.0])[-1] if py.get("funding") else 0.0),
    }


def _count_true(table: pa.Table, column: str) -> int:
    if column not in table.schema.names:
        return 0
    values = table[column].to_pylist()
    return int(sum(1 for v in values if bool(v)))


def _trigger_mask(table: pa.Table, column: str) -> np.ndarray:
    if table.num_rows == 0:
        return np.zeros(0, dtype=bool)
    if column not in table.schema.names:
        return np.zeros(int(table.num_rows), dtype=bool)
    values = table[column].to_pylist()
    return np.array([bool(v) for v in values], dtype=bool)


def _pnl_when(ledger: pa.Table, mask: np.ndarray) -> float:
    if ledger.num_rows == 0 or mask.size == 0:
        return 0.0
    py = ledger.to_pydict()
    pnl = _to_array(py.get("total_pnl", []))
    if pnl.size == 0:
        return 0.0
    prev = np.concatenate(([0.0], pnl[:-1]))
    delta = pnl - prev
    # NaNは無視（欠損が混ざっても落とさない）
    return float(np.nansum(delta[mask]))


def _fills_when(trades: pa.Table, ledger: pa.Table, trigger_column: str) -> int:
    if trades.num_rows == 0 or ledger.num_rows == 0:
        return 0
    if trigger_column not in ledger.schema.names:
        return 0

    # join key: book_event_id を優先。なければ block_ts_ms。
    if "book_event_id" in trades.schema.names and "book_event_id" in ledger.schema.names:
        key = "book_event_id"
    elif "block_ts_ms" in trades.schema.names and "block_ts_ms" in ledger.schema.names:
        key = "block_ts_ms"
    else:
        return 0

    ledger_py = ledger.to_pydict()
    keys = ledger_py.get(key, []) or []
    trig = ledger_py.get(trigger_column, []) or []
    triggered = {k for k, t in zip(keys, trig) if k is not None and bool(t)}

    trades_py = trades.to_pydict()
    trade_keys = trades_py.get(key, []) or []
    return int(sum(1 for k in trade_keys if k in triggered))


def compute_metrics(
    ledger_path: str | Path,
    trades_path: str | Path,
    features_path: str | Path | None = None,
) -> dict:
    ledger = pq.read_table(ledger_path)
    trades = pq.read_table(trades_path)
    features = pq.read_table(features_path) if features_path else None
    # features が無い場合でも markout を出せるよう、ledger の mark_price を mid として代用する。
    # （厳密検証では blocks→features を作って features_path を渡すのが推奨）
    if features is None and ledger.num_rows and "block_ts_ms" in ledger.schema.names and "mark_price" in ledger.schema.names:
        py = ledger.to_pydict()
        features = pa.Table.from_pydict({"block_ts_ms": py.get("block_ts_ms", []), "mid": py.get("mark_price", [])})

    stop_mask = _trigger_mask(ledger, "stop_triggered")
    pull_mask = _trigger_mask(ledger, "pull_triggered")

    metrics = {
        "num_blocks": int(ledger.num_rows),
        "num_fills": int(trades.num_rows),
        "notional_traded": _notional_traded(trades),
        "fill_rate": _fill_rate(trades, ledger),
        "pnl": float(ledger.to_pydict().get("total_pnl", [0.0])[-1] if ledger.num_rows else 0.0),
        "max_drawdown": _max_drawdown(ledger),
        "stop_trigger_count": _count_true(ledger, "stop_triggered"),
        "pull_trigger_count": _count_true(ledger, "pull_triggered"),
        "stop_trigger_rate": float(_count_true(ledger, "stop_triggered") / ledger.num_rows)
        if ledger.num_rows
        else 0.0,
        "pull_trigger_rate": float(_count_true(ledger, "pull_triggered") / ledger.num_rows)
        if ledger.num_rows
        else 0.0,
        "fills_when_stop": _fills_when(trades, ledger, "stop_triggered"),
        "fills_when_pull": _fills_when(trades, ledger, "pull_triggered"),
        "pnl_when_stop": _pnl_when(ledger, stop_mask),
        "pnl_when_pull": _pnl_when(ledger, pull_mask),
        "realized_spread_1s": _realized_spread(trades, features, 1000),
        "realized_spread_5s": _realized_spread(trades, features, 5000),
        "realized_spread_15s": _realized_spread(trades, features, 15000),
        "markout_1s": _markout(trades, features, 1000),
        "markout_5s": _markout(trades, features, 5000),
        "markout_15s": _markout(trades, features, 15000),
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
