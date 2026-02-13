from __future__ import annotations

import argparse
import json
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq


def _load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"metrics.json が dict ではありません: {path}")
    return data


def _get_path(d: Dict[str, Any], key_path: str) -> Any:
    cur: Any = d
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _iter_run_ids(reports_dir: Path) -> List[str]:
    if not reports_dir.exists():
        return []
    run_ids = []
    for child in sorted(reports_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if (child / "metrics.json").exists():
            run_ids.append(child.name)
    return run_ids


def _load_mm_sim_tables(
    outputs_dir: Path,
    run_id: str,
    prefix: str,
    ledger_name: str,
    trades_name: str,
) -> Tuple[Any, Any]:
    out_dir = outputs_dir / f"{prefix}{run_id}"
    ledger_path = out_dir / ledger_name
    trades_path = out_dir / trades_name
    if not ledger_path.exists():
        raise FileNotFoundError(f"ledger.parquet が見つかりません: {ledger_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"trades.parquet が見つかりません: {trades_path}")
    return pq.read_table(ledger_path), pq.read_table(trades_path)


def _artifact_path(outputs_dir: Path, run_id: str, prefix: str, name: str) -> Path:
    return outputs_dir / f"{prefix}{run_id}" / name


def _sha256_hex(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _choose_key(ledger: Any, trades: Any) -> Optional[str]:
    names_l = set(getattr(ledger, "schema").names)
    names_t = set(getattr(trades, "schema").names)
    if "book_event_id" in names_l and "book_event_id" in names_t:
        return "book_event_id"
    if "block_ts_ms" in names_l and "block_ts_ms" in names_t:
        return "block_ts_ms"
    return None


def _window_keys(ledger: Any, key_col: str, trigger_col: str) -> Tuple[set, float]:
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    trig = py.get(trigger_col, []) or []
    if not keys or not trig:
        return set(), 0.0
    in_keys = {k for k, t in zip(keys, trig) if k is not None and bool(t)}
    rate = float(sum(1 for t in trig if bool(t)) / len(trig)) if trig else 0.0
    return in_keys, rate


def _delta_split(ledger: Any, key_col: str, window_keys: set, col: str) -> Tuple[float, float]:
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    vals = py.get(col, []) or []
    pnl_in = 0.0
    pnl_out = 0.0
    prev = 0.0
    for k, cur in zip(keys, vals):
        if cur is None:
            continue
        try:
            cur_f = float(cur)
        except (TypeError, ValueError):
            continue
        delta = cur_f - prev
        prev = cur_f
        if isinstance(delta, float) and math.isnan(delta):
            continue
        if k in window_keys:
            pnl_in += delta
        else:
            pnl_out += delta
    return float(pnl_in), float(pnl_out)


def _fills_split(trades: Any, key_col: str, window_keys: set) -> Tuple[int, int]:
    py = trades.to_pydict()
    keys = py.get(key_col, []) or []
    inside = int(sum(1 for k in keys if k in window_keys))
    return inside, int(len(keys) - inside)


def _fills_notional_for_keys(trades: Any, key_col: str, keys_set: set) -> Dict[str, float]:
    py = trades.to_pydict()
    keys = py.get(key_col, []) or []
    sizes = py.get("size", []) or []
    prices = py.get("price", []) or []
    n = min(len(keys), len(sizes), len(prices))
    fills = 0
    notional = 0.0
    for i in range(n):
        k = keys[i]
        if k not in keys_set:
            continue
        sz = sizes[i]
        px = prices[i]
        try:
            sz_f = float(sz)
            px_f = float(px)
        except (TypeError, ValueError):
            continue
        fills += 1
        notional += abs(sz_f * px_f)
    return {"fills": float(fills), "notional": float(notional)}


def _avg_abs_pos_for_keys(ledger: Any, key_col: str, keys_set: set) -> Optional[float]:
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    pos = py.get("position", []) or []
    n = min(len(keys), len(pos))
    vals: List[float] = []
    for i in range(n):
        k = keys[i]
        if k not in keys_set:
            continue
        p = pos[i]
        try:
            vals.append(abs(float(p)))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr))


def _inventory_split(ledger: Any, key_col: str, window_keys: set) -> Dict[str, float | None]:
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    pos = py.get("position", []) or []
    in_vals: List[float] = []
    out_vals: List[float] = []
    for k, p in zip(keys, pos):
        if p is None:
            continue
        try:
            p_f = float(p)
        except (TypeError, ValueError):
            continue
        if k in window_keys:
            in_vals.append(p_f)
        else:
            out_vals.append(p_f)

    def _stats(vals: List[float]) -> Dict[str, float | None]:
        if not vals:
            return {"mean": None, "p95": None, "max_abs": None}
        arr = np.array(vals, dtype=float)
        return {
            "mean": float(np.nanmean(arr)),
            "p95": float(np.nanpercentile(arr, 95)),
            "max_abs": float(np.nanmax(np.abs(arr))),
        }

    out = {}
    out.update({f"in_{k}": v for k, v in _stats(in_vals).items()})
    out.update({f"out_{k}": v for k, v in _stats(out_vals).items()})
    return out


def _pos_ret_split(ledger: Any, key_col: str, window_keys: set) -> Dict[str, float | None]:
    """
    outside の unrealized 崩壊が「在庫量」ではなく「posと値動きの相関」由来かを確認するための簡易指標。
    - pos_ret は Σ pos_{t-1} * Δmark_price_t（ブロック間の価格変化に対する方向性エクスポージャ）
    - corr は corr(pos_{t-1}, Δmark_price_t)
    """
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    pos = py.get("position", []) or []
    price = py.get("mark_price", []) or []
    n = min(len(keys), len(pos), len(price))
    if n < 2:
        return {
            "pos_ret_in": None,
            "pos_ret_out": None,
            "mean_pos_prev_in": None,
            "mean_pos_prev_out": None,
            "mean_dprice_in": None,
            "mean_dprice_out": None,
            "corr_pos_prev_dprice_in": None,
            "corr_pos_prev_dprice_out": None,
        }

    in_ret: List[float] = []
    out_ret: List[float] = []
    in_pos_prev: List[float] = []
    out_pos_prev: List[float] = []
    in_dprice: List[float] = []
    out_dprice: List[float] = []

    for i in range(1, n):
        k = keys[i]
        p_prev = pos[i - 1]
        px_prev = price[i - 1]
        px_cur = price[i]
        if px_prev is None or px_cur is None:
            continue
        try:
            dpx = float(px_cur) - float(px_prev)
        except (TypeError, ValueError):
            continue
        try:
            pos_prev_f = float(p_prev) if p_prev is not None else 0.0
        except (TypeError, ValueError):
            continue
        val = pos_prev_f * dpx
        if k in window_keys:
            in_ret.append(val)
            in_pos_prev.append(pos_prev_f)
            in_dprice.append(dpx)
        else:
            out_ret.append(val)
            out_pos_prev.append(pos_prev_f)
            out_dprice.append(dpx)

    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr))

    def _sum(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return float(np.nansum(arr))

    def _corr(xs: List[float], ys: List[float]) -> Optional[float]:
        if len(xs) < 2 or len(ys) < 2:
            return None
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        ok = ~np.isnan(x) & ~np.isnan(y)
        x = x[ok]
        y = y[ok]
        if x.size < 2:
            return None
        if float(np.nanstd(x)) == 0.0 or float(np.nanstd(y)) == 0.0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "pos_ret_in": _sum(in_ret),
        "pos_ret_out": _sum(out_ret),
        "mean_pos_prev_in": _mean(in_pos_prev),
        "mean_pos_prev_out": _mean(out_pos_prev),
        "mean_dprice_in": _mean(in_dprice),
        "mean_dprice_out": _mean(out_dprice),
        "corr_pos_prev_dprice_in": _corr(in_pos_prev, in_dprice),
        "corr_pos_prev_dprice_out": _corr(out_pos_prev, out_dprice),
    }


def _flag_rate_split(ledger: Any, key_col: str, window_keys: set, flag_col: str) -> Dict[str, float | int | None]:
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    flags = py.get(flag_col, []) or []
    if not keys or (flag_col not in py):
        return {"in_rate": None, "out_rate": None, "in_true": None, "out_true": None, "in_n": None, "out_n": None}

    in_true = 0
    out_true = 0
    in_n = 0
    out_n = 0
    for k, f in zip(keys, flags):
        if k in window_keys:
            in_n += 1
            if bool(f):
                in_true += 1
        else:
            out_n += 1
            if bool(f):
                out_true += 1

    in_rate = (in_true / in_n) if in_n else None
    out_rate = (out_true / out_n) if out_n else None
    return {
        "in_rate": float(in_rate) if in_rate is not None else None,
        "out_rate": float(out_rate) if out_rate is not None else None,
        "in_true": int(in_true),
        "out_true": int(out_true),
        "in_n": int(in_n),
        "out_n": int(out_n),
    }


def _mid_series_from_ledger(ledger: Any) -> Tuple[np.ndarray, np.ndarray]:
    py = ledger.to_pydict()
    ts = py.get("block_ts_ms", []) or []
    mid = py.get("mark_price", []) or []
    pairs: List[Tuple[int, float]] = []
    for t, m in zip(ts, mid):
        if t is None or m is None:
            continue
        try:
            pairs.append((int(t), float(m)))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)
    ts_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    mid_arr = np.array([p[1] for p in pairs], dtype=float)
    return ts_arr, mid_arr


def _markout_and_realized_spread_split(
    ledger: Any,
    trades: Any,
    key_col: str,
    window_keys: set,
    horizon_ms: int,
) -> Dict[str, float | None]:
    ts_arr, mid_arr = _mid_series_from_ledger(ledger)
    if ts_arr.size == 0 or trades.num_rows == 0:
        return {
            "markout_in": None,
            "markout_out": None,
            "realized_spread_in": None,
            "realized_spread_out": None,
        }

    tpy = trades.to_pydict()
    sides = tpy.get("side", []) or []
    ts = tpy.get("block_ts_ms", []) or []
    keys = tpy.get(key_col, []) or []
    prices = tpy.get("price", []) or []

    mark_in: List[float] = []
    mark_out: List[float] = []
    rs_in: List[float] = []
    rs_out: List[float] = []

    for side, t, k, px in zip(sides, ts, keys, prices):
        if side is None or t is None:
            continue
        try:
            ts_i = int(t)
        except (TypeError, ValueError):
            continue

        idx_now = int(np.searchsorted(ts_arr, ts_i, side="right") - 1)
        if idx_now < 0 or idx_now >= ts_arr.size:
            continue
        mid_now = float(mid_arr[idx_now])

        fut_ts = ts_i + int(horizon_ms)
        idx_fut = int(np.searchsorted(ts_arr, fut_ts, side="left"))
        if idx_fut < 0 or idx_fut >= ts_arr.size:
            continue
        mid_fut = float(mid_arr[idx_fut])

        is_in = k in window_keys
        if str(side) == "buy":
            mark = mid_fut - mid_now
            rs = None
            if px is not None:
                try:
                    rs = mid_fut - float(px)
                except (TypeError, ValueError):
                    rs = None
        else:
            mark = mid_now - mid_fut
            rs = None
            if px is not None:
                try:
                    rs = float(px) - mid_fut
                except (TypeError, ValueError):
                    rs = None

        if is_in:
            mark_in.append(float(mark))
            if rs is not None:
                rs_in.append(float(rs))
        else:
            mark_out.append(float(mark))
            if rs is not None:
                rs_out.append(float(rs))

    def _mean(vals: List[float]) -> float | None:
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr))

    return {
        "markout_in": _mean(mark_in),
        "markout_out": _mean(mark_out),
        "realized_spread_in": _mean(rs_in),
        "realized_spread_out": _mean(rs_out),
    }


def _build_rows(
    reports_dir: Path,
    outputs_dir: Path,
    mm_sim_prefix: str,
    ledger_name: str,
    trades_name: str,
    orders_name: str,
    pull_window_from: str,
    stop_window_from: str,
    baseline_run_id: str,
    run_ids: Iterable[str],
    metrics: List[str],
) -> Tuple[List[str], List[List[str]]]:
    metrics_cache: Dict[str, Dict[str, Any]] = {}
    tables_cache: Dict[str, Tuple[Any, Any]] = {}
    hash_cache: Dict[Tuple[str, str], Optional[str]] = {}
    window_cache: Dict[str, Dict[str, Any]] = {}
    window_stats_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def load_metrics(run_id: str) -> Dict[str, Any]:
        if run_id not in metrics_cache:
            metrics_cache[run_id] = _load_metrics(reports_dir / run_id / "metrics.json")
        return metrics_cache[run_id]

    def load_tables(run_id: str) -> Tuple[Any, Any]:
        if run_id not in tables_cache:
            tables_cache[run_id] = _load_mm_sim_tables(
                outputs_dir, run_id, mm_sim_prefix, ledger_name, trades_name
            )
        return tables_cache[run_id]

    def artifact_hash(run_id: str, name: str) -> Optional[str]:
        key = (run_id, name)
        if key in hash_cache:
            return hash_cache[key]
        path = _artifact_path(outputs_dir, run_id, mm_sim_prefix, name)
        hash_cache[key] = _sha256_hex(path)
        return hash_cache[key]

    def get_window(kind: str) -> Dict[str, Any]:
        if kind in window_cache:
            return window_cache[kind]
        if kind == "pull":
            from_run = pull_window_from
            trigger_col = "pull_triggered"
        else:
            from_run = stop_window_from
            trigger_col = "stop_triggered"

        ledger, trades = load_tables(from_run)
        key_col = _choose_key(ledger, trades)
        if key_col is None:
            raise SystemExit(f"window join key が見つかりません: run={from_run}")
        keys, rate = _window_keys(ledger, key_col, trigger_col)
        window_cache[kind] = {"from_run": from_run, "trigger_col": trigger_col, "key_col": key_col, "keys": keys, "rate": rate}
        return window_cache[kind]

    def metric_value(run_id: str, key: str) -> Any:
        if key in {
            "ledger_hash8",
            "sim_trades_hash8",
            "orders_hash8",
            "ledger_sha256",
            "sim_trades_sha256",
            "orders_sha256",
            "ledger_same_as_baseline",
            "sim_trades_same_as_baseline",
            "orders_same_as_baseline",
        }:
            if key.startswith("ledger_"):
                name = ledger_name
            elif key.startswith("sim_trades_"):
                name = trades_name
            else:
                name = orders_name

            h = artifact_hash(run_id, name)
            if key.endswith("_sha256"):
                return h
            if key.endswith("_hash8"):
                return (h or "-")[:8] if h else None
            # same_as_baseline
            hb = artifact_hash(baseline_run_id, name)
            if h is None or hb is None:
                return None
            return bool(h == hb)

        if key in {
            "fills_outside_when_unwind_active",
            "notional_outside_when_unwind_active",
            "avg_abs_pos_outside_when_unwind_active",
        }:
            # post-pull unwind は pull_window の外でのみ動く想定なので、pull_window を基準に outside を切る
            kind = "pull"
            win = get_window(kind)
            ledger, trades = load_tables(run_id)
            key_col = win["key_col"]
            py = ledger.to_pydict()
            keys = py.get(key_col, []) or []
            if "post_pull_unwind_active" not in py:
                return None
            flags = py.get("post_pull_unwind_active", []) or []
            n = min(len(keys), len(flags))
            active_outside = {k for k, f in zip(keys[:n], flags[:n]) if k is not None and (k not in win["keys"]) and bool(f)}
            if key.startswith("fills_") or key.startswith("notional_"):
                stats_key = (run_id, f"{kind}_unwind_active_outside")
                if stats_key not in window_stats_cache:
                    window_stats_cache[stats_key] = {}
                cache = window_stats_cache[stats_key]
                if "fills_notional" not in cache:
                    cache["fills_notional"] = _fills_notional_for_keys(trades, key_col, active_outside)
                fn = cache["fills_notional"]
                return fn["fills"] if key.startswith("fills_") else fn["notional"]
            return _avg_abs_pos_for_keys(ledger, key_col, active_outside)

        # derived: pull/stop の「市場条件ウィンドウ」を window_from の trigger で定義し、全runで同じ区間を集計する
        if key in {
            "pull_window_rate",
            "pnl_in_pull_window",
            "pnl_outside_pull_window",
            "price_pnl_in_pull_window",
            "price_pnl_outside_pull_window",
            "unrealized_pnl_in_pull_window",
            "unrealized_pnl_outside_pull_window",
            "fees_in_pull_window",
            "fees_outside_pull_window",
            "rebates_in_pull_window",
            "rebates_outside_pull_window",
            "funding_in_pull_window",
            "funding_outside_pull_window",
            "pos_ret_in_pull_window",
            "pos_ret_outside_pull_window",
            "mean_pos_prev_in_pull_window",
            "mean_pos_prev_outside_pull_window",
            "mean_dprice_in_pull_window",
            "mean_dprice_outside_pull_window",
            "corr_pos_prev_dprice_in_pull_window",
            "corr_pos_prev_dprice_outside_pull_window",
            "post_unwind_active_rate_in_pull_window",
            "post_unwind_active_rate_outside_pull_window",
            "post_unwind_active_count_in_pull_window",
            "post_unwind_active_count_outside_pull_window",
            "fills_in_pull_window",
            "fills_outside_pull_window",
            "inventory_mean_in_pull_window",
            "inventory_p95_in_pull_window",
            "inventory_max_abs_in_pull_window",
            "inventory_mean_outside_pull_window",
            "inventory_p95_outside_pull_window",
            "inventory_max_abs_outside_pull_window",
            "markout_5s_in_pull_window",
            "markout_5s_outside_pull_window",
            "realized_spread_5s_in_pull_window",
            "realized_spread_5s_outside_pull_window",
            "stop_window_rate",
            "pnl_in_stop_window",
            "pnl_outside_stop_window",
            "price_pnl_in_stop_window",
            "price_pnl_outside_stop_window",
            "unrealized_pnl_in_stop_window",
            "unrealized_pnl_outside_stop_window",
            "fees_in_stop_window",
            "fees_outside_stop_window",
            "rebates_in_stop_window",
            "rebates_outside_stop_window",
            "funding_in_stop_window",
            "funding_outside_stop_window",
            "pos_ret_in_stop_window",
            "pos_ret_outside_stop_window",
            "mean_pos_prev_in_stop_window",
            "mean_pos_prev_outside_stop_window",
            "mean_dprice_in_stop_window",
            "mean_dprice_outside_stop_window",
            "corr_pos_prev_dprice_in_stop_window",
            "corr_pos_prev_dprice_outside_stop_window",
            "post_unwind_active_rate_in_stop_window",
            "post_unwind_active_rate_outside_stop_window",
            "post_unwind_active_count_in_stop_window",
            "post_unwind_active_count_outside_stop_window",
            "fills_in_stop_window",
            "fills_outside_stop_window",
            "inventory_mean_in_stop_window",
            "inventory_p95_in_stop_window",
            "inventory_max_abs_in_stop_window",
            "inventory_mean_outside_stop_window",
            "inventory_p95_outside_stop_window",
            "inventory_max_abs_outside_stop_window",
            "markout_5s_in_stop_window",
            "markout_5s_outside_stop_window",
            "realized_spread_5s_in_stop_window",
            "realized_spread_5s_outside_stop_window",
        }:
            kind = "pull" if "pull" in key else "stop"
            win = get_window(kind)
            if key.endswith("_window_rate"):
                return win["rate"]
            ledger, trades = load_tables(run_id)
            key_col = win["key_col"]
            # key が一致している前提（同一 blocks.parquet を入力にする運用）
            pnl_cols = {
                "pnl": "total_pnl",
                "price_pnl": "price_pnl",
                "unrealized_pnl": "unrealized_pnl",
                "fees": "fees",
                "rebates": "rebates",
                "funding": "funding",
            }
            for prefix, col in pnl_cols.items():
                if key.startswith(f"{prefix}_"):
                    pnl_in, pnl_out = _delta_split(ledger, key_col, win["keys"], col=col)
                    return pnl_in if "_in_" in key else pnl_out
            if key.startswith("fills_"):
                fills_in, fills_out = _fills_split(trades, key_col, win["keys"])
                return fills_in if "_in_" in key else fills_out

            if key.startswith("inventory_"):
                stats_key = (run_id, kind)
                if stats_key not in window_stats_cache:
                    window_stats_cache[stats_key] = {}
                cache = window_stats_cache[stats_key]
                if "inventory" not in cache:
                    cache["inventory"] = _inventory_split(ledger, key_col, win["keys"])
                inv = cache["inventory"]
                if key.endswith("_in_pull_window") or key.endswith("_in_stop_window"):
                    suffix = key.split("inventory_")[1].split("_in_")[0]
                    return inv.get(f"in_{suffix}")
                suffix = key.split("inventory_")[1].split("_outside_")[0]
                return inv.get(f"out_{suffix}")

            if (
                key.startswith("pos_ret_")
                or key.startswith("mean_pos_prev_")
                or key.startswith("mean_dprice_")
                or key.startswith("corr_pos_prev_dprice_")
            ):
                stats_key = (run_id, kind)
                if stats_key not in window_stats_cache:
                    window_stats_cache[stats_key] = {}
                cache = window_stats_cache[stats_key]
                if "pos_ret" not in cache:
                    cache["pos_ret"] = _pos_ret_split(ledger, key_col, win["keys"])
                pr = cache["pos_ret"]
                inside = "_in_" in key
                if key.startswith("pos_ret_"):
                    return pr["pos_ret_in"] if inside else pr["pos_ret_out"]
                if key.startswith("mean_pos_prev_"):
                    return pr["mean_pos_prev_in"] if inside else pr["mean_pos_prev_out"]
                if key.startswith("mean_dprice_"):
                    return pr["mean_dprice_in"] if inside else pr["mean_dprice_out"]
                return pr["corr_pos_prev_dprice_in"] if inside else pr["corr_pos_prev_dprice_out"]

            if key.startswith("post_unwind_active_rate_") or key.startswith("post_unwind_active_count_"):
                stats_key = (run_id, kind)
                if stats_key not in window_stats_cache:
                    window_stats_cache[stats_key] = {}
                cache = window_stats_cache[stats_key]
                if "post_unwind_active" not in cache:
                    cache["post_unwind_active"] = _flag_rate_split(
                        ledger, key_col, win["keys"], flag_col="post_pull_unwind_active"
                    )
                fr = cache["post_unwind_active"]
                inside = "_in_" in key
                if key.startswith("post_unwind_active_rate_"):
                    return fr["in_rate"] if inside else fr["out_rate"]
                return fr["in_true"] if inside else fr["out_true"]

            if key.startswith("markout_5s") or key.startswith("realized_spread_5s"):
                stats_key = (run_id, kind)
                if stats_key not in window_stats_cache:
                    window_stats_cache[stats_key] = {}
                cache = window_stats_cache[stats_key]
                if "markout_rs_5s" not in cache:
                    cache["markout_rs_5s"] = _markout_and_realized_spread_split(
                        ledger, trades, key_col, win["keys"], horizon_ms=5000
                    )
                mr = cache["markout_rs_5s"]
                if key.startswith("markout_5s"):
                    return mr["markout_in"] if "_in_" in key else mr["markout_out"]
                return mr["realized_spread_in"] if "_in_" in key else mr["realized_spread_out"]

        return _get_path(load_metrics(run_id), key)

    headers: List[str] = ["run_id"]
    for key in metrics:
        headers.append(key)
        headers.append(f"Δ{key}")

    rows: List[List[str]] = []

    for run_id in run_ids:
        row: List[str] = [run_id]
        for key in metrics:
            v = metric_value(run_id, key)
            b = metric_value(baseline_run_id, key)
            v_f = _as_float(v)
            b_f = _as_float(b)
            dv = v_f - b_f if v_f is not None and b_f is not None else None
            row.append(_fmt(v, digits=6))
            row.append(_fmt(dv, digits=6))
        rows.append(row)

    return headers, rows


def _render_markdown(headers: List[str], rows: List[List[str]]) -> str:
    align = ["---"] + ["---:" for _ in headers[1:]]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def _render_tsv(headers: List[str], rows: List[List[str]]) -> str:
    out = ["\t".join(headers)]
    out.extend("\t".join(r) for r in rows)
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="reports/<run_id>/metrics.json を比較し、baselineとの差分を1枚の表にする"
    )
    parser.add_argument("--reports-dir", default="reports", help="reports ディレクトリ")
    parser.add_argument("--outputs-dir", default="outputs", help="mm-sim の出力ディレクトリ（outputs/mm_sim_<run_id>/...）")
    parser.add_argument("--mm-sim-prefix", default="mm_sim_", help="mm-sim の出力ディレクトリ接頭辞")
    parser.add_argument("--ledger-name", default="ledger.parquet", help="mm-sim 出力の ledger ファイル名")
    parser.add_argument("--trades-name", default="sim_trades.parquet", help="mm-sim 出力の trades ファイル名")
    parser.add_argument("--orders-name", default="orders.parquet", help="mm-sim 出力の orders ファイル名")
    parser.add_argument(
        "--pull-window-from",
        default="pull",
        help="pull_window を定義する run_id（この run の pull_triggered を市場条件ウィンドウとして使う）",
    )
    parser.add_argument(
        "--stop-window-from",
        default="stop",
        help="stop_window を定義する run_id（この run の stop_triggered を市場条件ウィンドウとして使う）",
    )
    parser.add_argument("--baseline", default="baseline", help="baseline の run_id（例: baseline）")
    parser.add_argument(
        "--runs",
        default=None,
        help="比較対象 run_id（カンマ区切り）。未指定なら reports/ 配下の全runを対象にする",
    )
    parser.add_argument(
        "--metrics",
        default="pnl,max_drawdown,fill_rate,num_fills,notional_traded,stop_trigger_rate,pull_trigger_rate",
        help="比較するメトリクス（カンマ区切り。ネストは inventory.mean のように指定）",
    )
    parser.add_argument("--format", choices=["md", "tsv"], default="md", help="出力形式")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    all_run_ids = _iter_run_ids(reports_dir)
    if not all_run_ids:
        raise SystemExit(f"run が見つかりません: {reports_dir}")

    baseline_run_id = str(args.baseline)
    if baseline_run_id not in all_run_ids:
        raise SystemExit(f"baseline が見つかりません: {reports_dir / baseline_run_id / 'metrics.json'}")

    if args.runs:
        run_ids = [r.strip() for r in str(args.runs).split(",") if r.strip()]
    else:
        run_ids = all_run_ids
    # baseline を先頭に並べたい（比較の見やすさ）
    if baseline_run_id in run_ids:
        run_ids = [baseline_run_id] + [r for r in run_ids if r != baseline_run_id]

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    headers, rows = _build_rows(
        reports_dir=reports_dir,
        outputs_dir=Path(args.outputs_dir),
        mm_sim_prefix=str(args.mm_sim_prefix),
        ledger_name=str(args.ledger_name),
        trades_name=str(args.trades_name),
        orders_name=str(args.orders_name),
        pull_window_from=str(args.pull_window_from),
        stop_window_from=str(args.stop_window_from),
        baseline_run_id=baseline_run_id,
        run_ids=run_ids,
        metrics=metrics,
    )

    if args.format == "tsv":
        text = _render_tsv(headers, rows)
    else:
        text = _render_markdown(headers, rows)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
