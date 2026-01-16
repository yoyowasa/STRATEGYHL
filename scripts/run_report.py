import argparse
import json
import math
from bisect import bisect_left, bisect_right
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_cloid(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, dict):
        for key in ("cloid", "client_oid", "clientOid", "value", "hex", "id"):
            if key in value:
                s = _normalize_cloid(value.get(key))
                if s:
                    return s
        return None
    try:
        s = str(value).strip()
    except Exception:
        return None
    return s or None


def _price_tick_from_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        dec = Decimal(str(value))
    except Exception:
        return None
    if dec.is_zero():
        return None
    exp = dec.as_tuple().exponent
    tick = Decimal(1).scaleb(exp)
    return float(tick)


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    weight = idx - lo
    return sorted_vals[lo] * (1 - weight) + sorted_vals[hi] * weight


def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    vals = sorted(values)
    mean = sum(vals) / len(vals)
    return {
        "count": len(vals),
        "mean": mean,
        "median": _percentile(vals, 0.5),
        "p10": _percentile(vals, 0.1),
        "p90": _percentile(vals, 0.9),
    }


def _find_mid_after(ts_list: List[int], mid_list: List[Optional[float]], ts_ms: int) -> Optional[float]:
    if not ts_list:
        return None
    idx = bisect_left(ts_list, ts_ms)
    if idx >= len(ts_list):
        return None
    return mid_list[idx]


def build_report(run_dir: Path, horizons_s: List[int]) -> Dict[str, Any]:
    fills = _read_jsonl(run_dir / "fills.jsonl")
    market = _read_jsonl(run_dir / "market_state.jsonl")
    decision = _read_jsonl(run_dir / "decision.jsonl")
    orders = _read_jsonl(run_dir / "orders.jsonl")

    orders_by_cloid: Dict[str, List[Tuple[int, Optional[float]]]] = {}
    for row in orders:
        action = row.get("action")
        if action not in {"new", "replace"}:
            continue
        cloid = _normalize_cloid(
            row.get("client_oid") or row.get("cloid") or row.get("clientOid") or row.get("clientOid")
        )
        ts = row.get("ts_ms")
        if cloid is None or not isinstance(ts, (int, float)):
            continue
        spread = _as_float(row.get("effective_spread_bps"))
        orders_by_cloid.setdefault(cloid, []).append((int(ts), spread))
    order_ts_by_cloid: Dict[str, List[int]] = {}
    order_spread_by_cloid: Dict[str, List[Optional[float]]] = {}
    for cloid, entries in orders_by_cloid.items():
        entries.sort(key=lambda item: item[0])
        order_ts_by_cloid[cloid] = [t for t, _ in entries]
        order_spread_by_cloid[cloid] = [s for _, s in entries]

    market_ts: List[int] = []
    market_mid: List[Optional[float]] = []
    market_bid: List[Optional[float]] = []
    market_ask: List[Optional[float]] = []
    market_mid_source: List[Optional[str]] = []
    spread_filter_total = 0
    spread_filter_active_count = 0
    for row in market:
        ts = row.get("ts_ms")
        if not isinstance(ts, (int, float)):
            continue
        best_bid = _as_float(row.get("best_bid"))
        best_ask = _as_float(row.get("best_ask"))
        mid = None
        mid_source = None
        if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
            mid = (best_bid + best_ask) / 2
            mid_source = "best_bid_ask"
        else:
            mid = _as_float(row.get("mid"))
            if mid is not None:
                mid_source = "market_state_mid"
            else:
                mark_px = _as_float(row.get("mark_px") or row.get("markPx"))
                if mark_px is not None:
                    mid = mark_px
                    mid_source = "mark_px"
        market_ts.append(int(ts))
        market_mid.append(mid)
        market_bid.append(best_bid)
        market_ask.append(best_ask)
        market_mid_source.append(mid_source)
        if "spread_filter_active" in row:
            spread_filter_total += 1
            if row.get("spread_filter_active"):
                spread_filter_active_count += 1
    market_ts_min = min(market_ts) if market_ts else None
    market_ts_max = max(market_ts) if market_ts else None

    decision_ticks = 0
    boost_active_count = 0
    for row in decision:
        ts = row.get("ts_ms")
        if not isinstance(ts, (int, float)):
            continue
        decision_ticks += 1
        if row.get("boost_active"):
            boost_active_count += 1
    boost_active_rate = (
        boost_active_count / decision_ticks if decision_ticks else None
    )

    decision_by_ts: Dict[int, Dict[str, Any]] = {}
    for row in decision:
        ts = row.get("ts_ms")
        if not isinstance(ts, (int, float)):
            continue
        decision_by_ts[int(ts)] = row

    maker_count = taker_count = 0
    maker_notional = taker_notional = 0.0
    maker_fee = taker_fee = 0.0
    fee_sum = 0.0
    notional_sum = 0.0
    edge_bps_values: List[float] = []
    fee_bps_values: List[float] = []
    net_bps_values: List[float] = []
    markout_by_h: Dict[int, List[float]] = {h: [] for h in horizons_s}
    fills_joined_count = 0
    fills_join_miss_count = 0
    fills_out_of_run_count = 0
    fills_join_dt_ms: List[float] = []
    mid_source_counts: Dict[str, int] = {}
    edge_sign_bad_count = 0
    edge_sign_zero_count = 0
    fills_cloid_missing_count = 0
    fills_cloid_unmatched_count = 0
    fills_cloid_matched_count = 0
    fills_cloid_join_dt_ms: List[float] = []
    fill_effective_spread_bps_values: List[float] = []
    net_bps_by_spread_bucket: Dict[str, List[float]] = {}

    for fill in fills:
        px = _as_float(fill.get("px"))
        sz = _as_float(fill.get("sz"))
        fee = _as_float(fill.get("fee")) or 0.0
        if px is None or sz is None or sz <= 0:
            continue
        notional = px * sz

        ts_ms = fill.get("time") or fill.get("ts_ms")
        if isinstance(ts_ms, (int, float)) and ts_ms < 1_000_000_000_000:
            ts_ms = int(ts_ms * 1000)
        ts_ms_i = int(ts_ms) if isinstance(ts_ms, (int, float)) else None
        if ts_ms_i is not None and market_ts_min is not None and market_ts_max is not None:
            if ts_ms_i < market_ts_min or ts_ms_i > market_ts_max:
                fills_out_of_run_count += 1
                continue

        notional_sum += notional
        fee_sum += fee
        crossed = bool(fill.get("crossed", False))
        if crossed:
            taker_count += 1
            taker_notional += notional
            taker_fee += fee
        else:
            maker_count += 1
            maker_notional += notional
            maker_fee += fee

        if ts_ms_i is None:
            fills_join_miss_count += 1
            continue
        idx = bisect_right(market_ts, ts_ms_i) - 1
        if idx < 0:
            fills_join_miss_count += 1
            continue
        mid = market_mid[idx]
        if mid is None or mid == 0:
            fills_join_miss_count += 1
            continue
        fills_joined_count += 1
        fills_join_dt_ms.append(ts_ms_i - market_ts[idx])
        mid_source = market_mid_source[idx]
        if mid_source:
            mid_source_counts[mid_source] = mid_source_counts.get(mid_source, 0) + 1
        side = str(fill.get("side") or "").upper()
        if side in {"B", "BUY", "BID"}:
            edge_bps = (mid - px) / mid * 1e4
        elif side in {"A", "SELL", "ASK"}:
            edge_bps = (px - mid) / mid * 1e4
        else:
            continue
        if edge_bps < 0:
            edge_sign_bad_count += 1
        elif edge_bps == 0:
            edge_sign_zero_count += 1
        fee_bps = fee / notional * 1e4 if notional > 0 else 0.0
        edge_bps_values.append(edge_bps)
        fee_bps_values.append(fee_bps)
        net_bps = edge_bps - fee_bps
        net_bps_values.append(net_bps)

        fill_cloid = _normalize_cloid(
            fill.get("cloid") or fill.get("client_oid") or fill.get("clientOid") or fill.get("clientOid")
        )
        if fill_cloid is None:
            fills_cloid_missing_count += 1
        else:
            ts_list = order_ts_by_cloid.get(fill_cloid)
            if not ts_list:
                fills_cloid_unmatched_count += 1
            else:
                oi = bisect_right(ts_list, ts_ms_i) - 1
                if oi < 0:
                    fills_cloid_unmatched_count += 1
                else:
                    fills_cloid_matched_count += 1
                    order_ts = ts_list[oi]
                    fills_cloid_join_dt_ms.append(ts_ms_i - order_ts)
                    spread = order_spread_by_cloid.get(fill_cloid, [None])[oi]
                    if spread is not None:
                        fill_effective_spread_bps_values.append(float(spread))
                        try:
                            bucket = format(Decimal(str(spread)).quantize(Decimal("0.01")), "f")
                        except Exception:
                            bucket = f"{float(spread):.2f}"
                        net_bps_by_spread_bucket.setdefault(bucket, []).append(net_bps)

        for h in horizons_s:
            mid_h = _find_mid_after(market_ts, market_mid, ts_ms_i + h * 1000)
            if mid_h is None or mid_h == 0:
                continue
            if side in {"B", "BUY", "BID"}:
                markout = (mid_h - px) / mid_h * 1e4
            else:
                markout = (px - mid_h) / mid_h * 1e4
            markout_by_h[h].append(markout)

    taker_notional_share = (
        taker_notional / notional_sum if notional_sum > 0 else None
    )
    fee_bps_total = fee_sum / notional_sum * 1e4 if notional_sum > 0 else None
    maker_fee_bps = maker_fee / maker_notional * 1e4 if maker_notional > 0 else None
    taker_fee_bps = taker_fee / taker_notional * 1e4 if taker_notional > 0 else None
    join_dt_sorted = sorted(fills_join_dt_ms)
    fills_join_dt_ms_p50 = _percentile(join_dt_sorted, 0.5)
    fills_join_dt_ms_p95 = _percentile(join_dt_sorted, 0.95)
    mid_source = None
    if mid_source_counts:
        if len(mid_source_counts) == 1:
            mid_source = next(iter(mid_source_counts))
        else:
            mid_source = "mixed"
    spread_filter_skip_rate = (
        spread_filter_active_count / spread_filter_total if spread_filter_total else None
    )
    spread_filter_active_rate = (
        (spread_filter_total - spread_filter_active_count) / spread_filter_total
        if spread_filter_total
        else None
    )

    touch_bid = 0
    touch_ask = 0
    touch_bid_total = 0
    touch_ask_total = 0
    for ts, row in decision_by_ts.items():
        idx = bisect_right(market_ts, ts) - 1
        if idx < 0:
            continue
        best_bid = market_bid[idx]
        best_ask = market_ask[idx]
        bid_px = _as_float(row.get("final_bid_px"))
        ask_px = _as_float(row.get("final_ask_px"))
        tick = _price_tick_from_value(best_bid) or _price_tick_from_value(best_ask)
        tol = (tick or 0.0) * 0.5
        if bid_px is not None and best_bid is not None:
            touch_bid_total += 1
            if abs(bid_px - best_bid) <= tol:
                touch_bid += 1
        if ask_px is not None and best_ask is not None:
            touch_ask_total += 1
            if abs(ask_px - best_ask) <= tol:
                touch_ask += 1

    action_counts: Dict[str, int] = {}
    order_ts: List[int] = []
    for row in orders:
        action = row.get("action")
        if isinstance(action, str):
            action_counts[action] = action_counts.get(action, 0) + 1
        ts = row.get("ts_ms")
        if isinstance(ts, (int, float)):
            order_ts.append(int(ts))
    duration_min = None
    if order_ts:
        duration_min = max((max(order_ts) - min(order_ts)) / 60000.0, 1e-6)
    action_rates = {
        k: (v / duration_min) if duration_min else None for k, v in action_counts.items()
    }

    fills_cloid_join_dt_sorted = sorted(fills_cloid_join_dt_ms)
    fills_cloid_join_dt_ms_p50 = _percentile(fills_cloid_join_dt_sorted, 0.5)
    fills_cloid_join_dt_ms_p95 = _percentile(fills_cloid_join_dt_sorted, 0.95)
    net_bps_by_effective_spread_bps = {
        k: _summary_stats(v) for k, v in sorted(net_bps_by_spread_bucket.items(), key=lambda item: item[0])
    }

    report = {
        "run_dir": str(run_dir),
        "decision_ticks": decision_ticks,
        "boost_active_count": boost_active_count,
        "boost_active_rate": boost_active_rate,
        "boost_trigger_count": boost_active_count,
        "boost_trigger_rate": boost_active_rate,
        "fills_total_count": len(fills),
        "fills_count": maker_count + taker_count,
        "fills_joined_count": fills_joined_count,
        "fills_join_miss_count": fills_join_miss_count,
        "fills_out_of_run_count": fills_out_of_run_count,
        "fills_join_dt_ms_p50": fills_join_dt_ms_p50,
        "fills_join_dt_ms_p95": fills_join_dt_ms_p95,
        "mid_source": mid_source,
        "mid_source_counts": mid_source_counts,
        "spread_filter_active_rate": spread_filter_active_rate,
        "spread_filter_skip_rate": spread_filter_skip_rate,
        "spread_filter_active_count": spread_filter_active_count,
        "spread_filter_total_count": spread_filter_total,
        "maker_fills": maker_count,
        "taker_fills": taker_count,
        "taker_guard_trip": taker_count > 0,
        "taker_notional_share": taker_notional_share,
        "notional_sum": notional_sum,
        "fee_sum": fee_sum,
        "fee_bps": fee_bps_total,
        "maker_fee_bps": maker_fee_bps,
        "taker_fee_bps": taker_fee_bps,
        "edge_sign_bad_count": edge_sign_bad_count,
        "edge_sign_zero_count": edge_sign_zero_count,
        "edge_bps": _summary_stats(edge_bps_values),
        "fee_bps_per_fill": _summary_stats(fee_bps_values),
        "net_bps": _summary_stats(net_bps_values),
        "fills_cloid_missing_count": fills_cloid_missing_count,
        "fills_cloid_unmatched_count": fills_cloid_unmatched_count,
        "fills_cloid_matched_count": fills_cloid_matched_count,
        "fills_cloid_join_dt_ms_p50": fills_cloid_join_dt_ms_p50,
        "fills_cloid_join_dt_ms_p95": fills_cloid_join_dt_ms_p95,
        "fill_effective_spread_bps": _summary_stats(fill_effective_spread_bps_values),
        "net_bps_by_effective_spread_bps": net_bps_by_effective_spread_bps,
        "markout_bps": {str(h): _summary_stats(vals) for h, vals in markout_by_h.items()},
        "at_touch_rate": {
            "bid": (touch_bid / touch_bid_total) if touch_bid_total else None,
            "ask": (touch_ask / touch_ask_total) if touch_ask_total else None,
            "bid_count": touch_bid,
            "ask_count": touch_ask,
            "bid_total": touch_bid_total,
            "ask_total": touch_ask_total,
        },
        "order_action_counts": action_counts,
        "order_action_per_min": action_rates,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="mm_live run の簡易レポート生成")
    parser.add_argument("--run-dir", required=True, help="log run ディレクトリ")
    parser.add_argument(
        "--horizons",
        default="5,30",
        help="markout の秒数（カンマ区切り）",
    )
    parser.add_argument("--out", help="出力JSONパス（未指定なら標準出力）")
    args = parser.parse_args()

    horizons = []
    for part in str(args.horizons).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            horizons.append(int(part))
        except ValueError:
            continue
    if not horizons:
        horizons = [5, 30]

    run_dir = Path(args.run_dir)
    report = build_report(run_dir, horizons)
    text = json.dumps(report, ensure_ascii=True, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
