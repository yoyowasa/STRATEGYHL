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


def _is_post_only_reject_error(message: Any) -> bool:
    if not isinstance(message, str):
        return False
    msg = message.lower()
    return (
        "post-only" in msg
        or "post only" in msg
        or "postonly" in msg
        or "would immediately match" in msg
        or "would take" in msg
        or "would cross" in msg
    )


def _is_rate_limit_error_message(message: Any) -> bool:
    if not isinstance(message, str):
        return False
    return "too many cumulative requests" in message.lower()


def _is_cannot_modify_error_message(message: Any) -> bool:
    if not isinstance(message, str):
        return False
    return "cannot modify canceled or filled order" in message.lower()


def _is_one_tick_spread(spread_ticks: Any, tol: float = 0.25) -> bool:
    v = _as_float(spread_ticks)
    if v is None:
        return False
    return abs(v - 1.0) <= tol


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
        return {
            "count": 0,
            "mean": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p10": None,
            "p90": None,
            "p99": None,
        }
    vals = sorted(values)
    mean = sum(vals) / len(vals)
    return {
        "count": len(vals),
        "mean": mean,
        "p25": _percentile(vals, 0.25),
        "median": _percentile(vals, 0.5),
        "p75": _percentile(vals, 0.75),
        "p10": _percentile(vals, 0.1),
        "p90": _percentile(vals, 0.9),
        "p99": _percentile(vals, 0.99),
    }


def _new_fill_bucket(horizons_s: List[int]) -> Dict[str, Any]:
    return {
        "realized_spread_bps": [],
        "net_after_fee_bps": [],
        "markout_bps": {h: [] for h in horizons_s},
    }


def _append_fill_bucket(
    bucket: Dict[str, Any],
    edge_bps: float,
    net_bps: float,
    markout_values: Dict[int, float],
) -> None:
    bucket["realized_spread_bps"].append(edge_bps)
    bucket["net_after_fee_bps"].append(net_bps)
    markout_bucket = bucket.get("markout_bps")
    if not isinstance(markout_bucket, dict):
        return
    for h, val in markout_values.items():
        vals = markout_bucket.get(h)
        if isinstance(vals, list):
            vals.append(val)


def _summarize_fill_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    markout = bucket.get("markout_bps")
    markout_summary: Dict[str, Dict[str, Optional[float]]] = {}
    if isinstance(markout, dict):
        for h, vals in markout.items():
            if isinstance(vals, list):
                markout_summary[str(h)] = _summary_stats(vals)
    return {
        "fills": len(bucket.get("realized_spread_bps", [])),
        "realized_spread_bps": _summary_stats(bucket.get("realized_spread_bps", [])),
        "net_after_fee_bps": _summary_stats(bucket.get("net_after_fee_bps", [])),
        "markout_bps": markout_summary,
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
    events = _read_jsonl(run_dir / "events.jsonl")

    orders_by_cloid: Dict[str, List[Tuple[int, Optional[float], bool, bool, str, str, bool, bool]]] = {}
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
        reject_backoff_active = bool(row.get("reject_backoff_active"))
        risk_escape_applied = bool(row.get("risk_escape_applied"))
        risk_escape_bucket = str(row.get("risk_escape_bucket") or "none")
        risk_escape_mode = str(row.get("risk_escape_mode") or "none")
        risk_size_down_applied = bool(row.get("risk_size_down_applied"))
        risk_escape_global_lag_quiet_trigger = bool(row.get("risk_escape_global_lag_quiet_trigger"))
        orders_by_cloid.setdefault(cloid, []).append(
            (
                int(ts),
                spread,
                reject_backoff_active,
                risk_escape_applied,
                risk_escape_bucket,
                risk_escape_mode,
                risk_size_down_applied,
                risk_escape_global_lag_quiet_trigger,
            )
        )
    order_ts_by_cloid: Dict[str, List[int]] = {}
    order_spread_by_cloid: Dict[str, List[Optional[float]]] = {}
    order_backoff_active_by_cloid: Dict[str, List[bool]] = {}
    order_risk_escape_applied_by_cloid: Dict[str, List[bool]] = {}
    order_risk_escape_bucket_by_cloid: Dict[str, List[str]] = {}
    order_risk_size_down_applied_by_cloid: Dict[str, List[bool]] = {}
    order_global_lag_quiet_by_cloid: Dict[str, List[bool]] = {}
    for cloid, entries in orders_by_cloid.items():
        entries.sort(key=lambda item: item[0])
        order_ts_by_cloid[cloid] = [t for t, _, _, _, _, _, _, _ in entries]
        order_spread_by_cloid[cloid] = [s for _, s, _, _, _, _, _, _ in entries]
        order_backoff_active_by_cloid[cloid] = [b for _, _, b, _, _, _, _, _ in entries]
        order_risk_escape_applied_by_cloid[cloid] = [r for _, _, _, r, _, _, _, _ in entries]
        order_risk_escape_bucket_by_cloid[cloid] = [k for _, _, _, _, k, _, _, _ in entries]
        order_risk_size_down_applied_by_cloid[cloid] = [sd for _, _, _, _, _, _, sd, _ in entries]
        order_global_lag_quiet_by_cloid[cloid] = [g for _, _, _, _, _, _, _, g in entries]

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
    decision_noop_count = 0
    decision_noop_halt_count = 0
    decision_noop_stop_count = 0
    decision_noop_spread_filter_count = 0
    for row in decision:
        ts = row.get("ts_ms")
        if not isinstance(ts, (int, float)):
            continue
        decision_ticks += 1
        if row.get("boost_active"):
            boost_active_count += 1
        raw_bid_sz = _as_float(row.get("raw_bid_sz"))
        raw_ask_sz = _as_float(row.get("raw_ask_sz"))
        if (raw_bid_sz is None or raw_bid_sz <= 0.0) and (raw_ask_sz is None or raw_ask_sz <= 0.0):
            decision_noop_count += 1
            if bool(row.get("halt_triggered")):
                decision_noop_halt_count += 1
            if bool(row.get("stop_triggered")):
                decision_noop_stop_count += 1
            if bool(row.get("spread_filter_active")):
                decision_noop_spread_filter_count += 1
    boost_active_rate = (
        boost_active_count / decision_ticks if decision_ticks else None
    )
    decision_noop_rate = (
        decision_noop_count / decision_ticks if decision_ticks else None
    )

    decision_by_ts: Dict[int, Dict[str, Any]] = {}
    for row in decision:
        ts = row.get("ts_ms")
        if not isinstance(ts, (int, float)):
            continue
        decision_by_ts[int(ts)] = row
    decision_ts_sorted = sorted(decision_by_ts.items(), key=lambda item: item[0])
    decision_ts_list: List[int] = [ts for ts, _ in decision_ts_sorted]
    decision_boost_list: List[bool] = [
        bool(row.get("boost_active", row.get("boost_triggered", False))) for _, row in decision_ts_sorted
    ]

    maker_count = taker_count = 0
    maker_notional = taker_notional = 0.0
    maker_fee = taker_fee = 0.0
    fee_sum = 0.0
    notional_sum = 0.0
    edge_bps_values: List[float] = []
    fee_bps_values: List[float] = []
    net_bps_values: List[float] = []
    markout_by_h: Dict[int, List[float]] = {h: [] for h in horizons_s}
    bucket_by_side: Dict[str, Dict[str, Any]] = {
        "buy": _new_fill_bucket(horizons_s),
        "sell": _new_fill_bucket(horizons_s),
    }
    bucket_by_boost: Dict[str, Dict[str, Any]] = {
        "boost": _new_fill_bucket(horizons_s),
        "base": _new_fill_bucket(horizons_s),
        "unknown": _new_fill_bucket(horizons_s),
    }
    bucket_by_boost_side: Dict[str, Dict[str, Any]] = {
        "boost_buy": _new_fill_bucket(horizons_s),
        "boost_sell": _new_fill_bucket(horizons_s),
        "base_buy": _new_fill_bucket(horizons_s),
        "base_sell": _new_fill_bucket(horizons_s),
        "unknown_buy": _new_fill_bucket(horizons_s),
        "unknown_sell": _new_fill_bucket(horizons_s),
    }
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
    send_latency_ms_values: List[float] = []
    pre_send_book_age_ms_values: List[float] = []
    pre_send_recv_age_ms_values: List[float] = []
    pre_send_recv_age_ms_skip_values: List[float] = []
    pre_send_recv_age_ms_all_values: List[float] = []
    post_only_reject_class1_count = 0
    post_only_reject_class2_count = 0
    post_only_reject_by_spread_ticks: Dict[str, int] = {}
    post_only_reject_by_round_mode: Dict[str, int] = {}
    post_only_reject_class2_by_spread_ticks: Dict[str, int] = {}
    post_only_reject_class2_by_round_mode: Dict[str, int] = {}
    post_only_reject_class2_by_side: Dict[str, int] = {}
    post_only_reject_class2_by_side_spread_ticks: Dict[str, int] = {}
    post_only_reject_by_side_spread_ticks: Dict[str, int] = {}
    post_only_reject_by_risk_escape_bucket: Dict[str, int] = {}
    post_only_reject_by_risk_escape_applied: Dict[str, int] = {}
    post_only_reject_by_bucket_applied: Dict[str, int] = {}
    post_only_reject_by_risk_size_down: Dict[str, int] = {}
    post_only_reject_by_bucket_size_down: Dict[str, int] = {}
    post_only_reject_class2_send_latency_ms_values: List[float] = []
    post_only_reject_class2_pre_send_recv_age_ms_values: List[float] = []
    post_only_reject_class2_sell_spread1_count = 0
    backoff_active_sent_count = 0
    backoff_inactive_sent_count = 0
    backoff_active_reject_count = 0
    backoff_inactive_reject_count = 0
    backoff_active_fills_count = 0
    backoff_inactive_fills_count = 0
    backoff_active_net_bps_values: List[float] = []
    backoff_inactive_net_bps_values: List[float] = []
    backoff_active_markout30_bps_values: List[float] = []
    backoff_inactive_markout30_bps_values: List[float] = []
    risk_escape_applied_sent_count = 0
    risk_escape_unapplied_sent_count = 0
    risk_escape_triggered_sent_count = 0
    risk_escape_untriggered_sent_count = 0
    risk_escape_applied_reject_count = 0
    risk_escape_unapplied_reject_count = 0
    risk_escape_applied_fills_count = 0
    risk_escape_unapplied_fills_count = 0
    risk_escape_applied_net_bps_values: List[float] = []
    risk_escape_unapplied_net_bps_values: List[float] = []
    risk_escape_applied_markout30_bps_values: List[float] = []
    risk_escape_unapplied_markout30_bps_values: List[float] = []
    risk_size_down_applied_sent_count = 0
    risk_size_down_inactive_sent_count = 0
    risk_size_down_applied_reject_count = 0
    risk_size_down_inactive_reject_count = 0
    risk_size_down_applied_fills_count = 0
    risk_size_down_inactive_fills_count = 0
    risk_size_down_applied_net_bps_values: List[float] = []
    risk_size_down_inactive_net_bps_values: List[float] = []
    risk_size_down_applied_markout30_bps_values: List[float] = []
    risk_size_down_inactive_markout30_bps_values: List[float] = []
    risk_size_down_applied_fills_count_by_bucket: Dict[str, int] = {}
    risk_size_down_inactive_fills_count_by_bucket: Dict[str, int] = {}
    risk_size_down_applied_net_bps_by_bucket: Dict[str, List[float]] = {}
    risk_size_down_inactive_net_bps_by_bucket: Dict[str, List[float]] = {}
    risk_size_down_applied_markout30_bps_by_bucket: Dict[str, List[float]] = {}
    risk_size_down_inactive_markout30_bps_by_bucket: Dict[str, List[float]] = {}
    global_lag_quiet_triggered_sent_count = 0
    global_lag_quiet_triggered_reject_count = 0
    global_lag_quiet_triggered_fills_count = 0
    global_lag_quiet_triggered_count_by_bucket: Dict[str, int] = {}
    global_lag_quiet_triggered_net_bps_values: List[float] = []
    global_lag_quiet_triggered_markout30_bps_values: List[float] = []
    risk_escape_triggered_count_by_side: Dict[str, int] = {}
    risk_escape_trigger_bucket_counts: Dict[str, int] = {}
    risk_escape_sent_count_by_mode: Dict[str, int] = {}
    risk_escape_applied_count_by_side: Dict[str, int] = {}
    risk_escape_applied_bucket_counts: Dict[str, int] = {}
    risk_escape_skip_sent_count = 0
    risk_escape_skip_count_by_side: Dict[str, int] = {}
    risk_escape_skip_bucket_counts: Dict[str, int] = {}
    risk_escape_skip_count_by_side_bucket: Dict[str, int] = {}
    risk_escape_applied_fills_count_by_bucket: Dict[str, int] = {}
    risk_escape_unapplied_fills_count_by_bucket: Dict[str, int] = {}
    risk_escape_applied_net_bps_by_bucket: Dict[str, List[float]] = {}
    risk_escape_unapplied_net_bps_by_bucket: Dict[str, List[float]] = {}
    risk_escape_applied_markout30_bps_by_bucket: Dict[str, List[float]] = {}
    risk_escape_unapplied_markout30_bps_by_bucket: Dict[str, List[float]] = {}
    risk_escape_delta_ticks_triggered_values: List[float] = []
    risk_escape_delta_ticks_applied_values: List[float] = []
    latency_bucket_cancel_sent_count = 0
    latency_bucket_cancel_success_count = 0
    latency_bucket_cancel_skip_count = 0
    latency_bucket_cancel_sent_by_side: Dict[str, int] = {}
    latency_bucket_cancel_sent_by_bucket: Dict[str, int] = {}
    latency_bucket_cancel_sent_by_side_bucket: Dict[str, int] = {}
    latency_bucket_cancel_success_by_side: Dict[str, int] = {}
    latency_bucket_cancel_success_by_bucket: Dict[str, int] = {}
    latency_bucket_cancel_skip_by_side: Dict[str, int] = {}
    latency_bucket_cancel_skip_by_bucket: Dict[str, int] = {}
    latency_bucket_cancel_skip_reason_counts: Dict[str, int] = {}
    latency_bucket_cancel_attempt_needed_count = 0
    latency_bucket_cancel_attempt_not_needed_count = 0
    latency_bucket_cancel_attempt_needed_by_side: Dict[str, int] = {}
    latency_bucket_cancel_attempt_needed_by_bucket: Dict[str, int] = {}
    latency_bucket_cancel_attempt_not_needed_by_side: Dict[str, int] = {}
    latency_bucket_cancel_attempt_not_needed_by_bucket: Dict[str, int] = {}
    risk_escape_cancel_and_skip_only_count = 0
    risk_escape_cancel_all_fallback_sent_count = 0
    risk_escape_cancel_all_fallback_success_count = 0
    risk_escape_cancel_all_fallback_skip_count = 0
    risk_escape_cancel_all_fallback_skip_reason_counts: Dict[str, int] = {}
    pre_send_marketable_sent_count = 0
    cannot_modify_count = 0
    cannot_modify_by_action: Dict[str, int] = {}
    rate_limit_error_count = 0

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
        side_key: Optional[str] = None
        if side in {"B", "BUY", "BID"}:
            side_key = "buy"
            edge_bps = (mid - px) / mid * 1e4
        elif side in {"A", "SELL", "ASK"}:
            side_key = "sell"
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

        fill_backoff_active: Optional[bool] = None
        fill_risk_escape_applied: Optional[bool] = None
        fill_risk_escape_bucket = "none"
        fill_risk_size_down_applied: Optional[bool] = None
        fill_global_lag_quiet_triggered: Optional[bool] = None
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
                    backoff_active_seq = order_backoff_active_by_cloid.get(fill_cloid, [])
                    backoff_active_fill = (
                        bool(backoff_active_seq[oi]) if oi < len(backoff_active_seq) else False
                    )
                    fill_backoff_active = backoff_active_fill
                    if backoff_active_fill:
                        backoff_active_fills_count += 1
                        backoff_active_net_bps_values.append(net_bps)
                    else:
                        backoff_inactive_fills_count += 1
                        backoff_inactive_net_bps_values.append(net_bps)
                    risk_escape_applied_seq = order_risk_escape_applied_by_cloid.get(fill_cloid, [])
                    risk_escape_bucket_seq = order_risk_escape_bucket_by_cloid.get(fill_cloid, [])
                    risk_size_down_applied_seq = order_risk_size_down_applied_by_cloid.get(
                        fill_cloid, []
                    )
                    global_lag_quiet_seq = order_global_lag_quiet_by_cloid.get(fill_cloid, [])
                    risk_escape_applied_fill = (
                        bool(risk_escape_applied_seq[oi]) if oi < len(risk_escape_applied_seq) else False
                    )
                    risk_size_down_applied_fill = (
                        bool(risk_size_down_applied_seq[oi])
                        if oi < len(risk_size_down_applied_seq)
                        else False
                    )
                    global_lag_quiet_fill = (
                        bool(global_lag_quiet_seq[oi]) if oi < len(global_lag_quiet_seq) else False
                    )
                    if oi < len(risk_escape_bucket_seq):
                        fill_risk_escape_bucket = str(risk_escape_bucket_seq[oi] or "none")
                    fill_risk_escape_applied = risk_escape_applied_fill
                    fill_risk_size_down_applied = risk_size_down_applied_fill
                    fill_global_lag_quiet_triggered = global_lag_quiet_fill
                    if risk_escape_applied_fill:
                        risk_escape_applied_fills_count += 1
                        risk_escape_applied_net_bps_values.append(net_bps)
                        risk_escape_applied_fills_count_by_bucket[fill_risk_escape_bucket] = (
                            risk_escape_applied_fills_count_by_bucket.get(fill_risk_escape_bucket, 0) + 1
                        )
                        risk_escape_applied_net_bps_by_bucket.setdefault(
                            fill_risk_escape_bucket, []
                        ).append(net_bps)
                    else:
                        risk_escape_unapplied_fills_count += 1
                        risk_escape_unapplied_net_bps_values.append(net_bps)
                        risk_escape_unapplied_fills_count_by_bucket[fill_risk_escape_bucket] = (
                            risk_escape_unapplied_fills_count_by_bucket.get(fill_risk_escape_bucket, 0) + 1
                        )
                        risk_escape_unapplied_net_bps_by_bucket.setdefault(
                            fill_risk_escape_bucket, []
                        ).append(net_bps)
                    if risk_size_down_applied_fill:
                        risk_size_down_applied_fills_count += 1
                        risk_size_down_applied_net_bps_values.append(net_bps)
                        risk_size_down_applied_fills_count_by_bucket[fill_risk_escape_bucket] = (
                            risk_size_down_applied_fills_count_by_bucket.get(
                                fill_risk_escape_bucket, 0
                            )
                            + 1
                        )
                        risk_size_down_applied_net_bps_by_bucket.setdefault(
                            fill_risk_escape_bucket, []
                        ).append(net_bps)
                    else:
                        risk_size_down_inactive_fills_count += 1
                        risk_size_down_inactive_net_bps_values.append(net_bps)
                        risk_size_down_inactive_fills_count_by_bucket[fill_risk_escape_bucket] = (
                            risk_size_down_inactive_fills_count_by_bucket.get(
                                fill_risk_escape_bucket, 0
                            )
                            + 1
                        )
                        risk_size_down_inactive_net_bps_by_bucket.setdefault(
                            fill_risk_escape_bucket, []
                        ).append(net_bps)
                    if global_lag_quiet_fill:
                        global_lag_quiet_triggered_fills_count += 1
                        global_lag_quiet_triggered_count_by_bucket[fill_risk_escape_bucket] = (
                            global_lag_quiet_triggered_count_by_bucket.get(
                                fill_risk_escape_bucket, 0
                            )
                            + 1
                        )
                        global_lag_quiet_triggered_net_bps_values.append(net_bps)
                    spread = order_spread_by_cloid.get(fill_cloid, [None])[oi]
                    if spread is not None:
                        fill_effective_spread_bps_values.append(float(spread))
                        try:
                            bucket = format(Decimal(str(spread)).quantize(Decimal("0.01")), "f")
                        except Exception:
                            bucket = f"{float(spread):.2f}"
                        net_bps_by_spread_bucket.setdefault(bucket, []).append(net_bps)

        boost_key = "unknown"
        if decision_ts_list:
            di = bisect_right(decision_ts_list, ts_ms_i) - 1
            if di >= 0:
                boost_key = "boost" if decision_boost_list[di] else "base"

        markout_this_fill: Dict[int, float] = {}
        for h in horizons_s:
            mid_h = _find_mid_after(market_ts, market_mid, ts_ms_i + h * 1000)
            if mid_h is None or mid_h == 0:
                continue
            if side in {"B", "BUY", "BID"}:
                markout = (mid_h - px) / mid_h * 1e4
            else:
                markout = (px - mid_h) / mid_h * 1e4
            markout_by_h[h].append(markout)
            markout_this_fill[h] = markout

        markout30 = markout_this_fill.get(30)
        if markout30 is not None:
            if fill_backoff_active is True:
                backoff_active_markout30_bps_values.append(markout30)
            elif fill_backoff_active is False:
                backoff_inactive_markout30_bps_values.append(markout30)
            if fill_risk_escape_applied is True:
                risk_escape_applied_markout30_bps_values.append(markout30)
                risk_escape_applied_markout30_bps_by_bucket.setdefault(
                    fill_risk_escape_bucket, []
                ).append(markout30)
            elif fill_risk_escape_applied is False:
                risk_escape_unapplied_markout30_bps_values.append(markout30)
                risk_escape_unapplied_markout30_bps_by_bucket.setdefault(
                    fill_risk_escape_bucket, []
                ).append(markout30)
            if fill_risk_size_down_applied is True:
                risk_size_down_applied_markout30_bps_values.append(markout30)
                risk_size_down_applied_markout30_bps_by_bucket.setdefault(
                    fill_risk_escape_bucket, []
                ).append(markout30)
            elif fill_risk_size_down_applied is False:
                risk_size_down_inactive_markout30_bps_values.append(markout30)
                risk_size_down_inactive_markout30_bps_by_bucket.setdefault(
                    fill_risk_escape_bucket, []
                ).append(markout30)
            if fill_global_lag_quiet_triggered is True:
                global_lag_quiet_triggered_markout30_bps_values.append(markout30)

        if side_key is not None:
            _append_fill_bucket(bucket_by_side[side_key], edge_bps, net_bps, markout_this_fill)
            _append_fill_bucket(bucket_by_boost[boost_key], edge_bps, net_bps, markout_this_fill)
            boost_side_key = f"{boost_key}_{side_key}"
            if boost_side_key in bucket_by_boost_side:
                _append_fill_bucket(bucket_by_boost_side[boost_side_key], edge_bps, net_bps, markout_this_fill)

    for row in orders:
        action = row.get("action")
        is_sent_order = action in {"new", "replace"}
        error_text = row.get("error")
        if _is_cannot_modify_error_message(error_text):
            cannot_modify_count += 1
            action_key = str(action or "-")
            cannot_modify_by_action[action_key] = cannot_modify_by_action.get(action_key, 0) + 1
        if _is_rate_limit_error_message(error_text):
            rate_limit_error_count += 1
        risk_escape_reason = str(row.get("reason") or "").lower()
        if action == "cancel" and risk_escape_reason == "risk_escape_cancel_and_skip":
            side = str(row.get("side") or "-").lower()
            side_key = side if side in {"buy", "sell"} else "-"
            bucket = str(row.get("risk_escape_bucket") or "none")
            side_bucket_key = f"{side_key}|{bucket}"
            latency_bucket_cancel_sent_count += 1
            latency_bucket_cancel_sent_by_side[side_key] = (
                latency_bucket_cancel_sent_by_side.get(side_key, 0) + 1
            )
            latency_bucket_cancel_sent_by_bucket[bucket] = (
                latency_bucket_cancel_sent_by_bucket.get(bucket, 0) + 1
            )
            latency_bucket_cancel_sent_by_side_bucket[side_bucket_key] = (
                latency_bucket_cancel_sent_by_side_bucket.get(side_bucket_key, 0) + 1
            )
            if str(row.get("status") or "") != "error":
                latency_bucket_cancel_success_count += 1
                latency_bucket_cancel_success_by_side[side_key] = (
                    latency_bucket_cancel_success_by_side.get(side_key, 0) + 1
                )
                latency_bucket_cancel_success_by_bucket[bucket] = (
                    latency_bucket_cancel_success_by_bucket.get(bucket, 0) + 1
                )
        if action == "cancel_all" and risk_escape_reason == "risk_escape_cancel_and_skip_fallback":
            risk_escape_cancel_all_fallback_sent_count += 1
            if str(row.get("status") or "") != "error":
                risk_escape_cancel_all_fallback_success_count += 1
        reject_backoff_active = bool(row.get("reject_backoff_active"))
        risk_escape_triggered = bool(row.get("risk_escape_triggered"))
        risk_escape_applied = bool(row.get("risk_escape_applied"))
        risk_escape_bucket = str(row.get("risk_escape_bucket") or "none")
        risk_size_down_applied = bool(row.get("risk_size_down_applied"))
        global_lag_quiet_triggered = bool(row.get("risk_escape_global_lag_quiet_trigger"))
        risk_escape_delta_ticks = _as_float(row.get("risk_escape_delta_ticks"))
        if is_sent_order:
            if reject_backoff_active:
                backoff_active_sent_count += 1
            else:
                backoff_inactive_sent_count += 1
            if risk_size_down_applied:
                risk_size_down_applied_sent_count += 1
            else:
                risk_size_down_inactive_sent_count += 1
            if global_lag_quiet_triggered:
                global_lag_quiet_triggered_sent_count += 1
            side_key = str(row.get("side") or "-").lower()
            risk_escape_mode = str(row.get("risk_escape_mode") or "none").lower()
            risk_escape_sent_count_by_mode[risk_escape_mode] = (
                risk_escape_sent_count_by_mode.get(risk_escape_mode, 0) + 1
            )
            if risk_escape_triggered:
                risk_escape_triggered_sent_count += 1
                if side_key in {"buy", "sell"}:
                    risk_escape_triggered_count_by_side[side_key] = (
                        risk_escape_triggered_count_by_side.get(side_key, 0) + 1
                    )
                risk_escape_trigger_bucket_counts[risk_escape_bucket] = (
                    risk_escape_trigger_bucket_counts.get(risk_escape_bucket, 0) + 1
                )
                if risk_escape_delta_ticks is not None:
                    risk_escape_delta_ticks_triggered_values.append(risk_escape_delta_ticks)
            else:
                risk_escape_untriggered_sent_count += 1
            if risk_escape_applied:
                risk_escape_applied_sent_count += 1
                if side_key in {"buy", "sell"}:
                    risk_escape_applied_count_by_side[side_key] = (
                        risk_escape_applied_count_by_side.get(side_key, 0) + 1
                    )
                risk_escape_applied_bucket_counts[risk_escape_bucket] = (
                    risk_escape_applied_bucket_counts.get(risk_escape_bucket, 0) + 1
                )
                if risk_escape_delta_ticks is not None:
                    risk_escape_delta_ticks_applied_values.append(risk_escape_delta_ticks)
            else:
                risk_escape_unapplied_sent_count += 1
        send_latency_ms = _as_float(row.get("send_latency_ms"))
        if send_latency_ms is not None and send_latency_ms >= 0:
            send_latency_ms_values.append(send_latency_ms)
        pre_send_book_age_ms = _as_float(row.get("pre_send_book_age_ms"))
        if pre_send_book_age_ms is not None and pre_send_book_age_ms >= 0:
            pre_send_book_age_ms_values.append(pre_send_book_age_ms)
        pre_send_recv_age_ms = _as_float(row.get("pre_send_recv_age_ms"))
        if pre_send_recv_age_ms is not None and pre_send_recv_age_ms >= 0:
            pre_send_recv_age_ms_values.append(pre_send_recv_age_ms)
            pre_send_recv_age_ms_all_values.append(pre_send_recv_age_ms)
        if is_sent_order and bool(row.get("pre_send_marketable")):
            pre_send_marketable_sent_count += 1
        if not _is_post_only_reject_error(row.get("error")):
            continue
        side = str(row.get("side") or "-").lower()
        if side not in {"buy", "sell"}:
            side = "-"
        if reject_backoff_active:
            backoff_active_reject_count += 1
        else:
            backoff_inactive_reject_count += 1
        if risk_escape_applied:
            risk_escape_applied_reject_count += 1
        else:
            risk_escape_unapplied_reject_count += 1
        if risk_size_down_applied:
            risk_size_down_applied_reject_count += 1
        else:
            risk_size_down_inactive_reject_count += 1
        if global_lag_quiet_triggered:
            global_lag_quiet_triggered_reject_count += 1
        if bool(row.get("pre_send_marketable")):
            post_only_reject_class1_count += 1
        else:
            post_only_reject_class2_count += 1
            post_only_reject_class2_by_side[side] = post_only_reject_class2_by_side.get(side, 0) + 1
        spread_ticks = _as_float(row.get("pre_send_spread_ticks"))
        spread_bucket = "-" if spread_ticks is None else f"{spread_ticks:.1f}"
        post_only_reject_by_spread_ticks[spread_bucket] = (
            post_only_reject_by_spread_ticks.get(spread_bucket, 0) + 1
        )
        side_spread_key_all = f"{side}|{spread_bucket}"
        post_only_reject_by_side_spread_ticks[side_spread_key_all] = (
            post_only_reject_by_side_spread_ticks.get(side_spread_key_all, 0) + 1
        )
        risk_escape_bucket_key = str(risk_escape_bucket or "none")
        post_only_reject_by_risk_escape_bucket[risk_escape_bucket_key] = (
            post_only_reject_by_risk_escape_bucket.get(risk_escape_bucket_key, 0) + 1
        )
        applied_key = "applied" if risk_escape_applied else "inactive"
        post_only_reject_by_risk_escape_applied[applied_key] = (
            post_only_reject_by_risk_escape_applied.get(applied_key, 0) + 1
        )
        bucket_applied_key = f"{risk_escape_bucket_key}|{applied_key}"
        post_only_reject_by_bucket_applied[bucket_applied_key] = (
            post_only_reject_by_bucket_applied.get(bucket_applied_key, 0) + 1
        )
        size_down_key = "size_down_applied" if risk_size_down_applied else "size_down_inactive"
        post_only_reject_by_risk_size_down[size_down_key] = (
            post_only_reject_by_risk_size_down.get(size_down_key, 0) + 1
        )
        bucket_size_down_key = f"{risk_escape_bucket_key}|{size_down_key}"
        post_only_reject_by_bucket_size_down[bucket_size_down_key] = (
            post_only_reject_by_bucket_size_down.get(bucket_size_down_key, 0) + 1
        )
        round_mode = row.get("pre_send_round_mode")
        round_key = str(round_mode) if round_mode is not None else "-"
        post_only_reject_by_round_mode[round_key] = post_only_reject_by_round_mode.get(round_key, 0) + 1
        if not bool(row.get("pre_send_marketable")):
            post_only_reject_class2_by_spread_ticks[spread_bucket] = (
                post_only_reject_class2_by_spread_ticks.get(spread_bucket, 0) + 1
            )
            side_spread_key = f"{side}|{spread_bucket}"
            post_only_reject_class2_by_side_spread_ticks[side_spread_key] = (
                post_only_reject_class2_by_side_spread_ticks.get(side_spread_key, 0) + 1
            )
            post_only_reject_class2_by_round_mode[round_key] = (
                post_only_reject_class2_by_round_mode.get(round_key, 0) + 1
            )
            if side == "sell" and _is_one_tick_spread(spread_ticks):
                post_only_reject_class2_sell_spread1_count += 1
            if send_latency_ms is not None and send_latency_ms >= 0:
                post_only_reject_class2_send_latency_ms_values.append(send_latency_ms)
            if pre_send_recv_age_ms is not None and pre_send_recv_age_ms >= 0:
                post_only_reject_class2_pre_send_recv_age_ms_values.append(pre_send_recv_age_ms)

    replace_skip_recent_cancel_all = 0
    replace_skip_oid_missing_stage2 = 0
    replace_skip_dead_order_id = 0
    replace_skip_cancel_inflight = 0
    dead_order_id_mark_count = 0
    cancel_inflight_mark_count = 0
    cancel_inflight_clear_count = 0
    cancel_inflight_mark_count_by_side: Dict[str, int] = {}
    cancel_inflight_clear_count_by_side: Dict[str, int] = {}
    dead_order_id_mark_by_side: Dict[str, int] = {}
    resync_ok = 0
    resync_fail = 0
    last_sync_ts_for_age: Optional[int] = None
    open_orders_age_ms_samples: List[float] = []
    quote_gap_ms_by_side: Dict[str, List[float]] = {"buy": [], "sell": []}
    quote_gap_start_ms: Dict[str, Optional[int]] = {"buy": None, "sell": None}
    last_event_ts: Optional[int] = None
    send_skip_reason_counts: Dict[str, int] = {}
    quote_skip_reason_counts: Dict[str, int] = {}
    pre_send_book_stale_skip_count = 0
    pre_send_guard_all_skipped_count = 0
    reject_backoff_trigger_count = 0
    reject_backoff_clear_count = 0
    reject_backoff_trigger_by_side: Dict[str, int] = {}
    reject_backoff_active_ms_total = 0
    reject_backoff_active_ms_by_side: Dict[str, int] = {"buy": 0, "sell": 0}
    for row in events:
        ts = row.get("ts_ms")
        if not isinstance(ts, (int, float)):
            continue
        ts_i = int(ts)
        last_event_ts = ts_i
        event = str(row.get("event") or "")
        reason = str(row.get("reason") or "")
        reason_lc = reason.lower()

        if event == "send_skip":
            reason_key = reason if reason else "-"
            send_skip_reason_counts[reason_key] = send_skip_reason_counts.get(reason_key, 0) + 1
            if reason_lc == "pre_send_stale_book":
                pre_send_book_stale_skip_count += 1
                pre_send_recv_age_ms = _as_float(row.get("pre_send_recv_age_ms"))
                if pre_send_recv_age_ms is not None and pre_send_recv_age_ms >= 0:
                    pre_send_recv_age_ms_skip_values.append(pre_send_recv_age_ms)
                    pre_send_recv_age_ms_all_values.append(pre_send_recv_age_ms)
            elif reason_lc == "pre_send_guard_all_skipped":
                pre_send_guard_all_skipped_count += 1
            elif reason_lc == "risk_escape_cancel_and_skip_only":
                risk_escape_cancel_and_skip_only_count += 1
        elif event == "risk_escape_skip":
            risk_escape_skip_sent_count += 1
            side = str(row.get("side") or "-").lower()
            side_key = side if side in {"buy", "sell"} else "-"
            bucket = str(row.get("bucket") or "none")
            risk_escape_skip_count_by_side[side_key] = (
                risk_escape_skip_count_by_side.get(side_key, 0) + 1
            )
            risk_escape_skip_bucket_counts[bucket] = (
                risk_escape_skip_bucket_counts.get(bucket, 0) + 1
            )
            side_bucket_key = f"{side_key}|{bucket}"
            risk_escape_skip_count_by_side_bucket[side_bucket_key] = (
                risk_escape_skip_count_by_side_bucket.get(side_bucket_key, 0) + 1
            )
        elif event == "risk_escape_cancel_and_skip":
            side = str(row.get("side") or "-").lower()
            side_key = side if side in {"buy", "sell"} else "-"
            bucket = str(row.get("bucket") or "none")
            decision = str(row.get("decision") or "").lower()
            cancel_needed = bool(row.get("cancel_needed"))
            if decision in {"skip_only", "cancel_enqueued"}:
                if cancel_needed:
                    latency_bucket_cancel_attempt_needed_count += 1
                    latency_bucket_cancel_attempt_needed_by_side[side_key] = (
                        latency_bucket_cancel_attempt_needed_by_side.get(side_key, 0) + 1
                    )
                    latency_bucket_cancel_attempt_needed_by_bucket[bucket] = (
                        latency_bucket_cancel_attempt_needed_by_bucket.get(bucket, 0) + 1
                    )
                else:
                    latency_bucket_cancel_attempt_not_needed_count += 1
                    latency_bucket_cancel_attempt_not_needed_by_side[side_key] = (
                        latency_bucket_cancel_attempt_not_needed_by_side.get(side_key, 0) + 1
                    )
                    latency_bucket_cancel_attempt_not_needed_by_bucket[bucket] = (
                        latency_bucket_cancel_attempt_not_needed_by_bucket.get(bucket, 0) + 1
                    )
            if decision == "skip_only":
                latency_bucket_cancel_skip_count += 1
                latency_bucket_cancel_skip_by_side[side_key] = (
                    latency_bucket_cancel_skip_by_side.get(side_key, 0) + 1
                )
                latency_bucket_cancel_skip_by_bucket[bucket] = (
                    latency_bucket_cancel_skip_by_bucket.get(bucket, 0) + 1
                )
                skip_reason = str(row.get("skip_reason") or "unknown")
                latency_bucket_cancel_skip_reason_counts[skip_reason] = (
                    latency_bucket_cancel_skip_reason_counts.get(skip_reason, 0) + 1
                )
        elif event == "risk_escape_cancel_all_fallback":
            decision = str(row.get("decision") or "").lower()
            if decision in {"cooldown_skip", "rate_limit_cooldown_skip", "run_cap_skip"}:
                risk_escape_cancel_all_fallback_skip_count += 1
                skip_reason = decision
                risk_escape_cancel_all_fallback_skip_reason_counts[skip_reason] = (
                    risk_escape_cancel_all_fallback_skip_reason_counts.get(skip_reason, 0) + 1
                )
        elif event == "quote_skip":
            reason_key = reason if reason else "-"
            quote_skip_reason_counts[reason_key] = quote_skip_reason_counts.get(reason_key, 0) + 1
        elif event == "reject_backoff":
            reject_backoff_trigger_count += 1
            side = str(row.get("side") or "-").lower()
            side_key = side if side in {"buy", "sell"} else "-"
            reject_backoff_trigger_by_side[side_key] = (
                reject_backoff_trigger_by_side.get(side_key, 0) + 1
            )
            cooldown_ms = _as_float(row.get("cooldown_ms"))
            if cooldown_ms is not None and cooldown_ms >= 0:
                cooldown_ms_i = int(cooldown_ms)
                reject_backoff_active_ms_total += cooldown_ms_i
                if side_key in reject_backoff_active_ms_by_side:
                    reject_backoff_active_ms_by_side[side_key] += cooldown_ms_i
        elif event == "reject_backoff_clear":
            reject_backoff_clear_count += 1

        if event == "replace_guard":
            if "replace->skip (recent_cancel_all)" in reason_lc:
                replace_skip_recent_cancel_all += 1
            if "replace->skip (order_id_missing_stage2)" in reason_lc:
                replace_skip_oid_missing_stage2 += 1
            if "replace->skip (dead_order_id)" in reason_lc:
                replace_skip_dead_order_id += 1
            if "replace->skip (cancel_inflight)" in reason_lc:
                replace_skip_cancel_inflight += 1
        elif event == "dead_order_id_mark":
            dead_order_id_mark_count += 1
            side = str(row.get("side") or "-").lower()
            side_key = side if side in {"buy", "sell"} else "-"
            dead_order_id_mark_by_side[side_key] = (
                dead_order_id_mark_by_side.get(side_key, 0) + 1
            )
        elif event == "cancel_inflight_mark":
            cancel_inflight_mark_count += 1
            side = str(row.get("side") or "-").lower()
            side_key = side if side in {"buy", "sell"} else "-"
            cancel_inflight_mark_count_by_side[side_key] = (
                cancel_inflight_mark_count_by_side.get(side_key, 0) + 1
            )
        elif event == "cancel_inflight_clear":
            cancel_inflight_clear_count += 1
            side = str(row.get("side") or "-").lower()
            side_key = side if side in {"buy", "sell"} else "-"
            cancel_inflight_clear_count_by_side[side_key] = (
                cancel_inflight_clear_count_by_side.get(side_key, 0) + 1
            )

        if event == "open_orders_sync":
            last_sync_ts_for_age = ts_i
            if reason_lc != "periodic":
                resync_ok += 1
            bid_present = row.get("open_bid_present")
            ask_present = row.get("open_ask_present")
            for side_name, present in (("buy", bid_present), ("sell", ask_present)):
                if not isinstance(present, bool):
                    continue
                if not present:
                    if quote_gap_start_ms[side_name] is None:
                        quote_gap_start_ms[side_name] = ts_i
                else:
                    start_ms = quote_gap_start_ms[side_name]
                    if start_ms is not None and ts_i >= start_ms:
                        quote_gap_ms_by_side[side_name].append(float(ts_i - start_ms))
                        quote_gap_start_ms[side_name] = None
        elif event == "open_orders_error":
            if reason_lc != "periodic":
                resync_fail += 1

        age_ref = row.get("last_open_orders_sync_ts")
        if isinstance(age_ref, (int, float)):
            age_ms = ts_i - int(age_ref)
            if age_ms >= 0:
                open_orders_age_ms_samples.append(float(age_ms))
        elif event in {"quote_skip", "send_skip", "replace_guard"} and last_sync_ts_for_age is not None:
            age_ms = ts_i - last_sync_ts_for_age
            if age_ms >= 0:
                open_orders_age_ms_samples.append(float(age_ms))

    if last_event_ts is not None:
        for side_name in ("buy", "sell"):
            start_ms = quote_gap_start_ms.get(side_name)
            if start_ms is not None and last_event_ts >= start_ms:
                quote_gap_ms_by_side[side_name].append(float(last_event_ts - start_ms))

    quote_gap_ms_all = quote_gap_ms_by_side["buy"] + quote_gap_ms_by_side["sell"]
    resync_triggered = resync_ok + resync_fail

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
    backoff_active_reject_rate = (
        backoff_active_reject_count / backoff_active_sent_count
        if backoff_active_sent_count > 0
        else None
    )
    backoff_inactive_reject_rate = (
        backoff_inactive_reject_count / backoff_inactive_sent_count
        if backoff_inactive_sent_count > 0
        else None
    )
    backoff_active_fill_rate = (
        backoff_active_fills_count / backoff_active_sent_count
        if backoff_active_sent_count > 0
        else None
    )
    backoff_inactive_fill_rate = (
        backoff_inactive_fills_count / backoff_inactive_sent_count
        if backoff_inactive_sent_count > 0
        else None
    )
    risk_escape_applied_reject_rate = (
        risk_escape_applied_reject_count / risk_escape_applied_sent_count
        if risk_escape_applied_sent_count > 0
        else None
    )
    risk_escape_unapplied_reject_rate = (
        risk_escape_unapplied_reject_count / risk_escape_unapplied_sent_count
        if risk_escape_unapplied_sent_count > 0
        else None
    )
    risk_escape_applied_fill_rate = (
        risk_escape_applied_fills_count / risk_escape_applied_sent_count
        if risk_escape_applied_sent_count > 0
        else None
    )
    risk_escape_unapplied_fill_rate = (
        risk_escape_unapplied_fills_count / risk_escape_unapplied_sent_count
        if risk_escape_unapplied_sent_count > 0
        else None
    )
    risk_size_down_applied_reject_rate = (
        risk_size_down_applied_reject_count / risk_size_down_applied_sent_count
        if risk_size_down_applied_sent_count > 0
        else None
    )
    risk_size_down_inactive_reject_rate = (
        risk_size_down_inactive_reject_count / risk_size_down_inactive_sent_count
        if risk_size_down_inactive_sent_count > 0
        else None
    )
    risk_size_down_applied_fill_rate = (
        risk_size_down_applied_fills_count / risk_size_down_applied_sent_count
        if risk_size_down_applied_sent_count > 0
        else None
    )
    risk_size_down_inactive_fill_rate = (
        risk_size_down_inactive_fills_count / risk_size_down_inactive_sent_count
        if risk_size_down_inactive_sent_count > 0
        else None
    )
    global_lag_quiet_triggered_reject_rate = (
        global_lag_quiet_triggered_reject_count / global_lag_quiet_triggered_sent_count
        if global_lag_quiet_triggered_sent_count > 0
        else None
    )
    global_lag_quiet_triggered_fill_rate = (
        global_lag_quiet_triggered_fills_count / global_lag_quiet_triggered_sent_count
        if global_lag_quiet_triggered_sent_count > 0
        else None
    )
    latency_bucket_cancel_gate_required = latency_bucket_cancel_attempt_needed_count > 0
    latency_bucket_cancel_gate_pass = (not latency_bucket_cancel_gate_required) or (
        latency_bucket_cancel_sent_count > 0
    )
    latency_bucket_cancel_no_match_split_counts = {
        k: latency_bucket_cancel_skip_reason_counts.get(k, 0)
        for k in (
            "no_order_on_side",
            "have_order_on_side_but_no_candidate",
            "have_candidate_but_dead_guarded",
            "open_orders_no_match",
            "open_orders_not_fresh",
            "resync_failed",
            "dead_order_guarded",
        )
        if latency_bucket_cancel_skip_reason_counts.get(k, 0) > 0
    }

    report = {
        "run_dir": str(run_dir),
        "decision_ticks": decision_ticks,
        "boost_active_count": boost_active_count,
        "boost_active_rate": boost_active_rate,
        "boost_trigger_count": boost_active_count,
        "boost_trigger_rate": boost_active_rate,
        "decision_noop_count": decision_noop_count,
        "decision_noop_rate": decision_noop_rate,
        "decision_quote_active_count": (decision_ticks - decision_noop_count),
        "decision_noop_halt_count": decision_noop_halt_count,
        "decision_noop_stop_count": decision_noop_stop_count,
        "decision_noop_spread_filter_count": decision_noop_spread_filter_count,
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
        "fill_decomp": {
            "by_side": {k: _summarize_fill_bucket(v) for k, v in bucket_by_side.items()},
            "by_boost": {k: _summarize_fill_bucket(v) for k, v in bucket_by_boost.items()},
            "by_boost_side": {k: _summarize_fill_bucket(v) for k, v in bucket_by_boost_side.items()},
        },
        "replace_skip_recent_cancel_all": replace_skip_recent_cancel_all,
        "replace_skip_recent_cancel_all_count": replace_skip_recent_cancel_all,
        "replace_skip_oid_missing_stage2": replace_skip_oid_missing_stage2,
        "replace_skip_order_id_missing_stage2_count": replace_skip_oid_missing_stage2,
        "replace_skip_dead_order_id_count": replace_skip_dead_order_id,
        "replace_skip_cancel_inflight_count": replace_skip_cancel_inflight,
        "dead_order_id_skip_count": replace_skip_dead_order_id,
        "dead_order_id_mark_count": dead_order_id_mark_count,
        "dead_order_id_mark_by_side": {
            k: dead_order_id_mark_by_side[k]
            for k in sorted(dead_order_id_mark_by_side.keys())
        },
        "cancel_inflight_mark_count": cancel_inflight_mark_count,
        "cancel_inflight_clear_count": cancel_inflight_clear_count,
        "cancel_inflight_mark_count_by_side": {
            k: cancel_inflight_mark_count_by_side[k]
            for k in sorted(cancel_inflight_mark_count_by_side.keys())
        },
        "cancel_inflight_clear_count_by_side": {
            k: cancel_inflight_clear_count_by_side[k]
            for k in sorted(cancel_inflight_clear_count_by_side.keys())
        },
        "cannot_modify_count": cannot_modify_count,
        "cannot_modify_by_action": {
            k: cannot_modify_by_action[k]
            for k in sorted(cannot_modify_by_action.keys())
        },
        "rate_limit_error_count": rate_limit_error_count,
        "rate_limit_count": rate_limit_error_count,
        "send_skip_reason_counts": {
            k: send_skip_reason_counts[k]
            for k in sorted(send_skip_reason_counts.keys())
        },
        "quote_skip_reason_counts": {
            k: quote_skip_reason_counts[k]
            for k in sorted(quote_skip_reason_counts.keys())
        },
        "pre_send_book_stale_skip_count": pre_send_book_stale_skip_count,
        "pre_send_guard_all_skipped_count": pre_send_guard_all_skipped_count,
        "reject_backoff_trigger_count": reject_backoff_trigger_count,
        "reject_backoff_clear_count": reject_backoff_clear_count,
        "reject_backoff_trigger_by_side": {
            k: reject_backoff_trigger_by_side[k]
            for k in sorted(reject_backoff_trigger_by_side.keys())
        },
        "reject_backoff_active_ms_total": reject_backoff_active_ms_total,
        "reject_backoff_active_ms_by_side": {
            "buy": reject_backoff_active_ms_by_side.get("buy", 0),
            "sell": reject_backoff_active_ms_by_side.get("sell", 0),
        },
        "resync_triggered": resync_triggered,
        "resync_ok": resync_ok,
        "resync_fail": resync_fail,
        "open_orders_age_ms": _summary_stats(open_orders_age_ms_samples),
        "send_latency_ms": _summary_stats(send_latency_ms_values),
        "pre_send_book_age_ms": _summary_stats(pre_send_book_age_ms_values),
        "pre_send_recv_age_ms": _summary_stats(pre_send_recv_age_ms_values),
        "pre_send_recv_age_ms_skip": _summary_stats(pre_send_recv_age_ms_skip_values),
        "pre_send_recv_age_ms_all": _summary_stats(pre_send_recv_age_ms_all_values),
        "pre_send_marketable_sent_count": pre_send_marketable_sent_count,
        "post_only_reject_class1_count": post_only_reject_class1_count,
        "post_only_reject_class2_count": post_only_reject_class2_count,
        "post_only_reject_by_spread_ticks": {
            k: post_only_reject_by_spread_ticks[k]
            for k in sorted(post_only_reject_by_spread_ticks.keys())
        },
        "post_only_reject_by_side_spread_ticks": {
            k: post_only_reject_by_side_spread_ticks[k]
            for k in sorted(post_only_reject_by_side_spread_ticks.keys())
        },
        "post_only_reject_by_risk_escape_bucket": {
            k: post_only_reject_by_risk_escape_bucket[k]
            for k in sorted(post_only_reject_by_risk_escape_bucket.keys())
        },
        "post_only_reject_by_risk_escape_applied": {
            k: post_only_reject_by_risk_escape_applied[k]
            for k in sorted(post_only_reject_by_risk_escape_applied.keys())
        },
        "post_only_reject_by_bucket_applied": {
            k: post_only_reject_by_bucket_applied[k]
            for k in sorted(post_only_reject_by_bucket_applied.keys())
        },
        "post_only_reject_by_risk_size_down": {
            k: post_only_reject_by_risk_size_down[k]
            for k in sorted(post_only_reject_by_risk_size_down.keys())
        },
        "post_only_reject_by_bucket_size_down": {
            k: post_only_reject_by_bucket_size_down[k]
            for k in sorted(post_only_reject_by_bucket_size_down.keys())
        },
        "post_only_reject_by_round_mode": {
            k: post_only_reject_by_round_mode[k]
            for k in sorted(post_only_reject_by_round_mode.keys())
        },
        "post_only_reject_class2_by_side": {
            k: post_only_reject_class2_by_side[k]
            for k in sorted(post_only_reject_class2_by_side.keys())
        },
        "post_only_reject_class2_by_side_spread_ticks": {
            k: post_only_reject_class2_by_side_spread_ticks[k]
            for k in sorted(post_only_reject_class2_by_side_spread_ticks.keys())
        },
        "post_only_reject_class2_by_spread_ticks": {
            k: post_only_reject_class2_by_spread_ticks[k]
            for k in sorted(post_only_reject_class2_by_spread_ticks.keys())
        },
        "post_only_reject_class2_by_round_mode": {
            k: post_only_reject_class2_by_round_mode[k]
            for k in sorted(post_only_reject_class2_by_round_mode.keys())
        },
        "post_only_reject_class2_sell_spread1_count": post_only_reject_class2_sell_spread1_count,
        "post_only_reject_class2_send_latency_ms": _summary_stats(
            post_only_reject_class2_send_latency_ms_values
        ),
        "post_only_reject_class2_pre_send_recv_age_ms": _summary_stats(
            post_only_reject_class2_pre_send_recv_age_ms_values
        ),
        "backoff_active_sent_count": backoff_active_sent_count,
        "backoff_inactive_sent_count": backoff_inactive_sent_count,
        "backoff_active_reject_count": backoff_active_reject_count,
        "backoff_inactive_reject_count": backoff_inactive_reject_count,
        "backoff_active_reject_rate": backoff_active_reject_rate,
        "backoff_inactive_reject_rate": backoff_inactive_reject_rate,
        "backoff_active_fills_count": backoff_active_fills_count,
        "backoff_inactive_fills_count": backoff_inactive_fills_count,
        "backoff_active_fill_rate": backoff_active_fill_rate,
        "backoff_inactive_fill_rate": backoff_inactive_fill_rate,
        "backoff_active_net_bps": _summary_stats(backoff_active_net_bps_values),
        "backoff_inactive_net_bps": _summary_stats(backoff_inactive_net_bps_values),
        "backoff_active_markout30_bps": _summary_stats(backoff_active_markout30_bps_values),
        "backoff_inactive_markout30_bps": _summary_stats(backoff_inactive_markout30_bps_values),
        "risk_escape_applied_count": risk_escape_applied_sent_count,
        "risk_escape_triggered_sent_count": risk_escape_triggered_sent_count,
        "risk_escape_untriggered_sent_count": risk_escape_untriggered_sent_count,
        "risk_escape_triggered_count_by_side": {
            k: risk_escape_triggered_count_by_side[k]
            for k in sorted(risk_escape_triggered_count_by_side.keys())
        },
        "risk_escape_sent_count_by_mode": {
            k: risk_escape_sent_count_by_mode[k]
            for k in sorted(risk_escape_sent_count_by_mode.keys())
        },
        "risk_escape_trigger_bucket_counts": {
            k: risk_escape_trigger_bucket_counts[k]
            for k in sorted(risk_escape_trigger_bucket_counts.keys())
        },
        "risk_escape_applied_count_by_side": {
            k: risk_escape_applied_count_by_side[k]
            for k in sorted(risk_escape_applied_count_by_side.keys())
        },
        "risk_escape_applied_bucket_counts": {
            k: risk_escape_applied_bucket_counts[k]
            for k in sorted(risk_escape_applied_bucket_counts.keys())
        },
        "risk_escape_skip_count": risk_escape_skip_sent_count,
        "risk_escape_skip_sent_count": risk_escape_skip_sent_count,
        "risk_escape_skip_count_by_side": {
            k: risk_escape_skip_count_by_side[k]
            for k in sorted(risk_escape_skip_count_by_side.keys())
        },
        "risk_escape_skip_bucket_counts": {
            k: risk_escape_skip_bucket_counts[k]
            for k in sorted(risk_escape_skip_bucket_counts.keys())
        },
        "risk_escape_skip_count_by_side_bucket": {
            k: risk_escape_skip_count_by_side_bucket[k]
            for k in sorted(risk_escape_skip_count_by_side_bucket.keys())
        },
        "latency_bucket_cancel_sent_count": latency_bucket_cancel_sent_count,
        "latency_bucket_cancel_success_count": latency_bucket_cancel_success_count,
        "latency_bucket_cancel_skip_count": latency_bucket_cancel_skip_count,
        "latency_bucket_cancel_sent_count_by_side": {
            k: latency_bucket_cancel_sent_by_side[k]
            for k in sorted(latency_bucket_cancel_sent_by_side.keys())
        },
        "latency_bucket_cancel_sent_count_by_bucket": {
            k: latency_bucket_cancel_sent_by_bucket[k]
            for k in sorted(latency_bucket_cancel_sent_by_bucket.keys())
        },
        "latency_bucket_cancel_sent_count_by_side_bucket": {
            k: latency_bucket_cancel_sent_by_side_bucket[k]
            for k in sorted(latency_bucket_cancel_sent_by_side_bucket.keys())
        },
        "latency_bucket_cancel_success_count_by_side": {
            k: latency_bucket_cancel_success_by_side[k]
            for k in sorted(latency_bucket_cancel_success_by_side.keys())
        },
        "latency_bucket_cancel_success_count_by_bucket": {
            k: latency_bucket_cancel_success_by_bucket[k]
            for k in sorted(latency_bucket_cancel_success_by_bucket.keys())
        },
        "latency_bucket_cancel_skip_count_by_side": {
            k: latency_bucket_cancel_skip_by_side[k]
            for k in sorted(latency_bucket_cancel_skip_by_side.keys())
        },
        "latency_bucket_cancel_skip_count_by_bucket": {
            k: latency_bucket_cancel_skip_by_bucket[k]
            for k in sorted(latency_bucket_cancel_skip_by_bucket.keys())
        },
        "latency_bucket_cancel_skip_reason_counts": {
            k: latency_bucket_cancel_skip_reason_counts[k]
            for k in sorted(latency_bucket_cancel_skip_reason_counts.keys())
        },
        "latency_bucket_cancel_no_match_split_counts": latency_bucket_cancel_no_match_split_counts,
        "latency_bucket_cancel_attempt_needed_count": latency_bucket_cancel_attempt_needed_count,
        "latency_bucket_cancel_attempt_not_needed_count": latency_bucket_cancel_attempt_not_needed_count,
        "latency_bucket_cancel_attempt_needed_count_by_side": {
            k: latency_bucket_cancel_attempt_needed_by_side[k]
            for k in sorted(latency_bucket_cancel_attempt_needed_by_side.keys())
        },
        "latency_bucket_cancel_attempt_needed_count_by_bucket": {
            k: latency_bucket_cancel_attempt_needed_by_bucket[k]
            for k in sorted(latency_bucket_cancel_attempt_needed_by_bucket.keys())
        },
        "latency_bucket_cancel_attempt_not_needed_count_by_side": {
            k: latency_bucket_cancel_attempt_not_needed_by_side[k]
            for k in sorted(latency_bucket_cancel_attempt_not_needed_by_side.keys())
        },
        "latency_bucket_cancel_attempt_not_needed_count_by_bucket": {
            k: latency_bucket_cancel_attempt_not_needed_by_bucket[k]
            for k in sorted(latency_bucket_cancel_attempt_not_needed_by_bucket.keys())
        },
        "latency_bucket_cancel_gate_required": latency_bucket_cancel_gate_required,
        "latency_bucket_cancel_gate_pass": latency_bucket_cancel_gate_pass,
        "risk_escape_cancel_and_skip_only_count": risk_escape_cancel_and_skip_only_count,
        "cancel_only_sent_count": risk_escape_cancel_and_skip_only_count,
        "risk_escape_cancel_all_fallback_sent_count": risk_escape_cancel_all_fallback_sent_count,
        "risk_escape_cancel_all_fallback_success_count": risk_escape_cancel_all_fallback_success_count,
        "risk_escape_cancel_all_fallback_skip_count": risk_escape_cancel_all_fallback_skip_count,
        "risk_escape_cancel_all_fallback_skip_reason_counts": {
            k: risk_escape_cancel_all_fallback_skip_reason_counts[k]
            for k in sorted(risk_escape_cancel_all_fallback_skip_reason_counts.keys())
        },
        "risk_escape_bucket_counts": {
            k: risk_escape_applied_bucket_counts[k]
            for k in sorted(risk_escape_applied_bucket_counts.keys())
        },
        "risk_escape_delta_ticks_triggered": _summary_stats(
            risk_escape_delta_ticks_triggered_values
        ),
        "risk_escape_delta_ticks_applied": _summary_stats(
            risk_escape_delta_ticks_applied_values
        ),
        "risk_escape_applied_sent_count": risk_escape_applied_sent_count,
        "risk_escape_unapplied_sent_count": risk_escape_unapplied_sent_count,
        "risk_escape_applied_reject_count": risk_escape_applied_reject_count,
        "risk_escape_unapplied_reject_count": risk_escape_unapplied_reject_count,
        "risk_escape_applied_reject_rate": risk_escape_applied_reject_rate,
        "risk_escape_unapplied_reject_rate": risk_escape_unapplied_reject_rate,
        "risk_escape_applied_fills_count": risk_escape_applied_fills_count,
        "risk_escape_unapplied_fills_count": risk_escape_unapplied_fills_count,
        "risk_escape_applied_fills_count_by_bucket": {
            k: risk_escape_applied_fills_count_by_bucket[k]
            for k in sorted(risk_escape_applied_fills_count_by_bucket.keys())
        },
        "risk_escape_unapplied_fills_count_by_bucket": {
            k: risk_escape_unapplied_fills_count_by_bucket[k]
            for k in sorted(risk_escape_unapplied_fills_count_by_bucket.keys())
        },
        "risk_escape_applied_fill_rate": risk_escape_applied_fill_rate,
        "risk_escape_unapplied_fill_rate": risk_escape_unapplied_fill_rate,
        "risk_escape_applied_net_bps": _summary_stats(risk_escape_applied_net_bps_values),
        "risk_escape_unapplied_net_bps": _summary_stats(risk_escape_unapplied_net_bps_values),
        "risk_escape_applied_net_bps_by_bucket": {
            k: _summary_stats(risk_escape_applied_net_bps_by_bucket[k])
            for k in sorted(risk_escape_applied_net_bps_by_bucket.keys())
        },
        "risk_escape_unapplied_net_bps_by_bucket": {
            k: _summary_stats(risk_escape_unapplied_net_bps_by_bucket[k])
            for k in sorted(risk_escape_unapplied_net_bps_by_bucket.keys())
        },
        "risk_escape_applied_markout30_bps": _summary_stats(risk_escape_applied_markout30_bps_values),
        "risk_escape_unapplied_markout30_bps": _summary_stats(
            risk_escape_unapplied_markout30_bps_values
        ),
        "risk_escape_applied_markout30_bps_by_bucket": {
            k: _summary_stats(risk_escape_applied_markout30_bps_by_bucket[k])
            for k in sorted(risk_escape_applied_markout30_bps_by_bucket.keys())
        },
        "risk_escape_unapplied_markout30_bps_by_bucket": {
            k: _summary_stats(risk_escape_unapplied_markout30_bps_by_bucket[k])
            for k in sorted(risk_escape_unapplied_markout30_bps_by_bucket.keys())
        },
        "risk_size_down_applied_sent_count": risk_size_down_applied_sent_count,
        "risk_size_down_inactive_sent_count": risk_size_down_inactive_sent_count,
        "size_down_applied_sent_count": risk_size_down_applied_sent_count,
        "risk_size_down_applied_reject_count": risk_size_down_applied_reject_count,
        "risk_size_down_inactive_reject_count": risk_size_down_inactive_reject_count,
        "risk_size_down_applied_reject_rate": risk_size_down_applied_reject_rate,
        "risk_size_down_inactive_reject_rate": risk_size_down_inactive_reject_rate,
        "risk_size_down_applied_fills_count": risk_size_down_applied_fills_count,
        "risk_size_down_inactive_fills_count": risk_size_down_inactive_fills_count,
        "size_down_applied_fills_count": risk_size_down_applied_fills_count,
        "risk_size_down_applied_fill_rate": risk_size_down_applied_fill_rate,
        "risk_size_down_inactive_fill_rate": risk_size_down_inactive_fill_rate,
        "risk_size_down_applied_fills_count_by_bucket": {
            k: risk_size_down_applied_fills_count_by_bucket[k]
            for k in sorted(risk_size_down_applied_fills_count_by_bucket.keys())
        },
        "risk_size_down_inactive_fills_count_by_bucket": {
            k: risk_size_down_inactive_fills_count_by_bucket[k]
            for k in sorted(risk_size_down_inactive_fills_count_by_bucket.keys())
        },
        "risk_size_down_applied_net_bps": _summary_stats(risk_size_down_applied_net_bps_values),
        "risk_size_down_inactive_net_bps": _summary_stats(risk_size_down_inactive_net_bps_values),
        "risk_size_down_applied_net_bps_by_bucket": {
            k: _summary_stats(risk_size_down_applied_net_bps_by_bucket[k])
            for k in sorted(risk_size_down_applied_net_bps_by_bucket.keys())
        },
        "risk_size_down_inactive_net_bps_by_bucket": {
            k: _summary_stats(risk_size_down_inactive_net_bps_by_bucket[k])
            for k in sorted(risk_size_down_inactive_net_bps_by_bucket.keys())
        },
        "risk_size_down_applied_markout30_bps": _summary_stats(
            risk_size_down_applied_markout30_bps_values
        ),
        "risk_size_down_inactive_markout30_bps": _summary_stats(
            risk_size_down_inactive_markout30_bps_values
        ),
        "risk_size_down_applied_markout30_bps_by_bucket": {
            k: _summary_stats(risk_size_down_applied_markout30_bps_by_bucket[k])
            for k in sorted(risk_size_down_applied_markout30_bps_by_bucket.keys())
        },
        "risk_size_down_inactive_markout30_bps_by_bucket": {
            k: _summary_stats(risk_size_down_inactive_markout30_bps_by_bucket[k])
            for k in sorted(risk_size_down_inactive_markout30_bps_by_bucket.keys())
        },
        "global_lag_quiet_triggered_sent_count": global_lag_quiet_triggered_sent_count,
        "global_lag_quiet_triggered_reject_count": global_lag_quiet_triggered_reject_count,
        "global_lag_quiet_triggered_reject_rate": global_lag_quiet_triggered_reject_rate,
        "global_lag_quiet_triggered_fills_count": global_lag_quiet_triggered_fills_count,
        "global_lag_quiet_triggered_fill_rate": global_lag_quiet_triggered_fill_rate,
        "global_lag_quiet_triggered_count_by_bucket": {
            k: global_lag_quiet_triggered_count_by_bucket[k]
            for k in sorted(global_lag_quiet_triggered_count_by_bucket.keys())
        },
        "global_lag_quiet_triggered_net_bps": _summary_stats(
            global_lag_quiet_triggered_net_bps_values
        ),
        "global_lag_quiet_triggered_markout30_bps": _summary_stats(
            global_lag_quiet_triggered_markout30_bps_values
        ),
        "quote_gap_ms": {
            "buy": _summary_stats(quote_gap_ms_by_side["buy"]),
            "sell": _summary_stats(quote_gap_ms_by_side["sell"]),
            "all": _summary_stats(quote_gap_ms_all),
        },
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
        default="1,5,30",
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
