from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return float(median(vals))


def main() -> int:
    parser = argparse.ArgumentParser(description="mm_shadow ログの合否チェック")
    parser.add_argument("--log-dir", required=True, help="mm_shadow のログディレクトリ")
    parser.add_argument("--min-trigger", type=float, default=0.03)
    parser.add_argument("--max-trigger", type=float, default=0.10)
    parser.add_argument("--boost-factor", type=float, default=1.5)
    parser.add_argument("--batch-sec", type=float, default=0.1)
    parser.add_argument("--size-eps", type=float, default=1e-12)
    parser.add_argument("--min-top-px-changes", type=int, default=20)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    market_path = log_dir / "market_state.jsonl"
    decision_path = log_dir / "decision.jsonl"
    orders_path = log_dir / "orders.jsonl"

    if not market_path.exists() or not decision_path.exists() or not orders_path.exists():
        raise SystemExit("market_state.jsonl / decision.jsonl / orders.jsonl が必要です")

    total = 0
    boost_on = 0
    reconnect_events = 0
    top_px_change_count = 0
    for rec in _iter_jsonl(market_path):
        total += 1
        if rec.get("boost_active"):
            boost_on += 1
        if rec.get("reconnect"):
            reconnect_events += 1
        if rec.get("top_px_change"):
            top_px_change_count += 1
    boost_rate = boost_on / total if total else 0.0

    boost_records = 0
    boost_applied = 0
    ratios = []
    crossed = 0
    for rec in _iter_jsonl(decision_path):
        if rec.get("crossed"):
            crossed += 1
        if not rec.get("boost_active"):
            continue
        boost_records += 1
        base_size = _safe_float(rec.get("base_size")) or 0.0
        bid_sz = _safe_float(rec.get("final_bid_sz")) or 0.0
        ask_sz = _safe_float(rec.get("final_ask_sz")) or 0.0
        if base_size > 0:
            ratios.append(bid_sz / base_size)
            ratios.append(ask_sz / base_size)
        if bid_sz > base_size + args.size_eps and ask_sz > base_size + args.size_eps:
            boost_applied += 1

    batch_ts = {}
    cancel_on_reconnect = 0
    for rec in _iter_jsonl(orders_path):
        batch_id = rec.get("batch_id")
        ts = rec.get("ts_ms")
        if batch_id is not None and ts is not None and batch_id not in batch_ts:
            batch_ts[batch_id] = int(ts)
        if rec.get("action") == "cancel_all" and rec.get("reason") == "reconnect":
            cancel_on_reconnect += 1

    batch_times = sorted(batch_ts.values())
    min_gap_ms = None
    if len(batch_times) >= 2:
        gaps = [b - a for a, b in zip(batch_times, batch_times[1:])]
        min_gap_ms = min(gaps) if gaps else None

    passed_trigger = args.min_trigger <= boost_rate <= args.max_trigger
    trigger_inconclusive = top_px_change_count < args.min_top_px_changes
    trigger_ok_effective = True if trigger_inconclusive else passed_trigger
    passed_size = boost_records == 0 or boost_applied == boost_records
    passed_cross = crossed == 0
    passed_batch = min_gap_ms is None or min_gap_ms >= int(args.batch_sec * 1000.0)
    passed_reconnect = reconnect_events == 0 or cancel_on_reconnect >= reconnect_events

    print("boost_trigger_rate", boost_rate)
    print("trigger_rate_ok", passed_trigger)
    print("trigger_rate_inconclusive", trigger_inconclusive)
    print("top_px_change_count", top_px_change_count)
    print("boost_records", boost_records)
    print("boost_applied_ok", passed_size)
    print("ratio_median", _median(ratios))
    print("crossed_count", crossed)
    print("crossed_ok", passed_cross)
    print("min_batch_gap_ms", min_gap_ms)
    print("batch_ok", passed_batch)
    print("reconnect_events", reconnect_events)
    print("reconnect_guard_ok", passed_reconnect)

    all_ok = trigger_ok_effective and passed_size and passed_cross and passed_batch and passed_reconnect
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
