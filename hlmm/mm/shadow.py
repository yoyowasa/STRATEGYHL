from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional

import pyarrow.parquet as pq

from hlmm.mm.rounding import MAX_DECIMALS, round_price, round_size
from hlmm.mm.strategy import StrategyParams, decide_orders


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _round_price_float(px: Optional[float], max_decimals: int | None) -> Optional[float]:
    if px is None:
        return None
    decimals = MAX_DECIMALS if max_decimals is None else int(max_decimals)
    return float(round_price(Decimal(str(px)), max_decimals=decimals))


def _round_size_float(sz: float, sz_decimals: int | None) -> float:
    decimals = MAX_DECIMALS if sz_decimals is None else int(sz_decimals)
    return float(round_size(Decimal(str(sz)), sz_decimals=decimals))


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def _changed(a: Optional[float], b: Optional[float], eps: float = 1e-12) -> bool:
    if a is None or b is None:
        return a != b
    return abs(a - b) > eps


def run_shadow(
    blocks_path: str | Path,
    out_dir: str | Path,
    params: StrategyParams,
    order_batch_sec: float = 0.1,
    size_decimals: int | None = None,
    price_decimals: int | None = None,
    reconnect_gap_ms: int | None = None,
    disconnect_guard: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    market_log = out_dir / "market_state.jsonl"
    decision_log = out_dir / "decision.jsonl"
    orders_log = out_dir / "orders.jsonl"

    table = pq.read_table(blocks_path)
    blocks = table.to_pylist()
    sorted_blocks = sorted(
        list(blocks), key=lambda b: (b.get("block_ts_ms") or 0, str(b.get("book_event_id") or ""))
    )

    state: Dict[str, object] = {
        "position": 0.0,
        "drawdown": 0.0,
        "mid": None,
        "abs_mid_ret": None,
        "market_spread_bps": None,
        "stop_triggered": False,
        "pull_triggered": False,
        "halt_triggered": False,
        "boost_triggered": False,
        "strategy_spread_bps": None,
        "strategy_size": None,
        "strategy_bid_spread_bps": None,
        "strategy_ask_spread_bps": None,
        "strategy_bid_size": None,
        "strategy_ask_size": None,
        "post_pull_unwind_active": False,
        "pull_side": None,
        "mid_ret": None,
    }

    active_orders: Dict[str, Optional[dict]] = {"buy": None, "sell": None}
    last_batch_ts: Optional[int] = None
    batch_id = 0
    prev_mid: Optional[float] = None
    prev_ts: Optional[int] = None
    prev_boost: Optional[bool] = None
    batch_gap_ms = int(float(order_batch_sec) * 1000.0) if order_batch_sec is not None else 0

    for block in sorted_blocks:
        ts_raw = block.get("block_ts_ms")
        try:
            ts_ms = int(ts_raw) if ts_raw is not None else 0
        except (TypeError, ValueError):
            ts_ms = 0

        reconnect = False
        if reconnect_gap_ms is not None and prev_ts is not None:
            try:
                if ts_ms - prev_ts >= int(reconnect_gap_ms):
                    reconnect = True
                    prev_mid = None
            except (TypeError, ValueError):
                reconnect = False

        top = block.get("book_top") or {}
        best_bid = _safe_float(top.get("bid_px"))
        best_ask = _safe_float(top.get("ask_px"))
        bid_sz1 = _safe_float(top.get("bid_sz"))
        ask_sz1 = _safe_float(top.get("ask_sz"))
        mid = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
        abs_mid_ret = None
        mid_ret = None
        if mid is not None and prev_mid not in (None, 0.0):
            mid_ret = (float(mid) - float(prev_mid)) / float(prev_mid)
            abs_mid_ret = abs(float(mid_ret))
        spread_bps = (
            10_000.0 * (best_ask - best_bid) / mid
            if best_bid is not None and best_ask is not None and mid not in (None, 0.0)
            else None
        )

        state["mid"] = mid
        state["mid_ret"] = mid_ret
        state["abs_mid_ret"] = abs_mid_ret
        state["market_spread_bps"] = spread_bps
        state["stop_triggered"] = False
        state["pull_triggered"] = False
        state["halt_triggered"] = False
        state["boost_triggered"] = False
        state["strategy_spread_bps"] = None
        state["strategy_size"] = None
        state["strategy_bid_spread_bps"] = None
        state["strategy_ask_spread_bps"] = None
        state["strategy_bid_size"] = None
        state["strategy_ask_size"] = None
        state["pull_side"] = None

        res = decide_orders(state, params)
        boost_active = bool(res.get("boost_triggered"))
        stop_triggered = bool(res.get("stop_triggered"))
        halt_triggered = bool(res.get("halt_triggered"))
        effective_bid_spread_bps = _safe_float(res.get("strategy_bid_spread_bps"))
        effective_ask_spread_bps = _safe_float(res.get("strategy_ask_spread_bps"))

        target: Dict[str, dict] = {
            "buy": {"price": None, "size": 0.0},
            "sell": {"price": None, "size": 0.0},
        }
        for order in res.get("orders") or []:
            try:
                side = str(order.side)
            except Exception:
                continue
            if side not in target:
                continue
            target[side] = {"price": _safe_float(order.price), "size": float(order.size)}

        raw_bid_px = target["buy"]["price"]
        raw_ask_px = target["sell"]["price"]
        raw_bid_sz = float(target["buy"]["size"])
        raw_ask_sz = float(target["sell"]["size"])
        final_bid_px = _round_price_float(raw_bid_px, price_decimals)
        final_ask_px = _round_price_float(raw_ask_px, price_decimals)
        final_bid_sz = _round_size_float(raw_bid_sz, size_decimals)
        final_ask_sz = _round_size_float(raw_ask_sz, size_decimals)

        crossed_bid = final_bid_px is not None and best_ask is not None and final_bid_px >= best_ask
        crossed_ask = final_ask_px is not None and best_bid is not None and final_ask_px <= best_bid
        crossed = crossed_bid or crossed_ask

        _write_jsonl(
            market_log,
            {
                "ts_ms": ts_ms,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_sz1": bid_sz1,
                "ask_sz1": ask_sz1,
                "mid": mid,
                "abs_mid_ret": abs_mid_ret,
                "boost_active": boost_active,
                "spread_bps": spread_bps,
                "reconnect": reconnect,
            },
        )
        _write_jsonl(
            decision_log,
            {
                "ts_ms": ts_ms,
                "base_size": float(params.base_size),
                "boost_size_factor": float(params.boost_size_factor),
                "raw_bid_sz": raw_bid_sz,
                "raw_ask_sz": raw_ask_sz,
                "final_bid_sz": final_bid_sz,
                "final_ask_sz": final_ask_sz,
                "raw_bid_px": raw_bid_px,
                "raw_ask_px": raw_ask_px,
                "final_bid_px": final_bid_px,
                "final_ask_px": final_ask_px,
                "position": float(state.get("position") or 0.0),
                "boost_active": boost_active,
                "stop_triggered": stop_triggered,
                "halt_triggered": halt_triggered,
                "crossed": crossed,
                "crossed_bid": crossed_bid,
                "crossed_ask": crossed_ask,
            },
        )

        if reconnect and disconnect_guard:
            if active_orders["buy"] is not None or active_orders["sell"] is not None:
                batch_id += 1
                _write_jsonl(
                    orders_log,
                    {
                        "ts_ms": ts_ms,
                        "action": "cancel_all",
                        "side": "both",
                        "px": None,
                        "sz": None,
                        "post_only": True,
                        "batch_id": batch_id,
                        "reason": "reconnect",
                        "crossed": False,
                    },
                )
                active_orders = {"buy": None, "sell": None}
                last_batch_ts = ts_ms
            prev_mid = mid
            prev_ts = ts_ms
            prev_boost = boost_active
            continue

        if batch_gap_ms > 0 and last_batch_ts is not None and ts_ms - last_batch_ts < batch_gap_ms:
            prev_mid = mid
            prev_ts = ts_ms
            prev_boost = boost_active
            continue

        actions = []
        for side in ("buy", "sell"):
            target_px = final_bid_px if side == "buy" else final_ask_px
            target_sz = final_bid_sz if side == "buy" else final_ask_sz
            active = active_orders[side]
            if target_sz <= 0:
                if active is not None:
                    actions.append(("cancel", side, None, None))
                    active_orders[side] = None
                continue
            if active is None:
                actions.append(("new", side, target_px, target_sz))
                active_orders[side] = {"price": target_px, "size": target_sz}
                continue
            if _changed(active.get("price"), target_px) or _changed(active.get("size"), target_sz):
                actions.append(("replace", side, target_px, target_sz))
                active_orders[side] = {"price": target_px, "size": target_sz}

        if actions:
            batch_id += 1
            reason = "price_or_size_change"
            if prev_boost is not None and prev_boost != boost_active:
                reason = "boost_toggle"
            for action, side, px, sz in actions:
                record = {
                    "ts_ms": ts_ms,
                    "action": action,
                    "side": side,
                    "px": px,
                    "sz": sz,
                    "post_only": True,
                    "batch_id": batch_id,
                    "reason": reason,
                    "crossed": crossed,
                }
                if action in {"new", "replace"}:
                    record["effective_spread_bps"] = (
                        effective_bid_spread_bps if side == "buy" else effective_ask_spread_bps
                    )
                _write_jsonl(
                    orders_log,
                    record,
                )
            last_batch_ts = ts_ms

        prev_mid = mid
        prev_ts = ts_ms
        prev_boost = boost_active

    return out_dir
