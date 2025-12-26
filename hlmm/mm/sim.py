from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.mm.fill_models import FillResult, fill_order_lower, fill_order_upper

@dataclass(frozen=True)
class Order:
    side: str  # "buy" or "sell"
    size: float
    price: Optional[float] = None  # 未指定なら板トップを使用
    post_only: bool = True


StrategyFn = Callable[[Mapping[str, object], Mapping[str, float]], List[Order]]


def _mid_from_top(top: Mapping[str, object] | None) -> Optional[float]:
    if not top:
        return None
    bid_px = top.get("bid_px")
    ask_px = top.get("ask_px")
    try:
        bid = float(bid_px) if bid_px is not None else None
        ask = float(ask_px) if ask_px is not None else None
    except (TypeError, ValueError):
        return None
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _crosses_post_only(order: Order, top: Mapping[str, object]) -> bool:
    """post-only前提で板クロスしてしまう注文を検知。"""
    bid_px = top.get("bid_px")
    ask_px = top.get("ask_px")
    try:
        bid = float(bid_px) if bid_px is not None else None
        ask = float(ask_px) if ask_px is not None else None
    except (TypeError, ValueError):
        return True
    if order.side == "buy" and ask is not None:
        price = float(order.price) if order.price is not None else bid
        return price is not None and price >= ask
    if order.side == "sell" and bid is not None:
        price = float(order.price) if order.price is not None else ask
        return price is not None and price <= bid
    return False


def simulate_blocks(
    blocks: Iterable[Mapping[str, object]],
    strategy: Optional[StrategyFn] = None,
    taker_fee_bps: float = 0.0,
    maker_rebate_bps: float = 0.0,
    max_abs_position: Optional[float] = None,
    fill_model: str = "lower",
    lower_alpha: float = 0.5,
    lower_nprints: Optional[int] = None,
    allow_top_fill: bool = False,
    signed_volume_window_s: Optional[float] = None,
) -> tuple[List[dict], List[dict], List[dict]]:
    """Bookクロックでイベントループを回し、fills・台帳・注文状態を返す。"""
    strategy = strategy or (lambda block, state: [])
    state = {
        "position": 0.0,
        "avg_cost": None,
        "realized_price_pnl": 0.0,
        "fees": 0.0,
        "rebates": 0.0,
        "funding": 0.0,
        "last_price": None,
        # 戦略側が参照できる補助状態
        "mid": None,
        "mid_ret": None,
        "abs_mid_ret": None,
        "market_spread_bps": None,
        "total_pnl": 0.0,
        "peak_pnl": 0.0,
        "drawdown": 0.0,
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
        "pull_side": None,
        "signed_volume": 0.0,
        "signed_volume_window": 0.0,
        "prev_pull_triggered": False,
        "post_pull_unwind_active": False,
        "micro_bias_bps": None,
        "imbalance": None,
        "micro_pos": None,
    }
    trades: List[dict] = []
    ledger: List[dict] = []
    orders_state: List[dict] = []

    sorted_blocks = sorted(
        list(blocks), key=lambda b: (b.get("block_ts_ms") or 0, str(b.get("book_event_id") or ""))
    )

    window_ms: Optional[int] = None
    if signed_volume_window_s is not None:
        try:
            window_ms = int(float(signed_volume_window_s) * 1000.0)
        except (TypeError, ValueError):
            window_ms = None
        if window_ms is not None and window_ms <= 0:
            window_ms = None

    sv_window = deque()
    sv_window_sum = 0.0

    order_counter = 0

    for block in sorted_blocks:
        top = block.get("book_top") or {}
        trades_bucket_for_signal = block.get("trade_bucket") or []
        # 戦略入力（mid / market_spread_bps）
        prev_mid = state.get("mid")
        bid = None
        ask = None
        bid_sz = None
        ask_sz = None
        try:
            bid = float(top.get("bid_px")) if top.get("bid_px") is not None else None
            ask = float(top.get("ask_px")) if top.get("ask_px") is not None else None
            bid_sz = float(top.get("bid_sz")) if top.get("bid_sz") is not None else None
            ask_sz = float(top.get("ask_sz")) if top.get("ask_sz") is not None else None
        except (TypeError, ValueError):
            bid = None
            ask = None
            bid_sz = None
            ask_sz = None
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
        mid_ret = None
        abs_mid_ret = None
        if mid is not None and prev_mid not in (None, 0.0):
            try:
                mid_ret = (float(mid) - float(prev_mid)) / float(prev_mid)
                abs_mid_ret = abs(float(mid_ret))
            except (TypeError, ValueError, ZeroDivisionError):
                mid_ret = None
                abs_mid_ret = None
        market_spread_bps = (
            10_000.0 * (ask - bid) / mid if bid is not None and ask is not None and mid not in (None, 0.0) else None
        )
        micro_bias_bps = None
        imbalance = None
        micro_pos = None
        microprice = None
        if bid_sz is not None and ask_sz is not None:
            denom = bid_sz + ask_sz
            if denom not in (0.0, -0.0):
                imbalance = (bid_sz - ask_sz) / denom
                if bid is not None and ask is not None:
                    microprice = (ask * bid_sz + bid * ask_sz) / denom
        if microprice is not None and mid not in (None, 0.0):
            micro_bias_bps = (microprice - mid) / mid * 10_000.0
        if microprice is not None and bid is not None and ask is not None and mid is not None:
            spread = ask - bid
            if spread not in (0.0, -0.0):
                micro_pos = (microprice - mid) / spread
        state["mid"] = mid
        state["mid_ret"] = mid_ret
        state["abs_mid_ret"] = abs_mid_ret
        state["market_spread_bps"] = market_spread_bps
        state["micro_bias_bps"] = micro_bias_bps
        state["imbalance"] = imbalance
        state["micro_pos"] = micro_pos
        # post-pull unwind 用に、前ブロックの pull 状態を保存
        state["prev_pull_triggered"] = bool(state.get("pull_triggered"))
        # 毎ブロックでリセット（strategy が必要に応じて True にする）
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

        # orderflow（約定側の符号付き出来高）: buyを+、sellを-
        signed_volume = 0.0
        for tr in trades_bucket_for_signal:
            if not isinstance(tr, dict):
                continue
            side = tr.get("side")
            sz = tr.get("sz")
            try:
                sz_f = float(sz)
            except (TypeError, ValueError):
                continue
            if sz_f <= 0:
                continue
            if side == "buy":
                signed_volume += sz_f
            elif side == "sell":
                signed_volume -= sz_f
        state["signed_volume"] = signed_volume
        # 窓（直近window秒）のローリング合計。window未指定ならブロック内の値をそのまま使う。
        if window_ms is None:
            state["signed_volume_window"] = signed_volume
        else:
            try:
                ts_i = int(block.get("block_ts_ms") or 0)
            except (TypeError, ValueError):
                ts_i = 0
            sv_window.append((ts_i, signed_volume))
            sv_window_sum += signed_volume
            cutoff = ts_i - window_ms
            while sv_window and sv_window[0][0] < cutoff:
                _, old = sv_window.popleft()
                sv_window_sum -= old
            state["signed_volume_window"] = sv_window_sum

        # funding イベント（あれば即時反映）
        funding_amt = float(block.get("funding", 0.0) or 0.0)
        state["funding"] += funding_amt

        # ステート更新（未来を参照しない）
        orders = strategy(block, state)
        for order in orders:
            order_counter += 1
            status = "open"
            if order.size == 0 or order.side not in ("buy", "sell"):
                status = "invalid"
                orders_state.append(
                    {
                        "order_id": f"o{order_counter}",
                        "block_ts_ms": block.get("block_ts_ms"),
                        "book_event_id": block.get("book_event_id"),
                        "side": order.side,
                        "size": order.size,
                        "price": order.price,
                        "status": status,
                    }
                )
                continue
            # post-only で板クロス検知
            if order.post_only and _crosses_post_only(order, top):
                status = "invalid_cross"
                orders_state.append(
                    {
                        "order_id": f"o{order_counter}",
                        "block_ts_ms": block.get("block_ts_ms"),
                        "book_event_id": block.get("book_event_id"),
                        "side": order.side,
                        "size": order.size,
                        "price": order.price,
                        "status": status,
                    }
                )
                continue

            # フィル決定
            if order.price is not None:
                try:
                    limit_px = float(order.price)
                except (TypeError, ValueError):
                    status = "invalid"
                    orders_state.append(
                        {
                            "order_id": f"o{order_counter}",
                            "block_ts_ms": block.get("block_ts_ms"),
                            "book_event_id": block.get("book_event_id"),
                            "side": order.side,
                            "size": order.size,
                            "price": order.price,
                            "status": status,
                        }
                    )
                    continue
            else:
                px_key = "ask_px" if order.side == "buy" else "bid_px"
                px_val = top.get(px_key)
                if px_val is None:
                    status = "invalid_no_liquidity"
                    orders_state.append(
                        {
                            "order_id": f"o{order_counter}",
                            "block_ts_ms": block.get("block_ts_ms"),
                            "book_event_id": block.get("book_event_id"),
                            "side": order.side,
                            "size": order.size,
                            "price": order.price,
                            "status": status,
                        }
                    )
                    continue
                try:
                    limit_px = float(px_val)
                except (TypeError, ValueError):
                    status = "invalid"
                    orders_state.append(
                        {
                            "order_id": f"o{order_counter}",
                            "block_ts_ms": block.get("block_ts_ms"),
                            "book_event_id": block.get("book_event_id"),
                            "side": order.side,
                            "size": order.size,
                            "price": order.price,
                            "status": status,
                        }
                    )
                    continue

            trades_bucket = block.get("trade_bucket") or []
            if fill_model == "upper":
                fill: Optional[FillResult] = fill_order_upper(
                    order.side, order.size, limit_px, trades_bucket, allow_top_fill=allow_top_fill
                )
            else:
                fill = fill_order_lower(
                    order.side,
                    order.size,
                    limit_px,
                    trades_bucket,
                    alpha=lower_alpha,
                    nprints=lower_nprints,
                    allow_top_fill=allow_top_fill,
                )
            if fill is None or fill.size <= 0:
                orders_state.append(
                    {
                        "order_id": f"o{order_counter}",
                        "block_ts_ms": block.get("block_ts_ms"),
                        "book_event_id": block.get("book_event_id"),
                        "side": order.side,
                        "size": order.size,
                        "price": order.price if order.price is not None else limit_px,
                        "status": "no_fill",
                    }
                )
                continue

            fill_px = fill.price
            fill_sz = fill.size

            signed_size = fill_sz if order.side == "buy" else -fill_sz

            # ポジション制限チェック
            if max_abs_position is not None:
                new_pos = state["position"] + signed_size
                if abs(new_pos) > max_abs_position + 1e-9:
                    status = "rejected_position_limit"
                    orders_state.append(
                        {
                            "order_id": f"o{order_counter}",
                            "block_ts_ms": block.get("block_ts_ms"),
                            "book_event_id": block.get("book_event_id"),
                            "side": order.side,
                            "size": order.size,
                            "price": fill_px,
                            "status": status,
                        }
                    )
                    continue

            # 手数料/リベート
            fee = 0.0
            rebate = 0.0
            if order.post_only:
                rebate = abs(fill_sz) * fill_px * maker_rebate_bps / 10_000
            else:
                fee = abs(fill_sz) * fill_px * taker_fee_bps / 10_000

            # PnL計算とポジション更新
            pos = state["position"]
            avg_cost = state["avg_cost"]
            realized = state["realized_price_pnl"]

            if pos == 0:
                new_pos = signed_size
                avg_cost = fill_px
            elif (pos > 0 and signed_size > 0) or (pos < 0 and signed_size < 0):
                # 同方向で積み増し
                new_pos = pos + signed_size
                avg_cost = (pos * avg_cost + signed_size * fill_px) / new_pos
            else:
                # 反対方向でクローズ/反転
                closing = min(abs(pos), abs(signed_size))
                if pos > 0:  # 長→売りで決済
                    realized += (fill_px - avg_cost) * closing
                else:  # 短→買いで決済
                    realized += (avg_cost - fill_px) * closing
                new_pos = pos + signed_size
                if new_pos == 0:
                    avg_cost = None
                else:
                    avg_cost = fill_px  # 反転後は最新約定でコストリセット

            state["position"] = new_pos
            state["avg_cost"] = avg_cost
            state["realized_price_pnl"] = realized
            state["fees"] += fee
            state["rebates"] += rebate
            state["last_price"] = fill_px

            status = "filled"
            trades.append(
                {
                    "order_id": f"o{order_counter}",
                    "block_ts_ms": block.get("block_ts_ms"),
                    "book_event_id": block.get("book_event_id"),
                    "side": order.side,
                    "size": fill_sz,
                    "price": fill_px,
                    "fee": fee,
                    "rebate": rebate,
                }
            )
            orders_state.append(
                {
                    "order_id": f"o{order_counter}",
                    "block_ts_ms": block.get("block_ts_ms"),
                    "book_event_id": block.get("book_event_id"),
                    "side": order.side,
                    "size": fill_sz,
                    "price": fill_px,
                    "status": status,
                }
            )

        mark_price = mid if mid is not None else state["last_price"]
        unrealized = 0.0
        if state["position"] != 0 and mark_price is not None and state["avg_cost"] is not None:
            unrealized = (mark_price - state["avg_cost"]) * state["position"]
        total_pnl = (
            state["realized_price_pnl"] + state["fees"] + state["rebates"] + state["funding"] + unrealized
        )
        ledger.append(
            {
                "block_ts_ms": block.get("block_ts_ms"),
                "book_event_id": block.get("book_event_id"),
                "position": state["position"],
                "avg_cost": state["avg_cost"],
                "price_pnl": state["realized_price_pnl"],
                "fees": state["fees"],
                "rebates": state["rebates"],
                "funding": state["funding"],
                "unrealized_pnl": unrealized,
                "total_pnl": total_pnl,
                "mark_price": mark_price,
                "mid_ret": state.get("mid_ret"),
                "abs_mid_ret": state.get("abs_mid_ret"),
                "market_spread_bps": state.get("market_spread_bps"),
                "stop_triggered": bool(state.get("stop_triggered")),
                "pull_triggered": bool(state.get("pull_triggered")),
                "halt_triggered": bool(state.get("halt_triggered")),
                "boost_triggered": bool(state.get("boost_triggered")),
                "strategy_spread_bps": state.get("strategy_spread_bps"),
                "strategy_size": state.get("strategy_size"),
                "strategy_bid_spread_bps": state.get("strategy_bid_spread_bps"),
                "strategy_ask_spread_bps": state.get("strategy_ask_spread_bps"),
                "strategy_bid_size": state.get("strategy_bid_size"),
                "strategy_ask_size": state.get("strategy_ask_size"),
                "pull_side": state.get("pull_side"),
                "signed_volume": state.get("signed_volume"),
                "signed_volume_window": state.get("signed_volume_window"),
                "post_pull_unwind_active": bool(state.get("post_pull_unwind_active")),
            }
        )

        # 次ブロックの stop 判定に使うため、損益系列を state に保持
        state["total_pnl"] = total_pnl
        state["peak_pnl"] = max(float(state.get("peak_pnl", 0.0) or 0.0), float(total_pnl))
        state["drawdown"] = float(total_pnl) - float(state["peak_pnl"])

    return trades, ledger, orders_state


def run_mm_sim(
    blocks_path: str | Path,
    out_dir: str | Path = "mm_sim_out",
    strategy: Optional[StrategyFn] = None,
    taker_fee_bps: float = 0.0,
    maker_rebate_bps: float = 0.0,
    max_abs_position: Optional[float] = None,
    fill_model: str = "lower",
    lower_alpha: float = 0.5,
    lower_nprints: Optional[int] = None,
    allow_top_fill: bool = False,
    signed_volume_window_s: Optional[float] = None,
) -> tuple[Path, Path, Path]:
    """blocks.parquet を読み込み、シミュレーションを実行してトレード/台帳をParquet出力。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "sim_trades.parquet"
    ledger_path = out_dir / "ledger.parquet"
    orders_path = out_dir / "orders.parquet"

    table = pq.read_table(blocks_path)
    blocks = table.to_pylist()
    trades, ledger, orders = simulate_blocks(
        blocks,
        strategy=strategy,
        taker_fee_bps=taker_fee_bps,
        maker_rebate_bps=maker_rebate_bps,
        max_abs_position=max_abs_position,
        fill_model=fill_model,
        lower_alpha=lower_alpha,
        lower_nprints=lower_nprints,
        allow_top_fill=allow_top_fill,
        signed_volume_window_s=signed_volume_window_s,
    )
    pq.write_table(pa.Table.from_pylist(trades), trades_path)
    pq.write_table(pa.Table.from_pylist(ledger), ledger_path)
    pq.write_table(pa.Table.from_pylist(orders), orders_path)
    return trades_path, ledger_path, orders_path
