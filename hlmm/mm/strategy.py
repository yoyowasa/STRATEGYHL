from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from hlmm.mm import Order


@dataclass(frozen=True)
class StrategyParams:
    base_spread_bps: float = 5.0
    base_size: float = 1.0
    inventory_skew_bps: float = 2.0
    inventory_target: float = 0.0
    max_abs_position: Optional[float] = None
    strategy_variant: int = 1  # 1: baseline+inventory, 2/3: 拡張用


def _clamp_position(size: float, params: StrategyParams) -> float:
    if params.max_abs_position is None:
        return size
    if abs(size) > params.max_abs_position:
        return params.max_abs_position if size > 0 else -params.max_abs_position
    return size


def decide_orders(state: Mapping[str, object], params: StrategyParams) -> Dict[str, object]:
    """
    Strategy1: ベーススプレッドに inventory skew を加味した bid/ask を返す。
    state: mid/spread/imbalance/signed_volume/basis_bps/inventory を含む辞書を想定。
    """
    mid = state.get("mid")
    position = float(state.get("position", 0.0) or 0.0)
    if mid is None:
        return {"orders": [], "halt": True}
    try:
        mid_f = float(mid)
    except (TypeError, ValueError):
        return {"orders": [], "halt": True}

    # スプレッド計算
    spread_bps = params.base_spread_bps
    skew = (position - params.inventory_target) * params.inventory_skew_bps
    bid_px = mid_f * (1 - (spread_bps + max(skew, 0)) / 10_000)
    ask_px = mid_f * (1 + (spread_bps + max(-skew, 0)) / 10_000)

    size = params.base_size
    size = _clamp_position(size, params)

    orders = [
        Order(side="buy", size=size, price=bid_px, post_only=True),
        Order(side="sell", size=size, price=ask_px, post_only=True),
    ]
    return {
        "orders": orders,
        "halt": False,
        "bid_px": bid_px,
        "ask_px": ask_px,
        "skew": skew,
    }
