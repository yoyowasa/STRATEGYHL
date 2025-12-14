from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class FillResult:
    size: float
    price: float
    trades_used: int


def _eligible_trades(order_side: str, limit_price: float, trades: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    eligible = []
    for tr in trades:
        side = tr.get("side")
        px = tr.get("px")
        sz = tr.get("sz")
        try:
            px_f = float(px)
            sz_f = float(sz)
        except (TypeError, ValueError):
            continue
        if sz_f <= 0:
            continue
        if order_side == "buy":
            if side == "sell" and px_f <= limit_price:
                eligible.append({"px": px_f, "sz": sz_f})
        else:
            if side == "buy" and px_f >= limit_price:
                eligible.append({"px": px_f, "sz": sz_f})
    return eligible


def _weighted_price(trades: Sequence[Mapping[str, float]]) -> float:
    num = sum(t["px"] * t["sz"] for t in trades)
    den = sum(t["sz"] for t in trades)
    return num / den if den else 0.0


def fill_order_upper(
    side: str,
    size: float,
    limit_price: float,
    trades: Sequence[Mapping[str, object]],
    allow_top_fill: bool = True,
) -> Optional[FillResult]:
    """最も楽観的なフィル: 全対象取引量を使用し、サイズ上限まで約定。"""
    eligible = _eligible_trades(side, limit_price, trades)
    if not eligible:
        if not allow_top_fill:
            return None
        if limit_price is None:
            return None
        return FillResult(size=size, price=limit_price, trades_used=0)

    total_sz = sum(t["sz"] for t in eligible)
    fill_sz = min(size, total_sz)
    px = _weighted_price(eligible)
    return FillResult(size=fill_sz, price=px, trades_used=len(eligible))


def fill_order_lower(
    side: str,
    size: float,
    limit_price: float,
    trades: Sequence[Mapping[str, object]],
    alpha: float = 0.5,
    nprints: Optional[int] = None,
    allow_top_fill: bool = True,
) -> Optional[FillResult]:
    """控えめなフィル: 対象取引量の alpha 倍、nprints 件まで使用。"""
    eligible = _eligible_trades(side, limit_price, trades)
    if nprints is not None:
        eligible = eligible[: int(nprints)]
    if not eligible:
        if not allow_top_fill:
            return None
        if limit_price is None:
            return None
        return FillResult(size=size, price=limit_price, trades_used=0)

    total_sz = sum(t["sz"] for t in eligible)
    target_sz = total_sz * alpha
    fill_sz = min(size, target_sz)
    px = _weighted_price(eligible)
    return FillResult(size=fill_sz, price=px, trades_used=len(eligible))
