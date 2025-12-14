from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

SIG_DIGITS = 5
MAX_DECIMALS = 6  # パーペチュアル想定の最大小数桁


def _require_decimal(value: Decimal, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} は Decimal 型である必要があります")
    if value.is_nan() or value.is_infinite():
        raise ValueError(f"{name} が有限の数値ではありません")
    return value


def round_price(px: Decimal, max_decimals: int = MAX_DECIMALS) -> Decimal:
    """価格を丸める（最大有効桁数5、最大小数桁 max_decimals）。"""
    px = _require_decimal(px, "px")
    if px.is_zero():
        return Decimal("0")
    if max_decimals < 0:
        raise ValueError("max_decimals は 0 以上である必要があります")

    # 有効桁数から量子化指数を計算し、小数桁上限でクリップする
    adjusted = px.copy_abs().normalize().adjusted()
    exp_for_sig = adjusted - (SIG_DIGITS - 1)
    exp = max(exp_for_sig, -max_decimals)
    quantum = Decimal(1).scaleb(exp)
    try:
        return px.quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"価格の丸めに失敗しました: {exc}") from exc


def round_size(sz: Decimal, sz_decimals: int) -> Decimal:
    """サイズを sz_decimals 小数桁に丸める。"""
    sz = _require_decimal(sz, "sz")
    if sz_decimals < 0:
        raise ValueError("sz_decimals は 0 以上である必要があります")
    quantum = Decimal(1).scaleb(-sz_decimals)
    try:
        return sz.quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"サイズの丸めに失敗しました: {exc}") from exc
