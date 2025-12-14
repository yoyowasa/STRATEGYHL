from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, List, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


def _to_decimal(val) -> Decimal | None:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    try:
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _to_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _calc_mid(top: Mapping[str, object]) -> tuple[float, bool]:
    bid_px = _to_float(top.get("bid_px"))
    ask_px = _to_float(top.get("ask_px"))
    if bid_px is None or ask_px is None:
        return math.nan, True
    return (bid_px + ask_px) / 2.0, False


def _calc_spread(top: Mapping[str, object]) -> tuple[float, bool]:
    bid_px = _to_float(top.get("bid_px"))
    ask_px = _to_float(top.get("ask_px"))
    if bid_px is None or ask_px is None:
        return math.nan, True
    return ask_px - bid_px, False


def _calc_microprice(top: Mapping[str, object]) -> tuple[float, bool]:
    bid_px = _to_decimal(top.get("bid_px"))
    ask_px = _to_decimal(top.get("ask_px"))
    bid_sz = _to_decimal(top.get("bid_sz"))
    ask_sz = _to_decimal(top.get("ask_sz"))
    if None in (bid_px, ask_px, bid_sz, ask_sz):
        return math.nan, True
    denom = bid_sz + ask_sz
    if denom == 0:
        return math.nan, True
    micro = (bid_px * ask_sz + ask_px * bid_sz) / denom
    return float(micro), False


def _calc_imbalance(top: Mapping[str, object]) -> tuple[float, bool]:
    bid_sz = _to_decimal(top.get("bid_sz"))
    ask_sz = _to_decimal(top.get("ask_sz"))
    if None in (bid_sz, ask_sz):
        return math.nan, True
    denom = bid_sz + ask_sz
    if denom == 0:
        return math.nan, True
    imbalance = (bid_sz - ask_sz) / denom
    return float(imbalance), False


def _calc_signed_volume(trades: list[Mapping[str, object]]) -> tuple[float, bool, float | None]:
    if not trades:
        return math.nan, True, None
    total = 0.0
    last_px = None
    for tr in trades:
        side = tr.get("side")
        sz = _to_float(tr.get("sz"))
        px = _to_float(tr.get("px"))
        if sz is None or side not in ("buy", "sell"):
            return math.nan, True, None
        sign = 1.0 if side == "buy" else -1.0
        total += sign * sz
        if px is not None:
            last_px = px
    return total, False, last_px


def _calc_basis_bps(mid: float, mid_missing: bool, last_trade_px: float | None) -> tuple[float, bool]:
    if mid_missing or mid is None or math.isnan(mid) or last_trade_px is None:
        return math.nan, True
    if mid == 0:
        return math.nan, True
    basis = (last_trade_px - mid) / mid * 10_000
    return basis, False


def compute_features(blocks: Iterable[Mapping[str, object]]) -> List[dict]:
    results: List[dict] = []
    for block in blocks:
        top = block.get("book_top") or {}
        trades = block.get("trade_bucket") or []
        missing_book = bool(block.get("missing_book"))
        missing_trades = bool(block.get("missing_trades"))

        mid, mid_missing = _calc_mid(top)
        spread, spread_missing = _calc_spread(top)
        microprice, micro_missing = _calc_microprice(top)
        imbalance, imbalance_missing = _calc_imbalance(top)
        signed_vol, signed_vol_missing, last_trade_px = _calc_signed_volume(trades)
        basis_bps, basis_missing = _calc_basis_bps(mid, mid_missing, last_trade_px)

        result = {
            "block_ts_ms": block.get("block_ts_ms"),
            "book_event_id": block.get("book_event_id"),
            "mid": mid,
            "spread": spread,
            "microprice": microprice,
            "imbalance": imbalance,
            "signed_volume": signed_vol,
            "basis_bps": basis_bps,
            "missing_mid": mid_missing,
            "missing_spread": spread_missing,
            "missing_microprice": micro_missing,
            "missing_imbalance": imbalance_missing,
            "missing_signed_volume": signed_vol_missing,
            "missing_basis_bps": basis_missing,
            "missing_book": missing_book,
            "missing_trades": missing_trades,
        }
        results.append(result)
    # 決定的な並び
    results.sort(key=lambda r: (r.get("block_ts_ms") or 0, str(r.get("book_event_id") or "")))
    return results


def save_features_parquet(features: Iterable[dict], output_path: str | Path) -> Path:
    table = pa.Table.from_pylist(list(features))
    output = Path(output_path)
    pq.write_table(table, output)
    return output
