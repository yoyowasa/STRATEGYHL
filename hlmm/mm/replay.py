from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


def apply_replay(
    fills: Iterable[Mapping[str, object]],
    fundings: Iterable[Mapping[str, object]] | None = None,
) -> pa.Table:
    """UserFill/UserFunding の列から台帳を再構成する。"""
    position = 0.0
    avg_cost = None
    price_pnl = 0.0
    fees = 0.0
    rebates = 0.0
    funding_acc = 0.0
    last_price = None
    rows: List[dict] = []

    fundings_list = sorted(list(fundings or []), key=lambda x: x.get("ts_ms", 0))
    funding_idx = 0

    fills_list = sorted(list(fills), key=lambda x: x.get("ts_ms", 0))

    for fill in fills_list:
        ts = fill.get("ts_ms")
        side = fill.get("side")
        px = float(fill.get("px"))
        sz = float(fill.get("sz"))
        fee = float(fill.get("fee", 0.0))
        trade_id = fill.get("trade_id")

        # funding を時系列順で適用
        while funding_idx < len(fundings_list) and fundings_list[funding_idx].get("ts_ms", 0) <= ts:
            funding_acc += float(fundings_list[funding_idx].get("amount", 0.0))
            funding_idx += 1

        signed_sz = sz if side == "buy" else -sz
        if position == 0:
            position = signed_sz
            avg_cost = px
        elif (position > 0 and signed_sz > 0) or (position < 0 and signed_sz < 0):
            new_pos = position + signed_sz
            avg_cost = (position * avg_cost + signed_sz * px) / new_pos
            position = new_pos
        else:
            closing = min(abs(position), abs(signed_sz))
            if position > 0:
                price_pnl += (px - avg_cost) * closing
            else:
                price_pnl += (avg_cost - px) * closing
            position += signed_sz
            if position == 0:
                avg_cost = None
            else:
                avg_cost = px

        # fee は買いも売りも控除とする
        fees -= fee
        last_price = px

        unrealized = 0.0
        if position != 0 and avg_cost is not None:
            unrealized = (last_price - avg_cost) * position
        total_pnl = price_pnl + fees + rebates + funding_acc + unrealized

        rows.append(
            {
                "ts_ms": ts,
                "trade_id": trade_id,
                "position": position,
                "avg_cost": avg_cost,
                "price_pnl": price_pnl,
                "fees": fees,
                "rebates": rebates,
                "funding": funding_acc,
                "unrealized_pnl": unrealized,
                "total_pnl": total_pnl,
                "side": side,
                "px": px,
                "sz": sz,
            }
        )

    # 残りの funding
    while funding_idx < len(fundings_list):
        funding_acc += float(fundings_list[funding_idx].get("amount", 0.0))
        funding_idx += 1
        total_pnl = price_pnl + fees + rebates + funding_acc
        rows.append(
            {
                "ts_ms": fundings_list[funding_idx - 1].get("ts_ms"),
                "trade_id": None,
                "position": position,
                "avg_cost": avg_cost,
                "price_pnl": price_pnl,
                "fees": fees,
                "rebates": rebates,
                "funding": funding_acc,
                "unrealized_pnl": (last_price - avg_cost) * position if position != 0 and avg_cost else 0.0,
                "total_pnl": total_pnl,
                "side": None,
                "px": None,
                "sz": None,
            }
        )

    return pa.Table.from_pylist(rows)


def run_replay(
    fills_path: str | Path,
    fundings_path: str | Path | None,
    out_dir: str | Path = "reports",
    run_id: str = "replay",
) -> Path:
    fills_table = pq.read_table(fills_path)
    fundings_table = pq.read_table(fundings_path) if fundings_path else None
    ledger = apply_replay(
        fills_table.to_pylist(), fundings_table.to_pylist() if fundings_table is not None else None
    )
    out_dir = Path(out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "ledger_replay.parquet"
    pq.write_table(ledger, ledger_path)
    return ledger_path
