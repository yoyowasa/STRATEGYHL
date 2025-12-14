from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Iterable, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.data import AssetCtxEvent, BboEvent, BookEvent, TradeEvent


def _event_ts(event) -> int:
    ts = getattr(event, "ts_ms", None)
    if ts is None:
        return event.recv_ts_ms
    return ts


def _dec_to_str(val):
    if isinstance(val, Decimal):
        return format(val, "f")
    return val


def _book_top(book: BookEvent) -> dict:
    best_bid = max(book.bids, key=lambda x: x[0], default=None)
    best_ask = min(book.asks, key=lambda x: x[0], default=None)
    return {
        "bid_px": _dec_to_str(best_bid[0]) if best_bid else None,
        "bid_sz": _dec_to_str(best_bid[1]) if best_bid else None,
        "ask_px": _dec_to_str(best_ask[0]) if best_ask else None,
        "ask_sz": _dec_to_str(best_ask[1]) if best_ask else None,
    }


def align_blocks(
    book_events: Sequence[BookEvent],
    trade_events: Sequence[TradeEvent] | None = None,
    ctx_events: Sequence[AssetCtxEvent] | None = None,
    bbo_events: Sequence[BboEvent] | None = None,
) -> List[dict]:
    books = sorted(book_events, key=lambda e: (_event_ts(e), e.event_id))
    trades = sorted(trade_events or [], key=lambda e: (_event_ts(e), e.event_id))
    ctxs = sorted(ctx_events or [], key=lambda e: (_event_ts(e), e.event_id))
    bbo_list = sorted(bbo_events or [], key=lambda e: (_event_ts(e), e.event_id))

    trade_idx = 0
    ctx_idx = 0
    bbo_idx = 0
    last_ctx = None
    last_bbo = None
    blocks: List[dict] = []

    for book in books:
        block_ts = _event_ts(book)

        # trade bucket: 前回ブロックより後、現在ブロック時刻以下
        bucket: List[TradeEvent] = []
        while trade_idx < len(trades) and _event_ts(trades[trade_idx]) <= block_ts:
            bucket.append(trades[trade_idx])
            trade_idx += 1

        # ctx: 最後に得られたもの
        while ctx_idx < len(ctxs) and _event_ts(ctxs[ctx_idx]) <= block_ts:
            last_ctx = ctxs[ctx_idx]
            ctx_idx += 1

        # bbo: 任意で保持（book_topが欠ける場合に備えて付帯情報として残す）
        while bbo_idx < len(bbo_list) and _event_ts(bbo_list[bbo_idx]) <= block_ts:
            last_bbo = bbo_list[bbo_idx]
            bbo_idx += 1

        if hasattr(book, "bids"):
            missing_book = not book.bids and not book.asks
            book_top = _book_top(book)
        else:
            missing_book = False
            book_top = {
                "bid_px": _dec_to_str(getattr(book, "bid_px", None)),
                "bid_sz": _dec_to_str(getattr(book, "bid_sz", None)),
                "ask_px": _dec_to_str(getattr(book, "ask_px", None)),
                "ask_sz": _dec_to_str(getattr(book, "ask_sz", None)),
            }

        record = {
            "block_ts_ms": block_ts,
            "book_event_id": book.event_id,
            "book_top": book_top,
            "trade_bucket": [
                {
                    "px": _dec_to_str(t.px),
                    "sz": _dec_to_str(t.sz),
                    "side": t.side,
                    "trade_id": t.trade_id,
                    "ts_ms": t.ts_ms,
                }
                for t in bucket
            ],
            "ctx_last": None,
            "missing_book": missing_book,
            "missing_trades": len(bucket) == 0,
        }
        if last_ctx is not None:
            record["ctx_last"] = {
                "symbol": last_ctx.symbol,
                "status": last_ctx.status,
                "ts_ms": getattr(last_ctx, "ts_ms", None),
                "metadata": last_ctx.metadata,
            }
        if last_bbo is not None:
            record["bbo_last"] = {
                "symbol": last_bbo.symbol,
                "bid_px": _dec_to_str(last_bbo.bid_px),
                "bid_sz": _dec_to_str(last_bbo.bid_sz),
                "ask_px": _dec_to_str(last_bbo.ask_px),
                "ask_sz": _dec_to_str(last_bbo.ask_sz),
                "ts_ms": getattr(last_bbo, "ts_ms", None),
            }

        blocks.append(record)

    return blocks


def save_blocks_parquet(blocks: Iterable[dict], output_path: str | Path) -> Path:
    table = pa.Table.from_pylist(list(blocks))
    output = Path(output_path)
    pq.write_table(table, output)
    return output
