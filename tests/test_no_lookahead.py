from decimal import Decimal

from hlmm.data import BookEvent, TradeEvent
from hlmm.features import align_blocks


def test_no_lookahead_trades():
    books = [
        BookEvent(
            event_id="b1",
            recv_ts_ms=1,
            symbol="BTC",
            bids=[(Decimal("100"), Decimal("1"))],
            asks=[(Decimal("101"), Decimal("2"))],
            ts_ms=1000,
        ),
        BookEvent(
            event_id="b2",
            recv_ts_ms=2,
            symbol="BTC",
            bids=[(Decimal("100"), Decimal("1"))],
            asks=[(Decimal("101"), Decimal("2"))],
            ts_ms=2000,
        ),
    ]
    trades = [
        TradeEvent(
            event_id="t1",
            recv_ts_ms=1,
            symbol="BTC",
            px=Decimal("100.5"),
            sz=Decimal("1"),
            side="buy",
            trade_id="tid1",
            ts_ms=1000,
        ),
        TradeEvent(
            event_id="t2",
            recv_ts_ms=2,
            symbol="BTC",
            px=Decimal("100.6"),
            sz=Decimal("1"),
            side="buy",
            trade_id="tid2",
            ts_ms=1500,
        ),
    ]

    blocks = align_blocks(books, trades)

    # 最初のブロックは t1 のみ、t2 は次ブロックで参照される
    first_bucket = blocks[0]["trade_bucket"]
    assert len(first_bucket) == 1
    assert first_bucket[0]["trade_id"] == "tid1"
    second_bucket = blocks[1]["trade_bucket"]
    assert len(second_bucket) == 1
    assert second_bucket[0]["trade_id"] == "tid2"
