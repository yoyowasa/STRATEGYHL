from decimal import Decimal

from hlmm.data import BookEvent
from hlmm.features import align_blocks


def test_block_times_monotonic():
    books = [
        BookEvent(
            event_id="b2",
            recv_ts_ms=2,
            symbol="BTC",
            bids=[(Decimal("100"), Decimal("1"))],
            asks=[(Decimal("101"), Decimal("2"))],
            ts_ms=2000,
        ),
        BookEvent(
            event_id="b1",
            recv_ts_ms=1,
            symbol="BTC",
            bids=[(Decimal("100"), Decimal("1"))],
            asks=[(Decimal("101"), Decimal("2"))],
            ts_ms=1000,
        ),
    ]
    blocks = align_blocks(books)
    times = [b["block_ts_ms"] for b in blocks]
    assert times == sorted(times)
