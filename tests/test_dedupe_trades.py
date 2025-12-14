from decimal import Decimal

from hlmm.data import TradeEvent
from hlmm.io import dedupe_events


def test_trade_dedupe_by_symbol_tradeid_ts():
    e1 = TradeEvent(
        event_id="x1",
        recv_ts_ms=1,
        symbol="BTC",
        px=Decimal("1.0"),
        sz=Decimal("2.0"),
        side="buy",
        trade_id="t1",
        ts_ms=100,
    )
    e2 = TradeEvent(
        event_id="x2",
        recv_ts_ms=2,
        symbol="BTC",
        px=Decimal("1.0"),
        sz=Decimal("2.0"),
        side="buy",
        trade_id="t1",
        ts_ms=100,
    )
    deduped = dedupe_events([e1, e2])
    assert len(deduped) == 1
    assert deduped[0].dedupe_key == "BTC:t1:100"
