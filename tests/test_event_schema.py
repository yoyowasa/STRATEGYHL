from decimal import Decimal

import pytest

from hlmm.data import (
    AssetCtxEvent,
    BboEvent,
    BookEvent,
    TradeEvent,
    UserFillEvent,
    UserFundingEvent,
)


def test_book_event_requires_fields():
    with pytest.raises(ValueError):
        BookEvent(event_id="e1", recv_ts_ms=1, symbol="", bids=[], asks=[])
    with pytest.raises(ValueError):
        BookEvent(event_id="e1", recv_ts_ms=1, symbol="BTC", bids=None, asks=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BookEvent(event_id="e1", recv_ts_ms=1, symbol="BTC", bids=[("bad", 1)], asks=[])


def test_trade_event_requires_fields_and_decimal():
    evt = TradeEvent(
        event_id="t1",
        recv_ts_ms=1,
        symbol="BTC",
        px=Decimal("1.23"),
        sz="2.34",
        side="buy",
        trade_id="id1",
        ts_ms=2,
    )
    assert evt.px == Decimal("1.23")
    assert evt.sz == Decimal("2.34")
    with pytest.raises(ValueError):
        TradeEvent(
            event_id="t2",
            recv_ts_ms=1,
            symbol="BTC",
            px="bad",
            sz="1",
            side="buy",
            trade_id="id2",
            ts_ms=2,
        )


def test_bbo_event_requires_decimal():
    evt = BboEvent(
        event_id="b1",
        recv_ts_ms=1,
        symbol="BTC",
        bid_px="1.1",
        bid_sz="2.2",
        ask_px="1.2",
        ask_sz="2.3",
    )
    assert evt.bid_px == Decimal("1.1")
    assert evt.dedupe_key == evt.event_id


def test_asset_ctx_event_missing_fields():
    with pytest.raises(ValueError):
        AssetCtxEvent(event_id="a1", recv_ts_ms=1, symbol="", status="inactive")


def test_user_fill_requires_decimal():
    evt = UserFillEvent(
        event_id="f1",
        recv_ts_ms=1,
        symbol="BTC",
        px="1.23",
        sz="4.56",
        side="buy",
        fee="0.001",
        trade_id="tid",
        ts_ms=2,
    )
    assert evt.fee == Decimal("0.001")
    with pytest.raises(ValueError):
        UserFillEvent(
            event_id="f2",
            recv_ts_ms=1,
            symbol="BTC",
            px="bad",
            sz="1",
            side="buy",
            fee="0",
            trade_id="tid",
            ts_ms=2,
        )


def test_user_funding_requires_decimal():
    evt = UserFundingEvent(
        event_id="fu1",
        recv_ts_ms=1,
        symbol="BTC",
        amount="0.1",
        rate="0.0001",
        ts_ms=2,
        funding_id="fid",
    )
    assert evt.amount == Decimal("0.1")
    with pytest.raises(ValueError):
        UserFundingEvent(
            event_id="fu2",
            recv_ts_ms=1,
            symbol="BTC",
            amount="bad",
            rate="0.0001",
            ts_ms=2,
            funding_id="fid",
        )
