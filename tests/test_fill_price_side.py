from hlmm.mm import fill_order_lower, fill_order_upper


def test_fill_price_side_constraints():
    trades = [
        {"side": "buy", "px": 101.0, "sz": 1.0},
        {"side": "sell", "px": 100.5, "sz": 1.0},
        {"side": "sell", "px": 101.5, "sz": 1.0},
    ]

    # bid は trade_px <= bid_px の sell のみ対象
    upper_bid = fill_order_upper("buy", 5.0, 101.0, trades)
    assert upper_bid is not None
    assert upper_bid.size == 1.0  # only 100.5 sell trade
    assert upper_bid.price == 100.5

    # ask は trade_px >= ask_px の buy のみ対象
    upper_ask = fill_order_upper("sell", 5.0, 101.0, trades)
    assert upper_ask is not None
    assert upper_ask.size == 1.0
    assert upper_ask.price == 101.0

    # lower でも条件は同じ
    lower_bid = fill_order_lower("buy", 5.0, 101.0, trades, alpha=1.0)
    assert lower_bid is not None
    assert lower_bid.size == upper_bid.size
