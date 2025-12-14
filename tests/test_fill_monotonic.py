from hlmm.mm import fill_order_lower, fill_order_upper


def test_fill_monotonic():
    trades = [
        {"side": "sell", "px": 100.0, "sz": 1.0},
        {"side": "sell", "px": 99.5, "sz": 2.0},
    ]
    upper = fill_order_upper("buy", 5.0, 101.0, trades)
    lower = fill_order_lower("buy", 5.0, 101.0, trades, alpha=0.5)

    assert upper is not None
    assert lower is not None
    assert upper.size >= lower.size
    assert upper.trades_used >= lower.trades_used
