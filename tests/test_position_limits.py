from hlmm.mm import Order, simulate_blocks


def test_position_limits_and_cross_detection():
    blocks = [
        {
            "block_ts_ms": 1000,
            "book_event_id": "b1",
            "book_top": {"bid_px": 99.0, "bid_sz": 1, "ask_px": 101.0, "ask_sz": 1},
            "missing_book": False,
            "trade_bucket": [{"side": "sell", "px": 100.0, "sz": 2}],
        },
        {
            "block_ts_ms": 2000,
            "book_event_id": "b2",
            "book_top": {"bid_px": 100.0, "bid_sz": 1, "ask_px": 102.0, "ask_sz": 1},
            "missing_book": False,
        },
    ]

    def strat(block, state):
        if block["block_ts_ms"] == 1000:
            # 上限超過のため拒否される
            return [Order(side="buy", size=2, price=100.0, post_only=True)]
        if block["block_ts_ms"] == 2000:
            # bid>=ask のクロスなので無効
            return [Order(side="buy", size=1, price=102.0, post_only=True)]
        return []

    trades, ledger, orders = simulate_blocks(
        blocks, strategy=strat, max_abs_position=0.5, allow_top_fill=True, fill_model="upper"
    )
    # どちらの注文も拒否され、ポジションは0のまま
    assert len(trades) == 0
    assert ledger[-1]["position"] == 0.0
    statuses = {o["status"] for o in orders}
    assert "rejected_position_limit" in statuses
    assert "invalid_cross" in statuses
