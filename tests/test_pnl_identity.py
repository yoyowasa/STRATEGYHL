import math

from hlmm.mm import Order, simulate_blocks


def test_pnl_identity_holds():
    blocks = [
        {
            "block_ts_ms": 1000,
            "book_event_id": "b1",
            "book_top": {"bid_px": 99.0, "bid_sz": 1, "ask_px": 101.0, "ask_sz": 1},
            "missing_book": False,
        },
        {
            "block_ts_ms": 2000,
            "book_event_id": "b2",
            "book_top": {"bid_px": 101.0, "bid_sz": 1, "ask_px": 103.0, "ask_sz": 1},
            "missing_book": False,
        },
    ]

    def strat(block, state):
        if block["block_ts_ms"] == 1000:
            return [Order(side="buy", size=1, price=100.0, post_only=True)]
        if block["block_ts_ms"] == 2000:
            return [Order(side="sell", size=1, price=102.0, post_only=True)]
        return []

    trades, ledger, orders = simulate_blocks(
        blocks,
        strategy=strat,
        maker_rebate_bps=1.0,
        taker_fee_bps=0.0,
        fill_model="upper",
        allow_top_fill=True,
    )
    assert len(trades) == 2
    # 各行で total = price_pnl + fees + rebates + funding + unrealized が成立
    for row in ledger:
        total = row["total_pnl"]
        parts = row["price_pnl"] + row["fees"] + row["rebates"] + row["funding"] + row["unrealized_pnl"]
        assert math.isclose(total, parts, rel_tol=1e-9, abs_tol=1e-9)
