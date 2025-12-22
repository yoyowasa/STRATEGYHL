import pytest

from hlmm.mm import StrategyParams, decide_orders, simulate_blocks


def test_pull_triggers_by_abs_signed_volume_block():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=2.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        pull_when_abs_signed_volume_gt=5.0,
        pull_spread_add_bps=3.0,
        pull_size_factor=1.0,
        pull_mode="one_side",
    )
    state = {"mid": 100.0, "position": 0.0, "signed_volume": 6.0}
    res = decide_orders(state, params)
    assert res["pull_triggered"] is True
    assert res["pull_side"] == "sell"
    assert res["strategy_bid_spread_bps"] == pytest.approx(1.0)
    assert res["strategy_ask_spread_bps"] == pytest.approx(4.0)


def test_pull_uses_signed_volume_window_when_configured():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=2.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        pull_when_abs_signed_volume_gt=5.0,
        pull_signed_volume_window_s=2.0,
        pull_spread_add_bps=3.0,
        pull_size_factor=1.0,
        pull_mode="one_side",
    )
    # signed_volume は0だが、window は閾値超え
    state = {"mid": 100.0, "position": 0.0, "signed_volume": 0.0, "signed_volume_window": -6.0}
    res = decide_orders(state, params)
    assert res["pull_triggered"] is True
    assert res["pull_side"] == "buy"
    assert res["strategy_bid_spread_bps"] == pytest.approx(4.0)
    assert res["strategy_ask_spread_bps"] == pytest.approx(1.0)


def test_sim_signed_volume_window_rolls_over_time():
    blocks = [
        {"block_ts_ms": 1000, "book_event_id": "b1", "book_top": {"bid_px": 99.0, "ask_px": 101.0}, "trade_bucket": [{"side": "buy", "sz": 1}]},
        {"block_ts_ms": 2000, "book_event_id": "b2", "book_top": {"bid_px": 99.0, "ask_px": 101.0}, "trade_bucket": [{"side": "buy", "sz": 1}]},
        {"block_ts_ms": 4000, "book_event_id": "b3", "book_top": {"bid_px": 99.0, "ask_px": 101.0}, "trade_bucket": [{"side": "buy", "sz": 1}]},
    ]
    _, ledger, _ = simulate_blocks(blocks, strategy=lambda b, s: [], signed_volume_window_s=2.0)
    assert [row["signed_volume_window"] for row in ledger] == pytest.approx([1.0, 2.0, 2.0])
