import pytest

from hlmm.mm import StrategyParams, decide_orders


def test_pull_one_side_sell_on_positive_signed_volume():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=2.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.0,
        pull_spread_add_bps=3.0,
        pull_size_factor=0.5,
        pull_mode="one_side",
    )
    state = {"mid": 100.0, "position": 0.0, "market_spread_bps": 1.0, "signed_volume": 10.0}
    res = decide_orders(state, params)
    assert res["pull_triggered"] is True
    assert res["pull_side"] == "sell"
    assert res["strategy_bid_spread_bps"] == pytest.approx(1.0)
    assert res["strategy_ask_spread_bps"] == pytest.approx(4.0)
    assert res["strategy_bid_size"] == pytest.approx(2.0)
    assert res["strategy_ask_size"] == pytest.approx(1.0)


def test_pull_one_side_buy_on_negative_signed_volume():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=2.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.0,
        pull_spread_add_bps=3.0,
        pull_size_factor=0.5,
        pull_mode="one_side",
    )
    state = {"mid": 100.0, "position": 0.0, "market_spread_bps": 1.0, "signed_volume": -10.0}
    res = decide_orders(state, params)
    assert res["pull_triggered"] is True
    assert res["pull_side"] == "buy"
    assert res["strategy_bid_spread_bps"] == pytest.approx(4.0)
    assert res["strategy_ask_spread_bps"] == pytest.approx(1.0)
    assert res["strategy_bid_size"] == pytest.approx(1.0)
    assert res["strategy_ask_size"] == pytest.approx(2.0)


def test_pull_one_side_falls_back_to_symmetric_on_zero_flow():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=2.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.0,
        pull_spread_add_bps=3.0,
        pull_size_factor=0.5,
        pull_mode="one_side",
    )
    state = {"mid": 100.0, "position": 0.0, "market_spread_bps": 1.0, "signed_volume": 0.0}
    res = decide_orders(state, params)
    assert res["pull_triggered"] is True
    assert res["pull_side"] is None
    assert res["strategy_bid_spread_bps"] == pytest.approx(4.0)
    assert res["strategy_ask_spread_bps"] == pytest.approx(4.0)
    assert res["strategy_bid_size"] == pytest.approx(1.0)
    assert res["strategy_ask_size"] == pytest.approx(1.0)

