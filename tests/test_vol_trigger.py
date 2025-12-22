import pytest

from hlmm.mm import StrategyParams, decide_orders


def test_pull_triggers_by_abs_mid_ret():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=1.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        pull_when_abs_mid_ret_gt=0.01,
        pull_spread_add_bps=2.0,
        pull_size_factor=1.0,
        pull_mode="symmetric",
    )
    res = decide_orders(
        {"mid": 100.0, "position": 0.0, "market_spread_bps": 0.0, "abs_mid_ret": 0.02, "signed_volume": 0.0}, params
    )
    assert res["pull_triggered"] is True
    assert res["strategy_bid_spread_bps"] == pytest.approx(3.0)
    assert res["strategy_ask_spread_bps"] == pytest.approx(3.0)


def test_stop_triggers_by_abs_mid_ret():
    params = StrategyParams(stop_when_abs_mid_ret_gt=0.01, stop_mode="halt")
    res = decide_orders({"mid": 100.0, "position": 0.0, "abs_mid_ret": 0.02}, params)
    assert res["halt"] is True
    assert res["stop_triggered"] is True
    assert res["orders"] == []

