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


def test_halt_triggers_by_abs_mid_ret():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=1.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        halt_when_abs_mid_ret_gt=0.01,
        halt_size_factor=0.0,
    )
    res = decide_orders(
        {"mid": 100.0, "position": 0.0, "market_spread_bps": 0.0, "abs_mid_ret": 0.02, "signed_volume": 0.0},
        params,
    )
    assert res["halt_triggered"] is True
    assert res["halt"] is True
    assert res["orders"] == []


def test_boost_triggers_by_abs_mid_ret():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=1.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        boost_when_abs_mid_ret_gt=0.01,
        boost_size_factor=1.5,
    )
    res = decide_orders(
        {"mid": 100.0, "position": 0.0, "market_spread_bps": 0.0, "abs_mid_ret": 0.02, "signed_volume": 0.0},
        params,
    )
    assert res["boost_triggered"] is True
    assert res["strategy_bid_size"] == pytest.approx(1.5)
    assert res["strategy_ask_size"] == pytest.approx(1.5)


def test_boost_guard_by_inventory():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=1.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        boost_when_abs_mid_ret_gt=0.01,
        boost_size_factor=1.5,
        boost_only_if_abs_pos_lt=0.3,
    )
    res = decide_orders({"mid": 100.0, "position": 0.5, "abs_mid_ret": 0.02}, params)
    assert res["boost_triggered"] is False
    assert res["strategy_bid_size"] == pytest.approx(1.0)
    assert res["strategy_ask_size"] == pytest.approx(1.0)
