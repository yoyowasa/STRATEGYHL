from dataclasses import replace

import pytest

from hlmm.mm import StrategyParams, decide_orders


def test_pull_window_max_abs_position_clamps_sizes_by_inventory():
    params = StrategyParams(
        base_spread_bps=0.0,
        base_size=0.4,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.0,  # 強制発動
        pull_spread_add_bps=0.0,
        pull_size_factor=1.0,
        pull_mode="symmetric",
        pull_window_max_abs_position=0.5,
    )
    out = decide_orders({"mid": 100.0, "position": 0.3, "market_spread_bps": 1.0, "signed_volume": 0.0}, params)
    assert out["pull_triggered"] is True
    assert out["strategy_bid_size"] == pytest.approx(0.2)  # cap-inv = 0.5-0.3
    assert out["strategy_ask_size"] == pytest.approx(0.4)  # cap+inv = 0.5+0.3


def test_pull_window_inventory_skew_mult_strengthens_skew_only_in_window():
    base = StrategyParams(
        base_spread_bps=0.0,
        base_size=1.0,
        inventory_skew_bps=10.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.0,  # 強制発動
        pull_spread_add_bps=0.0,
        pull_size_factor=1.0,
        pull_mode="symmetric",
    )
    boosted = replace(base, pull_window_inventory_skew_mult=2.0)

    state = {"mid": 100.0, "position": 1.0, "market_spread_bps": 1.0, "signed_volume": 0.0}
    out_base = decide_orders(state, base)
    out_boost = decide_orders(state, boosted)
    assert out_base["pull_triggered"] is True
    assert out_boost["pull_triggered"] is True
    assert out_boost["bid_px"] < out_base["bid_px"]
    assert out_boost["ask_px"] < out_base["ask_px"]
