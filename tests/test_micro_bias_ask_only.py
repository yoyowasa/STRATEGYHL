import pytest

from hlmm.mm import StrategyParams, decide_orders


def test_micro_bias_positive_only_reduces_ask_size():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=2.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        micro_bias_thr_pos_bps=0.5,
        micro_bias_ask_only_size_factor=0.5,
    )

    res_pos = decide_orders({"mid": 100.0, "position": 0.0, "micro_bias_bps": 0.6}, params)
    assert res_pos["strategy_bid_size"] == pytest.approx(2.0)
    assert res_pos["strategy_ask_size"] == pytest.approx(1.0)

    res_neg = decide_orders({"mid": 100.0, "position": 0.0, "micro_bias_bps": -0.6}, params)
    assert res_neg["strategy_bid_size"] == pytest.approx(2.0)
    assert res_neg["strategy_ask_size"] == pytest.approx(2.0)
