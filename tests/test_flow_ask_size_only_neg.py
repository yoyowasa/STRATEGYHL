import pytest

from hlmm.mm import StrategyParams, decide_orders


def test_ask_size_is_reduced_only_on_negative_extreme_sv():
    params = StrategyParams(
        base_spread_bps=1.0,
        base_size=2.0,
        inventory_skew_bps=0.0,
        inventory_target=0.0,
        pull_when_abs_signed_volume_gt=10.0,  # thr（sv<=-10 で発火）
        pull_signed_volume_window_s=2.0,
        ask_size_factor_when_sv_neg=0.5,
    )

    # sv が負方向に極端 → ASKだけサイズ半分
    res_neg = decide_orders(
        {"mid": 100.0, "position": 0.0, "signed_volume_window": -11.0, "signed_volume": 0.0},
        params,
    )
    assert res_neg["halt"] is False
    assert res_neg["ask_sv_neg_triggered"] is True
    assert res_neg["strategy_bid_size"] == pytest.approx(2.0)
    assert res_neg["strategy_ask_size"] == pytest.approx(1.0)

    # sv が正方向に極端 → 何もしない（ASKは落とさない）
    res_pos = decide_orders(
        {"mid": 100.0, "position": 0.0, "signed_volume_window": 11.0, "signed_volume": 0.0},
        params,
    )
    assert res_pos["halt"] is False
    assert res_pos["ask_sv_neg_triggered"] is False
    assert res_pos["strategy_bid_size"] == pytest.approx(2.0)
    assert res_pos["strategy_ask_size"] == pytest.approx(2.0)

