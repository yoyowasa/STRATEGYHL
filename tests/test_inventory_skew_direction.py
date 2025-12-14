from hlmm.mm import StrategyParams, decide_orders


def test_inventory_skew_direction():
    params = StrategyParams(base_spread_bps=0.0, inventory_skew_bps=10.0, inventory_target=0.0)
    state_long = {"mid": 100.0, "position": 1.0}
    state_short = {"mid": 100.0, "position": -1.0}

    out_long = decide_orders(state_long, params)
    out_short = decide_orders(state_short, params)

    # ロングならbidを下げ、ショートならaskを上げる方向に歪む
    assert out_long["bid_px"] < out_short["bid_px"]
    assert out_short["ask_px"] > out_long["ask_px"]
