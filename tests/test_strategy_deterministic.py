from hlmm.mm import StrategyParams, decide_orders


def test_strategy_deterministic():
    params = StrategyParams()
    state = {"mid": 100.0, "position": 0.0}
    out1 = decide_orders(state, params)
    out2 = decide_orders(state, params)
    assert out1["halt"] is False
    assert out1["bid_px"] == out2["bid_px"]
    assert out1["ask_px"] == out2["ask_px"]
    assert len(out1["orders"]) == 2
