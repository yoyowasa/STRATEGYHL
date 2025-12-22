from hlmm.mm import StrategyParams, decide_orders


def test_stop_triggers_by_market_spread():
    params = StrategyParams(
        base_size=0.5,
        inventory_target=0.0,
        stop_when_market_spread_bps_gt=0.3,
        stop_mode="halt",
    )

    out = decide_orders({"mid": 100.0, "position": 0.0, "market_spread_bps": 0.31}, params)
    assert out["halt"] is True
    assert out["stop_triggered"] is True
    assert out["orders"] == []


def test_stop_triggers_by_market_spread_lt():
    params = StrategyParams(
        base_size=0.5,
        inventory_target=0.0,
        stop_when_market_spread_bps_lt=0.3,
        stop_mode="halt",
    )

    out = decide_orders({"mid": 100.0, "position": 0.0, "market_spread_bps": 0.29}, params)
    assert out["halt"] is True
    assert out["stop_triggered"] is True
    assert out["orders"] == []
