from hlmm.mm import StrategyParams, decide_orders


def test_stop_unwind_only_reduces_inventory_side_only():
    params = StrategyParams(
        base_size=0.5,
        inventory_target=0.0,
        stop_max_abs_position=0.0,  # 強制発動
        stop_mode="unwind_only",
    )

    out_long = decide_orders({"mid": 100.0, "position": 1.5}, params)
    assert out_long["halt"] is False
    assert out_long["stop_triggered"] is True
    assert len(out_long["orders"]) == 1
    assert out_long["orders"][0].side == "sell"
    assert out_long["orders"][0].size == 0.5

    out_short = decide_orders({"mid": 100.0, "position": -0.3}, params)
    assert out_short["halt"] is False
    assert out_short["stop_triggered"] is True
    assert len(out_short["orders"]) == 1
    assert out_short["orders"][0].side == "buy"
    assert out_short["orders"][0].size == 0.3

