import pytest

from hlmm.mm import StrategyParams, decide_orders


def test_post_pull_unwind_activates_after_leaving_pull_window():
    params = StrategyParams(
        base_spread_bps=0.0,
        base_size=1.0,
        inventory_skew_bps=10.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.5,
        pull_spread_add_bps=0.0,
        pull_size_factor=1.0,
        pull_mode="symmetric",
        post_pull_unwind_enable=True,
        post_pull_unwind_until_abs_pos_lt=0.05,
        post_pull_inventory_skew_mult=3.0,
    )

    # pull window から抜けた直後（prev_pull_triggered=True, current pull_triggered=False）
    state = {
        "mid": 100.0,
        "position": 0.2,  # ロング
        "market_spread_bps": 0.0,  # pull条件を満たさない
        "prev_pull_triggered": True,
        "post_pull_unwind_active": False,
    }
    out = decide_orders(state, params)
    assert out["pull_triggered"] is False
    assert out["post_pull_unwind_active"] is True

    # skew が強くなり、ロングの unwind を促す方向（価格が下に寄る）
    base = StrategyParams(**{**params.__dict__, "post_pull_unwind_enable": False})
    out_base = decide_orders({**state, "prev_pull_triggered": True}, base)
    assert out["ask_px"] < out_base["ask_px"]
    assert out["bid_px"] < out_base["bid_px"]


def test_post_pull_unwind_can_tighten_unwind_side_only():
    params = StrategyParams(
        base_spread_bps=5.0,
        base_size=1.0,
        inventory_skew_bps=0.0,  # 価格シフトは入れず、片側spread調整だけを見る
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.5,
        post_pull_unwind_enable=True,
        post_pull_unwind_until_abs_pos_lt=0.05,
        post_pull_unwind_spread_add_bps=-2.0,  # unwind側をタイト化
        post_pull_unwind_size_factor=2.0,
        post_pull_unwind_other_side_size_factor=0.5,
    )

    state = {
        "mid": 100.0,
        "position": 0.2,  # ロング（=売り側をタイトにしたい）
        "market_spread_bps": 0.0,
        "prev_pull_triggered": True,
        "post_pull_unwind_active": False,
    }
    out = decide_orders(state, params)
    assert out["post_pull_unwind_active"] is True

    base = StrategyParams(
        **{
            **params.__dict__,
            "post_pull_unwind_spread_add_bps": 0.0,
            "post_pull_unwind_size_factor": 1.0,
            "post_pull_unwind_other_side_size_factor": 1.0,
        }
    )
    out_base = decide_orders(state, base)

    # askだけタイトになり、bidは変わらない
    assert out["ask_px"] < out_base["ask_px"]
    assert out["bid_px"] == pytest.approx(out_base["bid_px"])
    # サイズも片側だけ調整される
    assert out["strategy_ask_size"] == pytest.approx(2.0)
    assert out["strategy_bid_size"] == pytest.approx(0.5)


def test_post_pull_unwind_skew_mult_zero_disables_inventory_shift():
    base = StrategyParams(
        base_spread_bps=0.0,
        base_size=1.0,
        inventory_skew_bps=10.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.5,
        post_pull_unwind_enable=True,
        post_pull_unwind_until_abs_pos_lt=0.05,
        post_pull_inventory_skew_mult=1.0,
    )
    zero = StrategyParams(**{**base.__dict__, "post_pull_inventory_skew_mult": 0.0})

    state = {
        "mid": 100.0,
        "position": 1.0,
        "market_spread_bps": 0.0,
        "prev_pull_triggered": True,
        "post_pull_unwind_active": False,
    }
    out_base = decide_orders(state, base)
    out_zero = decide_orders(state, zero)

    assert out_base["post_pull_unwind_active"] is True
    assert out_zero["post_pull_unwind_active"] is True
    # mult=0 なら skew が消え、mid_adj=mid になる（spread=0なので bid/ask=mid）
    assert out_zero["bid_px"] == pytest.approx(100.0)
    assert out_zero["ask_px"] == pytest.approx(100.0)
    # base では skew が効くので 100 からズレる
    assert out_base["bid_px"] != pytest.approx(100.0)
    assert out_base["ask_px"] != pytest.approx(100.0)


def test_post_pull_unwind_stops_below_threshold():
    params = StrategyParams(
        base_spread_bps=0.0,
        base_size=1.0,
        inventory_skew_bps=10.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.5,
        post_pull_unwind_enable=True,
        post_pull_unwind_until_abs_pos_lt=0.05,
        post_pull_inventory_skew_mult=3.0,
    )
    # すでに active でも、pos が小さければ止まる
    out = decide_orders(
        {
            "mid": 100.0,
            "position": 0.01,
            "market_spread_bps": 0.0,
            "prev_pull_triggered": False,
            "post_pull_unwind_active": True,
        },
        params,
    )
    assert out["post_pull_unwind_active"] is False


def test_post_pull_unwind_persists_until_threshold():
    params = StrategyParams(
        base_spread_bps=0.0,
        base_size=1.0,
        inventory_skew_bps=10.0,
        inventory_target=0.0,
        pull_when_market_spread_bps_gt=0.5,
        post_pull_unwind_enable=True,
        post_pull_unwind_until_abs_pos_lt=0.05,
        post_pull_inventory_skew_mult=3.0,
    )
    out = decide_orders(
        {
            "mid": 100.0,
            "position": -0.2,
            "market_spread_bps": 0.0,
            "prev_pull_triggered": False,
            "post_pull_unwind_active": True,
        },
        params,
    )
    assert out["post_pull_unwind_active"] is True
    # ショートなら mid_adj が上がり、bid/ask が上に寄る（買い戻しを促す）
    assert out["bid_px"] > 100.0
    assert out["ask_px"] > 100.0
