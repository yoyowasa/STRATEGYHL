from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from .sim import Order


@dataclass(frozen=True)
class StrategyParams:
    base_spread_bps: float = 5.0
    base_size: float = 1.0
    inventory_skew_bps: float = 2.0
    inventory_target: float = 0.0
    max_abs_position: Optional[float] = None
    # stop（baseline + stop）
    stop_max_abs_position: Optional[float] = None
    stop_max_intraday_drawdown_usdc: Optional[float] = None
    stop_when_market_spread_bps_gt: Optional[float] = None
    stop_when_market_spread_bps_lt: Optional[float] = None
    stop_when_abs_mid_ret_gt: Optional[float] = None
    stop_mode: str = "halt"  # "halt" | "unwind_only"
    # halt（two-sided halt）
    halt_when_market_spread_bps_gt: Optional[float] = None
    halt_when_market_spread_bps_lt: Optional[float] = None
    halt_when_abs_mid_ret_gt: Optional[float] = None
    halt_size_factor: float = 0.0
    # boost（risk-on）
    boost_when_abs_mid_ret_gt: Optional[float] = None
    boost_size_factor: float = 1.0
    boost_spread_add_bps: float = 0.0
    boost_only_if_abs_pos_lt: Optional[float] = None
    quote_only_in_boost: bool = False
    # pull（baseline + pull）
    pull_when_market_spread_bps_gt: Optional[float] = None
    pull_when_market_spread_bps_lt: Optional[float] = None
    pull_when_abs_mid_ret_gt: Optional[float] = None
    # orderflow（符号付き出来高）で pull window を作る
    # - signed_volume は「ブロック内」の集計（buy:+ / sell:-）
    # - signed_volume_window は「直近 window 秒」のローリング合計（sim 側で計算）
    pull_when_abs_signed_volume_gt: Optional[float] = None
    pull_signed_volume_window_s: Optional[float] = None
    pull_spread_add_bps: float = 0.0
    pull_size_factor: float = 1.0
    pull_mode: str = "symmetric"  # "symmetric" | "one_side"
    # SVが「負方向に極端」なときだけ ASK 側のサイズを落とす（価格は触らない）
    # 条件は sv_window <= -thr（thrは pull_when_abs_signed_volume_gt を使う）
    ask_size_factor_when_sv_neg: Optional[float] = None
    # pull window 中だけ在庫を壊さないためのガード
    # - pull_window_max_abs_position: inventory_target からの偏り(|pos-target|)上限（window中のみ適用）
    # - pull_window_inventory_skew_mult: inventory_skew_bps の倍率（window中のみ適用）
    pull_window_max_abs_position: Optional[float] = None
    pull_window_inventory_skew_mult: float = 1.0
    # pull window を抜けた後、在庫を素早く target へ戻す（outside の directional/unrealized 崩壊を抑える）
    post_pull_unwind_enable: bool = False
    post_pull_unwind_until_abs_pos_lt: Optional[float] = None
    post_pull_inventory_skew_mult: float = 1.0
    # post-pull unwind 中の「刺さりやすさ」調整（在庫を減らす側だけタイトにする等）
    # - spread_add_bps は unwind 側の片側にだけ加算する（負ならタイト化）
    # - size_factor は unwind 側の片側サイズにだけ乗算する
    # - other_side_size_factor は反対側サイズにだけ乗算する（0で反対側を止められる）
    post_pull_unwind_spread_add_bps: float = 0.0
    post_pull_unwind_size_factor: float = 1.0
    post_pull_unwind_other_side_size_factor: float = 1.0
    # micro_bias_bps が閾値を超えた側だけ size-only で縮小
    micro_bias_thr_bps: Optional[float] = None
    micro_bias_size_factor: float = 1.0
    # micro_bias_bps が正側閾値を超えたときだけ ASK サイズを落とす
    micro_bias_thr_pos_bps: Optional[float] = None
    micro_bias_ask_only_size_factor: float = 1.0
    # imbalance（L1サイズ偏り）で危ない側だけ size-only
    imbalance_thr: Optional[float] = None
    imbalance_size_factor: float = 1.0
    # micro_pos（スプレッド内位置）で危ない側だけ size-only
    micro_pos_thr: Optional[float] = None
    micro_pos_size_factor: float = 1.0
    strategy_variant: int = 1  # 1: baseline+inventory, 2/3: 拡張用


def _clamp_position(size: float, params: StrategyParams) -> float:
    if params.max_abs_position is None:
        return size
    if abs(size) > params.max_abs_position:
        return params.max_abs_position if size > 0 else -params.max_abs_position
    return size


def decide_orders(state: Mapping[str, object], params: StrategyParams) -> Dict[str, object]:
    """
    Strategy1: ベーススプレッドに inventory skew を加味した bid/ask を返す。
    state: mid/spread/imbalance/micro_pos/signed_volume/basis_bps/inventory を含む辞書を想定。
    """
    mid = state.get("mid")
    position = float(state.get("position", 0.0) or 0.0)
    if mid is None:
        return {"orders": [], "halt": True, "post_pull_unwind_active": False}
    try:
        mid_f = float(mid)
    except (TypeError, ValueError):
        return {"orders": [], "halt": True, "post_pull_unwind_active": False}

    inv = position - params.inventory_target

    # halt 条件（two-sided）
    halt_triggered = False
    if (
        params.halt_when_market_spread_bps_gt is not None
        or params.halt_when_market_spread_bps_lt is not None
    ):
        ms = state.get("market_spread_bps")
        try:
            ms_f = float(ms) if ms is not None else None
        except (TypeError, ValueError):
            ms_f = None
        if ms_f is not None:
            if params.halt_when_market_spread_bps_gt is not None and ms_f > float(params.halt_when_market_spread_bps_gt):
                halt_triggered = True
            if params.halt_when_market_spread_bps_lt is not None and ms_f < float(params.halt_when_market_spread_bps_lt):
                halt_triggered = True
    if not halt_triggered and params.halt_when_abs_mid_ret_gt is not None:
        amr = state.get("abs_mid_ret")
        try:
            amr_f = float(amr) if amr is not None else None
        except (TypeError, ValueError):
            amr_f = None
        if amr_f is not None and amr_f > float(params.halt_when_abs_mid_ret_gt):
            halt_triggered = True

    # boost 条件（risk-on）
    boost_triggered = False
    if params.boost_when_abs_mid_ret_gt is not None:
        amr = state.get("abs_mid_ret")
        try:
            amr_f = float(amr) if amr is not None else None
        except (TypeError, ValueError):
            amr_f = None
        if amr_f is not None and amr_f > float(params.boost_when_abs_mid_ret_gt):
            boost_triggered = True
    if boost_triggered and params.boost_only_if_abs_pos_lt is not None:
        try:
            cap = float(params.boost_only_if_abs_pos_lt)
        except (TypeError, ValueError):
            cap = None
        if cap is not None and cap >= 0 and abs(inv) >= cap:
            boost_triggered = False

    # stop 条件
    stop_triggered = False
    if params.stop_max_abs_position is not None and abs(position) >= float(params.stop_max_abs_position):
        stop_triggered = True
    if (
        not stop_triggered
        and (params.stop_when_market_spread_bps_gt is not None or params.stop_when_market_spread_bps_lt is not None)
    ):
        ms = state.get("market_spread_bps")
        try:
            ms_f = float(ms) if ms is not None else None
        except (TypeError, ValueError):
            ms_f = None
        if ms_f is not None:
            if params.stop_when_market_spread_bps_gt is not None and ms_f > float(params.stop_when_market_spread_bps_gt):
                stop_triggered = True
            if params.stop_when_market_spread_bps_lt is not None and ms_f < float(params.stop_when_market_spread_bps_lt):
                stop_triggered = True
    if not stop_triggered and params.stop_when_abs_mid_ret_gt is not None:
        amr = state.get("abs_mid_ret")
        try:
            amr_f = float(amr) if amr is not None else None
        except (TypeError, ValueError):
            amr_f = None
        if amr_f is not None and amr_f > float(params.stop_when_abs_mid_ret_gt):
            stop_triggered = True
    if not stop_triggered and params.stop_max_intraday_drawdown_usdc is not None:
        dd = state.get("drawdown")
        try:
            dd_f = float(dd) if dd is not None else None
        except (TypeError, ValueError):
            dd_f = None
        if dd_f is not None and dd_f <= float(params.stop_max_intraday_drawdown_usdc):
            stop_triggered = True

    if stop_triggered:
        # 最小仕様: "halt" は新規注文を止めるだけ、"unwind_only" は在庫を減らす側だけ出す
        if str(params.stop_mode).lower() == "unwind_only":
            inv = position - params.inventory_target
            if abs(inv) <= 1e-12:
                return {
                    "orders": [],
                    "halt": True,
                    "stop_triggered": True,
                    "pull_triggered": False,
                    "post_pull_unwind_active": False,
                    "halt_triggered": halt_triggered,
                    "boost_triggered": boost_triggered,
                }

            # 在庫を減らす方向にのみ注文（ロングなら sell、ショートなら buy）
            side = "sell" if inv > 0 else "buy"
            size = float(params.base_size)
            size = min(size, abs(inv))
            size = max(0.0, size)
            size = _clamp_position(size, params)
            if size <= 0:
                return {
                    "orders": [],
                    "halt": True,
                    "stop_triggered": True,
                    "pull_triggered": False,
                    "post_pull_unwind_active": False,
                    "halt_triggered": halt_triggered,
                    "boost_triggered": boost_triggered,
                }

            return {
                "orders": [Order(side=side, size=size, price=mid_f, post_only=True)],
                "halt": False,
                "stop_triggered": True,
                "pull_triggered": False,
                "post_pull_unwind_active": False,
                "halt_triggered": halt_triggered,
                "boost_triggered": boost_triggered,
                "strategy_spread_bps": None,
                "strategy_size": size,
            }
        return {
            "orders": [],
            "halt": True,
            "stop_triggered": True,
            "pull_triggered": False,
            "post_pull_unwind_active": False,
            "halt_triggered": halt_triggered,
            "boost_triggered": boost_triggered,
        }

    if params.quote_only_in_boost and not boost_triggered:
        return {
            "orders": [],
            "halt": True,
            "stop_triggered": False,
            "pull_triggered": False,
            "post_pull_unwind_active": False,
            "halt_triggered": halt_triggered,
            "boost_triggered": boost_triggered,
        }

    # スプレッド/サイズ（pullで変える）
    base_spread_bps = float(params.base_spread_bps)
    base_size = float(params.base_size)
    bid_spread_bps = base_spread_bps
    ask_spread_bps = base_spread_bps
    bid_size = base_size
    ask_size = base_size
    pull_triggered = False
    pull_side: Optional[str] = None
    if (
        params.pull_when_market_spread_bps_gt is not None
        or params.pull_when_market_spread_bps_lt is not None
        or params.pull_when_abs_mid_ret_gt is not None
        or params.pull_when_abs_signed_volume_gt is not None
    ):
        ms = state.get("market_spread_bps")
        try:
            ms_f = float(ms) if ms is not None else None
        except (TypeError, ValueError):
            ms_f = None
        cond = False
        if ms_f is not None:
            if params.pull_when_market_spread_bps_gt is not None and ms_f > float(params.pull_when_market_spread_bps_gt):
                cond = True
            if params.pull_when_market_spread_bps_lt is not None and ms_f < float(params.pull_when_market_spread_bps_lt):
                cond = True
        if not cond and params.pull_when_abs_mid_ret_gt is not None:
            amr = state.get("abs_mid_ret")
            try:
                amr_f = float(amr) if amr is not None else None
            except (TypeError, ValueError):
                amr_f = None
            if amr_f is not None and amr_f > float(params.pull_when_abs_mid_ret_gt):
                cond = True
        if not cond and params.pull_when_abs_signed_volume_gt is not None:
            # signed_volume_window がある場合は優先して使う（window_s を指定した時のみ）
            sv_key = (
                "signed_volume_window"
                if params.pull_signed_volume_window_s is not None and state.get("signed_volume_window") is not None
                else "signed_volume"
            )
            sv = state.get(sv_key)
            try:
                sv_f = float(sv) if sv is not None else 0.0
            except (TypeError, ValueError):
                sv_f = 0.0
            if abs(sv_f) > float(params.pull_when_abs_signed_volume_gt):
                cond = True
        if cond:
            pull_triggered = True
            mode = str(params.pull_mode).lower().strip()
            if mode == "one_side":
                sv_key = (
                    "signed_volume_window"
                    if params.pull_signed_volume_window_s is not None and state.get("signed_volume_window") is not None
                    else "signed_volume"
                )
                sv = state.get(sv_key)
                try:
                    sv_f = float(sv) if sv is not None else 0.0
                except (TypeError, ValueError):
                    sv_f = 0.0
                if sv_f > 0:
                    pull_side = "sell"
                elif sv_f < 0:
                    pull_side = "buy"

                if pull_side == "sell":
                    ask_spread_bps = ask_spread_bps + float(params.pull_spread_add_bps)
                    ask_size = ask_size * float(params.pull_size_factor)
                elif pull_side == "buy":
                    bid_spread_bps = bid_spread_bps + float(params.pull_spread_add_bps)
                    bid_size = bid_size * float(params.pull_size_factor)
                else:
                    # 判定できない場合は対称pullへフォールバック
                    bid_spread_bps = bid_spread_bps + float(params.pull_spread_add_bps)
                    ask_spread_bps = ask_spread_bps + float(params.pull_spread_add_bps)
                    bid_size = bid_size * float(params.pull_size_factor)
                    ask_size = ask_size * float(params.pull_size_factor)
            else:
                # symmetric（両側同時）
                bid_spread_bps = bid_spread_bps + float(params.pull_spread_add_bps)
                ask_spread_bps = ask_spread_bps + float(params.pull_spread_add_bps)
                bid_size = bid_size * float(params.pull_size_factor)
                ask_size = ask_size * float(params.pull_size_factor)

            bid_size = max(0.0, float(bid_size))
            ask_size = max(0.0, float(ask_size))

    # 「悪い組だけ回避」: sv が負方向に極端なときだけ ASK サイズを落とす（BIDは触らない）
    ask_sv_neg_triggered = False
    if params.ask_size_factor_when_sv_neg is not None and params.pull_when_abs_signed_volume_gt is not None:
        try:
            thr = float(params.pull_when_abs_signed_volume_gt)
        except (TypeError, ValueError):
            thr = 0.0
        if thr > 0:
            sv_key = (
                "signed_volume_window"
                if params.pull_signed_volume_window_s is not None and state.get("signed_volume_window") is not None
                else "signed_volume"
            )
            sv = state.get(sv_key)
            try:
                sv_f = float(sv) if sv is not None else 0.0
            except (TypeError, ValueError):
                sv_f = 0.0
            if sv_f <= -thr:
                try:
                    f = float(params.ask_size_factor_when_sv_neg)
                except (TypeError, ValueError):
                    f = 1.0
                if f < 0:
                    f = 0.0
                ask_size = max(0.0, float(ask_size) * f)
                ask_sv_neg_triggered = True

    # micro_bias で ASK だけ size-only（>+thr_pos の時のみ）
    if params.micro_bias_thr_pos_bps is not None:
        try:
            thr = float(params.micro_bias_thr_pos_bps)
        except (TypeError, ValueError):
            thr = 0.0
        if thr > 0:
            mb = state.get("micro_bias_bps")
            try:
                mb_f = float(mb) if mb is not None else 0.0
            except (TypeError, ValueError):
                mb_f = 0.0
            try:
                f = float(params.micro_bias_ask_only_size_factor)
            except (TypeError, ValueError):
                f = 1.0
            if f < 0:
                f = 0.0
            if mb_f >= thr:
                ask_size = max(0.0, float(ask_size) * f)
    # micro_bias で危ない側だけ size-only（>+thr: ASK / <-thr: BID）
    elif params.micro_bias_thr_bps is not None:
        try:
            thr = float(params.micro_bias_thr_bps)
        except (TypeError, ValueError):
            thr = 0.0
        if thr > 0:
            mb = state.get("micro_bias_bps")
            try:
                mb_f = float(mb) if mb is not None else 0.0
            except (TypeError, ValueError):
                mb_f = 0.0
            try:
                f = float(params.micro_bias_size_factor)
            except (TypeError, ValueError):
                f = 1.0
            if f < 0:
                f = 0.0
            if mb_f >= thr:
                ask_size = max(0.0, float(ask_size) * f)
            elif mb_f <= -thr:
                bid_size = max(0.0, float(bid_size) * f)

    # imbalance で危ない側だけ size-only（>+thr: ASK / <-thr: BID）
    if params.imbalance_thr is not None:
        try:
            thr = float(params.imbalance_thr)
        except (TypeError, ValueError):
            thr = 0.0
        if thr > 0:
            imb = state.get("imbalance")
            try:
                imb_f = float(imb) if imb is not None else 0.0
            except (TypeError, ValueError):
                imb_f = 0.0
            try:
                f = float(params.imbalance_size_factor)
            except (TypeError, ValueError):
                f = 1.0
            if f < 0:
                f = 0.0
            if imb_f >= thr:
                ask_size = max(0.0, float(ask_size) * f)
            elif imb_f <= -thr:
                bid_size = max(0.0, float(bid_size) * f)

    # micro_pos で危ない側だけ size-only（>+thr: ASK / <-thr: BID）
    if params.micro_pos_thr is not None:
        try:
            thr = float(params.micro_pos_thr)
        except (TypeError, ValueError):
            thr = 0.0
        if thr > 0:
            mp = state.get("micro_pos")
            try:
                mp_f = float(mp) if mp is not None else 0.0
            except (TypeError, ValueError):
                mp_f = 0.0
            try:
                f = float(params.micro_pos_size_factor)
            except (TypeError, ValueError):
                f = 1.0
            if f < 0:
                f = 0.0
            if mp_f >= thr:
                ask_size = max(0.0, float(ask_size) * f)
            elif mp_f <= -thr:
                bid_size = max(0.0, float(bid_size) * f)

    # inventory skew: mid を在庫方向にシフトし、bid/ask は対称に張る
    if pull_triggered and params.pull_window_max_abs_position is not None:
        cap = float(params.pull_window_max_abs_position)
        # 1回の約定で cap を超えないよう、サイズをクランプする
        bid_size = min(float(bid_size), max(0.0, cap - inv))
        ask_size = min(float(ask_size), max(0.0, cap + inv))

    # pull window を抜けた後の「在庫戻し」モード（状態は外側が保持する）
    post_pull_unwind_active = False
    if not pull_triggered and bool(params.post_pull_unwind_enable):
        prev_pull = bool(state.get("prev_pull_triggered", False))
        active_prev = bool(state.get("post_pull_unwind_active", False))
        thr = params.post_pull_unwind_until_abs_pos_lt
        try:
            thr_f = float(thr) if thr is not None else 0.0
        except (TypeError, ValueError):
            thr_f = 0.0
        if (active_prev or prev_pull) and abs(inv) > thr_f:
            post_pull_unwind_active = True

    # post-pull unwind 中の片側クオート調整（約定を出して在庫を戻す）
    if post_pull_unwind_active and inv != 0:
        try:
            spread_add = float(params.post_pull_unwind_spread_add_bps)
        except (TypeError, ValueError):
            spread_add = 0.0
        try:
            size_factor = float(params.post_pull_unwind_size_factor)
        except (TypeError, ValueError):
            size_factor = 1.0
        try:
            other_size_factor = float(params.post_pull_unwind_other_side_size_factor)
        except (TypeError, ValueError):
            other_size_factor = 1.0

        # 変な値を防ぐ（負のサイズは0扱い）
        if size_factor < 0:
            size_factor = 0.0
        if other_size_factor < 0:
            other_size_factor = 0.0

        if inv > 0:
            # ロング在庫を減らす（ask側をタイトに / bid側を抑える）
            ask_spread_bps = max(0.0, float(ask_spread_bps) + spread_add)
            ask_size = max(0.0, float(ask_size) * size_factor)
            bid_size = max(0.0, float(bid_size) * other_size_factor)
        else:
            # ショート在庫を減らす（bid側をタイトに / ask側を抑える）
            bid_spread_bps = max(0.0, float(bid_spread_bps) + spread_add)
            bid_size = max(0.0, float(bid_size) * size_factor)
            ask_size = max(0.0, float(ask_size) * other_size_factor)

    if boost_triggered:
        try:
            boost_spread_add = float(params.boost_spread_add_bps)
        except (TypeError, ValueError):
            boost_spread_add = 0.0
        bid_spread_bps = max(0.0, float(bid_spread_bps) + boost_spread_add)
        ask_spread_bps = max(0.0, float(ask_spread_bps) + boost_spread_add)

    skew_mult = 1.0
    if pull_triggered:
        try:
            skew_mult = float(params.pull_window_inventory_skew_mult)
        except (TypeError, ValueError):
            skew_mult = 1.0
        # 0 は「skew無効」として許容する（極端テスト/切り分け用）
        if skew_mult < 0:
            skew_mult = 1.0
    elif post_pull_unwind_active:
        try:
            skew_mult = float(params.post_pull_inventory_skew_mult)
        except (TypeError, ValueError):
            skew_mult = 1.0
        if skew_mult < 0:
            skew_mult = 1.0

    skew_bps = inv * params.inventory_skew_bps * skew_mult
    mid_adj = mid_f * (1 - skew_bps / 10_000)
    bid_px = mid_adj * (1 - bid_spread_bps / 10_000)
    ask_px = mid_adj * (1 + ask_spread_bps / 10_000)

    bid_size = _clamp_position(float(bid_size), params)
    ask_size = _clamp_position(float(ask_size), params)

    if boost_triggered:
        try:
            boost_factor = float(params.boost_size_factor)
        except (TypeError, ValueError):
            boost_factor = 1.0
        if boost_factor < 0:
            boost_factor = 0.0
        bid_size = max(0.0, float(bid_size) * boost_factor)
        ask_size = max(0.0, float(ask_size) * boost_factor)
        bid_size = _clamp_position(float(bid_size), params)
        ask_size = _clamp_position(float(ask_size), params)

    if halt_triggered:
        try:
            halt_factor = float(params.halt_size_factor)
        except (TypeError, ValueError):
            halt_factor = 0.0
        if halt_factor < 0:
            halt_factor = 0.0
        bid_size = max(0.0, float(bid_size) * halt_factor)
        ask_size = max(0.0, float(ask_size) * halt_factor)
        bid_size = _clamp_position(float(bid_size), params)
        ask_size = _clamp_position(float(ask_size), params)
        if bid_size <= 0 and ask_size <= 0:
            return {
                "orders": [],
                "halt": True,
                "bid_px": bid_px,
                "ask_px": ask_px,
                "skew": skew_bps,
                "stop_triggered": False,
                "pull_triggered": pull_triggered,
                "post_pull_unwind_active": post_pull_unwind_active,
                "pull_side": pull_side,
                "ask_sv_neg_triggered": ask_sv_neg_triggered,
                "strategy_bid_spread_bps": bid_spread_bps,
                "strategy_ask_spread_bps": ask_spread_bps,
                "strategy_bid_size": bid_size,
                "strategy_ask_size": ask_size,
                "strategy_spread_bps": max(bid_spread_bps, ask_spread_bps),
                "strategy_size": max(bid_size, ask_size),
                "halt_triggered": True,
                "boost_triggered": boost_triggered,
            }

    orders = [
        Order(side="buy", size=bid_size, price=bid_px, post_only=True),
        Order(side="sell", size=ask_size, price=ask_px, post_only=True),
    ]
    return {
        "orders": orders,
        "halt": False,
        "bid_px": bid_px,
        "ask_px": ask_px,
        "skew": skew_bps,
        "stop_triggered": False,
        "pull_triggered": pull_triggered,
        "post_pull_unwind_active": post_pull_unwind_active,
        "pull_side": pull_side,
        "ask_sv_neg_triggered": ask_sv_neg_triggered,
        "strategy_bid_spread_bps": bid_spread_bps,
        "strategy_ask_spread_bps": ask_spread_bps,
        "strategy_bid_size": bid_size,
        "strategy_ask_size": ask_size,
        "strategy_spread_bps": max(bid_spread_bps, ask_spread_bps),
        "strategy_size": max(bid_size, ask_size),
        "halt_triggered": halt_triggered,
        "boost_triggered": boost_triggered,
    }
