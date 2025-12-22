from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pyarrow.parquet as pq


def _safe_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mid_from_top(top: dict) -> float | None:
    bid = _safe_float(top.get("bid_px"))
    ask = _safe_float(top.get("ask_px"))
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _signed_volume(trade_bucket: list) -> float:
    total = 0.0
    for tr in trade_bucket:
        if not isinstance(tr, dict):
            continue
        side = tr.get("side")
        sz = _safe_float(tr.get("sz"))
        if sz is None or sz <= 0:
            continue
        if side == "buy":
            total += sz
        elif side == "sell":
            total -= sz
    return total


def _cond_mean(x: np.ndarray, cond: np.ndarray) -> Tuple[float, int]:
    mask = (~np.isnan(x)) & cond
    if not np.any(mask):
        return float("nan"), 0
    return float(np.nanmean(x[mask])), int(np.sum(mask))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = (~np.isnan(x)) & (~np.isnan(y))
    if int(np.sum(ok)) < 3:
        return float("nan")
    xs = x[ok]
    ys = y[ok]
    if float(np.nanstd(xs)) == 0.0 or float(np.nanstd(ys)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _quantiles(abs_vals: np.ndarray, qs: List[float]) -> List[Tuple[float, float]]:
    ok = ~np.isnan(abs_vals)
    v = abs_vals[ok]
    if v.size == 0:
        return [(q, float("nan")) for q in qs]
    return [(q, float(np.quantile(v, q))) for q in qs]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="signed_volume の符号が「次の値動き」と整合しているかを確認する（SV>0 が上圧か？）"
    )
    parser.add_argument("--blocks", required=True, help="入力 blocks.parquet（例: data/blocks_long_*.parquet）")
    parser.add_argument(
        "--window-s",
        type=float,
        default=2.0,
        help="signed_volume のローリング窓（秒）。SV_window を作るのに使う（デフォルト: 2.0）",
    )
    parser.add_argument(
        "--quantiles",
        default="0.5,0.8,0.9,0.95",
        help="|SV| 分位点（カンマ区切り）。閾値選定の目安（デフォルト: 0.5,0.8,0.9,0.95）",
    )
    args = parser.parse_args()

    blocks_path = Path(args.blocks)
    table = pq.read_table(blocks_path)
    blocks = table.to_pylist()
    blocks.sort(key=lambda b: (b.get("block_ts_ms") or 0, str(b.get("book_event_id") or "")))

    try:
        window_ms = int(float(args.window_s) * 1000.0)
    except (TypeError, ValueError):
        window_ms = 0
    if window_ms <= 0:
        raise SystemExit("--window-s は正の数である必要があります")

    qs = []
    for part in str(args.quantiles).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            qs.append(float(part))
        except ValueError:
            continue
    if not qs:
        qs = [0.5, 0.8, 0.9, 0.95]

    mids: List[float] = []
    sv_block: List[float] = []
    sv_win: List[float] = []

    q = deque()
    win_sum = 0.0

    for b in blocks:
        ts = b.get("block_ts_ms")
        try:
            ts_i = int(ts) if ts is not None else 0
        except (TypeError, ValueError):
            ts_i = 0

        top = b.get("book_top") or {}
        mid = _mid_from_top(top) if isinstance(top, dict) else None
        mids.append(float(mid) if mid is not None else float("nan"))

        bucket = b.get("trade_bucket") or []
        sv = _signed_volume(bucket) if isinstance(bucket, list) else 0.0
        sv_block.append(float(sv))

        q.append((ts_i, sv))
        win_sum += sv
        cutoff = ts_i - window_ms
        while q and q[0][0] < cutoff:
            _, old = q.popleft()
            win_sum -= old
        sv_win.append(float(win_sum))

    mid_arr = np.array(mids, dtype=float)
    sv_block_arr = np.array(sv_block, dtype=float)
    sv_win_arr = np.array(sv_win, dtype=float)

    # ret_now: mid_t / mid_{t-1} - 1
    ret_now = np.full_like(mid_arr, np.nan)
    for i in range(1, int(mid_arr.size)):
        if not math.isnan(mid_arr[i]) and not math.isnan(mid_arr[i - 1]) and mid_arr[i - 1] != 0:
            ret_now[i] = mid_arr[i] / mid_arr[i - 1] - 1.0

    # fwd_ret: mid_{t+1} / mid_t - 1
    fwd_ret = np.full_like(mid_arr, np.nan)
    for i in range(0, int(mid_arr.size) - 1):
        if not math.isnan(mid_arr[i]) and not math.isnan(mid_arr[i + 1]) and mid_arr[i] != 0:
            fwd_ret[i] = mid_arr[i + 1] / mid_arr[i] - 1.0

    for name, sv_arr in [("sv_block", sv_block_arr), ("sv_window", sv_win_arr)]:
        pos = sv_arr > 0
        neg = sv_arr < 0
        m_now_pos, n_now_pos = _cond_mean(ret_now, pos)
        m_now_neg, n_now_neg = _cond_mean(ret_now, neg)
        m_fwd_pos, n_fwd_pos = _cond_mean(fwd_ret, pos)
        m_fwd_neg, n_fwd_neg = _cond_mean(fwd_ret, neg)

        print(f"=== {name} ===")
        print(f"E[ret_now | sv>0] = {m_now_pos}  n={n_now_pos}")
        print(f"E[ret_now | sv<0] = {m_now_neg}  n={n_now_neg}")
        print(f"E[fwd_ret | sv>0] = {m_fwd_pos}  n={n_fwd_pos}")
        print(f"E[fwd_ret | sv<0] = {m_fwd_neg}  n={n_fwd_neg}")
        print(f"corr(sv, ret_now) = {_corr(sv_arr, ret_now)}")
        print(f"corr(sv, fwd_ret) = {_corr(sv_arr, fwd_ret)}")

        abs_sv = np.abs(sv_arr)
        print("|SV| quantiles:")
        for qv, thr in _quantiles(abs_sv, qs):
            print(f"  q={qv}: {thr}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

