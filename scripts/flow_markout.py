from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq


def _as_int(x: object) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _as_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pick_join_key(ledger, trades) -> str:
    names_l = set(ledger.schema.names)
    names_t = set(trades.schema.names)
    if "book_event_id" in names_l and "book_event_id" in names_t:
        return "book_event_id"
    if "block_ts_ms" in names_l and "block_ts_ms" in names_t:
        return "block_ts_ms"
    raise SystemExit("join key が見つかりません（book_event_id / block_ts_ms のどちらも揃っていません）")


def _build_mid_series(ledger) -> Tuple[np.ndarray, np.ndarray]:
    py = ledger.to_pydict()
    ts_raw = py.get("block_ts_ms", []) or []
    mid_raw = py.get("mark_price", []) or []
    pairs = []
    for ts, mid in zip(ts_raw, mid_raw):
        ts_i = _as_int(ts)
        mid_f = _as_float(mid)
        if ts_i is None or mid_f is None or math.isnan(mid_f):
            continue
        pairs.append((ts_i, mid_f))
    if not pairs:
        raise SystemExit("ledger から mid series（block_ts_ms/mark_price）を作れません")
    pairs.sort(key=lambda x: x[0])
    ts_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    mid_arr = np.array([p[1] for p in pairs], dtype=float)
    return ts_arr, mid_arr


def _markout_for_fill(ts_arr: np.ndarray, mid_arr: np.ndarray, ts_ms: int, side: str, horizon_ms: int) -> Optional[float]:
    idx_now = int(np.searchsorted(ts_arr, int(ts_ms), side="right") - 1)
    if idx_now < 0 or idx_now >= int(ts_arr.size):
        return None
    idx_fut = int(np.searchsorted(ts_arr, int(ts_ms) + int(horizon_ms), side="left"))
    if idx_fut < 0 or idx_fut >= int(ts_arr.size):
        return None
    mid_now = float(mid_arr[idx_now])
    mid_fut = float(mid_arr[idx_fut])
    if side == "buy":
        return float(mid_fut - mid_now)
    if side == "sell":
        return float(mid_now - mid_fut)
    return None


def _stats(xs: List[float]) -> Dict[str, object]:
    if not xs:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    arr = np.array(xs, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.nanmean(arr)),
        "p10": float(np.nanpercentile(arr, 10)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p90": float(np.nanpercentile(arr, 90)),
    }


def _fmt(x: object) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SV（signed_volume_window）条件別に、fill の markout を side ごとに集計する（逆選択の有無を1回で確認）"
    )
    parser.add_argument("--ledger", default=None, help="入力 ledger.parquet（未指定なら --outputs-dir/--run-id を使用）")
    parser.add_argument(
        "--trades", default=None, help="入力 sim_trades.parquet（未指定なら --outputs-dir/--run-id を使用）"
    )
    parser.add_argument("--outputs-dir", default="outputs", help="mm-sim 出力ディレクトリ（デフォルト: outputs/）")
    parser.add_argument("--run-id", default=None, help="run_id（例: flow_base__20251221_0737）")
    parser.add_argument("--prefix", default="mm_sim_", help="outputs_dir 配下の prefix（デフォルト: mm_sim_）")
    parser.add_argument("--horizon-ms", type=int, default=5000, help="markout のホライズン（ミリ秒）。デフォルト 5000")
    parser.add_argument(
        "--sv-col",
        default="signed_volume_window",
        help="ledger から参照する SV 列名（デフォルト signed_volume_window）",
    )
    thr_group = parser.add_mutually_exclusive_group(required=True)
    thr_group.add_argument("--thr", type=float, default=None, help="SV 閾値（例: 4.2382）。条件は sv>+thr / sv<-thr")
    thr_group.add_argument(
        "--thr-quantile",
        type=float,
        default=None,
        help="|SV| の分位点で閾値を作る（例: 0.95）。thr は ledger の |sv_col| から算出",
    )
    parser.add_argument("--format", choices=["md", "tsv"], default="md", help="出力形式")
    args = parser.parse_args()

    if args.ledger and args.trades:
        ledger_path = Path(str(args.ledger))
        trades_path = Path(str(args.trades))
    else:
        if not args.run_id:
            raise SystemExit("--ledger/--trades を指定しない場合、--run-id が必要です")
        out_dir = Path(args.outputs_dir) / f"{args.prefix}{args.run_id}"
        ledger_path = out_dir / "ledger.parquet"
        trades_path = out_dir / "sim_trades.parquet"

    ledger = pq.read_table(ledger_path)
    trades = pq.read_table(trades_path)

    if ledger.num_rows == 0 or trades.num_rows == 0:
        raise SystemExit("ledger または trades が空です（集計できません）")

    key_col = _pick_join_key(ledger, trades)
    if args.sv_col not in ledger.schema.names:
        raise SystemExit(f"ledger に {args.sv_col} がありません（列名を確認してください）")

    # SVマップ（fill時の条件判定に使う）
    led_py = ledger.to_pydict()
    led_keys = led_py.get(key_col, []) or []
    led_sv = led_py.get(args.sv_col, []) or []
    sv_by_key: Dict[object, float] = {}
    for k, v in zip(led_keys, led_sv):
        if k is None:
            continue
        sv_f = _as_float(v)
        if sv_f is None or math.isnan(sv_f):
            continue
        sv_by_key[k] = float(sv_f)
    if not sv_by_key:
        raise SystemExit("SVマップを作れません（sv_colが全て欠損の可能性）")

    # 閾値
    if args.thr is not None:
        thr = float(args.thr)
    else:
        q = float(args.thr_quantile)
        if not (0.0 < q < 1.0):
            raise SystemExit("--thr-quantile は 0〜1 の範囲で指定してください")
        abs_vals = np.array([abs(v) for v in sv_by_key.values()], dtype=float)
        abs_vals = abs_vals[~np.isnan(abs_vals)]
        if abs_vals.size == 0:
            raise SystemExit("thr を計算できません（|SV| が全て NaN）")
        thr = float(np.quantile(abs_vals, q))

    ts_arr, mid_arr = _build_mid_series(ledger)

    tr_py = trades.to_pydict()
    tr_keys = tr_py.get(key_col, []) or []
    tr_ts = tr_py.get("block_ts_ms", []) or []
    tr_side = tr_py.get("side", []) or []

    horizon_ms = int(args.horizon_ms)
    if horizon_ms <= 0:
        raise SystemExit("--horizon-ms は正の数である必要があります")

    groups: Dict[str, List[float]] = {
        "ask|sv>+thr": [],
        "ask|sv<-thr": [],
        "bid|sv<-thr": [],
        "bid|sv>+thr": [],
    }

    for k, ts, side in zip(tr_keys, tr_ts, tr_side):
        if k is None:
            continue
        ts_i = _as_int(ts)
        if ts_i is None:
            continue
        s = str(side)
        if s not in ("buy", "sell"):
            continue

        sv = sv_by_key.get(k)
        if sv is None:
            continue

        markout = _markout_for_fill(ts_arr, mid_arr, ts_i, s, horizon_ms=horizon_ms)
        if markout is None or (isinstance(markout, float) and math.isnan(markout)):
            continue

        is_ask = s == "sell"
        if sv > thr:
            if is_ask:
                groups["ask|sv>+thr"].append(markout)
            else:
                groups["bid|sv>+thr"].append(markout)
        elif sv < -thr:
            if is_ask:
                groups["ask|sv<-thr"].append(markout)
            else:
                groups["bid|sv<-thr"].append(markout)

    rows = []
    for name, xs in groups.items():
        st = _stats(xs)
        rows.append((name, st))

    print(f"ledger={ledger_path}")
    print(f"trades={trades_path}")
    print(f"sv_col={args.sv_col}  horizon_ms={horizon_ms}  thr={thr}")
    print("")

    if str(args.format) == "md":
        print("| group | n | mean | p10 | p50 | p90 |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name, st in rows:
            print(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(st["n"]),
                        _fmt(st["mean"]),
                        _fmt(st["p10"]),
                        _fmt(st["p50"]),
                        _fmt(st["p90"]),
                    ]
                )
                + " |"
            )
    else:
        print("group\tn\tmean\tp10\tp50\tp90")
        for name, st in rows:
            print(
                "\t".join(
                    [
                        name,
                        _fmt(st["n"]),
                        _fmt(st["mean"]),
                        _fmt(st["p10"]),
                        _fmt(st["p50"]),
                        _fmt(st["p90"]),
                    ]
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

