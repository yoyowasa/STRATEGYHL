from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import numpy as np

from hlmm.cli import main as hlmm_main
from hlmm.config import load_config
from hlmm.research import generate_report


@dataclass(frozen=True)
class WindowSpec:
    start_ms: int
    end_ms: int
    tag: str
    blocks_path: Path


def _fmt_tag(ts_ms: int, tz: str) -> str:
    if tz == "utc":
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    else:
        # ローカル時刻（ユーザーの Get-Date と揃える想定）
        dt = datetime.fromtimestamp(ts_ms / 1000.0)
    return dt.strftime("%Y%m%d_%H%M")


def _clip_trade_bucket(blocks: List[dict], start_ms: int) -> None:
    # 2h窓の先頭ブロックに「窓開始より前のtrade」が混ざるのを防ぐ（窓内で閉じる）。
    for b in blocks:
        bucket = b.get("trade_bucket")
        if not isinstance(bucket, list):
            continue
        kept = []
        for tr in bucket:
            if not isinstance(tr, dict):
                continue
            ts = tr.get("ts_ms")
            if ts is None:
                continue
            try:
                ts_i = int(ts)
            except (TypeError, ValueError):
                continue
            if ts_i >= start_ms:
                kept.append(tr)
        b["trade_bucket"] = kept


def build_windows(
    table: pa.Table,
    out_root: Path,
    window_ms: int,
    step_ms: int,
    n_windows: int,
    tz: str,
) -> List[WindowSpec]:
    if "block_ts_ms" not in table.schema.names:
        raise SystemExit("blocks.parquet に block_ts_ms がありません")

    ts_col = table["block_ts_ms"]
    min_ts = pc.min(ts_col).as_py()
    max_ts = pc.max(ts_col).as_py()
    if min_ts is None or max_ts is None:
        raise SystemExit("block_ts_ms の min/max を取れません")

    start0 = int(min_ts)
    end_max = int(max_ts)

    out_root.mkdir(parents=True, exist_ok=True)

    windows: List[WindowSpec] = []
    for i in range(int(n_windows)):
        start = start0 + i * int(step_ms)
        end = start + int(window_ms)
        if end > end_max:
            break

        tag = _fmt_tag(start, tz=tz)
        window_dir = out_root / tag
        window_dir.mkdir(parents=True, exist_ok=True)
        blocks_path = window_dir / "blocks.parquet"

        mask = pc.and_(pc.greater_equal(ts_col, start), pc.less(ts_col, end))
        win_table = table.filter(mask)
        if win_table.num_rows == 0:
            continue

        blocks = win_table.to_pylist()
        _clip_trade_bucket(blocks, start_ms=start)
        pq.write_table(pa.Table.from_pylist(blocks), blocks_path)
        windows.append(WindowSpec(start_ms=start, end_ms=end, tag=tag, blocks_path=blocks_path))

    return windows


def _run_mm_sim_and_report(
    config_path: Path,
    blocks_path: Path,
    run_id: str,
    outputs_dir: Path,
    reports_dir: Path,
    fill_model: str,
    lower_alpha: float,
    lower_nprints: int | None,
    allow_top_fill: bool,
) -> None:
    out_dir = outputs_dir / f"mm_sim_{run_id}"
    argv = [
        "--config",
        str(config_path),
        "mm-sim",
        "--blocks",
        str(blocks_path),
        "--out-dir",
        str(out_dir),
        "--fill-model",
        str(fill_model),
        "--lower-alpha",
        str(lower_alpha),
    ]
    if lower_nprints is not None:
        argv.extend(["--lower-nprints", str(int(lower_nprints))])
    if allow_top_fill:
        argv.append("--allow-top-fill")

    code = hlmm_main(argv)
    if code != 0:
        raise SystemExit(f"mm-sim failed (code={code}) run_id={run_id}")

    ledger_path = out_dir / "ledger.parquet"
    trades_path = out_dir / "sim_trades.parquet"
    generate_report(
        run_id=run_id,
        ledger_path=ledger_path,
        trades_path=trades_path,
        reports_dir=reports_dir,
    )


def _load_pnl(reports_dir: Path, run_id: str) -> float:
    path = reports_dir / run_id / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return float(data.get("pnl", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _load_metrics(reports_dir: Path, run_id: str) -> dict:
    path = reports_dir / run_id / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tables(outputs_dir: Path, run_id: str) -> tuple[pa.Table, pa.Table]:
    out_dir = outputs_dir / f"mm_sim_{run_id}"
    ledger_path = out_dir / "ledger.parquet"
    trades_path = out_dir / "sim_trades.parquet"
    if not ledger_path.exists():
        raise SystemExit(f"ledger.parquet が見つかりません: {ledger_path}")
    if not trades_path.exists():
        raise SystemExit(f"sim_trades.parquet が見つかりません: {trades_path}")
    return pq.read_table(ledger_path), pq.read_table(trades_path)


def _choose_common_key(ledger_a: pa.Table, trades_a: pa.Table, ledger_b: pa.Table, trades_b: pa.Table) -> str:
    for key in ("book_event_id", "block_ts_ms"):
        if (
            key in ledger_a.schema.names
            and key in trades_a.schema.names
            and key in ledger_b.schema.names
            and key in trades_b.schema.names
        ):
            return key
    raise SystemExit("window join key が見つかりません（book_event_id/block_ts_ms）")


def _window_keys(ledger: pa.Table, key_col: str, trigger_col: str) -> tuple[set, float]:
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    trig = py.get(trigger_col, []) or []
    if not keys or not trig:
        return set(), 0.0
    in_keys = {k for k, t in zip(keys, trig) if k is not None and bool(t)}
    rate = float(sum(1 for t in trig if bool(t)) / len(trig)) if trig else 0.0
    return in_keys, rate


def _halt_window_keys(
    ledger: pa.Table, key_col: str, halt_abs_mid_ret_thr: float | None
) -> tuple[set, float]:
    if "halt_triggered" in ledger.schema.names:
        return _window_keys(ledger, key_col, "halt_triggered")
    if halt_abs_mid_ret_thr is None or "abs_mid_ret" not in ledger.schema.names:
        return set(), 0.0
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    vals = py.get("abs_mid_ret", []) or []
    if not keys or not vals:
        return set(), 0.0
    in_keys = set()
    trig = 0
    total = 0
    for k, v in zip(keys, vals):
        total += 1
        try:
            v_f = float(v) if v is not None else None
        except (TypeError, ValueError):
            v_f = None
        if v_f is not None and v_f > float(halt_abs_mid_ret_thr):
            if k is not None:
                in_keys.add(k)
            trig += 1
    rate = float(trig / total) if total else 0.0
    return in_keys, rate


def _boost_window_keys(
    ledger: pa.Table, key_col: str, boost_abs_mid_ret_thr: float | None
) -> tuple[set, float]:
    if "boost_triggered" in ledger.schema.names:
        return _window_keys(ledger, key_col, "boost_triggered")
    if boost_abs_mid_ret_thr is None or "abs_mid_ret" not in ledger.schema.names:
        return set(), 0.0
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    vals = py.get("abs_mid_ret", []) or []
    if not keys or not vals:
        return set(), 0.0
    in_keys = set()
    trig = 0
    total = 0
    for k, v in zip(keys, vals):
        total += 1
        try:
            v_f = float(v) if v is not None else None
        except (TypeError, ValueError):
            v_f = None
        if v_f is not None and v_f > float(boost_abs_mid_ret_thr):
            if k is not None:
                in_keys.add(k)
            trig += 1
    rate = float(trig / total) if total else 0.0
    return in_keys, rate


def _pnl_split(ledger: pa.Table, key_col: str, window_keys: set) -> tuple[float, float]:
    py = ledger.to_pydict()
    keys = py.get(key_col, []) or []
    pnl = py.get("total_pnl", []) or []
    pnl_in = 0.0
    pnl_out = 0.0
    prev = 0.0
    for k, cur in zip(keys, pnl):
        if cur is None:
            continue
        try:
            cur_f = float(cur)
        except (TypeError, ValueError):
            continue
        delta = cur_f - prev
        prev = cur_f
        if isinstance(delta, float) and math.isnan(delta):
            continue
        if k in window_keys:
            pnl_in += delta
        else:
            pnl_out += delta
    return float(pnl_in), float(pnl_out)


def _fills_notional_for_keys(trades: pa.Table, key_col: str, keys_set: set) -> tuple[int, float]:
    py = trades.to_pydict()
    keys = py.get(key_col, []) or []
    if "price" in py and "size" in py:
        prices = py.get("price", []) or []
        sizes = py.get("size", []) or []
    elif "px" in py and "sz" in py:
        prices = py.get("px", []) or []
        sizes = py.get("sz", []) or []
    else:
        return 0, 0.0
    n = min(len(keys), len(prices), len(sizes))
    fills = 0
    notional = 0.0
    for i in range(n):
        if keys[i] not in keys_set:
            continue
        px = prices[i]
        sz = sizes[i]
        if px is None or sz is None:
            continue
        try:
            px_f = float(px)
            sz_f = float(sz)
        except (TypeError, ValueError):
            continue
        fills += 1
        notional += abs(px_f * sz_f)
    return fills, float(notional)


def _mid_series_from_ledger(ledger: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    py = ledger.to_pydict()
    ts = py.get("block_ts_ms", []) or []
    mid = py.get("mark_price", []) or []
    pairs: list[tuple[int, float]] = []
    for t, m in zip(ts, mid):
        if t is None or m is None:
            continue
        try:
            pairs.append((int(t), float(m)))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)
    ts_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    mid_arr = np.array([p[1] for p in pairs], dtype=float)
    return ts_arr, mid_arr


def _markout_and_realized_spread_split(
    ledger: pa.Table,
    trades: pa.Table,
    key_col: str,
    window_keys: set,
    horizon_ms: int,
) -> dict[str, float | None]:
    ts_arr, mid_arr = _mid_series_from_ledger(ledger)
    if ts_arr.size == 0 or trades.num_rows == 0:
        return {
            "markout_in": None,
            "markout_out": None,
            "realized_spread_in": None,
            "realized_spread_out": None,
        }

    tpy = trades.to_pydict()
    sides = tpy.get("side", []) or []
    ts = tpy.get("block_ts_ms", []) or []
    keys = tpy.get(key_col, []) or []
    prices = tpy.get("price", []) or []

    mark_in: list[float] = []
    mark_out: list[float] = []
    rs_in: list[float] = []
    rs_out: list[float] = []

    for side, t, k, px in zip(sides, ts, keys, prices):
        if side is None or t is None:
            continue
        try:
            ts_i = int(t)
        except (TypeError, ValueError):
            continue

        idx_now = int(np.searchsorted(ts_arr, ts_i, side="right") - 1)
        if idx_now < 0 or idx_now >= ts_arr.size:
            continue
        mid_now = float(mid_arr[idx_now])

        fut_ts = ts_i + int(horizon_ms)
        idx_fut = int(np.searchsorted(ts_arr, fut_ts, side="left"))
        if idx_fut < 0 or idx_fut >= ts_arr.size:
            continue
        mid_fut = float(mid_arr[idx_fut])

        is_in = k in window_keys
        if str(side) == "buy":
            mark = mid_fut - mid_now
            rs = None
            if px is not None:
                try:
                    rs = mid_fut - float(px)
                except (TypeError, ValueError):
                    rs = None
        else:
            mark = mid_now - mid_fut
            rs = None
            if px is not None:
                try:
                    rs = float(px) - mid_fut
                except (TypeError, ValueError):
                    rs = None

        if is_in:
            mark_in.append(float(mark))
            if rs is not None:
                rs_in.append(float(rs))
        else:
            mark_out.append(float(mark))
            if rs is not None:
                rs_out.append(float(rs))

    def _mean(vals: list[float]) -> float | None:
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr))

    return {
        "markout_in": _mean(mark_in),
        "markout_out": _mean(mark_out),
        "realized_spread_in": _mean(rs_in),
        "realized_spread_out": _mean(rs_out),
    }


def _median(values: Sequence[float | None]) -> float | None:
    vals = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        vals.append(float(v))
    if not vals:
        return None
    vals = sorted(vals)
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{value:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="1つの blocks.parquet から 2h窓を切り出し、A/B（baseline vs candidate）をまとめて回して Δpnl を集計する"
    )
    parser.add_argument("--blocks", required=True, help="長時間 blocks.parquet のパス")
    parser.add_argument("--baseline-config", required=True, help="baseline の YAML（例: configs/strategy_pull_vol.yaml）")
    parser.add_argument(
        "--candidate-configs",
        required=True,
        help="candidate の YAML（カンマ区切りで複数可。例: configs/strategy_pull_vol_post_unwind_thr0p02.yaml）",
    )
    parser.add_argument("--window-sec", type=int, default=7200, help="窓長（秒）。デフォルト 2h=7200")
    parser.add_argument("--step-sec", type=int, default=1800, help="ずらし幅（秒）。デフォルト 30m=1800")
    parser.add_argument("--n-windows", type=int, default=20, help="窓数（デフォルト 20）")
    parser.add_argument("--tz", choices=["local", "utc"], default="local", help="窓タグの時刻系")
    parser.add_argument("--fill-model", choices=["upper", "lower"], default="lower")
    parser.add_argument("--lower-alpha", type=float, default=0.5)
    parser.add_argument("--lower-nprints", type=int, default=None)
    parser.add_argument("--allow-top-fill", action="store_true")
    parser.add_argument(
        "--windows-out",
        default="data/windows",
        help="切り出した窓 blocks の出力先（デフォルト: data/windows/<tag>/blocks.parquet）",
    )
    parser.add_argument("--outputs-dir", default="outputs", help="mm-sim 出力ディレクトリ（デフォルト: outputs/）")
    parser.add_argument("--reports-dir", default="reports", help="レポート出力ディレクトリ（デフォルト: reports/）")
    parser.add_argument("--format", choices=["tsv", "md"], default="tsv", help="出力形式（tsv/md）")
    args = parser.parse_args()

    blocks_path = Path(args.blocks)
    baseline_cfg = Path(args.baseline_config)
    candidate_cfgs = [Path(p.strip()) for p in str(args.candidate_configs).split(",") if p.strip()]
    if not candidate_cfgs:
        raise SystemExit("--candidate-configs が空です")

    baseline_name = load_config(baseline_cfg).strategy.name
    candidate_configs = {cfg: load_config(cfg) for cfg in candidate_cfgs}
    candidate_names = {cfg: cfg_obj.strategy.name for cfg, cfg_obj in candidate_configs.items()}
    candidate_halt_abs_mid_ret = {}
    candidate_boost_abs_mid_ret = {}
    for cfg, cfg_obj in candidate_configs.items():
        thr = None
        try:
            if "halt_when_abs_mid_ret_gt" in (cfg_obj.strategy.extra_params or {}):
                thr = float(cfg_obj.strategy.extra_params["halt_when_abs_mid_ret_gt"])
        except (TypeError, ValueError):
            thr = None
        candidate_halt_abs_mid_ret[candidate_names[cfg]] = thr
        boost_thr = None
        try:
            if "boost_when_abs_mid_ret_gt" in (cfg_obj.strategy.extra_params or {}):
                boost_thr = float(cfg_obj.strategy.extra_params["boost_when_abs_mid_ret_gt"])
        except (TypeError, ValueError):
            boost_thr = None
        candidate_boost_abs_mid_ret[candidate_names[cfg]] = boost_thr

    table = pq.read_table(blocks_path)
    windows = build_windows(
        table=table,
        out_root=Path(args.windows_out),
        window_ms=int(args.window_sec) * 1000,
        step_ms=int(args.step_sec) * 1000,
        n_windows=int(args.n_windows),
        tz=str(args.tz),
    )
    if not windows:
        raise SystemExit("窓を1つも作れませんでした（blocksの長さ不足の可能性）")
    if len(windows) < int(args.n_windows):
        print(
            f"[warn] 窓数が不足しています: built={len(windows)} requested={int(args.n_windows)}"
            f"（window_sec={int(args.window_sec)} step_sec={int(args.step_sec)}）"
        )
    else:
        print(f"[info] built windows: {len(windows)}（window_sec={int(args.window_sec)} step_sec={int(args.step_sec)}）")

    outputs_dir = Path(args.outputs_dir)
    reports_dir = Path(args.reports_dir)
    fill_model = str(args.fill_model)
    lower_alpha = float(args.lower_alpha)
    lower_nprints = args.lower_nprints
    allow_top_fill = bool(args.allow_top_fill)

    fmt = str(args.format)

    # A/B 実行（窓ごとの表）
    if fmt == "md":
        print("| tag | baseline_pnl | candidate | candidate_pnl | Δpnl |")
        print("| --- | ---: | --- | ---: | ---: |")
    else:
        print("tag\tbaseline_pnl\tcandidate\tcandidate_pnl\tΔpnl")
    per_candidate_deltas: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_wins: dict[str, int] = {candidate_names[c]: 0 for c in candidate_cfgs}
    per_candidate_max_drawdown: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_inventory_max_abs: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_notional_traded: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_pnl_in_boost_window: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_pnl_outside_boost_window: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_realized_spread_in_boost: dict[str, List[float | None]] = {
        candidate_names[c]: [] for c in candidate_cfgs
    }
    per_candidate_realized_spread_out_boost: dict[str, List[float | None]] = {
        candidate_names[c]: [] for c in candidate_cfgs
    }
    per_candidate_markout_in_boost: dict[str, List[float | None]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_markout_out_boost: dict[str, List[float | None]] = {
        candidate_names[c]: [] for c in candidate_cfgs
    }
    per_candidate_halt_trigger_rates: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_fills_when_halt_active: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_notional_when_halt_active: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_pnl_in_halt_window: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_pnl_outside_halt_window: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_baseline_pnl_in_halt_window: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}
    per_candidate_baseline_pnl_outside_halt_window: dict[str, List[float]] = {candidate_names[c]: [] for c in candidate_cfgs}

    for w in windows:
        base_run_id = f"{baseline_name}__{w.tag}"
        _run_mm_sim_and_report(
            config_path=baseline_cfg,
            blocks_path=w.blocks_path,
            run_id=base_run_id,
            outputs_dir=outputs_dir,
            reports_dir=reports_dir,
            fill_model=fill_model,
            lower_alpha=lower_alpha,
            lower_nprints=lower_nprints,
            allow_top_fill=allow_top_fill,
        )
        base_pnl = _load_pnl(reports_dir, base_run_id)
        base_ledger, base_trades = _load_tables(outputs_dir, base_run_id)

        for cfg in candidate_cfgs:
            cand_name = candidate_names[cfg]
            cand_run_id = f"{cand_name}__{w.tag}"
            _run_mm_sim_and_report(
                config_path=cfg,
                blocks_path=w.blocks_path,
                run_id=cand_run_id,
                outputs_dir=outputs_dir,
                reports_dir=reports_dir,
                fill_model=fill_model,
                lower_alpha=lower_alpha,
                lower_nprints=lower_nprints,
                allow_top_fill=allow_top_fill,
            )
            cand_pnl = _load_pnl(reports_dir, cand_run_id)
            cand_metrics = _load_metrics(reports_dir, cand_run_id)
            cand_ledger, cand_trades = _load_tables(outputs_dir, cand_run_id)
            key_col = _choose_common_key(base_ledger, base_trades, cand_ledger, cand_trades)
            boost_keys, _boost_rate = _boost_window_keys(
                cand_ledger, key_col, candidate_boost_abs_mid_ret.get(cand_name)
            )
            boost_pnl_in, boost_pnl_out = _pnl_split(cand_ledger, key_col, boost_keys)
            boost_rs_mark = _markout_and_realized_spread_split(
                cand_ledger, cand_trades, key_col, boost_keys, horizon_ms=5000
            )
            halt_keys, halt_rate = _halt_window_keys(
                cand_ledger, key_col, candidate_halt_abs_mid_ret.get(cand_name)
            )
            fills, notional = _fills_notional_for_keys(cand_trades, key_col, halt_keys)
            cand_pnl_in, cand_pnl_out = _pnl_split(cand_ledger, key_col, halt_keys)
            base_pnl_in, base_pnl_out = _pnl_split(base_ledger, key_col, halt_keys)
            delta = float(cand_pnl - base_pnl)
            per_candidate_deltas[cand_name].append(delta)
            if "max_drawdown" in cand_metrics:
                try:
                    per_candidate_max_drawdown[cand_name].append(float(cand_metrics.get("max_drawdown", 0.0)))
                except (TypeError, ValueError):
                    pass
            inv = cand_metrics.get("inventory") or {}
            if isinstance(inv, dict) and "max_abs" in inv:
                try:
                    per_candidate_inventory_max_abs[cand_name].append(float(inv.get("max_abs")))
                except (TypeError, ValueError):
                    pass
            if "notional_traded" in cand_metrics:
                try:
                    per_candidate_notional_traded[cand_name].append(float(cand_metrics.get("notional_traded", 0.0)))
                except (TypeError, ValueError):
                    pass
            per_candidate_pnl_in_boost_window[cand_name].append(float(boost_pnl_in))
            per_candidate_pnl_outside_boost_window[cand_name].append(float(boost_pnl_out))
            per_candidate_realized_spread_in_boost[cand_name].append(boost_rs_mark.get("realized_spread_in"))
            per_candidate_realized_spread_out_boost[cand_name].append(boost_rs_mark.get("realized_spread_out"))
            per_candidate_markout_in_boost[cand_name].append(boost_rs_mark.get("markout_in"))
            per_candidate_markout_out_boost[cand_name].append(boost_rs_mark.get("markout_out"))
            per_candidate_halt_trigger_rates[cand_name].append(float(halt_rate))
            per_candidate_fills_when_halt_active[cand_name].append(float(fills))
            per_candidate_notional_when_halt_active[cand_name].append(float(notional))
            per_candidate_pnl_in_halt_window[cand_name].append(float(cand_pnl_in))
            per_candidate_pnl_outside_halt_window[cand_name].append(float(cand_pnl_out))
            per_candidate_baseline_pnl_in_halt_window[cand_name].append(float(base_pnl_in))
            per_candidate_baseline_pnl_outside_halt_window[cand_name].append(float(base_pnl_out))
            if delta > 0:
                per_candidate_wins[cand_name] += 1
            if fmt == "md":
                print(f"| {w.tag} | {base_pnl:.6g} | {cand_name} | {cand_pnl:.6g} | {delta:.6g} |")
            else:
                print(f"{w.tag}\t{base_pnl:.6g}\t{cand_name}\t{cand_pnl:.6g}\t{delta:.6g}")

    # 集計（median と勝率）
    print("")
    headers = [
        "candidate",
        "N",
        "median_Δpnl",
        "wins",
        "win_rate",
        "max_drawdown",
        "inventory_max_abs",
        "notional_traded",
        "pnl_in_boost_window",
        "pnl_outside_boost_window",
        "realized_spread_5s_in_boost",
        "realized_spread_5s_out_boost",
        "markout_5s_in_boost",
        "markout_5s_out_boost",
        "halt_trigger_rate",
        "fills_when_halt_active",
        "notional_when_halt_active",
        "pnl_in_halt_window",
        "pnl_outside_halt_window",
        "baseline_pnl_in_halt_window",
        "baseline_pnl_outside_halt_window",
    ]
    if fmt == "md":
        align = ["---"] + ["---:" for _ in headers[1:]]
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(align) + " |")
    else:
        print("\t".join(headers))
    for cand_name, deltas in per_candidate_deltas.items():
        if not deltas:
            continue
        deltas_sorted = sorted(deltas)
        median = _median(deltas_sorted)
        wins = int(per_candidate_wins.get(cand_name, 0))
        win_rate = wins / len(deltas_sorted) if deltas_sorted else float("nan")
        max_drawdown = _median(per_candidate_max_drawdown[cand_name])
        inv_max_abs = _median(per_candidate_inventory_max_abs[cand_name])
        notional_traded = _median(per_candidate_notional_traded[cand_name])
        pnl_in_boost = _median(per_candidate_pnl_in_boost_window[cand_name])
        pnl_out_boost = _median(per_candidate_pnl_outside_boost_window[cand_name])
        rs_in_boost = _median(per_candidate_realized_spread_in_boost[cand_name])
        rs_out_boost = _median(per_candidate_realized_spread_out_boost[cand_name])
        mo_in_boost = _median(per_candidate_markout_in_boost[cand_name])
        mo_out_boost = _median(per_candidate_markout_out_boost[cand_name])
        halt_rate = _median(per_candidate_halt_trigger_rates[cand_name])
        fills = _median(per_candidate_fills_when_halt_active[cand_name])
        notional = _median(per_candidate_notional_when_halt_active[cand_name])
        pnl_in = _median(per_candidate_pnl_in_halt_window[cand_name])
        pnl_out = _median(per_candidate_pnl_outside_halt_window[cand_name])
        base_pnl_in = _median(per_candidate_baseline_pnl_in_halt_window[cand_name])
        base_pnl_out = _median(per_candidate_baseline_pnl_outside_halt_window[cand_name])
        row = [
            cand_name,
            str(len(deltas_sorted)),
            _fmt_num(median),
            str(wins),
            _fmt_num(win_rate),
            _fmt_num(max_drawdown),
            _fmt_num(inv_max_abs),
            _fmt_num(notional_traded),
            _fmt_num(pnl_in_boost),
            _fmt_num(pnl_out_boost),
            _fmt_num(rs_in_boost),
            _fmt_num(rs_out_boost),
            _fmt_num(mo_in_boost),
            _fmt_num(mo_out_boost),
            _fmt_num(halt_rate),
            _fmt_num(fills),
            _fmt_num(notional),
            _fmt_num(pnl_in),
            _fmt_num(pnl_out),
            _fmt_num(base_pnl_in),
            _fmt_num(base_pnl_out),
        ]
        if fmt == "md":
            print("| " + " | ".join(row) + " |")
        else:
            print("\t".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
