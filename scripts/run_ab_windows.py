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
    candidate_names = {cfg: load_config(cfg).strategy.name for cfg in candidate_cfgs}

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
            delta = float(cand_pnl - base_pnl)
            per_candidate_deltas[cand_name].append(delta)
            if delta > 0:
                per_candidate_wins[cand_name] += 1
            if fmt == "md":
                print(f"| {w.tag} | {base_pnl:.6g} | {cand_name} | {cand_pnl:.6g} | {delta:.6g} |")
            else:
                print(f"{w.tag}\t{base_pnl:.6g}\t{cand_name}\t{cand_pnl:.6g}\t{delta:.6g}")

    # 集計（median と勝率）
    print("")
    if fmt == "md":
        print("| candidate | N | median_Δpnl | wins | win_rate |")
        print("| --- | ---: | ---: | ---: | ---: |")
    else:
        print("candidate\tN\tmedian_Δpnl\twins\twin_rate")
    for cand_name, deltas in per_candidate_deltas.items():
        if not deltas:
            continue
        deltas_sorted = sorted(deltas)
        mid = len(deltas_sorted) // 2
        if len(deltas_sorted) % 2 == 1:
            median = deltas_sorted[mid]
        else:
            median = 0.5 * (deltas_sorted[mid - 1] + deltas_sorted[mid])
        wins = int(per_candidate_wins.get(cand_name, 0))
        win_rate = wins / len(deltas_sorted) if deltas_sorted else float("nan")
        if fmt == "md":
            print(f"| {cand_name} | {len(deltas_sorted)} | {median:.6g} | {wins} | {win_rate:.6g} |")
        else:
            print(f"{cand_name}\t{len(deltas_sorted)}\t{median:.6g}\t{wins}\t{win_rate:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
