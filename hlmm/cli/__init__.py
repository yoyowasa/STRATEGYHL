"""Command line interface for HLMM."""

from __future__ import annotations

import argparse
from importlib import metadata
import sys
from typing import Iterable, Optional

import yaml

from hlmm.config import ConfigError, load_config
from hlmm.mm import run_mm_sim, run_replay
from hlmm.research import build_dataset, run_edge_screen
from hlmm.research.report import generate_report


def _get_version() -> str:
    """Return the installed package version, or a placeholder when unavailable."""
    try:
        return metadata.version("hlmm")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hlmm",
        description="HLMM command line entrypoint (skeleton).",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Optional path to a configuration file.",
        default=None,
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Load YAML config,正規化した内容を標準出力へ出力。",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")

    dataset_parser = subparsers.add_parser(
        "dataset", help="features.parquet から目的変数付き dataset.parquet を生成"
    )
    dataset_parser.add_argument(
        "--features",
        required=True,
        help="入力 features.parquet のパス",
    )
    dataset_parser.add_argument(
        "--dataset-out",
        default="dataset.parquet",
        help="出力先 dataset.parquet パス",
    )
    dataset_parser.add_argument(
        "--splits-out",
        default="splits.json",
        help="出力先 splits.json パス",
    )
    dataset_parser.add_argument(
        "--horizons",
        default="1,5,15,60",
        help="予測ホライズン秒（カンマ区切り）",
    )

    edge_parser = subparsers.add_parser(
        "edge", help="単変量エッジ探索を実行し edge_report.json とプロットを出力"
    )
    edge_parser.add_argument("--dataset", required=True, help="dataset.parquet のパス")
    edge_parser.add_argument("--splits", required=True, help="splits.json のパス")
    edge_parser.add_argument(
        "--target",
        default=None,
        help="ターゲット列（指定しない場合は y_* の先頭）",
    )
    edge_parser.add_argument(
        "--out-dir",
        default="edge_output",
        help="レポート・プロット出力先ディレクトリ",
    )
    edge_parser.add_argument(
        "--ic-threshold",
        type=float,
        default=0.01,
        help="keep/drop 判定に使う IC 閾値",
    )

    sim_parser = subparsers.add_parser(
        "mm-sim", help="blocks.parquet を用いた簡易マーケットメイクシミュレーション"
    )
    sim_parser.add_argument("--blocks", required=True, help="入力 blocks.parquet")
    sim_parser.add_argument("--out-dir", default="mm_sim_out", help="出力ディレクトリ")
    sim_parser.add_argument(
        "--taker-fee-bps",
        type=float,
        default=0.0,
        help="フィル時の手数料bps（例: 5=0.05%）",
    )
    sim_parser.add_argument(
        "--maker-rebate-bps",
        type=float,
        default=0.0,
        help="メイカーリベートbps（post-only前提）",
    )
    sim_parser.add_argument(
        "--max-abs-position",
        type=float,
        default=None,
        help="ポジション絶対値の上限（超える注文は拒否）",
    )
    sim_parser.add_argument(
        "--fill-model",
        choices=["upper", "lower"],
        default="lower",
        help="フィルモデルを切替（upper:楽観/ lower:控えめ）",
    )
    sim_parser.add_argument(
        "--lower-alpha",
        type=float,
        default=0.5,
        help="lowerフィルで使用する alpha（対象約定量への比率）",
    )
    sim_parser.add_argument(
        "--lower-nprints",
        type=int,
        default=None,
        help="lowerフィルで使用する最大約定件数",
    )

    replay_parser = subparsers.add_parser(
        "mm-replay", help="user fills/fundings を再生して台帳を構築しレポート生成"
    )
    replay_parser.add_argument("--fills", required=True, help="user fills の Parquet")
    replay_parser.add_argument("--fundings", default=None, help="user fundings の Parquet（任意）")
    replay_parser.add_argument("--out-dir", default="reports", help="出力ディレクトリ（reports/<run_id>/）")
    replay_parser.add_argument("--run-id", default="replay", help="レポート出力名")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset":
        horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
        build_dataset(
            features_path=args.features,
            dataset_out=args.dataset_out,
            splits_out=args.splits_out,
            horizons_sec=horizons,
        )
        return 0

    if args.command == "edge":
        run_edge_screen(
            dataset_path=args.dataset,
            splits_path=args.splits,
            out_dir=args.out_dir,
            target=args.target,
            ic_threshold=args.ic_threshold,
        )
        return 0

    if args.command == "mm-sim":
        run_mm_sim(
            blocks_path=args.blocks,
            out_dir=args.out_dir,
            taker_fee_bps=args.taker_fee_bps,
            maker_rebate_bps=args.maker_rebate_bps,
            max_abs_position=args.max_abs_position,
            fill_model=args.fill_model,
            lower_alpha=args.lower_alpha,
            lower_nprints=args.lower_nprints,
        )
        return 0

    if args.command == "mm-replay":
        ledger_path = run_replay(
            fills_path=args.fills,
            fundings_path=args.fundings,
            out_dir=args.out_dir,
            run_id=args.run_id,
        )
        generate_report(run_id=args.run_id, ledger_path=ledger_path, trades_path=args.fills, reports_dir=args.out_dir)
        return 0

    if args.print_config:
        if not args.config:
            parser.error("--print-config を使うには --config で YAML を指定してください")
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            parser.exit(status=1, message=f"config error: {exc}\n")
        yaml.safe_dump(config.to_dict(), stream=sys.stdout, sort_keys=True)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
