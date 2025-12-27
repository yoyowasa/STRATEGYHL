"""Command line interface for HLMM."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import sys
from typing import Iterable, Mapping, Optional

import yaml

from hlmm.config import ConfigError, load_config
from hlmm.mm import StrategyParams, decide_orders, run_mm_sim, run_replay, run_shadow
from hlmm.research import build_dataset, run_edge_screen
from hlmm.research.report import generate_report


def _get_version() -> str:
    """Return the installed package version, or a placeholder when unavailable."""
    try:
        return metadata.version("hlmm")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return float(v) != 0.0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(v)


def _build_strategy_params(extra: Mapping[str, object], max_abs_position_override: Optional[float]) -> StrategyParams:
    effective_max_abs_position = (
        float(max_abs_position_override)
        if max_abs_position_override is not None
        else (float(extra["max_abs_position"]) if "max_abs_position" in extra else None)
    )

    # alias: one_side_pull_mode / one_side_pull_size_factor / one_side_pull_add_bps
    pull_mode = str(extra.get("pull_mode", "symmetric"))
    pull_spread_add_bps = float(extra.get("pull_spread_add_bps", 0.0))
    pull_size_factor = float(extra.get("pull_size_factor", 1.0))
    if "pull_mode" not in extra and "one_side_pull_mode" in extra:
        one_side_mode = str(extra.get("one_side_pull_mode") or "").lower().strip()
        if one_side_mode in ("size_only", "size"):
            pull_mode = "one_side"
            pull_spread_add_bps = 0.0
            pull_size_factor = float(extra.get("one_side_pull_size_factor", pull_size_factor))
        elif one_side_mode in ("widen_only", "widen", "spread_only", "spread"):
            pull_mode = "one_side"
            pull_spread_add_bps = float(extra.get("one_side_pull_add_bps", pull_spread_add_bps))
            pull_size_factor = 1.0
        elif one_side_mode in ("widen_and_size", "both"):
            pull_mode = "one_side"
            pull_spread_add_bps = float(extra.get("one_side_pull_add_bps", pull_spread_add_bps))
            pull_size_factor = float(extra.get("one_side_pull_size_factor", pull_size_factor))
    return StrategyParams(
        base_spread_bps=float(extra.get("base_spread_bps", 5.0)),
        base_size=float(extra.get("base_size", 1.0)),
        inventory_skew_bps=float(extra.get("inventory_skew_bps", 2.0)),
        inventory_target=float(extra.get("inventory_target", 0.0)),
        max_abs_position=effective_max_abs_position,
        stop_max_abs_position=(float(extra["stop_max_abs_position"]) if "stop_max_abs_position" in extra else None),
        stop_max_intraday_drawdown_usdc=(
            float(extra["stop_max_intraday_drawdown_usdc"]) if "stop_max_intraday_drawdown_usdc" in extra else None
        ),
        stop_when_market_spread_bps_gt=(
            float(extra["stop_when_market_spread_bps_gt"]) if "stop_when_market_spread_bps_gt" in extra else None
        ),
        stop_when_market_spread_bps_lt=(
            float(extra["stop_when_market_spread_bps_lt"]) if "stop_when_market_spread_bps_lt" in extra else None
        ),
        stop_when_abs_mid_ret_gt=(
            float(extra["stop_when_abs_mid_ret_gt"]) if "stop_when_abs_mid_ret_gt" in extra else None
        ),
        stop_mode=str(extra.get("stop_mode", "halt")),
        halt_when_market_spread_bps_gt=(
            float(extra["halt_when_market_spread_bps_gt"]) if "halt_when_market_spread_bps_gt" in extra else None
        ),
        halt_when_market_spread_bps_lt=(
            float(extra["halt_when_market_spread_bps_lt"]) if "halt_when_market_spread_bps_lt" in extra else None
        ),
        halt_when_abs_mid_ret_gt=(
            float(extra["halt_when_abs_mid_ret_gt"]) if "halt_when_abs_mid_ret_gt" in extra else None
        ),
        halt_size_factor=float(extra.get("halt_size_factor", 0.0)),
        boost_when_abs_mid_ret_gt=(
            float(extra["boost_when_abs_mid_ret_gt"]) if "boost_when_abs_mid_ret_gt" in extra else None
        ),
        boost_size_factor=float(extra.get("boost_size_factor", 1.0)),
        boost_only_if_abs_pos_lt=(
            float(extra["boost_only_if_abs_pos_lt"]) if "boost_only_if_abs_pos_lt" in extra else None
        ),
        pull_when_market_spread_bps_gt=(
            float(extra["pull_when_market_spread_bps_gt"]) if "pull_when_market_spread_bps_gt" in extra else None
        ),
        pull_when_market_spread_bps_lt=(
            float(extra["pull_when_market_spread_bps_lt"]) if "pull_when_market_spread_bps_lt" in extra else None
        ),
        pull_when_abs_mid_ret_gt=(
            float(extra["pull_when_abs_mid_ret_gt"]) if "pull_when_abs_mid_ret_gt" in extra else None
        ),
        pull_when_abs_signed_volume_gt=(
            float(extra["pull_when_abs_signed_volume_gt"])
            if "pull_when_abs_signed_volume_gt" in extra
            else (float(extra["flow_thr"]) if "flow_thr" in extra else None)
        ),
        pull_signed_volume_window_s=(
            float(extra["pull_signed_volume_window_s"])
            if "pull_signed_volume_window_s" in extra
            else (float(extra["flow_window_s"]) if "flow_window_s" in extra else None)
        ),
        ask_size_factor_when_sv_neg=(
            float(extra["ask_size_factor_when_sv_neg"]) if "ask_size_factor_when_sv_neg" in extra else None
        ),
        pull_spread_add_bps=float(pull_spread_add_bps),
        pull_size_factor=float(pull_size_factor),
        pull_mode=str(pull_mode),
        pull_window_max_abs_position=(
            float(extra["pull_window_max_abs_position"]) if "pull_window_max_abs_position" in extra else None
        ),
        pull_window_inventory_skew_mult=float(extra.get("pull_window_inventory_skew_mult", 1.0)),
        post_pull_unwind_enable=_as_bool(extra.get("post_pull_unwind_enable", False)),
        post_pull_unwind_until_abs_pos_lt=(
            float(extra["post_pull_unwind_until_abs_pos_lt"])
            if "post_pull_unwind_until_abs_pos_lt" in extra
            else None
        ),
        post_pull_inventory_skew_mult=float(extra.get("post_pull_inventory_skew_mult", 1.0)),
        post_pull_unwind_spread_add_bps=float(extra.get("post_pull_unwind_spread_add_bps", 0.0)),
        post_pull_unwind_size_factor=float(extra.get("post_pull_unwind_size_factor", 1.0)),
        post_pull_unwind_other_side_size_factor=float(extra.get("post_pull_unwind_other_side_size_factor", 1.0)),
        micro_bias_thr_bps=(float(extra["micro_bias_thr_bps"]) if "micro_bias_thr_bps" in extra else None),
        micro_bias_size_factor=float(extra.get("micro_bias_size_factor", 1.0)),
        micro_bias_thr_pos_bps=(float(extra["micro_bias_thr_pos_bps"]) if "micro_bias_thr_pos_bps" in extra else None),
        micro_bias_ask_only_size_factor=float(
            extra.get("micro_bias_ask_only_size_factor", extra.get("ask_only_size_factor", 1.0))
        ),
        imbalance_thr=(
            float(extra["imbalance_thr"])
            if "imbalance_thr" in extra
            else (float(extra["imb1_thr"]) if "imb1_thr" in extra else None)
        ),
        imbalance_size_factor=float(extra.get("imbalance_size_factor", extra.get("imb1_size_factor", 1.0))),
        micro_pos_thr=(float(extra["micro_pos_thr"]) if "micro_pos_thr" in extra else None),
        micro_pos_size_factor=float(extra.get("micro_pos_size_factor", 1.0)),
    )


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
    sim_parser.add_argument(
        "--blocks",
        default=None,
        help="入力 blocks.parquet（未指定の場合、--config があれば paths.data_dir/blocks.parquet を使用）",
    )
    sim_parser.add_argument("--out-dir", default="mm_sim_out", help="出力ディレクトリ")
    sim_parser.add_argument(
        "--taker-fee-bps",
        type=float,
        default=0.0,
        help="フィル時の手数料bps（例: 5=0.05%%）",
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
    sim_parser.add_argument(
        "--allow-top-fill",
        action="store_true",
        help="trade_bucketが空でも板トップで約定させる（デバッグ用。厳密検証ではOFF推奨）",
    )

    shadow_parser = subparsers.add_parser(
        "mm-shadow", help="blocks.parquet を用いた shadow ログ生成（注文は出さない）"
    )
    shadow_parser.add_argument("--blocks", required=True, help="入力 blocks.parquet")
    shadow_parser.add_argument("--out-dir", default="outputs_shadow", help="shadow ログ出力先")
    shadow_parser.add_argument("--order-batch-sec", type=float, default=None, help="更新バッチ間隔（秒）")
    shadow_parser.add_argument("--size-decimals", type=int, default=None, help="サイズ丸めの小数桁")
    shadow_parser.add_argument("--price-decimals", type=int, default=None, help="価格丸めの小数桁")
    shadow_parser.add_argument("--reconnect-gap-ms", type=int, default=None, help="再接続とみなす時系列ギャップ")
    shadow_parser.add_argument("--disconnect-guard", action="store_true", help="再接続時の cancel_all を強制")

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
        config = None
        if args.config:
            try:
                config = load_config(args.config)
            except ConfigError as exc:
                parser.exit(status=1, message=f"config error: {exc}\n")

        blocks_path = args.blocks
        if not blocks_path:
            if not config:
                parser.error("mm-sim には --blocks が必要です（または --config で paths.data_dir を指定）")
            blocks_path = str(Path(config.paths.data_dir) / "blocks.parquet")

        out_dir = args.out_dir
        if config and args.out_dir == "mm_sim_out":
            out_dir = str(Path(config.paths.output_dir) / f"mm_sim_{config.strategy.name}")

        strategy_fn = None
        if config:
            extra = dict(config.strategy.extra_params or {})
            params = _build_strategy_params(extra, args.max_abs_position)
            effective_max_abs_position = params.max_abs_position

            def strategy_fn(block, state):  # type: ignore[no-redef]
                top = block.get("book_top") or {}
                try:
                    bid = float(top.get("bid_px")) if top.get("bid_px") is not None else None
                    ask = float(top.get("ask_px")) if top.get("ask_px") is not None else None
                except (TypeError, ValueError):
                    bid = None
                    ask = None
                mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
                market_spread_bps = (
                    10_000.0 * (ask - bid) / mid
                    if bid is not None and ask is not None and mid not in (None, 0.0)
                    else None
                )
                # strategy が参照できる状態を更新
                state["mid"] = mid
                state["market_spread_bps"] = market_spread_bps
                res = decide_orders(state, params)
                state["stop_triggered"] = bool(res.get("stop_triggered"))
                state["pull_triggered"] = bool(res.get("pull_triggered"))
                state["halt_triggered"] = bool(res.get("halt_triggered"))
                state["boost_triggered"] = bool(res.get("boost_triggered"))
                state["pull_side"] = res.get("pull_side")
                state["strategy_spread_bps"] = res.get("strategy_spread_bps")
                state["strategy_size"] = res.get("strategy_size")
                state["strategy_bid_spread_bps"] = res.get("strategy_bid_spread_bps")
                state["strategy_ask_spread_bps"] = res.get("strategy_ask_spread_bps")
                state["strategy_bid_size"] = res.get("strategy_bid_size")
                state["strategy_ask_size"] = res.get("strategy_ask_size")
                state["post_pull_unwind_active"] = bool(res.get("post_pull_unwind_active"))
                if res.get("halt"):
                    return []
                return list(res.get("orders") or [])

        run_mm_sim(
            blocks_path=blocks_path,
            out_dir=out_dir,
            taker_fee_bps=args.taker_fee_bps,
            maker_rebate_bps=args.maker_rebate_bps,
            max_abs_position=effective_max_abs_position if config else args.max_abs_position,
            fill_model=args.fill_model,
            lower_alpha=args.lower_alpha,
            lower_nprints=args.lower_nprints,
            signed_volume_window_s=params.pull_signed_volume_window_s,
            allow_top_fill=bool(args.allow_top_fill),
            strategy=strategy_fn,
        )
        return 0

    if args.command == "mm-shadow":
        if not args.config:
            parser.error("mm-shadow には --config が必要です")
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            parser.exit(status=1, message=f"config error: {exc}\n")

        extra = dict(config.strategy.extra_params or {})
        params = _build_strategy_params(extra, None)

        order_batch_sec = (
            float(args.order_batch_sec)
            if args.order_batch_sec is not None
            else float(extra.get("order_batch_sec", 0.1))
        )
        size_decimals = (
            int(args.size_decimals)
            if args.size_decimals is not None
            else (int(extra["size_decimals"]) if "size_decimals" in extra else None)
        )
        price_decimals = (
            int(args.price_decimals)
            if args.price_decimals is not None
            else (int(extra["price_decimals"]) if "price_decimals" in extra else None)
        )
        reconnect_gap_ms = (
            int(args.reconnect_gap_ms)
            if args.reconnect_gap_ms is not None
            else (int(extra["reconnect_gap_ms"]) if "reconnect_gap_ms" in extra else None)
        )
        disconnect_guard = bool(args.disconnect_guard) or _as_bool(extra.get("disconnect_guard", False))

        run_shadow(
            blocks_path=args.blocks,
            out_dir=args.out_dir,
            params=params,
            order_batch_sec=order_batch_sec,
            size_decimals=size_decimals,
            price_decimals=price_decimals,
            reconnect_gap_ms=reconnect_gap_ms,
            disconnect_guard=disconnect_guard,
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
