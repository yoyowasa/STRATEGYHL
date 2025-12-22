from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


def _to_float(value: Any) -> float:
    return float(value)


def _candle_to_block(candle: Dict[str, Any], book_spread_bps: float) -> Dict[str, Any]:
    ts_ms = int(candle["T"])  # bar close を block 時刻として扱う（単調増加になりやすい）
    symbol = str(candle.get("s") or "")
    interval = str(candle.get("i") or "")
    start_ms = int(candle.get("t") or ts_ms)

    close_px = _to_float(candle["c"])

    half = float(book_spread_bps) / 10_000 / 2.0
    bid_px = close_px * (1.0 - half)
    ask_px = close_px * (1.0 + half)

    return {
        "block_ts_ms": ts_ms,
        "book_event_id": f"candle:{symbol}:{interval}:{start_ms}",
        "book_top": {"bid_px": bid_px, "bid_sz": 1.0, "ask_px": ask_px, "ask_sz": 1.0},
        "trade_bucket": [],
        "missing_book": False,
        "missing_trades": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HLのcandleSnapshot(JSON)を簡易 blocks.parquet に変換")
    parser.add_argument(
        "--input",
        required=True,
        help="入力 candleSnapshot JSON（例: data/hyperliquid/candles/data_eth_perp_1m.json）",
    )
    parser.add_argument(
        "--output",
        default="data/blocks.parquet",
        help="出力 blocks.parquet パス（デフォルト: data/blocks.parquet）",
    )
    parser.add_argument(
        "--book-spread-bps",
        type=float,
        default=0.0,
        help="板トップのbid/askスプレッド（bps）。0なら bid==ask==close。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先頭からN本だけ変換（デバッグ用）",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    candles = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(candles, list):
        raise SystemExit("入力JSONがlistではありません")

    if args.limit is not None:
        candles = candles[: int(args.limit)]

    blocks: List[Dict[str, Any]] = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        blocks.append(_candle_to_block(c, book_spread_bps=float(args.book_spread_bps)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(blocks), out_path)
    print(f"[ok] blocks: {len(blocks)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

