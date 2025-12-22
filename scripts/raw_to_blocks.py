from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable, List

from hlmm.data import AssetCtxEvent, BaseEvent, BboEvent, BookEvent, TradeEvent
from hlmm.features import align_blocks, save_blocks_parquet
from hlmm.io import dedupe_events, parse_events_from_file, save_events_parquet


def _iter_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    # 決定的な順序にする（ディレクトリ配下をパス順で処理）
    files: List[Path] = []
    for pattern in ("*.jsonl", "*.zst", "*.jsonl.zst"):
        files.extend(sorted(input_path.rglob(pattern)))
    files = sorted({p.resolve() for p in files}, key=lambda p: str(p).lower())
    return files


def _canonical_symbol(value: str) -> str:
    norm = re.sub(r"[^a-zA-Z0-9@]", "", value.strip()).upper()
    if norm.endswith("USDCPERP"):
        base = norm[: -len("USDCPERP")]
        return base or norm
    if norm.endswith("PERP"):
        base = norm[: -len("PERP")]
        return base or norm
    return norm


def _filter_symbol(events: Iterable[BaseEvent], symbol: str | None) -> List[BaseEvent]:
    if not symbol:
        return list(events)
    sym = _canonical_symbol(symbol)
    if not sym:
        return list(events)
    out: List[BaseEvent] = []
    for ev in events:
        if hasattr(ev, "symbol") and _canonical_symbol(str(getattr(ev, "symbol"))) == sym:
            out.append(ev)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="raw.jsonl(.zst) -> blocks.parquet を生成（l2Bookをクロックにtradeをバケット化）")
    parser.add_argument(
        "--input",
        required=True,
        help="入力raw（ファイル or ディレクトリ）",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="対象symbolでフィルタ（例: ETH / @151）",
    )
    parser.add_argument(
        "--out-blocks",
        default="data/blocks.parquet",
        help="出力 blocks.parquet（デフォルト: data/blocks.parquet）",
    )
    parser.add_argument(
        "--out-events",
        default=None,
        help="（任意）dedupe済み events.parquet を保存するパス",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    files = _iter_input_files(input_path)
    if not files:
        raise SystemExit(f"入力が見つかりません: {input_path}")

    events: List[BaseEvent] = []
    for path in files:
        events.extend(parse_events_from_file(path))

    events = dedupe_events(events)
    events = _filter_symbol(events, args.symbol)

    if args.out_events:
        save_events_parquet(events, args.out_events)

    books = [ev for ev in events if isinstance(ev, BookEvent)]
    trades = [ev for ev in events if isinstance(ev, TradeEvent)]
    ctxs = [ev for ev in events if isinstance(ev, AssetCtxEvent)]
    bbos = [ev for ev in events if isinstance(ev, BboEvent)]

    blocks = align_blocks(books, trades, ctxs, bbos)
    out_blocks = save_blocks_parquet(blocks, args.out_blocks)
    print(f"[ok] books={len(books)} trades={len(trades)} blocks={len(blocks)} -> {out_blocks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
