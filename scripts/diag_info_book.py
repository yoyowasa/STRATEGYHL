from __future__ import annotations

import argparse
import hashlib
import time
from typing import Optional

import hlmm.live.runner as live_runner


def _sleep_ms(interval_ms: int) -> None:
    time.sleep(max(0.0, float(interval_ms) / 1000.0))


def _book_hash8(
    best_bid: Optional[float],
    best_ask: Optional[float],
    bid_sz1: Optional[float],
    ask_sz1: Optional[float],
) -> str:
    payload = f"{best_bid}:{best_ask}:{bid_sz1}:{ask_sz1}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def diagnose(
    base_url: str,
    coin: str,
    attempts: int,
    interval_ms: int,
    book_depth: int,
) -> int:
    api_coin = live_runner._resolve_api_coin(base_url, coin)
    if api_coin != coin:
        market = "spot" if api_coin.startswith("@") else "perp"
        print(f"[info] coin alias ({market}): {coin} -> {api_coin}")

    prev_hash: Optional[str] = None
    unique_pairs: set[tuple[Optional[float], Optional[float]]] = set()
    unique_hashes: set[str] = set()
    changed = 0
    ok_samples = 0

    for i in range(attempts):
        recv_ts_ms = live_runner._now_ms()
        try:
            book = live_runner._post_json(base_url, {"type": "l2Book", "coin": api_coin})
            if not isinstance(book, dict):
                print(f"[warn] l2Book response not dict: {type(book)}")
                _sleep_ms(interval_ms)
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] l2Book fetch failed: {exc}")
            _sleep_ms(interval_ms)
            continue

        ts_ms = int(book.get("time") or recv_ts_ms)
        bids, asks = live_runner._extract_book_levels(book, depth=book_depth)
        best_bid, bid_sz1 = live_runner._best_level(bids)
        best_ask, ask_sz1 = live_runner._best_level(asks)

        hash8 = _book_hash8(best_bid, best_ask, bid_sz1, ask_sz1)
        book_change = prev_hash is not None and hash8 != prev_hash
        if book_change:
            changed += 1
        prev_hash = hash8

        unique_pairs.add((best_bid, best_ask))
        unique_hashes.add(hash8)
        ok_samples += 1

        print(
            "[{}/{}] ts={} bid={} ask={} bid_sz1={} ask_sz1={} hash8={} change={}".format(
                i + 1,
                attempts,
                ts_ms,
                best_bid,
                best_ask,
                bid_sz1,
                ask_sz1,
                hash8,
                book_change,
            )
        )
        _sleep_ms(interval_ms)

    print(
        "[summary] ok_samples={} unique_pairs={} unique_hash8={} changes={}".format(
            ok_samples,
            len(unique_pairs),
            len(unique_hashes),
            changed,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch /info l2Book repeatedly to confirm best bid/ask changes.",
    )
    parser.add_argument("--base-url", default="https://api.hyperliquid.xyz", help="Base URL")
    parser.add_argument("--coin", default="ETH", help="Coin (perp: ETH, spot: ETHUSDC)")
    parser.add_argument("--attempts", type=int, default=10, help="Number of l2Book requests")
    parser.add_argument("--interval-ms", type=int, default=1000, help="Sleep interval between requests (ms)")
    parser.add_argument("--book-depth", type=int, default=20, help="Book depth to parse")
    args = parser.parse_args()

    return diagnose(
        base_url=str(args.base_url),
        coin=str(args.coin),
        attempts=int(args.attempts),
        interval_ms=int(args.interval_ms),
        book_depth=int(args.book_depth),
    )


if __name__ == "__main__":
    raise SystemExit(main())
