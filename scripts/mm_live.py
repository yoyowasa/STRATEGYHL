from __future__ import annotations

import argparse

from hlmm.live import run_live


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live/shadow market making loop.")
    parser.add_argument("--config", required=True, help="Config YAML path.")
    parser.add_argument("--mode", required=True, choices=["live", "shadow"], help="live or shadow.")
    parser.add_argument("--run-id", required=True, help="Run identifier for log subdir.")
    parser.add_argument("--log-dir", required=True, help="Root log directory.")
    parser.add_argument("--secret-env", default="HL_PRIVATE_KEY", help="Env var for private key.")
    parser.add_argument("--base-url", default="https://api.hyperliquid.xyz", help="Hyperliquid base URL.")
    parser.add_argument("--coin", default="ETH", help="Market symbol (perp coin or spot).")
    parser.add_argument("--poll-interval-ms", type=int, default=1000, help="Polling interval in ms.")
    parser.add_argument("--book-depth", type=int, default=20, help="Order book depth.")
    parser.add_argument("--duration-sec", type=int, default=None, help="Optional duration limit.")
    parser.add_argument("--order-batch-sec", type=float, default=None, help="Override order batch seconds.")
    parser.add_argument("--size-decimals", type=int, default=None, help="Override size decimals.")
    parser.add_argument("--price-decimals", type=int, default=None, help="Override price decimals.")
    parser.add_argument("--reconnect-gap-ms", type=int, default=None, help="Override reconnect gap ms.")
    args = parser.parse_args()

    run_dir = run_live(
        config_path=args.config,
        mode=args.mode,
        run_id=args.run_id,
        log_dir=args.log_dir,
        secret_env=args.secret_env,
        base_url=args.base_url,
        coin=args.coin,
        poll_interval_ms=int(args.poll_interval_ms),
        book_depth=int(args.book_depth),
        duration_sec=args.duration_sec,
        order_batch_sec=args.order_batch_sec,
        size_decimals=args.size_decimals,
        price_decimals=args.price_decimals,
        reconnect_gap_ms=args.reconnect_gap_ms,
    )
    print(f"[ok] logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
