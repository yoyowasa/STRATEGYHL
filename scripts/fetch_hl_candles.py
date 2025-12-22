from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class CandleTask:
    label: str
    coin: str


def _post_json(url: str, body: Dict[str, Any], timeout_s: int = 30) -> Any:
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout_s) as resp:
        text = resp.read().decode("utf-8")
    return json.loads(text)


def _fetch_spot_meta(base_url: str) -> Dict[str, Any]:
    info_url = f"{base_url.rstrip('/')}/info"
    meta = _post_json(info_url, {"type": "spotMeta"})
    if not isinstance(meta, dict):
        raise RuntimeError(f"spotMeta のレスポンスが想定外です: {type(meta)}")
    return meta


def _resolve_ethusdc_spot_coin(spot_meta: Dict[str, Any]) -> str:
    tokens = spot_meta.get("tokens", [])
    universe = spot_meta.get("universe", [])

    # USDC は通常 index=0 だが、念のため tokens から引く
    usdc = next((t for t in tokens if t.get("name") == "USDC"), None)
    ueth = next((t for t in tokens if t.get("name") == "UETH"), None)
    if usdc is None or ueth is None:
        raise RuntimeError("spotMeta から USDC/UETH を特定できません")

    usdc_index = usdc["index"]
    ueth_index = ueth["index"]

    pair = next((u for u in universe if u.get("tokens") == [ueth_index, usdc_index]), None)
    if pair is None:
        raise RuntimeError("spotMeta から UETH/USDC ペアを特定できません")
    return pair["name"]


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _interval_ms(interval: str) -> int:
    # Hyperliquid docs の candleSnapshot interval に合わせる
    if interval.endswith("m"):
        return int(interval[:-1]) * 60_000
    if interval.endswith("h"):
        return int(interval[:-1]) * 3_600_000
    if interval.endswith("d"):
        return int(interval[:-1]) * 86_400_000
    raise ValueError(f"未対応 interval: {interval}")


def fetch_candles(
    base_url: str,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    info_url = f"{base_url.rstrip('/')}/info"
    body = {
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": end_ms},
    }
    data = _post_json(info_url, body)
    if not isinstance(data, list):
        raise RuntimeError(f"想定外レスポンス: {type(data)}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperliquid /info candleSnapshot を取得して保存")
    parser.add_argument("--base-url", default="https://api.hyperliquid.xyz")
    parser.add_argument(
        "--out-dir",
        default="data/hyperliquid/candles",
        help="出力先ディレクトリ",
    )
    parser.add_argument(
        "--spotmeta-out",
        default="data/hyperliquid/meta/spotMeta.json",
        help="spotMeta レスポンス保存先",
    )
    parser.add_argument(
        "--start-1d-ms",
        type=int,
        default=1704067200000,  # 2024-01-01T00:00:00Z
        help="日足取得の開始 epoch ms（デフォルト: 2024-01-01）",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=5000,
        help="1m/1h など短期足の最大本数（candleSnapshot制約に合わせる）",
    )
    parser.add_argument(
        "--intervals",
        default="1m,1h,1d",
        help="取得する interval（カンマ区切り）",
    )
    args = parser.parse_args()

    intervals = [x.strip() for x in str(args.intervals).split(",") if x.strip()]
    end_ms = _now_ms()

    spot_meta = _fetch_spot_meta(args.base_url)
    ethusdc_spot_coin = _resolve_ethusdc_spot_coin(spot_meta)

    tasks = [
        CandleTask(label="eth_perp", coin="ETH"),
        CandleTask(label="ethusdc", coin=ethusdc_spot_coin),
    ]

    spotmeta_out = Path(args.spotmeta_out)
    spotmeta_out.parent.mkdir(parents=True, exist_ok=True)
    spotmeta_out.write_text(json.dumps(spot_meta, ensure_ascii=False), encoding="utf-8")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for interval in intervals:
            if interval == "1d":
                start_ms = int(args.start_1d_ms)
            else:
                start_ms = end_ms - int(args.max_bars) * _interval_ms(interval)

            data = fetch_candles(
                base_url=args.base_url,
                coin=task.coin,
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            # 既存の命名に合わせる
            out_path = out_dir / f"data_{task.label}_{interval}.json"
            out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            print(f"[ok] {task.label} {interval}: {len(data)} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
