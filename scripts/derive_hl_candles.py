from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List


def _interval_ms(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60_000
    if interval.endswith("h"):
        return int(interval[:-1]) * 3_600_000
    if interval.endswith("d"):
        return int(interval[:-1]) * 86_400_000
    raise ValueError(f"未対応 interval: {interval}")


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Decimal変換できません: {value}") from exc


def _dec_to_str(value: Decimal) -> str:
    s = format(value, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


@dataclass(frozen=True)
class Candle:
    t: int
    T: int
    s: str
    i: str
    o: Decimal
    c: Decimal
    h: Decimal
    low: Decimal
    v: Decimal
    n: int


def _parse_candle(raw: Dict[str, Any]) -> Candle:
    return Candle(
        t=int(raw["t"]),
        T=int(raw.get("T", raw["t"])),
        s=str(raw["s"]),
        i=str(raw["i"]),
        o=_to_decimal(raw["o"]),
        c=_to_decimal(raw["c"]),
        h=_to_decimal(raw["h"]),
        low=_to_decimal(raw["l"]),
        v=_to_decimal(raw["v"]),
        n=int(raw.get("n", 0)),
    )


def derive_candles(base: List[Dict[str, Any]], target_interval: str) -> List[Dict[str, Any]]:
    if not base:
        return []
    candles = sorted((_parse_candle(x) for x in base), key=lambda c: c.t)
    target_ms = _interval_ms(target_interval)

    out: List[Dict[str, Any]] = []
    current_bucket = None
    bucket_items: List[Candle] = []

    def flush() -> None:
        nonlocal bucket_items, current_bucket
        if not bucket_items or current_bucket is None:
            bucket_items = []
            current_bucket = None
            return
        bucket_items.sort(key=lambda c: c.t)
        o = bucket_items[0].o
        c = bucket_items[-1].c
        h = max(x.h for x in bucket_items)
        low = min(x.low for x in bucket_items)
        v = sum((x.v for x in bucket_items), Decimal("0"))
        n = sum(x.n for x in bucket_items)
        s = bucket_items[0].s
        out.append(
            {
                "t": current_bucket,
                "T": current_bucket + target_ms - 1,
                "s": s,
                "i": target_interval,
                "o": _dec_to_str(o),
                "c": _dec_to_str(c),
                "h": _dec_to_str(h),
                "l": _dec_to_str(low),
                "v": _dec_to_str(v),
                "n": n,
            }
        )
        bucket_items = []
        current_bucket = None

    for candle in candles:
        bucket = (candle.t // target_ms) * target_ms
        if current_bucket is None:
            current_bucket = bucket
        if bucket != current_bucket:
            flush()
            current_bucket = bucket
        bucket_items.append(candle)
    flush()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperliquid candles JSON を集約して派生足を作る")
    parser.add_argument("--input", required=True, help="入力 candles JSON（list）")
    parser.add_argument("--output", required=True, help="出力 candles JSON（list）")
    parser.add_argument("--target-interval", required=True, help="派生 interval（例: 45m）")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    base = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(base, list):
        raise SystemExit("入力が list ではありません")
    derived = derive_candles(base, args.target_interval)
    out_path.write_text(json.dumps(derived, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {args.target_interval}: {len(base)} -> {len(derived)} ({out_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
