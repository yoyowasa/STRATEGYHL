import json
import gzip
import os
import pathlib
from datetime import datetime, timezone
from typing import Iterator, Dict, Any

import pandas as pd

# 生データを置いたルート（環境変数 RAW_ROOT があればそちらを優先）
RAW_ROOT = pathlib.Path(os.environ.get("RAW_ROOT", "./raw_data"))

# 期間: 2024-01-01 00:00:00 UTC から現在
START_MS = 1704067200000
END_MS = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

# シンボル: ETH-PERP と UETH/USDC(@151 spot)
COINS = ["ETH", "@151"]


def iter_trades(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    ts = pd.to_datetime(rec["time"], utc=True)
    return {
        "ts": ts,
        "px": float(rec["px"]),
        "sz": float(rec["sz"]),
        "coin": rec["coin"],
    }


def load_coin(coin: str) -> pd.DataFrame:
    rows = []
    pattern = RAW_ROOT.rglob("node_trades/hourly/*.json*")
    for p in pattern:
        for r in iter_trades(p):
            if r.get("coin") != coin:
                continue
            norm = normalize_record(r)
            ts_ms = int(norm["ts"].timestamp() * 1000)
            if START_MS <= ts_ms <= END_MS:
                rows.append(norm)
    return pd.DataFrame(rows)


def make_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("ts")
    agg = {"px": ["first", "max", "min", "last"], "sz": "sum"}
    res = df.resample(freq).agg(agg).dropna()
    res.columns = ["open", "high", "low", "close", "volume"]
    return res


def main() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for coin in COINS:
        df = load_coin(coin)
        if df.empty:
            print(f"[warn] no data for coin={coin} under {RAW_ROOT}")
            continue
        o1m = make_ohlcv(df, "1T")
        o1h = make_ohlcv(o1m.reset_index().rename(columns={"index": "ts"}), "1H") if not o1m.empty else pd.DataFrame()
        o1m.to_parquet(f"ohlcv_{coin.replace('@', 'at')}_1m.parquet")
        if not o1h.empty:
            o1h.to_parquet(f"ohlcv_{coin.replace('@', 'at')}_1h.parquet")
        print(f"[info] coin={coin} rows={len(df)} 1m={len(o1m)} 1h={len(o1h)}")


if __name__ == "__main__":
    main()
