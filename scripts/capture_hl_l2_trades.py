from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _now_ms() -> int:
    return int(time.time() * 1000)


def _post_json(base_url: str, body: Dict[str, Any], timeout_s: int = 30) -> Any:
    url = f"{base_url.rstrip('/')}/info"
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    backoff_s = 0.5
    for attempt in range(6):
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                text = resp.read().decode("utf-8")
            return json.loads(text)
        except HTTPError as exc:
            # レート制限/一時障害はリトライ（落とさない）
            if exc.code in (429, 500, 502, 503, 504):
                if attempt == 5:
                    raise
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 8.0)
                continue
            raise
        except URLError:
            if attempt == 5:
                raise
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2.0, 8.0)
            continue


def _fetch_spot_meta(base_url: str) -> Dict[str, Any]:
    meta = _post_json(base_url, {"type": "spotMeta"})
    if not isinstance(meta, dict):
        raise RuntimeError(f"spotMeta のレスポンスが想定外です: {type(meta)}")
    return meta


def _resolve_ethusdc_spot_coin(base_url: str) -> str:
    meta = _fetch_spot_meta(base_url)
    tokens = meta.get("tokens", [])
    universe = meta.get("universe", [])

    if not isinstance(tokens, list) or not isinstance(universe, list):
        raise RuntimeError("spotMeta の形式が想定外です（tokens/universe）")

    usdc = next((t for t in tokens if isinstance(t, dict) and t.get("name") == "USDC"), None)
    base = next(
        (t for t in tokens if isinstance(t, dict) and t.get("name") in {"UETH", "ETH"}),
        None,
    )
    if usdc is None or base is None:
        raise RuntimeError("spotMeta から USDC/(UETH|ETH) を特定できません")

    usdc_index = usdc.get("index")
    base_index = base.get("index")
    if not isinstance(usdc_index, int) or not isinstance(base_index, int):
        raise RuntimeError("spotMeta token index が不正です")

    pair = next(
        (
            u
            for u in universe
            if isinstance(u, dict)
            and u.get("tokens") == [base_index, usdc_index]
            and isinstance(u.get("name"), str)
        ),
        None,
    )
    if pair is None:
        raise RuntimeError("spotMeta から (UETH|ETH,USDC) のペアを特定できません")
    return str(pair["name"])


def _canonical_symbol(value: str) -> str:
    raw = value.strip()
    norm = re.sub(r"[^a-zA-Z0-9@]", "", raw).upper()
    # Perpの別名（例: ETH-USDCPERP, ETHUSDCPERP, ETH-PERP）
    if norm.endswith("USDCPERP"):
        base = norm[: -len("USDCPERP")]
        return base or norm
    if norm.endswith("PERP"):
        base = norm[: -len("PERP")]
        return base or norm
    if norm == "ETHUSDC":
        return "ETHUSDC"
    return norm


def _resolve_api_coin(base_url: str, coin: str) -> str:
    raw = coin.strip()
    norm = re.sub(r"[^a-zA-Z0-9@]", "", raw).upper()
    if norm.startswith("@"):
        return norm
    if norm == "ETHUSDC":
        return _resolve_ethusdc_spot_coin(base_url)
    if norm.endswith("USDCPERP"):
        base = norm[: -len("USDCPERP")]
        return base or norm
    if norm.endswith("PERP"):
        base = norm[: -len("PERP")]
        return base or norm
    return norm


def _as_levels(side_levels: Any, depth: int) -> List[List[str]]:
    if not isinstance(side_levels, list):
        return []
    out: List[List[str]] = []
    for item in side_levels[:depth]:
        if isinstance(item, dict) and "px" in item and "sz" in item:
            out.append([str(item["px"]), str(item["sz"])])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append([str(item[0]), str(item[1])])
    return out


def _extract_book_levels(book: Dict[str, Any], depth: int) -> Tuple[List[List[str]], List[List[str]]]:
    levels = book.get("levels")
    if isinstance(levels, list) and len(levels) == 2:
        bids = _as_levels(levels[0], depth)
        asks = _as_levels(levels[1], depth)
        return bids, asks
    bids = _as_levels(book.get("bids"), depth)
    asks = _as_levels(book.get("asks"), depth)
    return bids, asks


def _normalize_trade_side(side: Any) -> str | None:
    if not isinstance(side, str):
        return None
    s = side.strip().lower()
    # HL recentTrades は side が "B"(bid側=buy aggressor) / "A"(ask側=sell aggressor) の形式。
    if s in {"b", "buy", "bid"}:
        return "buy"
    if s in {"a", "s", "sell", "ask"}:
        return "sell"
    return None


def capture(
    base_url: str,
    coin: str,
    out_path: Path,
    duration_sec: int,
    poll_interval_ms: int,
    book_depth: int,
) -> None:
    symbol = _canonical_symbol(coin)
    api_coin = _resolve_api_coin(base_url, coin)
    if api_coin != coin:
        market = "spot" if api_coin.startswith("@") else "perp"
        print(f"[info] coin alias ({market}): {coin} -> {api_coin}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    end_at = time.monotonic() + float(duration_sec)
    seen_trade_ids: set[str] = set()
    cooldown_s = 0.0

    with out_path.open("a", encoding="utf-8") as fh:
        while time.monotonic() < end_at:
            recv_ts_ms = _now_ms()

            # l2Book（スナップショット）
            try:
                book = _post_json(base_url, {"type": "l2Book", "coin": api_coin})
                if isinstance(book, dict):
                    ts_ms = int(book.get("time") or recv_ts_ms)
                    bids, asks = _extract_book_levels(book, depth=book_depth)
                    rec = {
                        "channel": "l2Book",
                        "symbol": symbol,
                        "api_coin": str(book.get("coin") or api_coin),
                        "bids": bids,
                        "asks": asks,
                        "ts_ms": ts_ms,
                        "recv_ts_ms": recv_ts_ms,
                        "event_id": f"l2Book:{api_coin}:{ts_ms}",
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cooldown_s = max(0.0, cooldown_s - 0.5)
            except HTTPError as exc:
                if exc.code == 429:
                    cooldown_s = min(max(cooldown_s * 2.0, 1.0), 30.0)
                    print(f"[warn] rate limited (l2Book). sleep={cooldown_s:.1f}s")
                    time.sleep(cooldown_s)
                    continue
                print(f"[warn] l2Book fetch failed: {exc}")
                time.sleep(max(0.5, cooldown_s))
                continue
            except Exception as exc:
                print(f"[warn] l2Book fetch failed: {exc}")
                time.sleep(max(0.5, cooldown_s))
                continue

            # trades（recentTrades を高頻度でポーリングして取りこぼしを減らす）
            try:
                trades = _post_json(base_url, {"type": "recentTrades", "coin": api_coin})
                if isinstance(trades, list):
                    # 古い→新しい順に書く（整合が取りやすい）
                    def _k(x: Any) -> tuple:
                        if not isinstance(x, dict):
                            return (0, "")
                        return (int(x.get("time") or 0), str(x.get("tid") or x.get("hash") or ""))

                    trades_sorted = sorted([t for t in trades if isinstance(t, dict)], key=_k)
                    for tr in trades_sorted:
                        trade_id = tr.get("tid") or tr.get("hash")
                        if trade_id is None:
                            continue
                        trade_id_s = str(trade_id)
                        if trade_id_s in seen_trade_ids:
                            continue
                        side = _normalize_trade_side(tr.get("side"))
                        if side is None:
                            continue
                        ts_ms = int(tr.get("time") or recv_ts_ms)
                        rec = {
                            "channel": "trades",
                            "symbol": symbol,
                            "api_coin": str(tr.get("coin") or api_coin),
                            "px": str(tr.get("px")),
                            "sz": str(tr.get("sz")),
                            "side": side,
                            "trade_id": trade_id_s,
                            "ts_ms": ts_ms,
                            "recv_ts_ms": recv_ts_ms,
                            "event_id": f"trades:{api_coin}:{ts_ms}:{trade_id_s}",
                        }
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        seen_trade_ids.add(trade_id_s)
                    cooldown_s = max(0.0, cooldown_s - 0.5)
            except HTTPError as exc:
                if exc.code == 429:
                    cooldown_s = min(max(cooldown_s * 2.0, 1.0), 30.0)
                    print(f"[warn] rate limited (recentTrades). sleep={cooldown_s:.1f}s")
                    time.sleep(cooldown_s)
                else:
                    print(f"[warn] recentTrades fetch failed: {exc}")
            except Exception as exc:
                print(f"[warn] recentTrades fetch failed: {exc}")

            fh.flush()
            time.sleep(max(0.0, float(poll_interval_ms) / 1000.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperliquid /info から l2Book+recentTrades を収集して raw.jsonl を作る")
    parser.add_argument("--base-url", default="https://api.hyperliquid.xyz", help="ベースURL（mainnet）")
    parser.add_argument("--coin", default="ETH", help="perpは例: ETH / spotは例: ETHUSDC（内部で@indexへ解決）")
    parser.add_argument(
        "--out",
        default=None,
        help="出力raw.jsonl（未指定なら raw_data/hl_<coin>.jsonl）",
    )
    parser.add_argument("--duration-sec", type=int, default=60, help="収集時間（秒）")
    parser.add_argument("--poll-interval-ms", type=int, default=2000, help="ポーリング間隔（ms）")
    parser.add_argument("--book-depth", type=int, default=20, help="板の深さ（top N）")
    args = parser.parse_args()

    coin = str(args.coin)
    default_name = f"hl_{coin.lstrip('@').replace('/', '_')}.jsonl"
    out_path = Path(args.out) if args.out else Path("raw_data") / default_name

    capture(
        base_url=args.base_url,
        coin=coin,
        out_path=out_path,
        duration_sec=int(args.duration_sec),
        poll_interval_ms=int(args.poll_interval_ms),
        book_depth=int(args.book_depth),
    )
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
