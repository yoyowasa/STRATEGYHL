from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from hlmm.cli.__init__ import _build_strategy_params
from hlmm.config import load_config
from hlmm.mm import decide_orders
from hlmm.mm.rounding import MAX_DECIMALS, round_price, round_size


def _now_ms() -> int:
    return int(time.time() * 1000)


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _round_price_float(px: Optional[float], max_decimals: int | None) -> Optional[float]:
    if px is None:
        return None
    decimals = MAX_DECIMALS if max_decimals is None else int(max_decimals)
    return float(round_price(Decimal(str(px)), max_decimals=decimals))


def _round_size_float(sz: float, sz_decimals: int | None) -> float:
    decimals = MAX_DECIMALS if sz_decimals is None else int(sz_decimals)
    return float(round_size(Decimal(str(sz)), sz_decimals=decimals))


def _changed(a: Optional[float], b: Optional[float], eps: float = 1e-12) -> bool:
    if a is None or b is None:
        return a != b
    return abs(a - b) > eps


def _post_json(base_url: str, body: dict, timeout_s: int = 30) -> Any:
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


def _fetch_spot_meta(base_url: str) -> dict:
    meta = _post_json(base_url, {"type": "spotMeta"})
    if not isinstance(meta, dict):
        raise RuntimeError("spotMeta response is not a dict")
    return meta


def _resolve_ethusdc_spot_coin(base_url: str) -> str:
    meta = _fetch_spot_meta(base_url)
    tokens = meta.get("tokens", [])
    universe = meta.get("universe", [])

    if not isinstance(tokens, list) or not isinstance(universe, list):
        raise RuntimeError("spotMeta format unexpected")

    usdc = next((t for t in tokens if isinstance(t, dict) and t.get("name") == "USDC"), None)
    base = next((t for t in tokens if isinstance(t, dict) and t.get("name") in {"UETH", "ETH"}), None)
    if usdc is None or base is None:
        raise RuntimeError("spotMeta missing USDC or ETH")

    usdc_index = usdc.get("index")
    base_index = base.get("index")
    if not isinstance(usdc_index, int) or not isinstance(base_index, int):
        raise RuntimeError("spotMeta token index invalid")

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
        raise RuntimeError("spotMeta ETH/USDC pair not found")
    return str(pair["name"])


def _canonical_symbol(value: str) -> str:
    raw = value.strip()
    norm = re.sub(r"[^a-zA-Z0-9@]", "", raw).upper()
    if norm.endswith("USDCPERP"):
        base = norm[: -len("USDCPERP")]
        return base or norm
    if norm.endswith("PERP"):
        base = norm[: -len("PERP")]
        return base or norm
    if norm == "ETHUSDC":
        return "ETHUSDC"
    return norm


def _exchange_coin(value: str) -> str:
    norm = _canonical_symbol(value)
    if norm.endswith("USDC") and "/" not in norm:
        base = norm[: -len("USDC")]
        if base:
            return f"{base}/USDC"
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


def _as_levels(side_levels: Any, depth: int) -> list[list[str]]:
    if not isinstance(side_levels, list):
        return []
    out: list[list[str]] = []
    for item in side_levels[:depth]:
        if isinstance(item, dict) and "px" in item and "sz" in item:
            out.append([str(item["px"]), str(item["sz"])])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append([str(item[0]), str(item[1])])
    return out


def _extract_book_levels(book: dict, depth: int) -> tuple[list[list[str]], list[list[str]]]:
    levels = book.get("levels")
    if isinstance(levels, list) and len(levels) == 2:
        bids = _as_levels(levels[0], depth)
        asks = _as_levels(levels[1], depth)
        return bids, asks
    bids = _as_levels(book.get("bids"), depth)
    asks = _as_levels(book.get("asks"), depth)
    return bids, asks


def _best_level(levels: list[list[str]]) -> tuple[Optional[float], Optional[float]]:
    if not levels:
        return None, None
    return _safe_float(levels[0][0]), _safe_float(levels[0][1])


def _normalize_trade_side(side: object) -> Optional[str]:
    if not isinstance(side, str):
        return None
    s = side.strip().lower()
    if s in {"b", "buy", "bid"}:
        return "buy"
    if s in {"a", "s", "sell", "ask"}:
        return "sell"
    return None


class _OrderSender:
    def __init__(
        self,
        mode: str,
        base_url: str,
        coin: str,
        secret_env: str,
        event_cb: Callable[[str, dict], None],
    ) -> None:
        self._mode = mode
        self._event_cb = event_cb
        self._exchange = None
        self._coin = None
        self._account_address = None
        self._cloid_cls = None
        self._disabled_reason = None
        if mode != "live":
            self._disabled_reason = "shadow_mode"
            return

        exchange, trade_coin, account_address, cloid_cls, reason = self._init_client(
            base_url, coin, secret_env
        )
        if exchange is None or trade_coin is None:
            self._disabled_reason = reason or "sender_unavailable"
            self._event_cb("sender_disabled", {"reason": self._disabled_reason})
            return
        self._exchange = exchange
        self._coin = trade_coin
        self._account_address = account_address
        self._cloid_cls = cloid_cls

    @property
    def enabled(self) -> bool:
        return self._exchange is not None and self._coin is not None and self._mode == "live"

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def _init_client(self, base_url: str, coin: str, secret_env: str):
        try:
            from eth_account import Account
        except ImportError:
            return None, None, "missing_eth_account"
        try:
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils.types import Cloid
        except ImportError:
            return None, None, None, None, "missing_hyperliquid_sdk"

        private_key = os.environ.get(secret_env)
        if not private_key:
            return None, None, None, None, f"missing_env:{secret_env}"

        try:
            wallet = Account.from_key(private_key)
            address = wallet.address
        except Exception as exc:  # noqa: BLE001
            return None, None, None, None, f"invalid_private_key:{exc}"

        exchange = Exchange(wallet=wallet, base_url=base_url or None)
        trade_coin = _exchange_coin(coin)
        return exchange, trade_coin, address, Cloid, None

    def _skip_status(self) -> dict:
        if self._mode == "shadow":
            return {"status": "skipped_shadow"}
        return {"status": "skipped_no_sender", "reason": self._disabled_reason}

    def _parse_sdk_status(self, res: object) -> tuple[str, Optional[str], Optional[int]]:
        if not isinstance(res, dict):
            return "error", "invalid_response", None
        if res.get("status") != "ok":
            return "error", str(res), None
        data = res.get("response", {}).get("data", {})
        statuses = data.get("statuses", [])
        if isinstance(statuses, list) and statuses:
            st = statuses[0]
            if isinstance(st, dict):
                if "error" in st:
                    return "error", str(st.get("error")), None
                for key in ("resting", "filled", "partial", "done"):
                    info = st.get(key)
                    if isinstance(info, dict):
                        oid = info.get("oid") or info.get("orderId")
                        return "sent", None, oid
        return "sent", None, None

    def send_new(self, side: str, size: float, price: Optional[float], client_oid: str) -> dict:
        if not self.enabled:
            return self._skip_status()
        if price is None:
            return {"status": "error", "error": "missing_price"}
        try:
            cloid = self._cloid_cls.from_str(client_oid) if self._cloid_cls else None
            res = self._exchange.order(
                name=self._coin,
                is_buy=side == "buy",
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": "Alo"}},
                reduce_only=False,
                cloid=cloid,
            )
        except Exception as exc:  # noqa: BLE001
            self._event_cb("order_send_error", {"action": "new", "error": str(exc)})
            return {"status": "error", "error": str(exc)}
        status, error, exchange_id = self._parse_sdk_status(res)
        if status != "sent":
            return {"status": "error", "error": error or "order_rejected"}
        return {"status": "sent", "exchange_id": exchange_id}

    def send_cancel(self, client_oid: str) -> dict:
        if not self.enabled:
            return self._skip_status()
        try:
            cloid = self._cloid_cls.from_str(client_oid) if self._cloid_cls else None
            res = self._exchange.cancel_by_cloid(self._coin, cloid)
        except Exception as exc:  # noqa: BLE001
            self._event_cb("order_send_error", {"action": "cancel", "error": str(exc)})
            return {"status": "error", "error": str(exc)}
        status, error, _ = self._parse_sdk_status(res)
        if status != "sent":
            return {"status": "error", "error": error or "cancel_rejected"}
        return {"status": "sent"}

    def cancel_all(self) -> dict:
        if not self.enabled:
            return self._skip_status()
        try:
            if not self._account_address:
                raise RuntimeError("missing_account_address")
            open_orders = self._exchange.info.open_orders(self._account_address)
            cancels = [
                {"coin": o.get("coin"), "oid": o.get("oid")}
                for o in open_orders
                if o.get("coin") == self._coin and o.get("oid") is not None
            ]
            if cancels:
                res = self._exchange.bulk_cancel(cancels)
                status, error, _ = self._parse_sdk_status(res)
                if status != "sent":
                    return {"status": "error", "error": error or "cancel_rejected"}
        except Exception as exc:  # noqa: BLE001
            self._event_cb("order_send_error", {"action": "cancel_all", "error": str(exc)})
            return {"status": "error", "error": str(exc)}
        return {"status": "sent", "cancelled": len(cancels) if "cancels" in locals() else 0}


def run_live(
    config_path: str,
    mode: str,
    run_id: str,
    log_dir: str,
    secret_env: str = "HL_PRIVATE_KEY",
    base_url: str = "https://api.hyperliquid.xyz",
    coin: str = "ETH",
    poll_interval_ms: int = 1000,
    book_depth: int = 20,
    duration_sec: Optional[int] = None,
    order_batch_sec: Optional[float] = None,
    size_decimals: Optional[int] = None,
    price_decimals: Optional[int] = None,
    reconnect_gap_ms: Optional[int] = None,
) -> Path:
    mode = str(mode).strip().lower()
    if mode not in {"live", "shadow"}:
        raise ValueError("mode must be 'live' or 'shadow'")

    config = load_config(config_path)
    extra = dict(config.strategy.extra_params or {})
    params = _build_strategy_params(extra, None)

    order_batch_sec = (
        float(order_batch_sec)
        if order_batch_sec is not None
        else float(extra.get("order_batch_sec", 0.1))
    )
    if size_decimals is None and "size_decimals" in extra:
        size_decimals = int(extra["size_decimals"])
    if price_decimals is None and "price_decimals" in extra:
        price_decimals = int(extra["price_decimals"])
    if reconnect_gap_ms is None and "reconnect_gap_ms" in extra:
        reconnect_gap_ms = int(extra["reconnect_gap_ms"])
    disconnect_guard = _as_bool(extra.get("disconnect_guard", False))

    run_dir = Path(log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "mode": mode,
        "config": str(config_path),
        "log_dir": str(run_dir),
        "base_url": base_url,
        "coin": coin,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "order_batch_sec": order_batch_sec,
        "size_decimals": size_decimals,
        "price_decimals": price_decimals,
        "reconnect_gap_ms": reconnect_gap_ms,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), stream=fh, sort_keys=True)

    market_log = run_dir / "market_state.jsonl"
    decision_log = run_dir / "decision.jsonl"
    orders_log = run_dir / "orders.jsonl"
    events_log = run_dir / "events.jsonl"
    fills_log = run_dir / "fills.jsonl"
    fills_log.touch(exist_ok=True)

    def log_event(event: str, detail: dict) -> None:
        record = {"ts_ms": _now_ms(), "event": event}
        record.update(detail)
        _write_jsonl(events_log, record)

    api_coin = _resolve_api_coin(base_url, coin)
    symbol = _canonical_symbol(coin)

    sender = _OrderSender(mode=mode, base_url=base_url, coin=coin, secret_env=secret_env, event_cb=log_event)
    log_event(
        "startup",
        {
            "mode": mode,
            "run_id": run_id,
            "coin": coin,
            "api_coin": api_coin,
            "symbol": symbol,
            "sender_enabled": sender.enabled,
            "sender_disabled_reason": sender.disabled_reason,
        },
    )

    batch_id = 0
    last_batch_ts: Optional[int] = None
    batch_gap_ms = int(float(order_batch_sec) * 1000.0) if order_batch_sec is not None else 0

    active_orders: Dict[str, Optional[dict]] = {"buy": None, "sell": None}
    prev_mid: Optional[float] = None
    prev_best_bid: Optional[float] = None
    prev_best_ask: Optional[float] = None
    prev_ts: Optional[int] = None
    prev_boost: Optional[bool] = None
    nonce_counter = _now_ms()

    state: Dict[str, object] = {
        "position": 0.0,
        "drawdown": 0.0,
        "mid": None,
        "mid_ret": None,
        "abs_mid_ret": None,
        "market_spread_bps": None,
        "stop_triggered": False,
        "pull_triggered": False,
        "halt_triggered": False,
        "boost_triggered": False,
        "strategy_spread_bps": None,
        "strategy_size": None,
        "strategy_bid_spread_bps": None,
        "strategy_ask_spread_bps": None,
        "strategy_bid_size": None,
        "strategy_ask_size": None,
        "post_pull_unwind_active": False,
        "pull_side": None,
        "signed_volume": 0.0,
        "signed_volume_window": 0.0,
        "prev_pull_triggered": False,
        "micro_bias_bps": None,
        "imbalance": None,
        "micro_pos": None,
    }

    seen_trade_ids: set[str] = set()
    seen_trade_queue: Deque[str] = deque()
    max_seen = 5000
    poll_total = 0
    last_poll_ts: Optional[int] = None
    ws_connected = False
    ws_msg_total = 0
    ws_l2book_total = 0
    last_ws_msg_ts: Optional[int] = None
    prev_book_hash: Optional[str] = None

    window_ms: Optional[int] = None
    if params.pull_signed_volume_window_s is not None:
        try:
            window_ms = int(float(params.pull_signed_volume_window_s) * 1000.0)
        except (TypeError, ValueError):
            window_ms = None
        if window_ms is not None and window_ms <= 0:
            window_ms = None

    sv_window: Deque[tuple[int, float]] = deque()
    sv_window_sum = 0.0

    if disconnect_guard:
        batch_id += 1
        cancel_status = sender.cancel_all()
        _write_jsonl(
            orders_log,
            {
                "ts_ms": _now_ms(),
                "action": "cancel_all",
                "side": "both",
                "px": None,
                "sz": None,
                "post_only": True,
                "client_oid": None,
                "batch_id": batch_id,
                "reason": "startup",
                "status": cancel_status.get("status"),
                "error": cancel_status.get("error"),
            },
        )

    start_time = time.monotonic()

    try:
        while True:
            if duration_sec is not None and time.monotonic() - start_time >= float(duration_sec):
                log_event("shutdown", {"reason": "duration_elapsed"})
                break

            recv_ts_ms = _now_ms()
            try:
                book = _post_json(base_url, {"type": "l2Book", "coin": api_coin})
                if not isinstance(book, dict):
                    raise RuntimeError("l2Book response invalid")
            except Exception as exc:  # noqa: BLE001
                log_event("l2book_error", {"error": str(exc)})
                time.sleep(max(0.0, poll_interval_ms / 1000.0))
                continue

            ts_ms = int(book.get("time") or recv_ts_ms)
            bids, asks = _extract_book_levels(book, depth=book_depth)
            best_bid, bid_sz1 = _best_level(bids)
            best_ask, ask_sz1 = _best_level(asks)

            poll_total += 1
            last_poll_ts = ts_ms
            log_event(
                "poll_status",
                {
                    "poll_total": poll_total,
                    "last_poll_ts": last_poll_ts,
                    "last_ts": last_poll_ts,
                    "ws_connected": ws_connected,
                    "ws_msg_total": ws_msg_total,
                    "ws_l2book_total": ws_l2book_total,
                    "last_ws_msg_ts": last_ws_msg_ts,
                },
            )

            mid = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
            reconnect = False
            if reconnect_gap_ms is not None and prev_ts is not None:
                try:
                    if ts_ms - prev_ts >= int(reconnect_gap_ms):
                        reconnect = True
                        prev_mid = None
                        prev_best_bid = None
                        prev_best_ask = None
                except (TypeError, ValueError):
                    reconnect = False
            mid_prev = prev_mid
            top_px_change = False
            if prev_best_bid is not None and prev_best_ask is not None:
                top_px_change = best_bid != prev_best_bid or best_ask != prev_best_ask
            mid_change = False
            if mid_prev is not None and mid is not None:
                mid_change = mid != mid_prev
            mid_ret = None
            abs_mid_ret = None
            if mid is not None and mid_prev not in (None, 0.0):
                try:
                    mid_ret = (float(mid) - float(mid_prev)) / float(mid_prev)
                    abs_mid_ret = abs(float(mid_ret))
                except (TypeError, ValueError, ZeroDivisionError):
                    mid_ret = None
                    abs_mid_ret = None

            spread_bps = (
                10_000.0 * (best_ask - best_bid) / mid
                if best_bid is not None and best_ask is not None and mid not in (None, 0.0)
                else None
            )

            micro_bias_bps = None
            imbalance = None
            micro_pos = None
            if bid_sz1 is not None and ask_sz1 is not None:
                denom = bid_sz1 + ask_sz1
                if denom not in (0.0, -0.0):
                    imbalance = (bid_sz1 - ask_sz1) / denom
                    if best_bid is not None and best_ask is not None:
                        microprice = (best_ask * bid_sz1 + best_bid * ask_sz1) / denom
                        if mid not in (None, 0.0):
                            micro_bias_bps = (microprice - mid) / mid * 10_000.0
                        spread = best_ask - best_bid
                        if spread not in (0.0, -0.0) and mid is not None:
                            micro_pos = (microprice - mid) / spread

            signed_volume = 0.0
            trades_count = 0
            try:
                trades = _post_json(base_url, {"type": "recentTrades", "coin": api_coin})
                if isinstance(trades, list):
                    for tr in trades:
                        if not isinstance(tr, dict):
                            continue
                        trade_id = tr.get("tid") or tr.get("hash")
                        if trade_id is None:
                            continue
                        trade_id_s = str(trade_id)
                        if trade_id_s in seen_trade_ids:
                            continue
                        side = _normalize_trade_side(tr.get("side"))
                        if side is None:
                            continue
                        sz = _safe_float(tr.get("sz"))
                        if sz is None or sz <= 0:
                            continue
                        seen_trade_ids.add(trade_id_s)
                        seen_trade_queue.append(trade_id_s)
                        if len(seen_trade_queue) > max_seen:
                            old = seen_trade_queue.popleft()
                            seen_trade_ids.discard(old)
                        signed_volume += sz if side == "buy" else -sz
                        trades_count += 1
            except Exception as exc:  # noqa: BLE001
                log_event("recent_trades_error", {"error": str(exc)})

            if window_ms is None:
                signed_volume_window = signed_volume
            else:
                sv_window.append((ts_ms, signed_volume))
                sv_window_sum += signed_volume
                cutoff = ts_ms - window_ms
                while sv_window and sv_window[0][0] < cutoff:
                    _, old = sv_window.popleft()
                    sv_window_sum -= old
                signed_volume_window = sv_window_sum

            state["mid"] = mid
            state["mid_ret"] = mid_ret
            state["abs_mid_ret"] = abs_mid_ret
            state["market_spread_bps"] = spread_bps
            state["micro_bias_bps"] = micro_bias_bps
            state["imbalance"] = imbalance
            state["micro_pos"] = micro_pos
            state["signed_volume"] = signed_volume
            state["signed_volume_window"] = signed_volume_window
            state["prev_pull_triggered"] = bool(state.get("pull_triggered"))
            state["stop_triggered"] = False
            state["pull_triggered"] = False
            state["halt_triggered"] = False
            state["boost_triggered"] = False
            state["strategy_spread_bps"] = None
            state["strategy_size"] = None
            state["strategy_bid_spread_bps"] = None
            state["strategy_ask_spread_bps"] = None
            state["strategy_bid_size"] = None
            state["strategy_ask_size"] = None
            state["pull_side"] = None

            res = decide_orders(state, params)
            boost_active = bool(res.get("boost_triggered"))
            stop_triggered = bool(res.get("stop_triggered"))
            halt_triggered = bool(res.get("halt_triggered"))

            state["stop_triggered"] = stop_triggered
            state["pull_triggered"] = bool(res.get("pull_triggered"))
            state["halt_triggered"] = halt_triggered
            state["boost_triggered"] = boost_active
            state["pull_side"] = res.get("pull_side")
            state["strategy_spread_bps"] = res.get("strategy_spread_bps")
            state["strategy_size"] = res.get("strategy_size")
            state["strategy_bid_spread_bps"] = res.get("strategy_bid_spread_bps")
            state["strategy_ask_spread_bps"] = res.get("strategy_ask_spread_bps")
            state["strategy_bid_size"] = res.get("strategy_bid_size")
            state["strategy_ask_size"] = res.get("strategy_ask_size")
            state["post_pull_unwind_active"] = bool(res.get("post_pull_unwind_active"))

            target: Dict[str, dict] = {
                "buy": {"price": None, "size": 0.0},
                "sell": {"price": None, "size": 0.0},
            }
            for order in res.get("orders") or []:
                try:
                    side = str(order.side)
                except Exception:  # noqa: BLE001
                    continue
                if side not in target:
                    continue
                target[side] = {"price": _safe_float(order.price), "size": float(order.size)}

            raw_bid_px = target["buy"]["price"]
            raw_ask_px = target["sell"]["price"]
            raw_bid_sz = float(target["buy"]["size"])
            raw_ask_sz = float(target["sell"]["size"])
            final_bid_px = _round_price_float(raw_bid_px, price_decimals)
            final_ask_px = _round_price_float(raw_ask_px, price_decimals)
            final_bid_sz = _round_size_float(raw_bid_sz, size_decimals)
            final_ask_sz = _round_size_float(raw_ask_sz, size_decimals)

            crossed_bid = final_bid_px is not None and best_ask is not None and final_bid_px >= best_ask
            crossed_ask = final_ask_px is not None and best_bid is not None and final_ask_px <= best_bid
            crossed = crossed_bid or crossed_ask

            book_payload = f"{best_bid}:{best_ask}:{bid_sz1}:{ask_sz1}"
            book_hash8 = hashlib.sha256(book_payload.encode("utf-8")).hexdigest()[:8]
            book_change = prev_book_hash is not None and book_hash8 != prev_book_hash
            prev_book_hash = book_hash8

            _write_jsonl(
                market_log,
                {
                    "ts_ms": ts_ms,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "bid_sz1": bid_sz1,
                    "ask_sz1": ask_sz1,
                    "book_hash8": book_hash8,
                    "book_change": book_change,
                    "top_px_change": top_px_change,
                    "mid_change": mid_change,
                    "mid_prev": mid_prev,
                    "mid": mid,
                    "abs_mid_ret": abs_mid_ret,
                    "boost_active": boost_active,
                    "spread_bps": spread_bps,
                    "reconnect": reconnect,
                    "signed_volume": signed_volume,
                    "signed_volume_window": signed_volume_window,
                    "trades_count": trades_count,
                },
            )
            _write_jsonl(
                decision_log,
                {
                    "ts_ms": ts_ms,
                    "mid": mid,
                    "mid_prev": mid_prev,
                    "abs_mid_ret": abs_mid_ret,
                    "abs_mid_ret_src": "l2Book",
                    "base_size": float(params.base_size),
                    "boost_size_factor": float(params.boost_size_factor),
                    "raw_bid_sz": raw_bid_sz,
                    "raw_ask_sz": raw_ask_sz,
                    "final_bid_sz": final_bid_sz,
                    "final_ask_sz": final_ask_sz,
                    "raw_bid_px": raw_bid_px,
                    "raw_ask_px": raw_ask_px,
                    "final_bid_px": final_bid_px,
                    "final_ask_px": final_ask_px,
                    "position": float(state.get("position") or 0.0),
                    "boost_active": boost_active,
                    "stop_triggered": stop_triggered,
                    "halt_triggered": halt_triggered,
                    "crossed": crossed,
                    "crossed_bid": crossed_bid,
                    "crossed_ask": crossed_ask,
                },
            )

            if reconnect:
                log_event("reconnect", {"ts_ms": ts_ms})

            if reconnect and disconnect_guard:
                if active_orders["buy"] is not None or active_orders["sell"] is not None:
                    batch_id += 1
                    cancel_status = sender.cancel_all()
                    _write_jsonl(
                        orders_log,
                        {
                            "ts_ms": ts_ms,
                            "action": "cancel_all",
                            "side": "both",
                            "px": None,
                            "sz": None,
                            "post_only": True,
                            "client_oid": None,
                            "batch_id": batch_id,
                            "reason": "reconnect",
                            "status": cancel_status.get("status"),
                            "error": cancel_status.get("error"),
                        },
                    )
                    active_orders = {"buy": None, "sell": None}
                    last_batch_ts = ts_ms
                prev_mid = mid
                prev_best_bid = best_bid
                prev_best_ask = best_ask
                prev_ts = ts_ms
                prev_boost = boost_active
                time.sleep(max(0.0, poll_interval_ms / 1000.0))
                continue

            if batch_gap_ms > 0 and last_batch_ts is not None and ts_ms - last_batch_ts < batch_gap_ms:
                prev_mid = mid
                prev_best_bid = best_bid
                prev_best_ask = best_ask
                prev_ts = ts_ms
                prev_boost = boost_active
                time.sleep(max(0.0, poll_interval_ms / 1000.0))
                continue

            actions = []
            for side in ("buy", "sell"):
                target_px = final_bid_px if side == "buy" else final_ask_px
                target_sz = final_bid_sz if side == "buy" else final_ask_sz
                active = active_orders[side]
                if target_sz <= 0:
                    if active is not None:
                        actions.append(("cancel", side, None, None, active))
                    continue
                if active is None:
                    actions.append(("new", side, target_px, target_sz, None))
                    continue
                if _changed(active.get("price"), target_px) or _changed(active.get("size"), target_sz):
                    actions.append(("replace", side, target_px, target_sz, active))

            if actions:
                batch_id += 1
                reason = "price_or_size_change"
                if prev_boost is not None and prev_boost != boost_active:
                    reason = "boost_toggle"
                for action, side, px, sz, active in actions:
                    nonce_counter += 1
                    if action == "new":
                        client_oid = f"0x{os.urandom(16).hex()}"
                        send_status = sender.send_new(side, float(sz or 0.0), px, client_oid)
                        _write_jsonl(
                            orders_log,
                            {
                                "ts_ms": ts_ms,
                                "action": action,
                                "side": side,
                                "px": px,
                                "sz": sz,
                                "post_only": True,
                                "client_oid": client_oid,
                                "batch_id": batch_id,
                                "reason": reason,
                                "crossed": crossed,
                                "nonce": nonce_counter,
                                "status": send_status.get("status"),
                                "error": send_status.get("error"),
                                "exchange_id": send_status.get("exchange_id"),
                            },
                        )
                        if send_status.get("status") != "error":
                            active_orders[side] = {"price": px, "size": sz, "client_oid": client_oid}
                    elif action == "cancel":
                        client_oid = active.get("client_oid") if active else None
                        send_status = sender.send_cancel(client_oid) if client_oid else {"status": "error"}
                        _write_jsonl(
                            orders_log,
                            {
                                "ts_ms": ts_ms,
                                "action": action,
                                "side": side,
                                "px": None,
                                "sz": None,
                                "post_only": True,
                                "client_oid": client_oid,
                                "batch_id": batch_id,
                                "reason": reason,
                                "crossed": crossed,
                                "nonce": nonce_counter,
                                "status": send_status.get("status"),
                                "error": send_status.get("error"),
                            },
                        )
                        if send_status.get("status") != "error":
                            active_orders[side] = None
                    else:
                        prev_client_oid = active.get("client_oid") if active else None
                        cancel_status = (
                            sender.send_cancel(prev_client_oid) if prev_client_oid else {"status": "error"}
                        )
                        new_client_oid = f"0x{os.urandom(16).hex()}"
                        new_status = sender.send_new(side, float(sz or 0.0), px, new_client_oid)
                        _write_jsonl(
                            orders_log,
                            {
                                "ts_ms": ts_ms,
                                "action": action,
                                "side": side,
                                "px": px,
                                "sz": sz,
                                "post_only": True,
                                "client_oid": new_client_oid,
                                "prev_client_oid": prev_client_oid,
                                "batch_id": batch_id,
                                "reason": reason,
                                "crossed": crossed,
                                "nonce": nonce_counter,
                                "status": new_status.get("status"),
                                "cancel_status": cancel_status.get("status"),
                                "error": new_status.get("error") or cancel_status.get("error"),
                                "exchange_id": new_status.get("exchange_id"),
                            },
                        )
                        if new_status.get("status") != "error":
                            active_orders[side] = {"price": px, "size": sz, "client_oid": new_client_oid}
                        elif cancel_status.get("status") == "error":
                            active_orders[side] = active
                        else:
                            active_orders[side] = None
                last_batch_ts = ts_ms

            prev_mid = mid
            prev_best_bid = best_bid
            prev_best_ask = best_ask
            prev_ts = ts_ms
            prev_boost = boost_active
            time.sleep(max(0.0, poll_interval_ms / 1000.0))
    except KeyboardInterrupt:
        log_event("shutdown", {"reason": "keyboard_interrupt"})

    return run_dir
