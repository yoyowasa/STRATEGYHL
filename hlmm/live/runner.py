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


def _price_tick_from_value(value: Optional[float], max_decimals: int | None) -> Optional[float]:
    if value is None:
        return None
    dec = Decimal(str(value))
    if dec.is_zero():
        return None
    exp = dec.as_tuple().exponent
    tick = Decimal(1).scaleb(exp)
    if max_decimals is not None:
        tick = max(tick, Decimal(1).scaleb(-int(max_decimals)))
    return float(tick)


def _changed(a: Optional[float], b: Optional[float], eps: float = 1e-12) -> bool:
    if a is None or b is None:
        return a != b
    return abs(a - b) > eps


def _is_rate_limit_error(message: object) -> bool:
    if not isinstance(message, str):
        return False
    return "Too many cumulative requests" in message


def _extract_order_status_value(payload: object) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("status", "state", "orderStatus"):
            if key in payload:
                return str(payload.get(key))
    return None


def _summarize_order_status(result: dict) -> dict:
    if not isinstance(result, dict):
        return {"status": "error", "error": "invalid_response"}
    if result.get("status") == "ok" and "response" in result:
        inner = result.get("response")
        if isinstance(inner, dict):
            return _summarize_order_status(inner)
        return {"status": "error", "error": "invalid_response"}
    status_flag = result.get("status")
    if status_flag in {"order", "unknownOid"}:
        if status_flag == "unknownOid":
            return {"status": "ok", "order_status": "unknownOid"}
        order = result.get("order")
        status = _extract_order_status_value(order)
        if status is None and isinstance(order, dict):
            status = _extract_order_status_value(order.get("order"))
        if status is None:
            status = "order"
        return {"status": "ok", "order_status": status}
    if status_flag is not None and status_flag != "ok":
        return {"status": "error", "error": str(result)}
    if "order" in result:
        order = result.get("order")
        status = _extract_order_status_value(order)
        if status is None and isinstance(order, dict):
            status = _extract_order_status_value(order.get("order"))
        if status is None:
            status = "order"
        return {"status": "ok", "order_status": status}
    return {"status": "error", "error": "unknown_response"}


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


def _safe_state_name(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


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
        self._open_orders_user = None
        self._vault_address = None
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
        self._open_orders_user = os.environ.get("HL_OPEN_ORDERS_USER") or account_address
        self._vault_address = os.environ.get("HL_VAULT_ADDRESS")
        self._cloid_cls = cloid_cls

    @property
    def enabled(self) -> bool:
        return self._exchange is not None and self._coin is not None and self._mode == "live"

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    @property
    def coin(self) -> Optional[str]:
        return self._coin

    @property
    def open_orders_user(self) -> Optional[str]:
        return self._open_orders_user

    @property
    def account_address(self) -> Optional[str]:
        return self._account_address

    @property
    def vault_address(self) -> Optional[str]:
        return self._vault_address

    def _init_client(self, base_url: str, coin: str, secret_env: str):
        try:
            from eth_account import Account
        except ImportError:
            return None, None, None, None, "missing_eth_account"
        try:
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils.types import Cloid
        except ImportError:
            return None, None, None, None, "missing_hyperliquid_sdk"

        private_key = os.environ.get(secret_env)
        if not private_key:
            return None, None, None, None, f"missing_env:{secret_env}"

        account_address = os.environ.get("HL_ACCOUNT_ADDRESS")
        vault_address = os.environ.get("HL_VAULT_ADDRESS")
        open_orders_user = os.environ.get("HL_OPEN_ORDERS_USER")

        try:
            wallet = Account.from_key(private_key)
        except Exception as exc:  # noqa: BLE001
            return None, None, None, None, f"invalid_private_key:{exc}"

        address = account_address or wallet.address
        if vault_address is None and open_orders_user and account_address != open_orders_user:
            address = open_orders_user
        exchange = Exchange(
            wallet=wallet,
            base_url=base_url or None,
            account_address=address,
            vault_address=vault_address,
        )
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

    def _parse_sdk_statuses(self, res: object, expected: int) -> list[dict]:
        if expected <= 0:
            return []
        exchange_resp_status = res.get("status") if isinstance(res, dict) else None
        if not isinstance(res, dict):
            return [
                {
                    "status": "error",
                    "error": "invalid_response",
                    "status_kind": "error",
                    "exchange_resp_status": exchange_resp_status,
                    "raw_status": None,
                }
                for _ in range(expected)
            ]
        if res.get("status") != "ok":
            return [
                {
                    "status": "error",
                    "error": str(res),
                    "status_kind": "error",
                    "exchange_resp_status": exchange_resp_status,
                    "raw_status": None,
                }
                for _ in range(expected)
            ]
        response = res.get("response", {}) if isinstance(res, dict) else {}
        data = response.get("data", {}) if isinstance(response, dict) else {}
        response_type = response.get("type") if isinstance(response, dict) else None
        statuses = data.get("statuses")
        if not isinstance(statuses, list) or not statuses:
            return [
                {
                    "status": "error",
                    "error": "missing_statuses",
                    "status_kind": "error",
                    "exchange_resp_status": exchange_resp_status,
                    "raw_status": None,
                }
                for _ in range(expected)
            ]
        out: list[dict] = []
        for idx in range(expected):
            st = statuses[idx] if idx < len(statuses) else None
            status_kind = "unknown"
            if isinstance(st, dict):
                if "error" in st:
                    out.append(
                        {
                            "status": "error",
                            "error": str(st.get("error")),
                            "status_kind": "error",
                            "exchange_resp_status": exchange_resp_status,
                            "raw_status": st,
                        }
                    )
                    continue
                exchange_id = None
                for key in ("resting", "filled", "partial", "done"):
                    info = st.get(key)
                    if isinstance(info, dict):
                        status_kind = key
                        exchange_id = info.get("oid") or info.get("orderId")
                        break
                if status_kind != "unknown":
                    out.append(
                        {
                            "status": "sent",
                            "exchange_id": exchange_id,
                            "status_kind": status_kind,
                            "exchange_resp_status": exchange_resp_status,
                            "raw_status": st,
                        }
                    )
                    continue
                if response_type == "cancel":
                    out.append(
                        {
                            "status": "sent",
                            "exchange_id": None,
                            "status_kind": "cancel",
                            "exchange_resp_status": exchange_resp_status,
                            "raw_status": st,
                        }
                    )
                    continue
                out.append(
                    {
                        "status": "error",
                        "error": "missing_status_fields",
                        "status_kind": "error",
                        "exchange_resp_status": exchange_resp_status,
                        "raw_status": st,
                    }
                )
                continue
            if isinstance(st, str):
                if st.lower() == "success":
                    out.append(
                        {
                            "status": "sent",
                            "exchange_id": None,
                            "status_kind": "cancel" if response_type == "cancel" else "success",
                            "exchange_resp_status": exchange_resp_status,
                            "raw_status": st,
                        }
                    )
                    continue
                out.append(
                    {
                        "status": "error",
                        "error": f"unexpected_status:{st}",
                        "status_kind": "error",
                        "exchange_resp_status": exchange_resp_status,
                        "raw_status": st,
                    }
                )
                continue
            out.append(
                {
                    "status": "error",
                    "error": "missing_status",
                    "status_kind": "error",
                    "exchange_resp_status": exchange_resp_status,
                    "raw_status": st,
                }
            )
        return out

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

    def fetch_open_orders(self) -> list[dict]:
        if not self.enabled:
            return []
        if not self._open_orders_user:
            raise RuntimeError("missing_open_orders_user")
        dex = "spot" if self._coin and "/" in self._coin else ""
        open_orders = self._exchange.info.open_orders(self._open_orders_user, dex=dex)
        if isinstance(open_orders, list):
            return open_orders
        return []

    def query_order_status(self, client_oid: str) -> dict:
        if not self.enabled:
            return {"status": "skipped"}
        if not self._open_orders_user:
            return {"status": "error", "error": "missing_open_orders_user"}
        if not client_oid:
            return {"status": "error", "error": "missing_client_oid"}
        if not self._cloid_cls:
            return {"status": "error", "error": "missing_cloid_cls"}
        try:
            cloid = self._cloid_cls.from_str(client_oid)
            res = self._exchange.info.query_order_by_cloid(self._open_orders_user, cloid)
            return {"status": "ok", "response": res}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}

    def query_order_status_by_oid(self, oid: int) -> dict:
        if not self.enabled:
            return {"status": "skipped"}
        if not self._open_orders_user:
            return {"status": "error", "error": "missing_open_orders_user"}
        try:
            res = self._exchange.info.query_order_by_oid(self._open_orders_user, int(oid))
            return {"status": "ok", "response": res}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}

    def send_new_batch(self, orders: list[dict]) -> list[dict]:
        if not orders:
            return []
        if not self.enabled:
            return [self._skip_status() for _ in orders]
        try:
            order_requests = []
            for order in orders:
                cloid = self._cloid_cls.from_str(order["client_oid"]) if self._cloid_cls else None
                req = {
                    "coin": self._coin,
                    "is_buy": order["side"] == "buy",
                    "sz": float(order["size"]),
                    "limit_px": float(order["price"]),
                    "order_type": {"limit": {"tif": "Alo"}},
                    "reduce_only": False,
                }
                if cloid is not None:
                    req["cloid"] = cloid
                order_requests.append(req)
            res = self._exchange.bulk_orders(order_requests)
        except Exception as exc:  # noqa: BLE001
            self._event_cb("order_send_error", {"action": "new_batch", "error": str(exc)})
            return [{"status": "error", "error": str(exc)} for _ in orders]
        return self._parse_sdk_statuses(res, len(order_requests))

    def send_modify_batch(self, orders: list[dict]) -> list[dict]:
        if not orders:
            return []
        if not self.enabled:
            return [self._skip_status() for _ in orders]
        try:
            modify_requests = []
            for order in orders:
                client_oid = order.get("client_oid")
                cloid = self._cloid_cls.from_str(client_oid) if self._cloid_cls and client_oid else None
                exchange_id = order.get("exchange_id")
                if exchange_id is None:
                    raise RuntimeError("missing_exchange_id")
                req = {
                    "coin": self._coin,
                    "is_buy": order["side"] == "buy",
                    "sz": float(order["size"]),
                    "limit_px": float(order["price"]),
                    "order_type": {"limit": {"tif": "Alo"}},
                    "reduce_only": False,
                }
                if cloid is not None:
                    req["cloid"] = cloid
                modify_requests.append({"oid": exchange_id, "order": req})
            res = self._exchange.bulk_modify_orders_new(modify_requests)
        except Exception as exc:  # noqa: BLE001
            self._event_cb("order_send_error", {"action": "modify_batch", "error": str(exc)})
            return [{"status": "error", "error": str(exc)} for _ in orders]
        return self._parse_sdk_statuses(res, len(modify_requests))

    def send_cancel_batch(self, orders: list[dict]) -> list[dict]:
        if not orders:
            return []
        if not self.enabled:
            return [self._skip_status() for _ in orders]
        try:
            cancel_requests = []
            for order in orders:
                exchange_id = order.get("exchange_id")
                if exchange_id is None:
                    raise RuntimeError("missing_exchange_id")
                cancel_requests.append({"coin": self._coin, "oid": exchange_id})
            res = self._exchange.bulk_cancel(cancel_requests)
        except Exception as exc:  # noqa: BLE001
            self._event_cb("order_send_error", {"action": "cancel_batch", "error": str(exc)})
            return [{"status": "error", "error": str(exc)} for _ in orders]
        return self._parse_sdk_statuses(res, len(cancel_requests))

    def cancel_all(self) -> dict:
        if not self.enabled:
            return self._skip_status()
        try:
            open_orders = self.fetch_open_orders()
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
    gateb_fills_min: Optional[int] = None,
    gateb_notional_min: Optional[float] = None,
    order_batch_sec: Optional[float] = None,
    size_decimals: Optional[int] = None,
    price_decimals: Optional[int] = None,
    reconnect_gap_ms: Optional[int] = None,
) -> Path:
    mode = str(mode).strip().lower()
    if mode not in {"live", "shadow"}:
        raise ValueError("mode must be 'live' or 'shadow'")

    config = load_config(config_path)
    resolved_config = config.to_dict()
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

    if gateb_fills_min is None and "gateb_fills_min" in extra:
        try:
            gateb_fills_min = int(extra["gateb_fills_min"])
        except (TypeError, ValueError):
            gateb_fills_min = None
    if gateb_fills_min is not None and gateb_fills_min <= 0:
        gateb_fills_min = None
    if gateb_notional_min is None and "gateb_notional_min" in extra:
        try:
            gateb_notional_min = float(extra["gateb_notional_min"])
        except (TypeError, ValueError):
            gateb_notional_min = None
    if gateb_notional_min is not None and gateb_notional_min <= 0:
        gateb_notional_min = None
    disconnect_guard = _as_bool(extra.get("disconnect_guard", False))
    order_status_check = _as_bool(extra.get("order_status_check", True))
    try:
        min_book_spread_bps = (
            float(extra.get("min_book_spread_bps")) if "min_book_spread_bps" in extra else None
        )
    except (TypeError, ValueError):
        min_book_spread_bps = None
    if min_book_spread_bps is not None and min_book_spread_bps <= 0:
        min_book_spread_bps = None
    try:
        min_quote_lifetime_ms = int(extra.get("min_quote_lifetime_ms", 0))
    except (TypeError, ValueError):
        min_quote_lifetime_ms = 0
    if min_quote_lifetime_ms < 0:
        min_quote_lifetime_ms = 0
    try:
        rate_limit_backoff_ms = int(extra.get("rate_limit_backoff_ms", 0))
    except (TypeError, ValueError):
        rate_limit_backoff_ms = 0
    if rate_limit_backoff_ms < 0:
        rate_limit_backoff_ms = 0
    try:
        rate_limit_backoff_max_ms = int(
            extra.get("rate_limit_backoff_max_ms", rate_limit_backoff_ms)
        )
    except (TypeError, ValueError):
        rate_limit_backoff_max_ms = rate_limit_backoff_ms
    if rate_limit_backoff_max_ms < rate_limit_backoff_ms:
        rate_limit_backoff_max_ms = rate_limit_backoff_ms
    try:
        min_send_interval_ms = int(extra.get("min_send_interval_ms", 0))
    except (TypeError, ValueError):
        min_send_interval_ms = 0
    if min_send_interval_ms < 0:
        min_send_interval_ms = 0
    try:
        open_orders_sync_sec = float(extra.get("open_orders_sync_sec", 5))
    except (TypeError, ValueError):
        open_orders_sync_sec = 5.0
    if open_orders_sync_sec < 0:
        open_orders_sync_sec = 0.0
    open_orders_sync_ms = int(open_orders_sync_sec * 1000.0)
    try:
        fills_sync_sec = float(extra.get("fills_sync_sec", 5))
    except (TypeError, ValueError):
        fills_sync_sec = 5.0
    if fills_sync_sec < 0:
        fills_sync_sec = 0.0
    fills_sync_ms = int(fills_sync_sec * 1000.0)
    try:
        fills_lookback_sec = int(extra.get("fills_lookback_sec", 900))
    except (TypeError, ValueError):
        fills_lookback_sec = 900
    if fills_lookback_sec < 0:
        fills_lookback_sec = 0
    try:
        fills_max_pages = int(extra.get("fills_max_pages", 3))
    except (TypeError, ValueError):
        fills_max_pages = 3
    if fills_max_pages < 1:
        fills_max_pages = 1
    fills_aggregate_by_time = _as_bool(extra.get("fills_aggregate_by_time", False))
    try:
        fills_cursor_rewind_ms = int(extra.get("fills_cursor_rewind_ms", 60000))
    except (TypeError, ValueError):
        fills_cursor_rewind_ms = 60000
    if fills_cursor_rewind_ms < 0:
        fills_cursor_rewind_ms = 0
    try:
        extra_bid_ticks = int(extra.get("extra_bid_ticks", 0))
    except (TypeError, ValueError):
        extra_bid_ticks = 0
    if extra_bid_ticks < 0:
        extra_bid_ticks = 0
    try:
        extra_ask_ticks = int(extra.get("extra_ask_ticks", 0))
    except (TypeError, ValueError):
        extra_ask_ticks = 0
    if extra_ask_ticks < 0:
        extra_ask_ticks = 0
    try:
        open_orders_cancel_max_attempts = int(extra.get("open_orders_cancel_max_attempts", 3))
    except (TypeError, ValueError):
        open_orders_cancel_max_attempts = 3
    if open_orders_cancel_max_attempts < 1:
        open_orders_cancel_max_attempts = 1
    try:
        open_orders_cancel_wait_ms = int(extra.get("open_orders_cancel_wait_ms", 1000))
    except (TypeError, ValueError):
        open_orders_cancel_wait_ms = 1000
    if open_orders_cancel_wait_ms < 0:
        open_orders_cancel_wait_ms = 0
    open_orders_cancel_on_start = _as_bool(
        extra.get("open_orders_cancel_on_start", disconnect_guard)
    )
    open_orders_cancel_on_reconnect = _as_bool(
        extra.get("open_orders_cancel_on_reconnect", disconnect_guard)
    )

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
        "duration_sec": duration_sec,
        "gateb_fills_min": gateb_fills_min,
        "gateb_notional_min": gateb_notional_min,
        "order_batch_sec": order_batch_sec,
        "size_decimals": size_decimals,
        "price_decimals": price_decimals,
        "reconnect_gap_ms": reconnect_gap_ms,
        "min_quote_lifetime_ms": min_quote_lifetime_ms,
        "rate_limit_backoff_ms": rate_limit_backoff_ms,
        "rate_limit_backoff_max_ms": rate_limit_backoff_max_ms,
        "min_send_interval_ms": min_send_interval_ms,
        "open_orders_sync_sec": open_orders_sync_sec,
        "order_status_check": order_status_check,
        "open_orders_cancel_on_start": open_orders_cancel_on_start,
        "open_orders_cancel_on_reconnect": open_orders_cancel_on_reconnect,
        "open_orders_cancel_max_attempts": open_orders_cancel_max_attempts,
        "open_orders_cancel_wait_ms": open_orders_cancel_wait_ms,
        "extra_bid_ticks": extra_bid_ticks,
        "extra_ask_ticks": extra_ask_ticks,
        "fills_sync_sec": fills_sync_sec,
        "fills_lookback_sec": fills_lookback_sec,
        "fills_max_pages": fills_max_pages,
        "fills_aggregate_by_time": fills_aggregate_by_time,
        "fills_cursor_rewind_ms": fills_cursor_rewind_ms,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    market_log = run_dir / "market_state.jsonl"
    decision_log = run_dir / "decision.jsonl"
    orders_log = run_dir / "orders.jsonl"
    events_log = run_dir / "events.jsonl"
    fills_log = run_dir / "fills.jsonl"
    fills_log.touch(exist_ok=True)

    fills_cursor_ms = _now_ms() - (fills_lookback_sec * 1000)
    if fills_cursor_ms < 0:
        fills_cursor_ms = 0

    run_start_ts_ms: Optional[int] = None
    gateb_fills_count = 0
    gateb_notional_sum = 0.0

    def log_event(event: str, detail: dict) -> None:
        record = {"ts_ms": _now_ms(), "event": event}
        record.update(detail)
        _write_jsonl(events_log, record)

    def _fill_time_ms(fill: dict) -> Optional[int]:
        for key in ("time", "timestamp", "ts", "ts_ms", "time_ms"):
            val = fill.get(key)
            if isinstance(val, (int, float)):
                ts = int(val)
                if ts < 1_000_000_000_000:
                    ts = int(val * 1000)
                return ts
        return None

    def _fill_key(fill: dict, time_ms: Optional[int]) -> Optional[str]:
        tid = fill.get("tid") or fill.get("tradeId") or fill.get("hash")
        if tid is not None:
            return f"tid:{tid}"
        return None

    def sync_open_orders(reason: str, now_ms: int, force: bool = False) -> None:
        nonlocal last_open_orders_sync_ts, needs_open_orders_sync, active_orders, logged_open_orders_sample, last_missing_oid_by_side
        if not sender.enabled:
            return
        if open_orders_sync_ms <= 0 and not force:
            return
        if (
            not force
            and not needs_open_orders_sync
            and open_orders_sync_ms > 0
            and now_ms - last_open_orders_sync_ts < open_orders_sync_ms
        ):
            return
        try:
            open_orders = sender.fetch_open_orders()
        except Exception as exc:  # noqa: BLE001
            log_event("open_orders_error", {"reason": reason, "error": str(exc)})
            return
        trade_coin = sender.coin
        prev_active = {side: active_orders.get(side) for side in ("buy", "sell")}
        per_side: Dict[str, dict] = {}
        matched = 0
        sample: Optional[dict] = None
        for order in open_orders:
            if not isinstance(order, dict):
                continue
            if trade_coin and order.get("coin") != trade_coin:
                continue
            side_raw = str(order.get("side") or "").upper()
            side = None
            if side_raw in {"B", "BUY", "BID"}:
                side = "buy"
            elif side_raw in {"A", "SELL", "ASK"}:
                side = "sell"
            if side is None:
                continue
            matched += 1
            if sample is None:
                sample = order
            ts = order.get("timestamp")
            ts_ms = int(ts) if isinstance(ts, (int, float)) else 0
            existing = per_side.get(side)
            if existing is None or ts_ms >= existing.get("timestamp", 0):
                per_side[side] = {
                    "price": _safe_float(order.get("limitPx")),
                    "size": _safe_float(order.get("sz")),
                    "exchange_id": order.get("oid"),
                    "timestamp": ts_ms,
                }
        for side in ("buy", "sell"):
            if side in per_side:
                info = per_side[side]
                active_orders[side] = {
                    "price": info.get("price"),
                    "size": info.get("size"),
                    "client_oid": None,
                    "exchange_id": info.get("exchange_id"),
                }
            else:
                active_orders[side] = None
        needs_open_orders_sync = False
        last_open_orders_sync_ts = now_ms
        log_event(
            "open_orders_sync",
            {
                "reason": reason,
                "open_orders_count": matched,
                "buy_oid": active_orders["buy"]["exchange_id"] if active_orders["buy"] else None,
                "sell_oid": active_orders["sell"]["exchange_id"] if active_orders["sell"] else None,
                "open_bid_present": active_orders["buy"] is not None,
                "open_ask_present": active_orders["sell"] is not None,
                "missing_side": (
                    "both"
                    if active_orders["buy"] is None and active_orders["sell"] is None
                    else "bid"
                    if active_orders["buy"] is None
                    else "ask"
                    if active_orders["sell"] is None
                    else "none"
                ),
                "open_orders_user": sender.open_orders_user,
            },
        )
        if sample and not logged_open_orders_sample:
            log_event(
                "open_orders_sample",
                {
                    "coin": sample.get("coin"),
                    "side": sample.get("side"),
                    "limitPx": sample.get("limitPx"),
                    "sz": sample.get("sz"),
                    "oid": sample.get("oid"),
                    "timestamp": sample.get("timestamp"),
                },
            )
            oid = sample.get("oid")
            if oid is not None:
                status_res = sender.query_order_status_by_oid(int(oid))
                status_summary = _summarize_order_status(status_res)
                log_event(
                    "open_order_status_sample",
                    {
                        "oid": oid,
                        "status": status_summary.get("status"),
                        "order_status": status_summary.get("order_status"),
                        "error": status_summary.get("error"),
                        "order_status_raw": status_res.get("response")
                        if isinstance(status_res, dict)
                        else status_res,
                    },
                )
            logged_open_orders_sample = True
        for side in ("buy", "sell"):
            prev = prev_active.get(side)
            current = active_orders.get(side)
            prev_oid = prev.get("exchange_id") if prev else None
            if current is None:
                if prev_oid is not None and last_missing_oid_by_side.get(side) != prev_oid:
                    status_res = sender.query_order_status_by_oid(int(prev_oid))
                    status_summary = _summarize_order_status(status_res)
                    log_event(
                        "open_order_missing",
                        {
                            "side": side,
                            "exchange_id": prev_oid,
                            "status": status_summary.get("status"),
                            "order_status": status_summary.get("order_status"),
                            "error": status_summary.get("error"),
                            "order_status_raw": status_res.get("response")
                            if isinstance(status_res, dict)
                            else status_res,
                        },
                    )
                    last_missing_oid_by_side[side] = prev_oid
            else:
                last_missing_oid_by_side[side] = None

    def cancel_open_orders_until_clear(reason: str, now_ms: int) -> None:
        nonlocal batch_id, last_batch_ts, needs_open_orders_sync, active_orders
        if not sender.enabled:
            return
        wait_s = open_orders_cancel_wait_ms / 1000.0 if open_orders_cancel_wait_ms > 0 else 0.0
        for attempt in range(1, open_orders_cancel_max_attempts + 1):
            try:
                open_orders = sender.fetch_open_orders()
            except Exception as exc:  # noqa: BLE001
                log_event(
                    "open_orders_cancel_error",
                    {"reason": reason, "attempt": attempt, "error": str(exc)},
                )
                return
            cancels = []
            for order in open_orders:
                if not isinstance(order, dict):
                    continue
                if sender.coin and order.get("coin") != sender.coin:
                    continue
                oid = order.get("oid")
                if oid is None:
                    continue
                side_raw = str(order.get("side") or "").upper()
                side = None
                if side_raw in {"B", "BUY", "BID"}:
                    side = "buy"
                elif side_raw in {"A", "SELL", "ASK"}:
                    side = "sell"
                cancels.append({"exchange_id": oid, "side": side})
            if not cancels:
                active_orders["buy"] = None
                active_orders["sell"] = None
                needs_open_orders_sync = True
                sync_open_orders("open_orders_cleared", now_ms, force=True)
                log_event(
                    "open_orders_cleared",
                    {"reason": reason, "attempt": attempt - 1, "open_orders_count": 0},
                )
                return
            log_event(
                "open_orders_cancel_attempt",
                {"reason": reason, "attempt": attempt, "open_orders_count": len(cancels)},
            )
            batch_id += 1
            send_statuses = sender.send_cancel_batch(cancels)
            rate_limited = False
            for order, send_status in zip(cancels, send_statuses):
                _write_jsonl(
                    orders_log,
                    {
                        "ts_ms": _now_ms(),
                        "action": "cancel",
                        "side": order.get("side"),
                        "px": None,
                        "sz": None,
                        "post_only": True,
                        "client_oid": None,
                        "exchange_id": order.get("exchange_id"),
                        "batch_id": batch_id,
                        "reason": f"{reason}_cleanup",
                        "status": send_status.get("status"),
                        "status_kind": send_status.get("status_kind"),
                        "exchange_resp_status": send_status.get("exchange_resp_status"),
                        "exchange_status_raw": send_status.get("raw_status"),
                        "error": send_status.get("error"),
                    },
                )
                if _is_rate_limit_error(send_status.get("error")):
                    rate_limited = True
            needs_open_orders_sync = True
            sync_open_orders("open_orders_cancel", _now_ms(), force=True)
            last_batch_ts = now_ms
            if rate_limited:
                log_event(
                    "open_orders_cancel_rate_limited",
                    {"reason": reason, "attempt": attempt},
                )
                return
            if wait_s:
                time.sleep(wait_s)
        log_event(
            "open_orders_cancel_incomplete",
            {"reason": reason, "attempts": open_orders_cancel_max_attempts},
        )

    def sync_fills(reason: str, now_ms: int) -> None:
        nonlocal last_fills_sync_ts, fills_cursor_ms, gateb_fills_count, gateb_notional_sum
        if fills_sync_ms <= 0:
            return
        if now_ms - last_fills_sync_ts < fills_sync_ms:
            return
        if not sender.open_orders_user:
            log_event(
                "fills_sync_error",
                {"reason": reason, "error": "missing_open_orders_user"},
            )
            last_fills_sync_ts = now_ms
            return
        start_ms = max(0, fills_cursor_ms)
        req_start_ms = int(start_ms)
        new_count = 0
        skipped_before_start = 0
        gateb_new_fills = 0
        gateb_new_notional = 0.0
        resp_len = 0
        pages = 0
        missing_tid = 0
        while pages < fills_max_pages:
            body = {
                "type": "userFillsByTime",
                "user": sender.open_orders_user,
                "startTime": int(start_ms),
                "aggregateByTime": bool(fills_aggregate_by_time),
            }
            try:
                res = _post_json(base_url, body)
            except Exception as exc:  # noqa: BLE001
                log_event(
                    "fills_sync_error",
                    {"reason": reason, "error": str(exc), "start_time": int(start_ms)},
                )
                last_fills_sync_ts = now_ms
                return
            if not isinstance(res, list):
                log_event(
                    "fills_sync_error",
                    {
                        "reason": reason,
                        "error": "invalid_response",
                        "response": res,
                        "start_time": int(start_ms),
                    },
                )
                last_fills_sync_ts = now_ms
                return
            resp_len += len(res)
            pages += 1
            if not res:
                break
            max_time = None
            for fill in res:
                if not isinstance(fill, dict):
                    continue
                time_ms = _fill_time_ms(fill)
                if time_ms is not None and (max_time is None or time_ms > max_time):
                    max_time = time_ms
                if sender.coin and fill.get("coin") != sender.coin:
                    continue
                key = _fill_key(fill, time_ms)
                if key is None:
                    missing_tid += 1
                    continue
                if key in seen_fill_ids:
                    continue
                seen_fill_ids.add(key)
                seen_fill_queue.append(key)
                if len(seen_fill_queue) > max_seen_fills:
                    old = seen_fill_queue.popleft()
                    seen_fill_ids.discard(old)
                if run_start_ts_ms is not None and time_ms is not None and time_ms < run_start_ts_ms:
                    skipped_before_start += 1
                    continue
                _write_jsonl(fills_log, fill)
                new_count += 1
                px = _safe_float(fill.get("px"))
                sz = _safe_float(fill.get("sz"))
                if px is not None and sz is not None and px > 0 and sz > 0:
                    gateb_fills_count += 1
                    notional = px * sz
                    gateb_notional_sum += notional
                    gateb_new_fills += 1
                    gateb_new_notional += notional
            if max_time is None:
                break
            start_ms = int(max_time) + 1
            fills_cursor_ms = max(fills_cursor_ms, start_ms)
            if len(res) < 500:
                break
        last_fills_sync_ts = now_ms
        log_event(
            "fills_sync",
            {
                "reason": reason,
                "fills_sync_ok": True,
                "fills_req_start_time": req_start_ms,
                "fills_resp_len": resp_len,
                "fills_new_len": new_count,
                "fills_skipped_before_start": skipped_before_start,
                "fills_missing_tid": missing_tid,
                "fills_cursor_ms_after": int(fills_cursor_ms),
                "fills_pages": pages if resp_len else 0,
                "fills_user": sender.open_orders_user,
                "gateb_fills_count": gateb_fills_count,
                "gateb_notional_sum": gateb_notional_sum,
                "gateb_new_fills": gateb_new_fills,
                "gateb_new_notional": gateb_new_notional,
            },
        )
        try:
            fills_state = {
                "cursor_ms": int(fills_cursor_ms),
                "updated_at_ms": int(now_ms),
                "user": sender.open_orders_user,
                "coin": sender.coin,
                "base_url": base_url,
            }
            fills_state_path.write_text(
                json.dumps(fills_state, ensure_ascii=True) + "\n", encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            log_event(
                "fills_cursor_error",
                {"error": str(exc), "state_path": str(fills_state_path)},
            )

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
            "open_orders_user": sender.open_orders_user,
        },
    )
    runtime = resolved_config.setdefault("runtime", {})
    runtime.update(
        {
            "base_url": base_url,
            "fills_base_url": base_url,
            "network": "testnet" if "testnet" in base_url else "mainnet",
            "open_orders_user": sender.open_orders_user,
            "account_address": sender.account_address,
            "vault_address": sender.vault_address,
            "mode": mode,
        }
    )
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved_config, stream=fh, sort_keys=True)

    network_name = "testnet" if "testnet" in base_url else "mainnet"
    fills_state_path = (
        Path(log_dir)
        / f"fills_state_{network_name}_{_safe_state_name(sender.open_orders_user)}_{_safe_state_name(sender.coin)}.json"
    )
    if fills_state_path.exists():
        try:
            state = json.loads(fills_state_path.read_text(encoding="utf-8"))
            state_cursor = int(state.get("cursor_ms", 0)) if isinstance(state, dict) else 0
            if state_cursor > 0:
                prev_cursor = fills_cursor_ms
                fills_cursor_ms = max(state_cursor - fills_cursor_rewind_ms, 0)
                log_event(
                    "fills_cursor_loaded",
                    {
                        "cursor_ms": state_cursor,
                        "rewind_ms": fills_cursor_rewind_ms,
                        "cursor_ms_after": fills_cursor_ms,
                        "cursor_ms_before": prev_cursor,
                        "state_path": str(fills_state_path),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            log_event(
                "fills_cursor_error",
                {"error": str(exc), "state_path": str(fills_state_path)},
            )

    batch_id = 0
    last_batch_ts: Optional[int] = None
    batch_gap_ms = int(float(order_batch_sec) * 1000.0) if order_batch_sec is not None else 0

    active_orders: Dict[str, Optional[dict]] = {"buy": None, "sell": None}
    last_open_orders_sync_ts = 0
    needs_open_orders_sync = False
    logged_open_orders_sample = False
    spread_filter_skip_count = 0
    spread_filter_total_count = 0
    last_missing_oid_by_side: Dict[str, Optional[int]] = {"buy": None, "sell": None}
    prev_mid: Optional[float] = None
    prev_best_bid: Optional[float] = None
    prev_best_ask: Optional[float] = None
    prev_ts: Optional[int] = None
    prev_boost: Optional[bool] = None
    nonce_counter = _now_ms()
    last_quote_ts_ms: Dict[str, int] = {"buy": 0, "sell": 0}
    cooldown_until_ms = 0
    backoff_ms = rate_limit_backoff_ms
    last_send_ts_ms = 0

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
    seen_fill_ids: set[str] = set()
    seen_fill_queue: Deque[str] = deque()
    max_seen_fills = 5000
    poll_total = 0
    last_poll_ts: Optional[int] = None
    ws_connected = False
    ws_msg_total = 0
    ws_l2book_total = 0
    last_ws_msg_ts: Optional[int] = None
    prev_book_hash: Optional[str] = None
    last_fills_sync_ts = 0

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
        if open_orders_cancel_on_start:
            cancel_open_orders_until_clear("startup", _now_ms())
        else:
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
            needs_open_orders_sync = True

    start_time = time.monotonic()

    try:
        while True:
            if (
                (gateb_fills_min is not None or gateb_notional_min is not None)
                and (gateb_fills_min is None or gateb_fills_count >= gateb_fills_min)
                and (gateb_notional_min is None or gateb_notional_sum >= gateb_notional_min)
            ):
                log_event(
                    "shutdown",
                    {
                        "reason": "gateb_reached",
                        "gateb_fills_count": gateb_fills_count,
                        "gateb_notional_sum": gateb_notional_sum,
                        "gateb_fills_min": gateb_fills_min,
                        "gateb_notional_min": gateb_notional_min,
                    },
                )
                break
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
            if run_start_ts_ms is None:
                run_start_ts_ms = ts_ms
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
            spread_filter_active = False
            if min_book_spread_bps is not None and spread_bps is not None:
                spread_filter_total_count += 1
                if spread_bps < min_book_spread_bps:
                    spread_filter_active = True
                    spread_filter_skip_count += 1

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
            state["spread_filter_active"] = spread_filter_active
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
            effective_bid_spread_bps = _safe_float(state.get("strategy_bid_spread_bps"))
            effective_ask_spread_bps = _safe_float(state.get("strategy_ask_spread_bps"))

            if spread_filter_active:
                res["orders"] = []

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

            if raw_bid_px is not None and final_bid_px is not None and final_bid_px > raw_bid_px:
                tick = _price_tick_from_value(final_bid_px, price_decimals)
                if tick is not None:
                    final_bid_px = final_bid_px - tick
            if raw_ask_px is not None and final_ask_px is not None and final_ask_px < raw_ask_px:
                tick = _price_tick_from_value(final_ask_px, price_decimals)
                if tick is not None:
                    final_ask_px = final_ask_px + tick

            guard_ticks = int(extra.get("post_only_guard_ticks", 1))
            if guard_ticks < 0:
                guard_ticks = 0

            bid_tick = _price_tick_from_value(best_bid, price_decimals) or _price_tick_from_value(
                final_bid_px, price_decimals
            )
            ask_tick = _price_tick_from_value(best_ask, price_decimals) or _price_tick_from_value(
                final_ask_px, price_decimals
            )
            guard_tick = bid_tick or ask_tick
            skip_quote_cycle = False
            skip_reason = None
            if best_bid is None or best_ask is None:
                skip_quote_cycle = True
                skip_reason = "missing_best_bid_ask"
            elif guard_tick is None:
                skip_quote_cycle = True
                skip_reason = "missing_guard_tick"
            elif best_ask <= best_bid:
                skip_quote_cycle = True
                skip_reason = "crossed_or_stale_book"

            bid_guard = (guard_tick or 0.0) * guard_ticks
            ask_guard = (guard_tick or 0.0) * guard_ticks
            if not skip_quote_cycle:
                extra_bid = (guard_tick or 0.0) * extra_bid_ticks
                extra_ask = (guard_tick or 0.0) * extra_ask_ticks
                if final_bid_px is not None and best_ask is not None:
                    final_bid_px = min(final_bid_px, best_ask - bid_guard - extra_bid)
                if final_ask_px is not None and best_bid is not None:
                    ask_floor = best_bid + ask_guard
                    if best_ask is not None:
                        ask_floor = max(ask_floor, best_ask + extra_ask)
                    final_ask_px = max(final_ask_px, ask_floor)

                if final_bid_px is not None and best_ask is not None and final_bid_px >= best_ask:
                    tick = bid_tick or _price_tick_from_value(best_ask, price_decimals)
                    final_bid_px = best_ask - tick if tick is not None else None
                if final_ask_px is not None and best_bid is not None and final_ask_px <= best_bid:
                    tick = ask_tick or _price_tick_from_value(best_bid, price_decimals)
                    final_ask_px = best_bid + tick if tick is not None else None

            if final_bid_px is None:
                final_bid_sz = 0.0
            if final_ask_px is None:
                final_ask_sz = 0.0

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
                    "spread_filter_active": spread_filter_active,
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
                    "market_spread_bps": spread_bps,
                    "spread_filter_active": spread_filter_active,
                    "spread_filter_min_bps": min_book_spread_bps,
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
                if open_orders_cancel_on_reconnect:
                    cancel_open_orders_until_clear("reconnect", ts_ms)
                    last_batch_ts = ts_ms
                elif active_orders["buy"] is not None or active_orders["sell"] is not None:
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
                    needs_open_orders_sync = True
                    last_batch_ts = ts_ms
                prev_mid = mid
                prev_best_bid = best_bid
                prev_best_ask = best_ask
                prev_ts = ts_ms
                prev_boost = boost_active
                time.sleep(max(0.0, poll_interval_ms / 1000.0))
                continue

            if (
                open_orders_sync_ms > 0
                and (needs_open_orders_sync or ts_ms - last_open_orders_sync_ts >= open_orders_sync_ms)
            ):
                sync_open_orders("forced" if needs_open_orders_sync else "periodic", ts_ms)
            if fills_sync_ms > 0:
                sync_fills("periodic", ts_ms)
            if (
                (gateb_fills_min is not None or gateb_notional_min is not None)
                and (gateb_fills_min is None or gateb_fills_count >= gateb_fills_min)
                and (gateb_notional_min is None or gateb_notional_sum >= gateb_notional_min)
            ):
                log_event(
                    "shutdown",
                    {
                        "reason": "gateb_reached",
                        "gateb_fills_count": gateb_fills_count,
                        "gateb_notional_sum": gateb_notional_sum,
                        "gateb_fills_min": gateb_fills_min,
                        "gateb_notional_min": gateb_notional_min,
                    },
                )
                break
            if skip_quote_cycle and not (stop_triggered or halt_triggered):
                log_event(
                    "quote_skip",
                    {
                        "reason": skip_reason,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "guard_ticks": guard_ticks,
                        "tick": guard_tick,
                    },
                )
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
            if cooldown_until_ms and ts_ms < cooldown_until_ms:
                log_event(
                    "rate_limit_cooldown",
                    {
                        "cooldown_until_ms": cooldown_until_ms,
                        "remaining_ms": cooldown_until_ms - ts_ms,
                        "backoff_ms": backoff_ms,
                    },
                )
                prev_mid = mid
                prev_best_bid = best_bid
                prev_best_ask = best_ask
                prev_ts = ts_ms
                prev_boost = boost_active
                time.sleep(max(0.0, poll_interval_ms / 1000.0))
                continue

            force_open_orders_sync = False
            actions = []
            for side in ("buy", "sell"):
                target_px = final_bid_px if side == "buy" else final_ask_px
                target_sz = final_bid_sz if side == "buy" else final_ask_sz
                active = active_orders[side]
                active_exchange_id = active.get("exchange_id") if active else None
                action = None
                force_replenish = False
                if target_sz <= 0:
                    if active is not None:
                        if active_exchange_id is None:
                            log_event(
                                "quote_skip",
                                {
                                    "reason": "missing_exchange_id",
                                    "side": side,
                                    "action": "cancel",
                                },
                            )
                            continue
                        action = "cancel"
                elif active is None:
                    if target_px is not None:
                        action = "new"
                        force_replenish = True
                elif _changed(active.get("price"), target_px) or _changed(active.get("size"), target_sz):
                    if active_exchange_id is not None:
                        action = "replace"
                    else:
                        log_event(
                            "quote_skip",
                            {
                                "reason": "missing_exchange_id",
                                "side": side,
                                "action": "replace",
                            },
                        )
                        continue
                if action is None:
                    continue
                if spread_filter_active and action == "cancel":
                    force_replenish = True
                if min_quote_lifetime_ms > 0:
                    age_ms = ts_ms - last_quote_ts_ms[side]
                    if age_ms < min_quote_lifetime_ms and not force_replenish:
                        log_event(
                            "quote_skip",
                            {
                                "reason": "min_quote_lifetime",
                                "side": side,
                                "action": action,
                                "age_ms": age_ms,
                                "min_quote_lifetime_ms": min_quote_lifetime_ms,
                                "target_px": target_px,
                                "target_sz": target_sz,
                                "active_px": active.get("price") if active else None,
                                "active_sz": active.get("size") if active else None,
                                "replenish": force_replenish,
                            },
                        )
                        continue
                actions.append((action, side, target_px, target_sz, active, force_replenish))

            if actions:
                batch_action = None
                has_replenish = any(item[0] == "new" and item[-1] for item in actions)
                allow_cancel_first = stop_triggered or halt_triggered or bool(state.get("pull_triggered"))
                if has_replenish:
                    batch_action = "new"
                else:
                    candidates = ("cancel", "replace", "new") if allow_cancel_first else (
                        "new",
                        "replace",
                        "cancel",
                    )
                    for candidate in candidates:
                        if any(action == candidate for action, *_ in actions):
                            batch_action = candidate
                            break
                batch_actions = [item for item in actions if item[0] == batch_action] if batch_action else []
                if batch_actions:
                    if min_send_interval_ms > 0 and last_send_ts_ms:
                        age_ms = ts_ms - last_send_ts_ms
                        if age_ms < min_send_interval_ms:
                            log_event(
                                "send_skip",
                                {
                                    "reason": "min_send_interval",
                                    "age_ms": age_ms,
                                    "min_send_interval_ms": min_send_interval_ms,
                                    "batch_action": batch_action,
                                },
                            )
                            prev_mid = mid
                            prev_best_bid = best_bid
                            prev_best_ask = best_ask
                            prev_ts = ts_ms
                            prev_boost = boost_active
                            time.sleep(max(0.0, poll_interval_ms / 1000.0))
                            continue
                    batch_id += 1
                    reason = "price_or_size_change"
                    if prev_boost is not None and prev_boost != boost_active:
                        reason = "boost_toggle"
                    rate_limited = False
                    any_sent = False
                    if batch_action == "new":
                        new_orders = []
                        for _, side, px, sz, _active, replenish in batch_actions:
                            client_oid = f"0x{os.urandom(16).hex()}"
                            effective_spread_bps = (
                                effective_bid_spread_bps if side == "buy" else effective_ask_spread_bps
                            )
                            new_orders.append(
                                {
                                    "side": side,
                                    "price": px,
                                    "size": sz,
                                    "client_oid": client_oid,
                                    "replenish": replenish,
                                    "effective_spread_bps": effective_spread_bps,
                                }
                            )
                        send_statuses = sender.send_new_batch(new_orders)
                        for order, send_status in zip(new_orders, send_statuses):
                            nonce_counter += 1
                            _write_jsonl(
                                orders_log,
                                {
                                    "ts_ms": ts_ms,
                                    "action": batch_action,
                                    "side": order["side"],
                                    "px": order["price"],
                                    "sz": order["size"],
                                    "post_only": True,
                                    "client_oid": order["client_oid"],
                                    "effective_spread_bps": order.get("effective_spread_bps"),
                                    "batch_id": batch_id,
                                    "reason": reason,
                                    "crossed": crossed,
                                    "nonce": nonce_counter,
                                    "replenish": order.get("replenish", False),
                                    "status": send_status.get("status"),
                                    "status_kind": send_status.get("status_kind"),
                                    "exchange_resp_status": send_status.get("exchange_resp_status"),
                                    "exchange_status_raw": send_status.get("raw_status"),
                                    "error": send_status.get("error"),
                                    "exchange_id": send_status.get("exchange_id"),
                                },
                            )
                            status_kind = send_status.get("status_kind")
                            if send_status.get("status") != "error":
                                any_sent = True
                                if status_kind in {"resting", "partial"}:
                                    active_orders[order["side"]] = {
                                        "price": order["price"],
                                        "size": order["size"],
                                        "client_oid": order["client_oid"],
                                        "exchange_id": send_status.get("exchange_id"),
                                    }
                                else:
                                    active_orders[order["side"]] = None
                                if send_status.get("exchange_id") is None:
                                    needs_open_orders_sync = True
                                if order_status_check:
                                    status_res = sender.query_order_status(order["client_oid"])
                                    status_summary = _summarize_order_status(status_res)
                                    log_event(
                                        "order_status",
                                        {
                                            "side": order["side"],
                                            "client_oid": order["client_oid"],
                                            "exchange_id": send_status.get("exchange_id"),
                                            "status": status_summary.get("status"),
                                            "order_status": status_summary.get("order_status"),
                                            "error": status_summary.get("error"),
                                            "order_status_raw": status_res.get("response")
                                            if isinstance(status_res, dict)
                                            else status_res,
                                        },
                                    )
                            if _is_rate_limit_error(send_status.get("error")):
                                rate_limited = True
                        for order in new_orders:
                            last_quote_ts_ms[order["side"]] = ts_ms
                    elif batch_action == "cancel":
                        cancel_orders = []
                        for _, side, _px, _sz, active, _replenish in batch_actions:
                            exchange_id = active.get("exchange_id") if active else None
                            cancel_orders.append({"side": side, "exchange_id": exchange_id})
                        send_statuses = sender.send_cancel_batch(cancel_orders)
                        for order, send_status in zip(cancel_orders, send_statuses):
                            nonce_counter += 1
                            _write_jsonl(
                                orders_log,
                                {
                                    "ts_ms": ts_ms,
                                    "action": batch_action,
                                    "side": order["side"],
                                    "px": None,
                                    "sz": None,
                                    "post_only": True,
                                    "client_oid": None,
                                    "exchange_id": order["exchange_id"],
                                    "batch_id": batch_id,
                                    "reason": reason,
                                    "crossed": crossed,
                                    "nonce": nonce_counter,
                                    "status": send_status.get("status"),
                                    "status_kind": send_status.get("status_kind"),
                                    "exchange_resp_status": send_status.get("exchange_resp_status"),
                                    "exchange_status_raw": send_status.get("raw_status"),
                                    "error": send_status.get("error"),
                                },
                            )
                            if send_status.get("status") != "error":
                                any_sent = True
                                active_orders[order["side"]] = None
                            else:
                                needs_open_orders_sync = True
                            if _is_rate_limit_error(send_status.get("error")):
                                rate_limited = True
                        for order in cancel_orders:
                            last_quote_ts_ms[order["side"]] = ts_ms
                    else:
                        modify_orders = []
                        for _, side, px, sz, active, replenish in batch_actions:
                            client_oid = active.get("client_oid") if active else None
                            exchange_id = active.get("exchange_id") if active else None
                            effective_spread_bps = (
                                effective_bid_spread_bps if side == "buy" else effective_ask_spread_bps
                            )
                            modify_orders.append(
                                {
                                    "side": side,
                                    "price": px,
                                    "size": sz,
                                    "client_oid": client_oid,
                                    "exchange_id": exchange_id,
                                    "replenish": replenish,
                                    "effective_spread_bps": effective_spread_bps,
                                }
                            )
                        send_statuses = sender.send_modify_batch(modify_orders)
                        for order, send_status in zip(modify_orders, send_statuses):
                            nonce_counter += 1
                            _write_jsonl(
                                orders_log,
                                {
                                    "ts_ms": ts_ms,
                                    "action": batch_action,
                                    "side": order["side"],
                                    "px": order["price"],
                                    "sz": order["size"],
                                    "post_only": True,
                                    "client_oid": order["client_oid"],
                                    "effective_spread_bps": order.get("effective_spread_bps"),
                                    "batch_id": batch_id,
                                    "reason": reason,
                                    "crossed": crossed,
                                    "nonce": nonce_counter,
                                    "replenish": order.get("replenish", False),
                                    "status": send_status.get("status"),
                                    "status_kind": send_status.get("status_kind"),
                                    "exchange_resp_status": send_status.get("exchange_resp_status"),
                                    "exchange_status_raw": send_status.get("raw_status"),
                                    "error": send_status.get("error"),
                                    "exchange_id": send_status.get("exchange_id"),
                                },
                            )
                            if send_status.get("status") != "error":
                                any_sent = True
                                active_orders[order["side"]] = {
                                    "price": order["price"],
                                    "size": order["size"],
                                    "client_oid": order["client_oid"],
                                    "exchange_id": None,
                                }
                                needs_open_orders_sync = True
                                force_open_orders_sync = True
                            elif send_status.get("error") and "Cannot modify canceled or filled order" in str(
                                send_status.get("error")
                            ):
                                active_orders[order["side"]] = None
                                needs_open_orders_sync = True
                                force_open_orders_sync = True
                            if _is_rate_limit_error(send_status.get("error")):
                                rate_limited = True
                        for order in modify_orders:
                            last_quote_ts_ms[order["side"]] = ts_ms
                    if rate_limited and rate_limit_backoff_ms > 0:
                        cooldown_ms = backoff_ms
                        cooldown_until_ms = ts_ms + cooldown_ms
                        backoff_ms = min(backoff_ms * 2, rate_limit_backoff_max_ms)
                        log_event(
                            "rate_limited",
                            {
                                "cooldown_ms": cooldown_ms,
                                "cooldown_until_ms": cooldown_until_ms,
                                "next_backoff_ms": backoff_ms,
                            },
                        )
                    elif any_sent and rate_limit_backoff_ms > 0:
                        backoff_ms = rate_limit_backoff_ms
                    if force_open_orders_sync:
                        sync_open_orders("post_send", ts_ms, force=True)
                    last_send_ts_ms = ts_ms
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
