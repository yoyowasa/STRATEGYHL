from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable, List, Sequence, Tuple


def _ensure_str(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} は非空の文字列である必要があります")
    return value.strip()


def _ensure_int(value: int | None, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} は int である必要があります")
    return value


def _to_decimal(value: object, name: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} を Decimal に変換できません: {value}") from exc


def _ensure_decimal_tuple_pairs(
    levels: Iterable[Sequence[object]] | None, name: str
) -> List[Tuple[Decimal, Decimal]]:
    result: List[Tuple[Decimal, Decimal]] = []
    if levels is None:
        raise ValueError(f"{name} は None ではありません")
    for idx, item in enumerate(levels):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name}[{idx}] は (px, sz) の2要素シーケンスである必要があります")
        px = _to_decimal(item[0], f"{name}[{idx}].px")
        sz = _to_decimal(item[1], f"{name}[{idx}].sz")
        result.append((px, sz))
    return result


@dataclass(frozen=True)
class BaseEvent:
    EVENT_TYPE = "base"
    event_id: str
    recv_ts_ms: int

    def __post_init__(self) -> None:
        _ensure_str(self.event_id, "event_id")
        _ensure_int(self.recv_ts_ms, "recv_ts_ms")

    @property
    def dedupe_key(self) -> str:
        """重複排除用キー。"""
        return self.event_id

    @property
    def event_type(self) -> str:
        return self.EVENT_TYPE


@dataclass(frozen=True)
class BookEvent(BaseEvent):
    EVENT_TYPE = "book"
    symbol: str
    bids: List[Tuple[Decimal, Decimal]] = field(default_factory=list)
    asks: List[Tuple[Decimal, Decimal]] = field(default_factory=list)
    ts_ms: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _ensure_str(self.symbol, "symbol")
        _ensure_int(self.recv_ts_ms, "recv_ts_ms")
        object.__setattr__(self, "bids", _ensure_decimal_tuple_pairs(self.bids, "bids"))
        object.__setattr__(self, "asks", _ensure_decimal_tuple_pairs(self.asks, "asks"))
        if self.ts_ms is not None:
            _ensure_int(self.ts_ms, "ts_ms")


@dataclass(frozen=True)
class BboEvent(BaseEvent):
    EVENT_TYPE = "bbo"
    symbol: str
    bid_px: Decimal
    bid_sz: Decimal
    ask_px: Decimal
    ask_sz: Decimal
    ts_ms: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _ensure_str(self.symbol, "symbol")
        object.__setattr__(self, "bid_px", _to_decimal(self.bid_px, "bid_px"))
        object.__setattr__(self, "bid_sz", _to_decimal(self.bid_sz, "bid_sz"))
        object.__setattr__(self, "ask_px", _to_decimal(self.ask_px, "ask_px"))
        object.__setattr__(self, "ask_sz", _to_decimal(self.ask_sz, "ask_sz"))
        if self.ts_ms is not None:
            _ensure_int(self.ts_ms, "ts_ms")


@dataclass(frozen=True)
class TradeEvent(BaseEvent):
    EVENT_TYPE = "trade"
    symbol: str
    px: Decimal
    sz: Decimal
    side: str
    trade_id: str
    ts_ms: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _ensure_str(self.symbol, "symbol")
        _ensure_str(self.side, "side")
        _ensure_str(self.trade_id, "trade_id")
        object.__setattr__(self, "px", _to_decimal(self.px, "px"))
        object.__setattr__(self, "sz", _to_decimal(self.sz, "sz"))
        _ensure_int(self.ts_ms, "ts_ms")

    @property
    def dedupe_key(self) -> str:
        # symbol + trade_id + ts_ms が重複排除キーとして機能
        return f"{self.symbol}:{self.trade_id}:{self.ts_ms}"


@dataclass(frozen=True)
class AssetCtxEvent(BaseEvent):
    EVENT_TYPE = "asset_ctx"
    symbol: str
    status: str
    ts_ms: int | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _ensure_str(self.symbol, "symbol")
        _ensure_str(self.status, "status")
        if self.ts_ms is not None:
            _ensure_int(self.ts_ms, "ts_ms")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("metadata は dict である必要があります")


@dataclass(frozen=True)
class UserFillEvent(BaseEvent):
    EVENT_TYPE = "user_fill"
    symbol: str
    px: Decimal
    sz: Decimal
    side: str
    fee: Decimal
    trade_id: str
    ts_ms: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _ensure_str(self.symbol, "symbol")
        _ensure_str(self.side, "side")
        _ensure_str(self.trade_id, "trade_id")
        object.__setattr__(self, "px", _to_decimal(self.px, "px"))
        object.__setattr__(self, "sz", _to_decimal(self.sz, "sz"))
        object.__setattr__(self, "fee", _to_decimal(self.fee, "fee"))
        _ensure_int(self.ts_ms, "ts_ms")


@dataclass(frozen=True)
class UserFundingEvent(BaseEvent):
    EVENT_TYPE = "user_funding"
    symbol: str
    amount: Decimal
    rate: Decimal
    ts_ms: int
    funding_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _ensure_str(self.symbol, "symbol")
        _ensure_str(self.funding_id, "funding_id")
        object.__setattr__(self, "amount", _to_decimal(self.amount, "amount"))
        object.__setattr__(self, "rate", _to_decimal(self.rate, "rate"))
        _ensure_int(self.ts_ms, "ts_ms")
