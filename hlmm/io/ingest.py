from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.data import (
    AssetCtxEvent,
    BboEvent,
    BookEvent,
    TradeEvent,
    UserFillEvent,
    UserFundingEvent,
    BaseEvent,
)

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - zstd は任意依存
    zstd = None


def _to_int(record: dict, key: str) -> int:
    if key not in record:
        raise ValueError(f"{key} は必須です")
    value = record[key]
    if not isinstance(value, int):
        raise ValueError(f"{key} は int である必要があります")
    return value


def _to_str(record: dict, key: str) -> str:
    if key not in record:
        raise ValueError(f"{key} は必須です")
    value = record[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} は非空の文字列である必要があります")
    return value.strip()


def parse_event(record: dict) -> BaseEvent:
    channel = record.get("channel") or record.get("type")
    if not channel:
        raise ValueError("channel/type が必要です")
    recv_ts_ms = _to_int(record, "recv_ts_ms")
    symbol = _to_str(record, "symbol") if "symbol" in record else ""
    event_id = record.get("event_id") or f"{channel}:{symbol}:{record.get('ts_ms', recv_ts_ms)}"

    if channel == "l2Book":
        bids = record.get("bids")
        asks = record.get("asks")
        ts_ms = record.get("ts_ms")
        if ts_ms is None:
            raise ValueError("l2Book には ts_ms が必要です")
        return BookEvent(
            event_id=str(event_id),
            recv_ts_ms=recv_ts_ms,
            symbol=symbol,
            bids=bids,
            asks=asks,
            ts_ms=ts_ms,
        )
    if channel == "bbo":
        return BboEvent(
            event_id=str(event_id),
            recv_ts_ms=recv_ts_ms,
            symbol=symbol,
            bid_px=record.get("bid_px"),
            bid_sz=record.get("bid_sz"),
            ask_px=record.get("ask_px"),
            ask_sz=record.get("ask_sz"),
            ts_ms=record.get("ts_ms"),
        )
    if channel == "trades":
        ts_ms = _to_int(record, "ts_ms")
        trade_id = _to_str(record, "trade_id")
        return TradeEvent(
            event_id=str(event_id),
            recv_ts_ms=recv_ts_ms,
            symbol=symbol,
            px=record.get("px"),
            sz=record.get("sz"),
            side=_to_str(record, "side"),
            trade_id=trade_id,
            ts_ms=ts_ms,
        )
    if channel == "activeAssetCtx":
        return AssetCtxEvent(
            event_id=str(event_id),
            recv_ts_ms=recv_ts_ms,
            symbol=symbol,
            status=_to_str(record, "status"),
            ts_ms=record.get("ts_ms"),
            metadata=record.get("metadata"),
        )
    if channel == "userFills":
        return UserFillEvent(
            event_id=str(event_id),
            recv_ts_ms=recv_ts_ms,
            symbol=symbol,
            px=record.get("px"),
            sz=record.get("sz"),
            side=_to_str(record, "side"),
            fee=record.get("fee"),
            trade_id=_to_str(record, "trade_id"),
            ts_ms=_to_int(record, "ts_ms"),
        )
    if channel == "userFundings":
        return UserFundingEvent(
            event_id=str(event_id),
            recv_ts_ms=recv_ts_ms,
            symbol=symbol,
            amount=record.get("amount"),
            rate=record.get("rate"),
            ts_ms=_to_int(record, "ts_ms"),
            funding_id=_to_str(record, "funding_id"),
        )
    raise ValueError(f"未対応の channel: {channel}")


def iter_json_lines(path: Path) -> Iterable[dict]:
    if path.suffix == ".zst":
        if zstd is None:
            raise RuntimeError("zstandard がインストールされていません (.zst を読むには必要)")
        with path.open("rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def parse_events_from_file(path: str | Path) -> List[BaseEvent]:
    events: List[BaseEvent] = []
    for record in iter_json_lines(Path(path)):
        events.append(parse_event(record))
    return events


def dedupe_events(events: Iterable[BaseEvent]) -> List[BaseEvent]:
    seen = {}
    for ev in events:
        key = ev.dedupe_key
        if key not in seen:
            seen[key] = ev
    # 決定的な並びにするためキー順でソート
    return [seen[k] for k in sorted(seen.keys())]


def _decimal_to_str(val):
    try:
        from decimal import Decimal
    except Exception:  # pragma: no cover
        Decimal = None
    if isinstance(val, Decimal):
        return format(val, "f")
    return val


def event_to_record(event: BaseEvent) -> dict:
    data = {"event_type": event.event_type, "event_id": event.event_id, "recv_ts_ms": event.recv_ts_ms}
    if isinstance(event, BookEvent):
        data.update(
            {
                "symbol": event.symbol,
                "bids": [[_decimal_to_str(px), _decimal_to_str(sz)] for px, sz in event.bids],
                "asks": [[_decimal_to_str(px), _decimal_to_str(sz)] for px, sz in event.asks],
                "ts_ms": event.ts_ms,
            }
        )
    elif isinstance(event, BboEvent):
        data.update(
            {
                "symbol": event.symbol,
                "bid_px": _decimal_to_str(event.bid_px),
                "bid_sz": _decimal_to_str(event.bid_sz),
                "ask_px": _decimal_to_str(event.ask_px),
                "ask_sz": _decimal_to_str(event.ask_sz),
                "ts_ms": event.ts_ms,
            }
        )
    elif isinstance(event, TradeEvent):
        data.update(
            {
                "symbol": event.symbol,
                "px": _decimal_to_str(event.px),
                "sz": _decimal_to_str(event.sz),
                "side": event.side,
                "trade_id": event.trade_id,
                "ts_ms": event.ts_ms,
            }
        )
    elif isinstance(event, AssetCtxEvent):
        data.update(
            {
                "symbol": event.symbol,
                "status": event.status,
                "ts_ms": event.ts_ms,
                "metadata": event.metadata,
            }
        )
    elif isinstance(event, UserFillEvent):
        data.update(
            {
                "symbol": event.symbol,
                "px": _decimal_to_str(event.px),
                "sz": _decimal_to_str(event.sz),
                "side": event.side,
                "fee": _decimal_to_str(event.fee),
                "trade_id": event.trade_id,
                "ts_ms": event.ts_ms,
            }
        )
    elif isinstance(event, UserFundingEvent):
        data.update(
            {
                "symbol": event.symbol,
                "amount": _decimal_to_str(event.amount),
                "rate": _decimal_to_str(event.rate),
                "ts_ms": event.ts_ms,
                "funding_id": event.funding_id,
            }
        )
    else:  # pragma: no cover
        raise ValueError(f"未対応イベント: {event}")
    return data


def save_events_parquet(events: Iterable[BaseEvent], output_path: str | Path) -> Path:
    records = [event_to_record(ev) for ev in events]
    table = pa.Table.from_pylist(records)
    output = Path(output_path)
    pq.write_table(table, output)
    return output


def convert_raw_to_parquet(raw_path: str | Path, output_path: str | Path) -> Path:
    events = parse_events_from_file(raw_path)
    deduped = dedupe_events(events)
    return save_events_parquet(deduped, output_path)
