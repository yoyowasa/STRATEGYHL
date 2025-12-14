import json
from decimal import Decimal

from hlmm.features import align_blocks, save_blocks_parquet
from hlmm.io import convert_raw_to_parquet, event_to_record, parse_events_from_file


def test_parse_roundtrip(tmp_path):
    raw_lines = [
        {
            "channel": "trades",
            "symbol": "BTC",
            "px": "1.23",
            "sz": "4.56",
            "side": "buy",
            "trade_id": "t1",
            "ts_ms": 1000,
            "recv_ts_ms": 1001,
        },
        {
            "channel": "bbo",
            "symbol": "BTC",
            "bid_px": "1.2",
            "bid_sz": "3",
            "ask_px": "1.3",
            "ask_sz": "4",
            "ts_ms": 1000,
            "recv_ts_ms": 1002,
            "event_id": "b1",
        },
    ]
    raw_path = tmp_path / "raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for line in raw_lines:
            fh.write(json.dumps(line) + "\n")

    events = parse_events_from_file(raw_path)
    records = [event_to_record(ev) for ev in events]

    assert records[0]["event_type"] == "trade"
    assert records[0]["px"] == "1.23"
    assert records[1]["event_type"] == "bbo"
    assert records[1]["bid_px"] == "1.2"

    # parquet に書いても落ちず、決定的に出力される
    out1 = convert_raw_to_parquet(raw_path, tmp_path / "events1.parquet")
    out2 = convert_raw_to_parquet(raw_path, tmp_path / "events2.parquet")
    assert out1.read_bytes() == out2.read_bytes()

    # ブロックアライン後の Parquet も決定的
    from hlmm.data import BookEvent

    # テスト用に簡易な book イベントを生成
    books = [
        BookEvent(
            event_id="b1",
            recv_ts_ms=events[0].recv_ts_ms,
            symbol="BTC",
            bids=[(Decimal("1.2"), Decimal("1"))],
            asks=[(Decimal("1.3"), Decimal("1"))],
            ts_ms=events[0].ts_ms,
        )
    ]
    trades = [ev for ev in events if ev.event_type == "trade"]
    blocks = align_blocks(books, trades)
    out_block1 = save_blocks_parquet(blocks, tmp_path / "blocks1.parquet")
    out_block2 = save_blocks_parquet(blocks, tmp_path / "blocks2.parquet")
    assert out_block1.read_bytes() == out_block2.read_bytes()
