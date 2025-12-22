import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.mm import Order, run_mm_sim, simulate_blocks


def test_engine_runs_toy_blocks(tmp_path):
    blocks = [
        {
            "block_ts_ms": 1000,
            "book_event_id": "b1",
            "book_top": {"bid_px": 99.0, "bid_sz": 1, "ask_px": 101.0, "ask_sz": 1},
            "trade_bucket": [],
            "missing_book": False,
            "missing_trades": True,
        },
        {
            "block_ts_ms": 2000,
            "book_event_id": "b2",
            "book_top": {"bid_px": 100.0, "bid_sz": 1, "ask_px": 102.0, "ask_sz": 1},
            "trade_bucket": [],
            "missing_book": False,
            "missing_trades": True,
        },
    ]

    def strat(block, state):
        ts = block["block_ts_ms"]
        if ts == 1000:
            return [Order(side="buy", size=1)]
        if ts == 2000:
            return [Order(side="sell", size=1)]
        return []

    trades, ledger, orders = simulate_blocks(
        blocks, strategy=strat, taker_fee_bps=0.0, fill_model="upper", allow_top_fill=True
    )
    assert len(trades) == 2
    assert trades[0]["side"] == "buy"
    assert trades[1]["side"] == "sell"
    assert ledger[-1]["position"] == 0.0
    assert ledger[0]["block_ts_ms"] < ledger[1]["block_ts_ms"]

    blocks_path = tmp_path / "blocks.parquet"
    pq.write_table(pa.Table.from_pylist(blocks), blocks_path)
    out_dir = tmp_path / "out"
    trades_path, ledger_path, orders_path = run_mm_sim(
        blocks_path, out_dir=out_dir, taker_fee_bps=0.0, strategy=strat, allow_top_fill=True
    )
    assert trades_path.exists()
    assert ledger_path.exists()
    assert orders_path.exists()
    assert pq.read_table(trades_path).num_rows == 2
    assert pq.read_table(ledger_path).num_rows == len(blocks)
