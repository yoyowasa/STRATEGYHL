import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.research import compute_metrics


def test_metrics_schema(tmp_path: Path):
    ledger = [
        {"block_ts_ms": 0, "total_pnl": 0.0, "position": 0.0, "fees": 0.0, "rebates": 0.0, "funding": 0.0},
        {"block_ts_ms": 1000, "total_pnl": 1.0, "position": 1.0, "fees": 0.1, "rebates": 0.0, "funding": 0.0},
    ]
    trades = [
        {"block_ts_ms": 0, "side": "buy", "price": 100.0},
    ]
    ledger_path = tmp_path / "ledger.parquet"
    trades_path = tmp_path / "trades.parquet"
    pq.write_table(pa.Table.from_pylist(ledger), ledger_path)
    pq.write_table(pa.Table.from_pylist(trades), trades_path)

    metrics = compute_metrics(ledger_path, trades_path)
    for key in ["fill_rate", "pnl", "realized_spread_1s", "inventory", "fee_breakdown"]:
        assert key in metrics
    assert "mean" in metrics["inventory"]
    assert "fees" in metrics["fee_breakdown"]
