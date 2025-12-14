from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.research import generate_report


def test_report_artifacts_exist(tmp_path: Path):
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

    out_dir = generate_report("run1", ledger_path, trades_path, reports_dir=tmp_path)
    assert (out_dir / "metrics.json").exists()
    plots_dir = out_dir / "plots"
    assert (plots_dir / "equity.png").exists()
    assert (plots_dir / "inventory.png").exists()
