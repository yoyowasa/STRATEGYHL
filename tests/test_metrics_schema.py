from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlmm.research import compute_metrics


def test_metrics_schema(tmp_path: Path):
    ledger = [
        {"block_ts_ms": 0, "total_pnl": 0.0, "position": 0.0, "fees": 0.0, "rebates": 0.0, "funding": 0.0},
        {"block_ts_ms": 1000, "total_pnl": 1.0, "position": 1.0, "fees": 0.1, "rebates": 0.0, "funding": 0.0},
    ]
    trades = [
        {"block_ts_ms": 0, "side": "buy", "price": 100.0, "size": 2.0},
    ]
    ledger_path = tmp_path / "ledger.parquet"
    trades_path = tmp_path / "trades.parquet"
    pq.write_table(pa.Table.from_pylist(ledger), ledger_path)
    pq.write_table(pa.Table.from_pylist(trades), trades_path)

    metrics = compute_metrics(ledger_path, trades_path)
    for key in [
        "num_blocks",
        "num_fills",
        "notional_traded",
        "fill_rate",
        "pnl",
        "max_drawdown",
        "stop_trigger_count",
        "pull_trigger_count",
        "halt_trigger_count",
        "stop_trigger_rate",
        "pull_trigger_rate",
        "halt_trigger_rate",
        "fills_when_stop",
        "fills_when_pull",
        "fills_when_halt_active",
        "notional_when_halt_active",
        "pnl_when_stop",
        "pnl_when_pull",
        "pnl_in_halt_window",
        "pnl_outside_halt_window",
        "realized_spread_1s",
        "realized_spread_5s",
        "markout_5s",
        "inventory",
        "fee_breakdown",
    ]:
        assert key in metrics
    assert metrics["num_blocks"] == 2
    assert metrics["num_fills"] == 1
    assert metrics["notional_traded"] == pytest.approx(200.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert metrics["stop_trigger_count"] == 0
    assert metrics["pull_trigger_count"] == 0
    assert metrics["halt_trigger_count"] == 0
    assert metrics["halt_trigger_rate"] == pytest.approx(0.0)
    assert metrics["fills_when_stop"] == 0
    assert metrics["fills_when_pull"] == 0
    assert metrics["fills_when_halt_active"] == 0
    assert metrics["notional_when_halt_active"] == pytest.approx(0.0)
    assert metrics["pnl_when_stop"] == pytest.approx(0.0)
    assert metrics["pnl_when_pull"] == pytest.approx(0.0)
    assert metrics["pnl_in_halt_window"] == pytest.approx(0.0)
    assert metrics["pnl_outside_halt_window"] == pytest.approx(1.0)
    assert "mean" in metrics["inventory"]
    assert "p95" in metrics["inventory"]
    assert "max_abs" in metrics["inventory"]
    assert "fees" in metrics["fee_breakdown"]
