import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.features import align_blocks, compute_features
from hlmm.io import convert_raw_to_parquet, event_to_record, parse_event, save_events_parquet
from hlmm.mm import run_mm_sim, run_replay
from hlmm.research import build_dataset
from hlmm.research.report import generate_report


def test_e2e_three_modes(tmp_path: Path):
    # 1) research: raw events -> blocks -> features -> dataset
    raw_events = [
        {
            "channel": "trades",
            "symbol": "BTC",
            "px": "100",
            "sz": "1",
            "side": "buy",
            "trade_id": "t1",
            "ts_ms": 0,
            "recv_ts_ms": 0,
        },
        {
            "channel": "bbo",
            "symbol": "BTC",
            "bid_px": "99",
            "bid_sz": "1",
            "ask_px": "101",
            "ask_sz": "1",
            "ts_ms": 0,
            "recv_ts_ms": 0,
            "event_id": "b1",
        },
    ]
    raw_path = tmp_path / "raw.parquet"
    save_events_parquet([parse_event(e) for e in raw_events], raw_path)

    # blocks
    events = [parse_event(e) for e in raw_events]
    books = [
        events[1],  # bboをbook代わり
    ]
    trades = [events[0]]
    blocks = align_blocks(books, trades)
    blocks_path = tmp_path / "blocks.parquet"
    pq.write_table(pa.Table.from_pylist(blocks), blocks_path)

    # features
    features = compute_features(blocks)
    features_path = tmp_path / "features.parquet"
    pq.write_table(pa.Table.from_pylist(features), features_path)

    # dataset
    dataset_path, splits_path = build_dataset(
        features_path, dataset_out=tmp_path / "dataset.parquet", splits_out=tmp_path / "splits.json"
    )
    assert dataset_path.exists()
    assert splits_path.exists()

    # 2) mm_sim upper/lower
    run_mm_sim(blocks_path, out_dir=tmp_path / "sim_upper", fill_model="upper")
    run_mm_sim(blocks_path, out_dir=tmp_path / "sim_lower", fill_model="lower")
    assert (tmp_path / "sim_upper" / "sim_trades.parquet").exists()
    assert (tmp_path / "sim_lower" / "sim_trades.parquet").exists()

    # 3) mm_replay from upper fills (reuse trades as fills)
    upper_fills = tmp_path / "sim_upper" / "sim_trades.parquet"
    ledger_replay = run_replay(upper_fills, fundings_path=None, out_dir=tmp_path, run_id="replay")
    assert ledger_replay.exists()
    # レポート生成
    out_dir = generate_report("replay", ledger_replay, upper_fills, reports_dir=tmp_path)
    assert (out_dir / "metrics.json").exists()
