import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.research import run_edge_screen


def test_edge_runner_deterministic(tmp_path: Path):
    records = []
    for i in range(10):
        records.append(
            {
                "block_ts_ms": i * 1000,
                "feature_a": float(i),
                "feature_b": float(10 - i),
                "y_ret_1s": float(i) * 0.01,
                "split": "train",
            }
        )
    dataset_path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist(records), dataset_path)
    splits = {"train": {"start": 0, "end": 9000}, "valid": None, "test": None}
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits), encoding="utf-8")

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    report1 = run_edge_screen(dataset_path, splits_path, out_dir=out1, target="y_ret_1s")
    report2 = run_edge_screen(dataset_path, splits_path, out_dir=out2, target="y_ret_1s")

    assert report1.read_text(encoding="utf-8") == report2.read_text(encoding="utf-8")

    png1 = next((out1 / "edge_plots").glob("*.png"))
    png2 = next((out2 / "edge_plots").glob("*.png"))
    assert png1.read_bytes() == png2.read_bytes()
