import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hlmm.research import build_dataset


def test_targets_alignment(tmp_path: Path):
    records = [
        {
            "block_ts_ms": 0,
            "book_event_id": "b1",
            "mid": 100.0,
            "missing_mid": False,
        },
        {
            "block_ts_ms": 1000,
            "book_event_id": "b2",
            "mid": 110.0,
            "missing_mid": False,
        },
        {
            "block_ts_ms": 2000,
            "book_event_id": "b3",
            "mid": 120.0,
            "missing_mid": False,
        },
    ]
    features_path = tmp_path / "features.parquet"
    pq.write_table(pa.Table.from_pylist(records), features_path)

    dataset_path, splits_path = build_dataset(
        features_path=features_path,
        dataset_out=tmp_path / "dataset.parquet",
        splits_out=tmp_path / "splits.json",
        horizons_sec=[1],  # 1秒先
    )

    ds = pq.read_table(dataset_path).to_pylist()
    # 1秒ホライズンでは、各行は将来の mid を参照し、未来が無ければ NaN
    assert ds[0]["y_markout_1s"] == 10.0  # 100 -> 110
    assert ds[1]["y_markout_1s"] == 10.0  # 110 -> 120
    assert math.isnan(ds[2]["y_markout_1s"])  # 未来無し

    # split は時系列順で決定的
    assert splits_path.read_text(encoding="utf-8")  # ファイルが出力されていること
