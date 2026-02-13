from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


def _is_nan(val) -> bool:
    return isinstance(val, float) and math.isnan(val)


def _get_mid(record: Mapping[str, object]):
    mid = record.get("mid")
    if mid is None or _is_nan(mid):
        return None
    return float(mid)


def _future_mid(records: Sequence[Mapping[str, object]], start_idx: int, target_ts: int):
    for j in range(start_idx + 1, len(records)):
        ts = records[j].get("block_ts_ms")
        mid = _get_mid(records[j])
        if ts is None:
            continue
        if ts >= target_ts:
            if mid is None:
                return math.nan
            return mid
    return math.nan


def compute_targets(records: List[Mapping[str, object]], horizons_sec: Sequence[int]) -> List[dict]:
    horizons_ms = [int(h) * 1000 for h in horizons_sec]
    out: List[dict] = []
    for i, rec in enumerate(records):
        base = dict(rec)
        mid_now = _get_mid(rec)
        ts_now = rec.get("block_ts_ms")
        for h_sec, h_ms in zip(horizons_sec, horizons_ms):
            key_ret = f"y_ret_{h_sec}s"
            key_mark = f"y_markout_{h_sec}s"
            if mid_now is None or ts_now is None:
                base[key_ret] = math.nan
                base[key_mark] = math.nan
                continue
            fut_mid = _future_mid(records, i, ts_now + h_ms)
            if math.isnan(fut_mid):
                base[key_ret] = math.nan
                base[key_mark] = math.nan
                continue
            mark = fut_mid - mid_now
            base[key_mark] = mark
            if mid_now == 0:
                base[key_ret] = math.nan
            else:
                base[key_ret] = mark / mid_now
        out.append(base)
    return out


def assign_splits(records: List[dict], train_ratio: float = 0.7, valid_ratio: float = 0.15):
    if not records:
        return records, {"train": None, "valid": None, "test": None}
    times = [r.get("block_ts_ms") for r in records if r.get("block_ts_ms") is not None]
    if not times:
        return records, {"train": None, "valid": None, "test": None}
    start, end = min(times), max(times)
    span = max(end - start, 1)
    train_end = start + int(span * train_ratio)
    valid_end = start + int(span * (train_ratio + valid_ratio))
    splits = {
        "train": {"start": start, "end": train_end},
        "valid": {"start": train_end, "end": valid_end},
        "test": {"start": valid_end, "end": end},
    }
    for r in records:
        ts = r.get("block_ts_ms")
        split = "test"
        if ts is None:
            split = "unknown"
        elif ts < train_end:
            split = "train"
        elif ts < valid_end:
            split = "valid"
        r["split"] = split
    return records, splits


def build_dataset(
    features_path: str | Path,
    dataset_out: str | Path,
    splits_out: str | Path,
    horizons_sec: Sequence[int] = (1, 5, 15, 60),
) -> tuple[Path, Path]:
    table = pq.read_table(features_path)
    records = table.to_pylist()
    records_with_targets = compute_targets(records, horizons_sec)
    records_with_split, splits = assign_splits(records_with_targets)
    out_table = pa.Table.from_pylist(records_with_split)
    dataset_path = Path(dataset_out)
    splits_path = Path(splits_out)
    pq.write_table(out_table, dataset_path)
    splits_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_path, splits_path
