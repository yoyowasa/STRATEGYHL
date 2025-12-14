import pytest

from hlmm.features import compute_features


def test_signed_volume_and_basis():
    block = {
        "block_ts_ms": 1000,
        "book_event_id": "b1",
        "book_top": {"bid_px": "100", "bid_sz": "2", "ask_px": "101", "ask_sz": "1"},
        "trade_bucket": [
            {"px": "100", "sz": "2", "side": "buy", "trade_id": "t1", "ts_ms": 990},
            {"px": "101", "sz": "1", "side": "sell", "trade_id": "t2", "ts_ms": 995},
        ],
        "missing_book": False,
        "missing_trades": False,
    }
    feats = compute_features([block])[0]
    assert feats["missing_signed_volume"] is False
    assert feats["signed_volume"] == pytest.approx(1.0)  # 2 - 1

    # basis_bps = (last_px - mid)/mid*1e4
    expected_basis = (101 - 100.5) / 100.5 * 10_000
    assert feats["basis_bps"] == pytest.approx(expected_basis)
    assert feats["missing_basis_bps"] is False
