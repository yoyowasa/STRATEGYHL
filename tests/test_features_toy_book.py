import math

import pytest

from hlmm.features import compute_features


def test_toy_book_features():
    block = {
        "block_ts_ms": 1000,
        "book_event_id": "b1",
        "book_top": {"bid_px": "100", "bid_sz": "2", "ask_px": "101", "ask_sz": "1"},
        "trade_bucket": [],
        "missing_book": False,
        "missing_trades": True,
    }
    feats = compute_features([block])[0]
    assert feats["missing_mid"] is False
    assert feats["missing_spread"] is False
    assert feats["missing_microprice"] is False
    assert feats["missing_micro_bias_bps"] is False
    assert feats["missing_imbalance"] is False
    assert feats["missing_signed_volume"] is True

    assert feats["mid"] == pytest.approx(100.5)
    assert feats["spread"] == pytest.approx(1.0)
    assert feats["microprice"] == pytest.approx((100 * 1 + 101 * 2) / 3.0)
    assert feats["micro_bias_bps"] == pytest.approx(((100 * 1 + 101 * 2) / 3.0 - 100.5) / 100.5 * 1e4)
    assert feats["imbalance"] == pytest.approx((2 - 1) / 3.0)
    assert math.isnan(feats["signed_volume"])
