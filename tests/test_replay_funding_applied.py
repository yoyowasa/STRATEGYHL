
from hlmm.mm import apply_replay


def test_replay_funding_applied(tmp_path):
    fills = [
        {"ts_ms": 0, "side": "buy", "px": 100.0, "sz": 1.0, "fee": 0.1, "trade_id": "t1"},
    ]
    fundings = [
        {"ts_ms": 1000, "amount": 0.5},
        {"ts_ms": 2000, "amount": -0.25},
    ]
    ledger = apply_replay(fills, fundings)
    # fundingが順に加算されている
    assert ledger.to_pydict()["funding"][-1] == 0.25
