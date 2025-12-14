from decimal import Decimal

import pytest

from hlmm.mm import round_size


@pytest.mark.parametrize(
    "sz,decimals,expected",
    [
        (Decimal("1.234567"), 2, Decimal("1.23")),
        (Decimal("1.235"), 2, Decimal("1.24")),  # 端数は四捨五入
        (Decimal("0.000001234"), 6, Decimal("0.000001")),
        (Decimal("123456"), 0, Decimal("123456")),
    ],
)
def test_round_size_basic(sz, decimals, expected):
    assert round_size(sz, decimals) == expected


def test_round_size_deterministic():
    sz = Decimal("0.12345")
    out1 = round_size(sz, 3)
    out2 = round_size(sz, 3)
    assert out1 == out2
