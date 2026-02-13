from decimal import Decimal

import pytest

from hlmm.mm import round_price


@pytest.mark.parametrize(
    "px,expected",
    [
        (Decimal("123.456789"), Decimal("123.46")),  # 5有効桁へ
        (Decimal("12.345678"), Decimal("12.346")),  # 3小数で5有効桁
        (Decimal("0.0000123456"), Decimal("0.000012")),  # 小数桁上限でクリップ
        (Decimal("9999999"), Decimal("10000000")),  # 繰り上がり
    ],
)
def test_round_price_basic(px, expected):
    assert round_price(px) == expected


def test_round_price_respects_decimal_limit():
    px = Decimal("1.23456789")
    out = round_price(px, max_decimals=4)
    assert out == Decimal("1.2346")


def test_round_price_deterministic():
    px = Decimal("12.3456")
    out1 = round_price(px)
    out2 = round_price(px)
    assert out1 == out2
