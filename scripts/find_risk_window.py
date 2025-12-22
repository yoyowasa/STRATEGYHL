from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow.parquet as pq


def _to_float_array(values: Iterable[object]) -> np.ndarray:
    out = []
    for v in values:
        if v is None:
            out.append(np.nan)
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


def _pnl_delta(total_pnl: np.ndarray) -> np.ndarray:
    if total_pnl.size == 0:
        return np.zeros(0, dtype=float)
    prev = np.concatenate(([0.0], total_pnl[:-1]))
    return total_pnl - prev


def _abs_mid_ret(table) -> np.ndarray:
    # まず ledger 内の列を使う（無い場合は mark_price から計算）
    if "abs_mid_ret" in table.schema.names:
        return _to_float_array(table["abs_mid_ret"].to_pylist())
    if "mark_price" not in table.schema.names:
        return np.full(int(table.num_rows), np.nan, dtype=float)
    mid = _to_float_array(table["mark_price"].to_pylist())
    prev = np.concatenate(([np.nan], mid[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = (mid - prev) / prev
    return np.abs(ret)


@dataclass(frozen=True)
class WindowFit:
    feature: str
    mode: str  # "gt" | "lt"
    threshold: float
    window_rate: float
    precision: float
    recall: float
    f1: float
    mean_delta_in: float
    mean_delta_out: float


@dataclass(frozen=True)
class FixedRateWindow:
    feature: str
    mode: str  # "gt" | "lt"
    target_rate: float
    threshold: float
    window_rate: float
    mean_delta_in: float
    mean_delta_out: float
    precision: float
    recall: float
    f1: float


def _f1_window_fit(x: np.ndarray, risk: np.ndarray, feature: str) -> Optional[WindowFit]:
    # 危険ラベル(risk)に対して、単一閾値のwindowを当てはめる（F1最大）
    ok = ~np.isnan(x)
    x2 = x[ok]
    r2 = risk[ok]
    if x2.size == 0:
        return None

    best: Optional[WindowFit] = None
    qs = np.linspace(0.01, 0.99, 99)
    for q in qs:
        thr = float(np.quantile(x2, q))
        for mode in ("gt", "lt"):
            win = x2 >= thr if mode == "gt" else x2 <= thr
            tp = int(np.sum(win & r2))
            fp = int(np.sum(win & ~r2))
            fn = int(np.sum((~win) & r2))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            # mean pnl_delta は別で計算（winは okマスク後のインデックス）
            rate = float(np.mean(win))
            if best is None or f1 > best.f1:
                best = WindowFit(
                    feature=feature,
                    mode=mode,
                    threshold=thr,
                    window_rate=rate,
                    precision=float(precision),
                    recall=float(recall),
                    f1=float(f1),
                    mean_delta_in=float("nan"),
                    mean_delta_out=float("nan"),
                )

    return best


def _mean_delta_for_window(delta: np.ndarray, x: np.ndarray, mode: str, threshold: float) -> tuple[float, float]:
    ok = ~np.isnan(delta) & ~np.isnan(x)
    d2 = delta[ok]
    x2 = x[ok]
    if d2.size == 0:
        return float("nan"), float("nan")
    win = x2 >= float(threshold) if mode == "gt" else x2 <= float(threshold)
    if not np.any(win):
        return float("nan"), float(np.mean(d2))
    if np.all(win):
        return float(np.mean(d2)), float("nan")
    return float(np.mean(d2[win])), float(np.mean(d2[~win]))


def _prf(win: np.ndarray, risk: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(win & risk))
    fp = int(np.sum(win & ~risk))
    fn = int(np.sum((~win) & risk))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def _fixed_rate_windows(
    delta: np.ndarray, x: np.ndarray, risk: np.ndarray, feature: str, rates: list[float]
) -> list[FixedRateWindow]:
    ok = ~np.isnan(delta) & ~np.isnan(x)
    d2 = delta[ok]
    x2 = x[ok]
    r2 = risk[ok]
    if x2.size == 0:
        return []

    out: list[FixedRateWindow] = []
    for rate in rates:
        rate_f = float(rate)
        if not (0.0 < rate_f < 1.0):
            continue
        # gt: 上位rate
        thr_gt = float(np.quantile(x2, 1.0 - rate_f))
        win_gt = x2 >= thr_gt
        mean_in, mean_out = _mean_delta_for_window(d2, x2, "gt", thr_gt)
        prec, rec, f1 = _prf(win_gt, r2)
        out.append(
            FixedRateWindow(
                feature=feature,
                mode="gt",
                target_rate=rate_f,
                threshold=thr_gt,
                window_rate=float(np.mean(win_gt)),
                mean_delta_in=mean_in,
                mean_delta_out=mean_out,
                precision=prec,
                recall=rec,
                f1=f1,
            )
        )

        # lt: 下位rate
        thr_lt = float(np.quantile(x2, rate_f))
        win_lt = x2 <= thr_lt
        mean_in, mean_out = _mean_delta_for_window(d2, x2, "lt", thr_lt)
        prec, rec, f1 = _prf(win_lt, r2)
        out.append(
            FixedRateWindow(
                feature=feature,
                mode="lt",
                target_rate=rate_f,
                threshold=thr_lt,
                window_rate=float(np.mean(win_lt)),
                mean_delta_in=mean_in,
                mean_delta_out=mean_out,
                precision=prec,
                recall=rec,
                f1=f1,
            )
        )
    return out


def _render_table(rows: list[WindowFit]) -> str:
    headers = [
        "feature",
        "mode",
        "threshold",
        "window_rate",
        "precision",
        "recall",
        "f1",
        "mean_delta_in",
        "mean_delta_out",
    ]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    r.feature,
                    r.mode,
                    f"{r.threshold:.6g}",
                    f"{r.window_rate:.6g}",
                    f"{r.precision:.6g}",
                    f"{r.recall:.6g}",
                    f"{r.f1:.6g}",
                    f"{r.mean_delta_in:.6g}",
                    f"{r.mean_delta_out:.6g}",
                ]
            )
            + " |"
        )
    return "\n".join(out) + "\n"


def _render_fixed_table(rows: list[FixedRateWindow]) -> str:
    headers = [
        "feature",
        "mode",
        "target_rate",
        "window_rate",
        "threshold",
        "mean_delta_in",
        "mean_delta_out",
        "precision",
        "recall",
        "f1",
    ]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    r.feature,
                    r.mode,
                    f"{r.target_rate:.6g}",
                    f"{r.window_rate:.6g}",
                    f"{r.threshold:.6g}",
                    f"{r.mean_delta_in:.6g}",
                    f"{r.mean_delta_out:.6g}",
                    f"{r.precision:.6g}",
                    f"{r.recall:.6g}",
                    f"{r.f1:.6g}",
                ]
            )
            + " |"
        )
    return "\n".join(out) + "\n"


def _yaml_hint(row: WindowFit) -> Optional[str]:
    if row.feature == "market_spread_bps":
        key = "pull_when_market_spread_bps_gt" if row.mode == "gt" else "pull_when_market_spread_bps_lt"
        return f"{key}: {row.threshold:.6g}"
    if row.feature == "abs_mid_ret":
        if row.mode != "gt":
            return None
        return f"pull_when_abs_mid_ret_gt: {row.threshold:.6g}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="baseline の ledger.parquet から『危険window』候補を探す（pnl_deltaの悪い上位q%を説明する特徴量と閾値）"
    )
    parser.add_argument(
        "--ledger",
        default="outputs/mm_sim_baseline/ledger.parquet",
        help="入力 ledger.parquet（baseline推奨）",
    )
    parser.add_argument(
        "--risk-quantile",
        type=float,
        default=0.10,
        help="危険ラベルの分位（例: 0.10=最悪10%のpnl_deltaを危険とみなす）",
    )
    parser.add_argument(
        "--window-rates",
        default="0.05,0.10",
        help="固定レートwindowを評価する比率（カンマ区切り。例: 0.05,0.10）",
    )
    args = parser.parse_args()

    ledger = pq.read_table(Path(args.ledger))
    py = ledger.to_pydict()

    total_pnl = _to_float_array(py.get("total_pnl", []))
    delta = _pnl_delta(total_pnl)
    if delta.size == 0:
        raise SystemExit("ledger が空です")

    q = float(args.risk_quantile)
    if not (0.0 < q < 1.0):
        raise SystemExit("--risk-quantile は 0〜1 の間で指定してください")

    # 危険ラベル（最悪q%のpnl_delta）
    thr_delta = float(np.nanquantile(delta, q))
    risk = delta <= thr_delta

    print(f"[info] risk_quantile={q:.6g} pnl_delta_threshold={thr_delta:.6g} risk_rate={float(np.mean(risk)):.6g}")

    market_spread_bps = _to_float_array(py.get("market_spread_bps", []))
    signed_volume = _to_float_array(py.get("signed_volume", []))
    abs_signed_volume = np.abs(signed_volume)
    abs_mid_ret = _abs_mid_ret(ledger)

    candidates = [
        ("market_spread_bps", market_spread_bps),
        ("abs_signed_volume", abs_signed_volume),
        ("abs_mid_ret", abs_mid_ret),
    ]

    window_rates = [float(x.strip()) for x in str(args.window_rates).split(",") if x.strip()]

    rows: list[WindowFit] = []
    fixed_rows: list[FixedRateWindow] = []
    for feature, x in candidates:
        best = _f1_window_fit(x, risk, feature=feature)
        if best is None:
            continue
        mean_in, mean_out = _mean_delta_for_window(delta, x, best.mode, best.threshold)
        rows.append(
            WindowFit(
                feature=best.feature,
                mode=best.mode,
                threshold=best.threshold,
                window_rate=best.window_rate,
                precision=best.precision,
                recall=best.recall,
                f1=best.f1,
                mean_delta_in=mean_in,
                mean_delta_out=mean_out,
            )
        )
        fixed_rows.extend(_fixed_rate_windows(delta, x, risk, feature=feature, rates=window_rates))

    # 見やすくする（F1降順）
    rows.sort(key=lambda r: r.f1, reverse=True)
    print("[info] best threshold (F1) candidates:")
    print(_render_table(rows), end="")

    fixed_rows.sort(key=lambda r: (r.feature, r.target_rate, r.mode))
    print("[info] fixed-rate windows (使い勝手重視: window_rate を揃えて mean_delta_in を見る):")
    print(_render_fixed_table(fixed_rows), end="")

    # YAMLヒント（対応済みキーのみ）
    hints = [h for h in (_yaml_hint(r) for r in rows) if h]
    if hints:
        print("[hint] 対応済みキー（configs/*.yaml の extra_params に貼れる）:")
        for h in hints:
            print(f"  {h}")
    print("[note] abs_signed_volume で window を作りたい場合は、戦略側にトリガーキー追加が必要です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
