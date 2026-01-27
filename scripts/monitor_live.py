import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _get_nested(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _win_rate(values: List[float]) -> Optional[float]:
    if not values:
        return None
    wins = sum(1 for v in values if v > 0)
    return wins / len(values)


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _bool_to_str(value: Optional[bool]) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


def _safe_text(text: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    try:
        text.encode(enc)
        return text
    except UnicodeEncodeError:
        return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _safe_print(text: str) -> None:
    print(_safe_text(text))


@dataclass
class RunRecord:
    run_id: str
    path: Path
    mtime: float
    status: str
    gateb_pass: bool
    net_med: Optional[float]
    net_p10: Optional[float]
    markout30_med: Optional[float]
    fills_count: Optional[float]
    notional_sum: Optional[float]
    taker_guard_trip: Optional[bool]
    taker_notional_share: Optional[float]
    fills_joined_count: Optional[float]


def _classify_status(record: RunRecord) -> str:
    if record.taker_guard_trip:
        return "HARD_FAIL"
    if record.taker_notional_share is not None and record.taker_notional_share > 0.001:
        return "HARD_FAIL"
    if record.net_p10 is not None and record.net_p10 < -5:
        return "HARD_FAIL"
    if not record.gateb_pass:
        return "INCONCLUSIVE"
    if record.net_med is None:
        return "INCONCLUSIVE"
    return "PASS" if record.net_med > 0 else "FAIL"


def _load_report(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _make_record(path: Path, report: Dict[str, Any]) -> RunRecord:
    run_dir = report.get("run_dir")
    run_id = str(report.get("run_id") or "")
    if not run_id:
        if isinstance(run_dir, str) and run_dir:
            run_id = Path(run_dir).name
        else:
            run_id = path.parent.name

    fills_count = _as_float(report.get("fills_count"))
    notional_sum = _as_float(report.get("notional_sum"))
    fills_joined_count = _as_float(report.get("fills_joined_count"))
    gateb_pass = (
        (fills_count or 0) >= 10
        and (notional_sum or 0) >= 300
        and (fills_joined_count or 0) > 0
    )
    net_med = _as_float(_get_nested(report, ("net_bps", "median")))
    net_p10 = _as_float(_get_nested(report, ("net_bps", "p10")))
    markout30_med = _as_float(_get_nested(report, ("markout_bps", "30", "median")))
    taker_guard_trip = report.get("taker_guard_trip")
    taker_guard_trip_bool = bool(taker_guard_trip) if taker_guard_trip is not None else None
    taker_notional_share = _as_float(report.get("taker_notional_share"))

    record = RunRecord(
        run_id=run_id,
        path=path,
        mtime=path.stat().st_mtime,
        status="",
        gateb_pass=gateb_pass,
        net_med=net_med,
        net_p10=net_p10,
        markout30_med=markout30_med,
        fills_count=fills_count,
        notional_sum=notional_sum,
        taker_guard_trip=taker_guard_trip_bool,
        taker_notional_share=taker_notional_share,
        fills_joined_count=fills_joined_count,
    )
    record.status = _classify_status(record)
    return record


def _consecutive_warn(markout_flags: List[bool]) -> bool:
    if len(markout_flags) < 2:
        return False
    for idx in range(len(markout_flags) - 1):
        if markout_flags[idx] and markout_flags[idx + 1]:
            return True
    return False


def _build_summary(
    records: List[RunRecord],
    reports_root: Path,
    window: int,
    latest: int,
) -> Tuple[Dict[str, Any], List[RunRecord], List[RunRecord]]:
    records_sorted = sorted(records, key=lambda r: r.mtime, reverse=True)
    latest_runs = records_sorted[:latest]

    gateb_runs = [r for r in records_sorted if r.gateb_pass]
    last_gateb = gateb_runs[:window]

    net_med_vals = [r.net_med for r in last_gateb if r.net_med is not None]
    net_p10_vals = [r.net_p10 for r in last_gateb if r.net_p10 is not None]

    markout_flags = [
        (r.markout30_med is not None and r.markout30_med < -5) for r in last_gateb
    ]
    warn_count = sum(1 for flag in markout_flags if flag)
    warn_consecutive = _consecutive_warn(markout_flags)
    warn_need_action = warn_count >= 2 or warn_consecutive

    taker_guard_trip_count = sum(1 for r in last_gateb if r.taker_guard_trip)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports_root": str(reports_root),
        "window": window,
        "runs_total": len(records),
        "gateb_pass_total": len(gateb_runs),
        "last_10_gateb_pass": {
            "count": len(last_gateb),
            "run_ids": [r.run_id for r in last_gateb],
            "net_med_median": _median(net_med_vals),
            "net_med_win_rate": _win_rate(net_med_vals),
            "net_med_count": len(net_med_vals),
            "net_p10_min": min(net_p10_vals) if net_p10_vals else None,
            "net_p10_count": len(net_p10_vals),
            "warn_markout30_count": warn_count,
            "warn_markout30_consecutive": warn_consecutive,
            "warn_markout30_need_action": warn_need_action,
            "taker_guard_trip_count": taker_guard_trip_count,
        },
        "latest_runs": [
            {
                "run_id": r.run_id,
                "status": r.status,
                "net_med": r.net_med,
                "markout30_med": r.markout30_med,
                "fills_count": r.fills_count,
                "notional_sum": r.notional_sum,
                "path": str(r.path),
            }
            for r in latest_runs
        ],
        "status_counts": {
            "PASS": sum(1 for r in records if r.status == "PASS"),
            "FAIL": sum(1 for r in records if r.status == "FAIL"),
            "HARD_FAIL": sum(1 for r in records if r.status == "HARD_FAIL"),
            "INCONCLUSIVE": sum(1 for r in records if r.status == "INCONCLUSIVE"),
        },
    }
    return summary, last_gateb, latest_runs


def _print_summary(
    summary: Dict[str, Any],
    last_gateb: List[RunRecord],
    latest_runs: List[RunRecord],
) -> None:
    _safe_print("monitor_live: run_report 集計")
    _safe_print(f"  root: {summary['reports_root']}")
    _safe_print(f"  runs_total: {summary['runs_total']}")
    _safe_print(f"  gateb_pass_total: {summary['gateb_pass_total']}")

    last_info = summary["last_10_gateb_pass"]
    net_med_median = _fmt(last_info.get("net_med_median"))
    net_p10_min = _fmt(last_info.get("net_p10_min"))
    win_rate = last_info.get("net_med_win_rate")
    win_rate_str = "-" if win_rate is None else f"{win_rate:.2f}"
    warn_count = last_info.get("warn_markout30_count")
    warn_need_action = _bool_to_str(last_info.get("warn_markout30_need_action"))
    taker_trip = last_info.get("taker_guard_trip_count")

    _safe_print(f"  last_{summary['window']}_gateb_pass: {last_info.get('count')}")
    _safe_print(f"    median(net_med): {net_med_median}")
    _safe_print(f"    win_rate(net_med>0): {win_rate_str}")
    _safe_print(f"    min(net_p10): {net_p10_min}")
    _safe_print(f"    WARN_count(markout30_med<-5): {warn_count}")
    _safe_print(f"    WARN_need_action: {warn_need_action}")
    _safe_print(f"    taker_guard_trip_count: {taker_trip}")

    if last_gateb:
        _safe_print("  last_gateb_run_ids:")
        _safe_print("    " + ", ".join(r.run_id for r in last_gateb))

    if latest_runs:
        _safe_print("  latest_runs:")
        _safe_print("    run_id\tstatus\tnet_med\tmarkout30_med\tfills_count\tnotional_sum")
        for run in latest_runs:
            _safe_print(
                "    "
                + "\t".join(
                    [
                        run.run_id,
                        run.status,
                        _fmt(run.net_med),
                        _fmt(run.markout30_med),
                        _fmt(run.fills_count, digits=0),
                        _fmt(run.notional_sum, digits=1),
                    ]
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="mm_live run_report の監視集計")
    parser.add_argument(
        "--reports-root",
        default="reports_live_f15",
        help="run_report.json を探すルート",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="GateB_pass の集計本数",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=10,
        help="直近runの表示本数",
    )
    parser.add_argument(
        "--out",
        help="summary.json の出力先（未指定なら <reports_root>/_monitor/summary.json）",
    )
    args = parser.parse_args()

    reports_root = Path(args.reports_root)
    report_paths = [
        path
        for path in reports_root.glob("**/run_report.json")
        if "_monitor" not in path.parts
    ]

    records: List[RunRecord] = []
    for path in report_paths:
        report = _load_report(path)
        if report is None:
            continue
        try:
            record = _make_record(path, report)
        except OSError:
            continue
        records.append(record)

    summary, last_gateb, latest_runs = _build_summary(
        records, reports_root, window=args.window, latest=args.latest
    )

    _print_summary(summary, last_gateb, latest_runs)

    out_path = Path(args.out) if args.out else reports_root / "_monitor" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
