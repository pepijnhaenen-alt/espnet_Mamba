#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Aggregate true streaming interaction metrics from inference logs.

Expected log line format:
    ... INFO: stream_event: {"event": "update", ...}
"""

import argparse
import glob
import json
import math
import os
import statistics
from typing import Dict, List, Optional, Tuple


def _percentile(sorted_values: List[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    rank = (p / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def _series_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }

    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "mean": float(statistics.fmean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "p90": float(_percentile(sorted_values, 90)),
        "p95": float(_percentile(sorted_values, 95)),
        "p99": float(_percentile(sorted_values, 99)),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
    }


def _parse_stream_event_json(line: str) -> Optional[Dict]:
    marker = "stream_event:"
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def aggregate_metrics(
    log_dir: str,
    log_name: str,
    include_empty_updates: bool = False,
) -> Dict:
    pattern = os.path.join(log_dir, f"{log_name}.*.log")
    log_files = sorted(glob.glob(pattern))

    if not log_files:
        raise FileNotFoundError(f"No log files found: {pattern}")

    ftl_by_utt: Dict[str, float] = {}
    epd_by_utt: Dict[str, float] = {}
    lag_values: List[float] = []

    n_stream_events = 0
    n_update_events = 0
    n_final_events = 0
    n_first_token_events = 0

    for path in log_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                event = _parse_stream_event_json(line)
                if event is None:
                    continue

                n_stream_events += 1
                event_type = event.get("event")
                utt_id = event.get("utt_id")

                if event_type == "update":
                    n_update_events += 1
                    if utt_id is None:
                        continue
                    token_count = int(event.get("token_count", 0))
                    if include_empty_updates or token_count > 0:
                        lag_ms = event.get("lag_ms")
                        if lag_ms is not None:
                            lag_values.append(float(lag_ms))

                    if utt_id not in ftl_by_utt and token_count > 0:
                        emit_elapsed_ms = event.get("emit_elapsed_ms")
                        if emit_elapsed_ms is not None:
                            ftl_by_utt[utt_id] = float(emit_elapsed_ms)

                elif event_type == "first_token":
                    n_first_token_events += 1
                    if utt_id is None:
                        continue
                    ftl_ms = event.get("first_token_latency_ms")
                    if ftl_ms is not None:
                        ftl_by_utt[utt_id] = float(ftl_ms)

                elif event_type == "final":
                    n_final_events += 1
                    if utt_id is None:
                        continue
                    epd_ms = event.get("endpoint_delay_ms")
                    if epd_ms is not None:
                        epd_by_utt[utt_id] = float(epd_ms)

    summary = {
        "log_dir": log_dir,
        "log_name": log_name,
        "num_log_files": len(log_files),
        "num_stream_events": n_stream_events,
        "num_update_events": n_update_events,
        "num_first_token_events": n_first_token_events,
        "num_final_events": n_final_events,
        "num_utterances_with_ftl": len(ftl_by_utt),
        "num_utterances_with_epd": len(epd_by_utt),
        "ftl_ms": _series_stats(list(ftl_by_utt.values())),
        "lag_ms": _series_stats(lag_values),
        "epd_ms": _series_stats(list(epd_by_utt.values())),
    }
    return summary


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.3f}"


def _render_text(summary: Dict) -> str:
    lines = []
    lines.append("STREAMING INTERACTION METRICS")
    lines.append(f"log_dir: {summary['log_dir']}")
    lines.append(f"log_name: {summary['log_name']}")
    lines.append(f"num_log_files: {summary['num_log_files']}")
    lines.append(f"num_stream_events: {summary['num_stream_events']}")
    lines.append(f"num_update_events: {summary['num_update_events']}")
    lines.append(f"num_first_token_events: {summary['num_first_token_events']}")
    lines.append(f"num_final_events: {summary['num_final_events']}")
    lines.append("")

    for key, title in (("ftl_ms", "FTL [ms]"), ("lag_ms", "Lag [ms]"), ("epd_ms", "EPD [ms]")):
        s = summary[key]
        lines.append(f"{title}")
        lines.append(f"  count:  {s['count']}")
        lines.append(f"  mean:   {_fmt(s['mean'])}")
        lines.append(f"  median: {_fmt(s['median'])}")
        lines.append(f"  p90:    {_fmt(s['p90'])}")
        lines.append(f"  p95:    {_fmt(s['p95'])}")
        lines.append(f"  p99:    {_fmt(s['p99'])}")
        lines.append(f"  min:    {_fmt(s['min'])}")
        lines.append(f"  max:    {_fmt(s['max'])}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate true streaming interaction metrics")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to logging directory")
    parser.add_argument(
        "--log-name",
        type=str,
        default="asr_inference",
        help="Inference log file prefix. Log files are matched as <log-name>.*.log",
    )
    parser.add_argument(
        "--include-empty-updates",
        action="store_true",
        help="Include updates with zero emitted tokens in lag statistics",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write JSON summary",
    )
    parser.add_argument(
        "--output-text",
        type=str,
        default=None,
        help="Optional path to write text summary",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    summary = aggregate_metrics(
        log_dir=args.log_dir,
        log_name=args.log_name,
        include_empty_updates=args.include_empty_updates,
    )

    text = _render_text(summary)
    print(text, end="")

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    if args.output_text is not None:
        with open(args.output_text, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
