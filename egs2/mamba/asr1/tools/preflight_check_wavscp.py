#!/usr/bin/env python3
"""
Preflight checker for wav.scp files.
Checks existence, readability and optionally tries to open audio with soundfile.
Produces a report file `preflight_report.txt` next to this script.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import soundfile as sf
except Exception:
    sf = None


def check_path(path: str) -> str:
    # handle command pipelines (not supported) - treat as OK if not a file path
    if path.strip().startswith("|"):
        return "PIPELINE (skipped)"
    p = Path(path)
    if not p.exists():
        return "MISSING"
    if not os.access(p, os.R_OK):
        return "NO_READ_PERMISSION"
    if sf is not None:
        try:
            with sf.SoundFile(str(p)):
                return "OK"
        except Exception as e:
            return f"SOUNDFILE_ERROR: {e}"
    # fallback: try open binary
    try:
        with open(p, "rb"):
            return "OK"
    except Exception as e:
        return f"OPEN_ERROR: {e}"


def process_wavscp(wavscp_path: str, report_lines: list):
    wavscp = Path(wavscp_path)
    if not wavscp.exists():
        report_lines.append(f"FILE_NOT_FOUND: {wavscp}")
        return 1
    bad = 0
    with wavscp.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                utt, path = parts[0], ""
            else:
                utt, path = parts
            status = check_path(path)
            report_lines.append(f"{wavscp} | {i:6d} | {utt} | {path} | {status}")
            if status != "OK" and not status.startswith("PIPELINE"):
                bad += 1
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", "-f", nargs="+",
                        default=[
                            "data/train_lib360_copy/wav.scp",
                            "data/train_lib100_copy/wav.scp",
                            "data/dev_lib360/wav.scp",
                            "data/test_lib360/wav.scp",
                        ],
                        help="Paths to wav.scp files (workspace-relative or absolute)")
    parser.add_argument("--report", "-r", default="preflight_report.txt",
                        help="Report file path (created next to script by default)")
    args = parser.parse_args()

    report_lines = []
    total_bad = 0
    for f in args.files:
        bad = process_wavscp(f, report_lines)
        total_bad += bad

    report_path = Path(__file__).parent / args.report
    # ensure target directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")

    summary = [f"Checked {len(args.files)} wav.scp files", f"Total problematic entries: {total_bad}"]
    print("\n".join(summary))
    print(f"Report saved to: {report_path}")
    if total_bad > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
