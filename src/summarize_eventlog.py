"""Aggregate Spark task metrics from JSON event logs without rerunning jobs."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import pandas as pd


def lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def summarize(event_log: str):
    totals = {k: 0 for k in (
        "tasks", "executor_run_ms", "executor_cpu_ns", "jvm_gc_ms",
        "memory_spill_bytes", "disk_spill_bytes", "shuffle_read_bytes", "shuffle_write_bytes")}
    stage_tasks = {}
    paths = [Path(event_log)] if Path(event_log).is_file() else sorted(Path(event_log).rglob("*"))
    for path in paths:
        if not path.is_file() or path.name.startswith("."): continue
        for line in lines(path):
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            if event.get("Event") != "SparkListenerTaskEnd": continue
            metrics = event.get("Task Metrics", {})
            stage = event.get("Stage ID")
            row = stage_tasks.setdefault(stage, {k: 0 for k in totals})
            values = {
                "tasks": 1,
                "executor_run_ms": metrics.get("Executor Run Time", 0),
                "executor_cpu_ns": metrics.get("Executor CPU Time", 0),
                "jvm_gc_ms": metrics.get("JVM GC Time", 0),
                "memory_spill_bytes": metrics.get("Memory Bytes Spilled", 0),
                "disk_spill_bytes": metrics.get("Disk Bytes Spilled", 0),
                "shuffle_read_bytes": metrics.get("Shuffle Read Metrics", {}).get("Remote Bytes Read", 0) + metrics.get("Shuffle Read Metrics", {}).get("Local Bytes Read", 0),
                "shuffle_write_bytes": metrics.get("Shuffle Write Metrics", {}).get("Shuffle Bytes Written", 0),
            }
            for key, value in values.items(): totals[key] += value or 0; row[key] += value or 0
    return totals, [{"stage_id": stage, **metrics} for stage, metrics in sorted(stage_tasks.items())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("event_log")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    totals, stages = summarize(args.event_log)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stages).to_csv(out, index=False)
    out.with_suffix(".summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
