"""Terminal progress helpers for long Spark and filesystem operations."""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager


class SparkProgressMonitor:
    """Poll Spark's status tracker while the Python driver is blocked."""

    SPINNER = ("|", "/", "-", "\\")

    def __init__(self, spark, label: str, interval_seconds: float = 2.0, width: int = 28):
        self.spark = spark
        self.label = label
        self.interval_seconds = interval_seconds
        self.width = width
        self.started = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.is_tty = sys.stdout.isatty()
        self.last_log_bucket = -1

    def __enter__(self):
        self.started = time.monotonic()
        print(f"\n  {self.label}")
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_seconds + 0.5)
        elapsed = time.monotonic() - self.started
        if self.is_tty:
            sys.stdout.write("\r" + " " * 150 + "\r")
            sys.stdout.flush()
        outcome = "FAILED" if exc_type is not None else "complete"
        print(f"  [{outcome}] {self.label} ({elapsed:,.1f} s)")
        return False

    def _poll(self) -> None:
        tick = 0
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - self.started
            message = self._status_message(elapsed, tick)
            if self.is_tty:
                sys.stdout.write("\r" + message[:148].ljust(148))
                sys.stdout.flush()
            else:
                bucket = int(elapsed // 10)
                if bucket != self.last_log_bucket:
                    print(message)
                    self.last_log_bucket = bucket
            tick += 1
            self.stop_event.wait(self.interval_seconds)

    def _status_message(self, elapsed: float, tick: int) -> str:
        try:
            tracker = self.spark.sparkContext.statusTracker()
            infos = [tracker.getStageInfo(stage_id) for stage_id in tracker.getActiveStageIds()]
            infos = [info for info in infos if info is not None]
            total = sum(int(info.numTasks) for info in infos)
            completed = sum(int(info.numCompletedTasks) for info in infos)
            active = sum(int(info.numActiveTasks) for info in infos)
            failed = sum(int(info.numFailedTasks) for info in infos)
            if total > 0:
                fraction = min(1.0, completed / total)
                filled = int(round(self.width * fraction))
                bar = "#" * filled + "-" * (self.width - filled)
                stages = ",".join(str(info.stageId) for info in infos)
                return (
                    f"  [{bar}] {100.0 * fraction:5.1f}% | stages {stages} | "
                    f"tasks {completed}/{total}, active {active}, failed {failed} | "
                    f"elapsed {elapsed:,.0f}s"
                )
        except Exception as exc:
            return (
                f"  [{self.SPINNER[tick % len(self.SPINNER)]}] "
                f"Spark status unavailable: {exc} | elapsed {elapsed:,.0f}s"
            )
        return (
            f"  [{self.SPINNER[tick % len(self.SPINNER)]}] "
            f"waiting for/planning Spark stages | elapsed {elapsed:,.0f}s"
        )


@contextmanager
def driver_progress(label: str, interval_seconds: float = 5.0):
    """Show an elapsed-time heartbeat during hashing or filesystem copying."""
    started = time.monotonic()
    stop_event = threading.Event()
    is_tty = sys.stdout.isatty()

    def poll():
        tick = 0
        while not stop_event.is_set():
            elapsed = time.monotonic() - started
            message = f"  [{SparkProgressMonitor.SPINNER[tick % 4]}] {label} | elapsed {elapsed:,.0f}s"
            if is_tty:
                sys.stdout.write("\r" + message[:148].ljust(148))
                sys.stdout.flush()
            elif tick % max(1, int(10 / interval_seconds)) == 0:
                print(message)
            tick += 1
            stop_event.wait(interval_seconds)

    print(f"\n  {label}")
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=interval_seconds + 0.5)
        elapsed = time.monotonic() - started
        if is_tty:
            sys.stdout.write("\r" + " " * 150 + "\r")
            sys.stdout.flush()
        print(f"  [complete] {label} ({elapsed:,.1f} s)")

