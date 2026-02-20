#!/usr/bin/env python3
"""
Concurrent regression test for atomic_io:
- writer repeatedly atomic writes a larger JSON file
- reader repeatedly reads via safe_read_json
Goal:
- No JSONDecodeError should escape (0)
- Writer should not crash
- Reader should rarely see None after first write (some None allowed, but should be low)
Windows-friendly.
"""

from __future__ import annotations

import json
import os
import random
import string
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Adjust import path if needed
try:
    from atomic_io import atomic_write_json, safe_read_json
except Exception as e:
    print("[FATAL] Failed to import atomic_io:", e)
    sys.exit(2)


OUT_DIR = os.path.join("outputs", "_atomic_test")
TEST_JSON = os.path.join(OUT_DIR, "test.json")


@dataclass
class Stats:
    writer_ok: bool = True
    writer_exc: str = ""
    reader_jsondecode_errors: int = 0
    reader_other_errors: int = 0
    reader_none: int = 0
    reader_ok: int = 0


def _rand_payload(n: int = 8000) -> str:
    # Big payload increases chance of catching partial reads in non-atomic implementations.
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


def writer_thread(n_iters: int, ready_evt: threading.Event, stop_evt: threading.Event, stats: Stats) -> None:
    try:
        os.makedirs(OUT_DIR, exist_ok=True)

        # First write to ensure reader does not start before file exists.
        obj0 = {
            "i": 0,
            "ts": time.time(),
            "payload": _rand_payload(4000),
            "arr": list(range(1000)),
            "nested": {"a": 1, "b": "init"},
        }
        atomic_write_json(TEST_JSON, obj0, indent=2)
        ready_evt.set()

        for i in range(1, n_iters + 1):
            if stop_evt.is_set():
                break
            obj = {
                "i": i,
                "ts": time.time(),
                "payload": _rand_payload(8000),
                "arr": list(range(2000)),
                "nested": {"a": i, "b": f"run-{i}"},
            }
            atomic_write_json(TEST_JSON, obj, indent=2)

            # Tiny sleep to create more interleavings.
            if i % 5 == 0:
                time.sleep(0.001)

    except Exception:
        stats.writer_ok = False
        stats.writer_exc = traceback.format_exc()
    finally:
        stop_evt.set()


def reader_thread(n_iters: int, ready_evt: threading.Event, stop_evt: threading.Event, stats: Stats) -> None:
    # Wait until writer has produced the first complete file.
    if not ready_evt.wait(timeout=5.0):
        stats.reader_other_errors += 1
        print("[FATAL] Reader timeout waiting for initial write.")
        stop_evt.set()
        return

    for _ in range(n_iters):
        if stop_evt.is_set():
            break
        try:
            obj = safe_read_json(TEST_JSON, retries=3, sleep_ms=10)
            if obj is None:
                stats.reader_none += 1
            elif isinstance(obj, dict) and "i" in obj and "ts" in obj:
                stats.reader_ok += 1
            else:
                # Unexpected type/shape.
                stats.reader_other_errors += 1
        except json.JSONDecodeError:
            # This should never occur for atomic+safe_read flow.
            stats.reader_jsondecode_errors += 1
        except Exception:
            stats.reader_other_errors += 1

        # Tiny sleep to avoid pegging CPU.
        time.sleep(0.0005)


def main() -> int:
    random.seed(42)

    n_writer = int(os.environ.get("ATOMIC_TEST_WRITER_ITERS", "2000"))
    n_reader = int(os.environ.get("ATOMIC_TEST_READER_ITERS", "4000"))

    ready_evt = threading.Event()
    stop_evt = threading.Event()
    stats = Stats()

    wt = threading.Thread(target=writer_thread, args=(n_writer, ready_evt, stop_evt, stats), daemon=True)
    rt = threading.Thread(target=reader_thread, args=(n_reader, ready_evt, stop_evt, stats), daemon=True)

    t0 = time.time()
    wt.start()
    rt.start()
    wt.join(timeout=60.0)
    rt.join(timeout=60.0)
    dt = time.time() - t0

    print("==== atomic_io concurrent test ====")
    print("file:", TEST_JSON)
    print(f"elapsed: {dt:.2f}s")
    print(f"writer_ok: {stats.writer_ok}")
    if not stats.writer_ok:
        print("writer_exc:\n", stats.writer_exc)

    print(f"reader_ok: {stats.reader_ok}")
    print(f"reader_none: {stats.reader_none}")
    print(f"reader_jsondecode_errors: {stats.reader_jsondecode_errors}")
    print(f"reader_other_errors: {stats.reader_other_errors}")

    # Pass/fail policy.
    if not stats.writer_ok:
        print("[FAIL] writer crashed.")
        return 1
    if stats.reader_jsondecode_errors != 0:
        print("[FAIL] JSONDecodeError observed by reader.")
        return 1
    if stats.reader_other_errors != 0:
        print("[FAIL] reader saw unexpected errors/invalid objects.")
        return 1

    # None is allowed but should be low. We do not fail hard here.
    total_reads = max(stats.reader_ok + stats.reader_none, 1)
    none_ratio = stats.reader_none / total_reads
    if none_ratio > 0.05:
        print(f"[WARN] reader_none ratio is high: {none_ratio:.2%} (possible file access contention).")

    print("[PASS] atomic write + safe read appear robust under concurrency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
