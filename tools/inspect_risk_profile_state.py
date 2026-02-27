"""Inspect risk profile state file and validate requested profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _fmt_mtime(ts: float | None) -> str:
    if ts is None:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return "-"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect risk_profile_state.json")
    parser.add_argument("state_file_path", help="Absolute or relative state file path")
    parser.add_argument("--expected", default="high", help="Expected requested profile (default: high)")
    args = parser.parse_args()

    target = os.path.abspath(str(args.state_file_path or "").strip())
    expected = str(args.expected or "high").strip().lower() or "high"

    exists = os.path.exists(target)
    size = os.path.getsize(target) if exists else 0
    mtime = os.path.getmtime(target) if exists else None

    print(f"path={target}")
    print(f"exists={exists}")
    print(f"size={size}")
    print(f"mtime_unix={mtime if mtime is not None else '-'}")
    print(f"mtime_utc={_fmt_mtime(mtime)}")

    if not exists:
        print("FAIL: state file does not exist")
        return 1

    try:
        with open(target, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        print(f"FAIL: json parse error: {exc}")
        return 1

    if not isinstance(obj, dict):
        print("FAIL: state payload is not a JSON object")
        return 1

    requested = str(obj.get("requested", "") or "").strip().lower()
    set_at = str(obj.get("set_at", "") or "").strip()
    set_by = str(obj.get("set_by", "") or "").strip()
    version = str(obj.get("version", "") or "").strip()
    schema_version = obj.get("schema_version")

    print(
        "state="
        + json.dumps(
            {
                "requested": requested,
                "set_at": set_at,
                "set_by": set_by,
                "version": version,
                "schema_version": schema_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if requested != expected:
        print(f"FAIL: requested={requested!r}, expected={expected!r}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
