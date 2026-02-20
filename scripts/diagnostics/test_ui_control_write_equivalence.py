#!/usr/bin/env python3
"""T21: UI-equivalent runtime_control.json write/shape validation."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json


RISK_CHOICES = {"low", "mid", "high", "ultra"}


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _is_iso_tz(value: str) -> bool:
    try:
        dt = datetime.fromisoformat(str(value))
    except Exception:
        return False
    return dt.tzinfo is not None


def main() -> int:
    out_dir = ROOT / "outputs" / "test_ui_control_write_equivalence"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    control_path = out_dir / "runtime_control.json"

    requested = "high"
    if requested not in RISK_CHOICES:
        return _fail(f"requested profile invalid in test setup: {requested!r}")

    payload = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": uuid.uuid4().hex[:12],
        "requested_risk_profile": requested,
    }

    # UI-equivalent write.
    atomic_write_json(str(control_path), payload, indent=2)

    # File must be json-loadable.
    try:
        written = json.loads(control_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _fail(f"control file is not json-loadable: {exc}")

    if int(written.get("schema_version", -1)) != 1:
        return _fail(f"schema_version invalid: {written.get('schema_version')!r}")
    if not _is_iso_tz(written.get("updated_at_utc", "")):
        return _fail(f"updated_at_utc invalid/non-tz ISO8601: {written.get('updated_at_utc')!r}")
    request_id = str(written.get("request_id", "")).strip()
    if not request_id:
        return _fail("request_id is empty")
    req = str(written.get("requested_risk_profile", "")).strip().lower()
    if req not in RISK_CHOICES:
        return _fail(f"requested_risk_profile not in allowed set: {req!r}")

    print("[PASS] ui_control_write_equivalence")
    print(
        f"[INFO] path={control_path} schema={written.get('schema_version')} "
        f"requested={req} request_id={request_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

