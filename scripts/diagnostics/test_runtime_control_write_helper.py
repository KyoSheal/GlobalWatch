#!/usr/bin/env python3
"""T03: runtime_control write helper test (UI-equivalent write path)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_trading import PaperTradingEngine, RISK_PROFILE_CHOICES


def _build_engine_stub(control_path: Path) -> PaperTradingEngine:
    engine = PaperTradingEngine.__new__(PaperTradingEngine)
    engine.runtime_control_path = str(control_path)
    # _now/atomic_write_json methods are instance methods on class and do not
    # require full __init__ state for this test path.
    return engine


def _is_iso8601_with_timezone(value: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        dt = datetime.fromisoformat(value.strip())
    except Exception:
        return False
    return bool(dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    out_dir = ROOT / "outputs" / "test_runtime_control_write_helper"
    out_dir.mkdir(parents=True, exist_ok=True)
    control_path = out_dir / "runtime_control.json"

    engine = _build_engine_stub(control_path)
    choices = tuple(sorted(set(RISK_PROFILE_CHOICES)))
    failures: list[str] = []

    # Atomic-write smoke: repeat writes and assert file always JSON-loadable.
    loops = 100
    for i in range(loops):
        requested = choices[i % len(choices)] if choices else "mid"
        request_id = f"T03-{i:03d}"
        try:
            returned_path = engine.write_runtime_control_request(
                requested_risk_profile=requested,
                request_id=request_id,
            )
        except Exception as exc:
            failures.append(f"write raised at i={i}: {exc}")
            continue

        if str(control_path) != str(returned_path):
            failures.append(f"returned path mismatch at i={i}: {returned_path!r}")

        try:
            payload = _load_json(control_path)
        except Exception as exc:
            failures.append(f"json.load failed at i={i}: {exc}")
            continue

        if not isinstance(payload, dict):
            failures.append(f"payload type invalid at i={i}: {type(payload).__name__}")
            continue

        if int(payload.get("schema_version", -1)) != 1:
            failures.append(f"schema_version invalid at i={i}: {payload.get('schema_version')!r}")

        updated_at = payload.get("updated_at_utc")
        if not _is_iso8601_with_timezone(updated_at):
            failures.append(f"updated_at_utc invalid/no-timezone at i={i}: {updated_at!r}")

        rid = str(payload.get("request_id", "") or "").strip()
        if not rid:
            failures.append(f"request_id empty at i={i}")

        prof = str(payload.get("requested_risk_profile", "") or "").strip().lower()
        if prof not in set(choices):
            failures.append(f"requested_risk_profile invalid at i={i}: {prof!r}")

    if failures:
        print("[FAIL] runtime_control_write_helper")
        for item in failures[:20]:
            print(f"  - {item}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    final_payload = _load_json(control_path)
    print("[PASS] runtime_control_write_helper")
    print(f"[INFO] loops={loops} control_path={control_path}")
    print(
        "[INFO] final="
        f"schema={final_payload.get('schema_version')} "
        f"requested={final_payload.get('requested_risk_profile')} "
        f"request_id={final_payload.get('request_id')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

