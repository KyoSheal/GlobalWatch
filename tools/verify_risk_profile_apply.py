"""One-shot verifier: UI-equivalent write -> next-cycle engine apply."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json  # noqa: E402
from paper_trading import PaperTradingEngine  # noqa: E402
from risk_profile_state import (  # noqa: E402
    request_risk_profile_change,
    resolve_risk_profile_state_path,
)
from risk_profile_ui_utils import resolve_effective_reporting_cfg  # noqa: E402


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    sandbox = ROOT / "outputs" / "test_verify_risk_profile_apply"
    if sandbox.exists():
        shutil.rmtree(sandbox, ignore_errors=True)
    sandbox.mkdir(parents=True, exist_ok=True)

    base_cfg_path = ROOT / "paper_config.json"
    if not base_cfg_path.exists():
        return _fail(f"missing config: {base_cfg_path}")

    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    cfg["run_mode"] = "dryrun"
    cfg.setdefault("execution", {})["risk_profile"] = "mid"
    rep = cfg.setdefault("reporting", {})
    rep["base_out_dir"] = str((sandbox / "outputs").resolve())
    rep["out_dir"] = ""
    rep["runtime_control_path"] = ""
    rep["risk_profile_state_path"] = ""
    rep["snapshot_live_path"] = ""
    rep["telemetry_enabled"] = False
    rep["enable_daily_report"] = False
    rep["enable_eod_report"] = False

    cfg_path = sandbox / "paper_config_verify_rp_apply.json"
    atomic_write_json(str(cfg_path), cfg, indent=2)

    old_checkpoint = os.environ.get("GW_CHECKPOINT_ACTION")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    try:
        engine = PaperTradingEngine(str(cfg_path))
    finally:
        if old_checkpoint is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint

    ui_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ui_reporting = resolve_effective_reporting_cfg(ui_cfg)
    ui_state_path = resolve_risk_profile_state_path(ui_reporting)
    engine_state_path = str(engine.risk_profile_state_path)

    print(f"UI_STATE_PATH={os.path.abspath(ui_state_path)}")
    print(f"ENGINE_STATE_PATH={os.path.abspath(engine_state_path)}")

    cycle1 = 900
    engine.current_cycle = cycle1
    engine._maybe_apply_runtime_risk_profile(cycle_id=cycle1, now_utc=datetime.now(timezone.utc))

    request_id = "VERIFY-RP-HIGH"
    change = request_risk_profile_change(
        ui_state_path,
        requested="high",
        source="ui",
        actor="streamlit_sidebar",
        run_id=str(engine.run_id or ""),
        cycle_id=int(engine.current_cycle),
        extra_state={"request_id": request_id},
    )
    state_obj = change.get("state", {}) if isinstance(change, dict) else {}
    mtime = None
    try:
        mtime = os.path.getmtime(ui_state_path)
    except Exception:
        mtime = None
    print(
        "[RP_UI_WRITE] "
        f"requested=high path={os.path.abspath(ui_state_path)} "
        f"version={str(state_obj.get('version', '') or '')} "
        f"changed={bool((change or {}).get('changed', False))} mtime={mtime}"
    )

    cycle2 = cycle1 + 1
    engine.current_cycle = cycle2
    engine._maybe_apply_runtime_risk_profile(cycle_id=cycle2, now_utc=datetime.now(timezone.utc))

    applied = str(engine.active_risk_profile or "").strip().lower()
    requested = str(engine.requested_risk_profile or "").strip().lower()
    if applied != "high" or requested != "high":
        return _fail(
            f"expected requested/applied=high, got requested={requested!r} applied={applied!r}"
        )

    print(
        "[PASS] verify_risk_profile_apply "
        f"cycle={cycle2} requested={requested} applied={applied}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
