#!/usr/bin/env python3
"""A3-2: multi-cycle replay window + attribution."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quant_io_utils import parse_iso_to_utc, safe_read_json
from quant_replay import load_price_debug, run_single_cycle_replay, write_replay_outputs

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _normalize_ticker(t: Any) -> str:
    return str(t or "").strip().upper()


def _normalize_weights(weights: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (weights or {}).items():
        tk = _normalize_ticker(k)
        if not tk:
            continue
        w = _num_or_none(v)
        if w is None:
            continue
        out[tk] = max(0.0, float(w))
    if "CASH" not in out:
        out["CASH"] = 0.0
    total = sum(out.values())
    if total <= 0:
        return {"CASH": 1.0}
    for k in list(out.keys()):
        out[k] = float(out[k] / total)
    return out


def _extract_price_debug_from_obj(obj: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj, dict):
        for tk, row in obj.items():
            if not isinstance(row, dict):
                continue
            ticker = _normalize_ticker(tk)
            if not ticker:
                continue
            out[ticker] = {
                "price": _num_or_none(row.get("price") or row.get("current_price")),
                "price_ts": str(row.get("price_ts") or row.get("ts") or row.get("timestamp") or ""),
                "status": str(row.get("status") or ""),
                "source": str(row.get("source") or ""),
                "bar_interval": row.get("bar_interval"),
                "tz_ok": row.get("tz_ok"),
            }
    elif isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            ticker = _normalize_ticker(row.get("ticker"))
            if not ticker:
                continue
            out[ticker] = {
                "price": _num_or_none(row.get("price") or row.get("current_price")),
                "price_ts": str(row.get("price_ts") or row.get("ts") or row.get("timestamp") or ""),
                "status": str(row.get("status") or ""),
                "source": str(row.get("source") or ""),
                "bar_interval": row.get("bar_interval"),
                "tz_ok": row.get("tz_ok"),
            }
    return out


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {c: row.get(c, "") for c in columns}
            w.writerow(out)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_cycle_from_obj(obj: Dict[str, Any]) -> Optional[int]:
    c = _num_or_none(obj.get("cycle") or obj.get("cycle_id"))
    if c is None:
        return None
    try:
        return int(c)
    except Exception:
        return None


def _extract_time_utc(obj: Dict[str, Any]) -> str:
    for k in ("time_utc", "ts", "time", "timestamp", "now_utc", "snapshot_time"):
        if k not in obj:
            continue
        dt = parse_iso_to_utc(obj.get(k))
        if dt is not None:
            return dt.isoformat(timespec="seconds")
    return ""


def discover_cycles(
    *,
    run_dir: Path,
    cycles_spec: Optional[Tuple[int, int, int]],
    start_cycle: Optional[int],
    end_cycle: Optional[int],
    step: int,
    max_cycles: int,
) -> Tuple[List[int], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    snapshots: Dict[int, Dict[str, Any]] = {}
    source = ""

    # preferred cycle snapshots jsonl
    for candidate in (run_dir / "cycle_snapshots.jsonl", run_dir / "portfolio_snapshots.jsonl"):
        if not candidate.exists():
            continue
        source = str(candidate)
        with candidate.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if isinstance(obj.get("snapshot"), dict):
                    snap = dict(obj.get("snapshot"))
                    if "cycle" not in snap and "cycle_id" not in snap:
                        if obj.get("cycle") is not None:
                            snap["cycle"] = obj.get("cycle")
                        elif obj.get("cycle_id") is not None:
                            snap["cycle_id"] = obj.get("cycle_id")
                    obj = snap
                c = _parse_cycle_from_obj(obj)
                if c is None:
                    continue
                snapshots[c] = obj
        if snapshots:
            break

    if not snapshots:
        live = safe_read_json(run_dir / "snapshot_live.json")
        if isinstance(live, dict):
            c = _parse_cycle_from_obj(live) or 0
            snapshots[c] = live
            source = str((run_dir / "snapshot_live.json").resolve())

    if not snapshots:
        return [], {}, {"source": source, "total": 0}

    all_cycles = sorted(snapshots.keys())

    if cycles_spec is not None:
        s, e, st = cycles_spec
        selected = [c for c in all_cycles if c >= s and c <= e and ((c - s) % max(1, st) == 0)]
    elif start_cycle is not None or end_cycle is not None:
        s = start_cycle if start_cycle is not None else all_cycles[0]
        e = end_cycle if end_cycle is not None else all_cycles[-1]
        st = max(1, int(step or 1))
        selected = [c for c in all_cycles if c >= s and c <= e and ((c - s) % st == 0)]
    else:
        selected = list(all_cycles)

    if len(selected) > int(max_cycles):
        selected = selected[: int(max_cycles)]

    return selected, snapshots, {"source": source, "total": len(all_cycles)}


def _load_price_by_cycle(run_dir: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for name in ("price_debug_by_cycle.json", "price_debug_cycle_map.json"):
        p = (run_dir / name).resolve()
        if not p.exists():
            continue
        obj = safe_read_json(p)
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            try:
                cycle = int(float(str(k)))
            except Exception:
                continue
            rows = _extract_price_debug_from_obj(v)
            if rows:
                out[cycle] = rows
    return out


def _required_tickers(snapshot: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    pos = snapshot.get("positions") if isinstance(snapshot.get("positions"), dict) else {}
    for tk in pos.keys():
        t = _normalize_ticker(tk)
        if t and t != "CASH":
            out.append(t)
    for key in ("target_weights", "planned_target_weights"):
        w = snapshot.get(key)
        if isinstance(w, dict):
            for tk in w.keys():
                t = _normalize_ticker(tk)
                if t and t != "CASH":
                    out.append(t)
    return sorted(set(out))


def _parse_events_for_reference(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    events_path = run_dir / "telemetry" / "events.jsonl"
    if not events_path.exists():
        return out
    with events_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            c = _parse_cycle_from_obj(obj)
            if c is None:
                continue
            event = str(obj.get("event") or "")
            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
            if event in ("REBALANCE_PLAN", "CYCLE_METRICS"):
                ref = out.setdefault(c, {})
                if isinstance(payload.get("target_weights"), dict):
                    ref["target_weights"] = payload.get("target_weights")
                if isinstance(payload.get("planned_trades"), list):
                    ref["planned_trades"] = payload.get("planned_trades")
                if isinstance(payload.get("gate"), dict):
                    ref["gate"] = payload.get("gate")
                ref["source"] = f"telemetry:{event}"
    return out


def load_reference_for_cycle(
    *,
    cycle: int,
    snapshot: Dict[str, Any],
    run_dir: Path,
    events_ref: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    # A) snapshot embedded reference
    for key in ("reference", "replay_reference"):
        r = snapshot.get(key)
        if isinstance(r, dict):
            return {
                "status": "ok",
                "source": f"snapshot.{key}",
                "target_weights": r.get("target_weights") if isinstance(r.get("target_weights"), dict) else {},
                "planned_trades": r.get("planned_trades") if isinstance(r.get("planned_trades"), list) else [],
                "gate": r.get("gate") if isinstance(r.get("gate"), dict) else {},
                "price_source": str(r.get("price_source") or ""),
            }

    ref_w = None
    for key in ("reference_target_weights", "ref_target_weights", "target_weights_ref"):
        if isinstance(snapshot.get(key), dict):
            ref_w = snapshot.get(key)
            break
    ref_t = None
    for key in ("reference_planned_trades", "ref_planned_trades"):
        if isinstance(snapshot.get(key), list):
            ref_t = snapshot.get(key)
            break
    ref_g = None
    for key in ("reference_gate", "ref_gate"):
        if isinstance(snapshot.get(key), dict):
            ref_g = snapshot.get(key)
            break
    if ref_w is not None or ref_t is not None or ref_g is not None:
        return {
            "status": "ok",
            "source": "snapshot.inline_ref",
            "target_weights": ref_w or {},
            "planned_trades": ref_t or [],
            "gate": ref_g or {},
            "price_source": str(snapshot.get("reference_price_source") or ""),
        }

    # B) references_by_cycle file
    for name in ("references_by_cycle.json", "replay_reference_by_cycle.json"):
        p = (run_dir / name).resolve()
        if not p.exists():
            continue
        obj = safe_read_json(p)
        if not isinstance(obj, dict):
            continue
        row = obj.get(str(cycle))
        if not isinstance(row, dict):
            row = obj.get(int(cycle)) if isinstance(obj.get(int(cycle)), dict) else None
        if isinstance(row, dict):
            return {
                "status": "ok",
                "source": str(p),
                "target_weights": row.get("target_weights") if isinstance(row.get("target_weights"), dict) else {},
                "planned_trades": row.get("planned_trades") if isinstance(row.get("planned_trades"), list) else [],
                "gate": row.get("gate") if isinstance(row.get("gate"), dict) else {},
                "price_source": str(row.get("price_source") or ""),
            }

    # C) telemetry-derived reference
    row = events_ref.get(cycle)
    if isinstance(row, dict):
        return {
            "status": "ok",
            "source": row.get("source") or "telemetry",
            "target_weights": row.get("target_weights") if isinstance(row.get("target_weights"), dict) else {},
            "planned_trades": row.get("planned_trades") if isinstance(row.get("planned_trades"), list) else [],
            "gate": row.get("gate") if isinstance(row.get("gate"), dict) else {},
            "price_source": str(row.get("price_source") or ""),
        }

    return {"status": "missing", "source": "", "target_weights": {}, "planned_trades": [], "gate": {}, "price_source": ""}


def diff_weights(ref_weights: Dict[str, Any], replay_weights: Dict[str, Any]) -> Dict[str, Any]:
    ref = _normalize_weights(ref_weights if isinstance(ref_weights, dict) else {})
    rep = _normalize_weights(replay_weights if isinstance(replay_weights, dict) else {})
    keys = sorted(set(ref.keys()) | set(rep.keys()))
    deltas: List[Dict[str, Any]] = []
    l1 = 0.0
    for k in keys:
        a = float(ref.get(k, 0.0))
        b = float(rep.get(k, 0.0))
        d = b - a
        l1 += abs(d)
        if abs(d) > 1e-12:
            deltas.append({"ticker": k, "ref": a, "replay": b, "delta": d, "abs_delta": abs(d)})
    deltas.sort(key=lambda x: (-float(x["abs_delta"]), str(x["ticker"])))
    return {"weights_l1": l1, "top_deltas": deltas[:10], "count_changed": len(deltas)}


def _normalize_trade_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        tk = _normalize_ticker(row.get("ticker"))
        if not tk:
            continue
        side = str(row.get("side") or "").upper()
        if not side:
            continue
        v = _num_or_none(row.get("desired_trade_value") or row.get("notional") or row.get("trade_value"))
        if v is None:
            v = 0.0
        key = f"{tk}|{side}"
        out[key] = {"ticker": tk, "side": side, "value": float(v)}
    return out


def diff_trades(ref_trades: List[Dict[str, Any]], replay_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    ref = _normalize_trade_rows(ref_trades if isinstance(ref_trades, list) else [])
    rep = _normalize_trade_rows(replay_trades if isinstance(replay_trades, list) else [])

    ref_ticker_side = {(v["ticker"], v["side"]): v for v in ref.values()}
    rep_ticker_side = {(v["ticker"], v["side"]): v for v in rep.values()}

    ref_ticker_only: Dict[str, str] = {}
    rep_ticker_only: Dict[str, str] = {}
    for (t, s) in ref_ticker_side.keys():
        ref_ticker_only[t] = s
    for (t, s) in rep_ticker_side.keys():
        rep_ticker_only[t] = s

    added = sorted([k for k in rep.keys() if k not in ref.keys()])
    removed = sorted([k for k in ref.keys() if k not in rep.keys()])
    side_changes = sorted([t for t in set(ref_ticker_only.keys()) & set(rep_ticker_only.keys()) if ref_ticker_only[t] != rep_ticker_only[t]])

    all_keys = sorted(set(ref.keys()) | set(rep.keys()))
    notional_delta = 0.0
    for k in all_keys:
        a = float((ref.get(k) or {}).get("value", 0.0))
        b = float((rep.get(k) or {}).get("value", 0.0))
        notional_delta += abs(b - a)

    return {
        "added": added,
        "removed": removed,
        "side_changes": side_changes,
        "notional_delta": notional_delta,
        "count_ref": len(ref),
        "count_replay": len(rep),
    }


def attribute_diff(
    *,
    ref_status: str,
    replay_warnings: List[str],
    price_rows: int,
    required_tickers: int,
    source_mismatch: bool,
    weights_diff: Dict[str, Any],
    trades_diff: Dict[str, Any],
    gate_diff: Dict[str, Any],
    strict: bool,
) -> List[str]:
    tags: List[str] = []
    if ref_status != "ok":
        tags.append("REF_MISSING")
    if required_tickers > 0 and price_rows < required_tickers:
        tags.append("PRICE_COVERAGE_LOW")
    if source_mismatch:
        tags.append("SOURCE_MISMATCH")
    if float(weights_diff.get("weights_l1", 0.0) or 0.0) > 1e-9 or float(trades_diff.get("notional_delta", 0.0) or 0.0) > 1e-6 or bool(gate_diff.get("changed", False)):
        tags.append("INPUT_DRIFT")
    if replay_warnings and not strict:
        tags.append("NONDETERMINISM_WARNING")
    return sorted(set(tags))


def _parse_cycles_spec(spec: str) -> Optional[Tuple[int, int, int]]:
    text = str(spec or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        s = int(parts[0])
        e = int(parts[1])
        st = max(1, int(parts[2]))
        if e < s:
            s, e = e, s
        return s, e, st
    except Exception:
        return None


def run_replay_window(
    *,
    run_dir: Path,
    cycles_spec: Optional[str],
    start_cycle: Optional[int],
    end_cycle: Optional[int],
    step: int,
    out_dir: Path,
    strict: bool,
    compare_ref: bool,
    max_cycles: int,
    fail_on_drift: bool,
    verbose: bool,
) -> Tuple[int, Dict[str, Any]]:
    started = _now_utc_iso()
    spec = _parse_cycles_spec(cycles_spec or "")
    selected_cycles, snapshots_by_cycle, discover_info = discover_cycles(
        run_dir=run_dir,
        cycles_spec=spec,
        start_cycle=start_cycle,
        end_cycle=end_cycle,
        step=step,
        max_cycles=max_cycles,
    )

    if not selected_cycles:
        manifest = {
            "schema_version": 1,
            "status": "fail",
            "reason": "no_cycles_discovered",
            "run_dir": str(run_dir),
            "started_at_utc": started,
            "finished_at_utc": _now_utc_iso(),
            "steps_ok": False,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(out_dir / "replay_window_manifest.json", manifest)
        return 2, manifest

    price_by_cycle = _load_price_by_cycle(run_dir)
    events_ref = _parse_events_for_reference(run_dir) if compare_ref else {}

    per_cycle_rows: List[Dict[str, Any]] = []
    tag_counter: Counter = Counter()
    warnings: List[str] = []
    ref_ok = 0
    ref_missing = 0
    drift_fail = False

    for cycle in selected_cycles:
        snapshot = dict(snapshots_by_cycle.get(cycle) or {})
        if "cycle" not in snapshot and "cycle_id" not in snapshot:
            snapshot["cycle"] = cycle

        pd_map, pd_info = load_price_debug(snapshot, run_dir)
        if cycle in price_by_cycle:
            pd_map.update(price_by_cycle[cycle])
            pd_info["sources_checked"] = list(pd_info.get("sources_checked", [])) + ["price_debug_by_cycle.json"]
            pd_info["count"] = len(pd_map)

        replay = run_single_cycle_replay(
            snapshot=snapshot,
            price_debug=pd_map,
            strict=bool(strict),
            fail_on_gate=False,
        )

        cycle_dir = (out_dir / "per_cycle" / str(cycle)).resolve()
        out_paths = write_replay_outputs(
            out_dir=cycle_dir,
            result=replay,
            snapshot_source=str(discover_info.get("source") or ""),
            price_source=pd_info,
            strict=bool(strict),
        )

        time_utc = _extract_time_utc(snapshot)
        required = _required_tickers(snapshot)
        ref = load_reference_for_cycle(cycle=cycle, snapshot=snapshot, run_dir=run_dir, events_ref=events_ref) if compare_ref else {
            "status": "disabled",
            "source": "",
            "target_weights": {},
            "planned_trades": [],
            "gate": {},
            "price_source": "",
        }

        if ref.get("status") == "ok":
            ref_ok += 1
        elif compare_ref:
            ref_missing += 1

        weights_diff = diff_weights(ref.get("target_weights") or {}, replay.target_weights)
        trades_diff = diff_trades(ref.get("planned_trades") or [], replay.planned_trades)
        gate_ref = ref.get("gate") if isinstance(ref.get("gate"), dict) else {}
        gate_diff = {
            "changed": bool(gate_ref) and (
                bool(gate_ref.get("gate_fail", False)) != bool(replay.gate.get("gate_fail", False))
                or str(gate_ref.get("reason", "")) != str(replay.gate.get("reason", ""))
            )
        }

        ref_price_source = str(ref.get("price_source") or "")
        replay_sources_text = "|".join([str(x) for x in (pd_info.get("sources_checked") or [])])
        source_mismatch = bool(ref_price_source) and (ref_price_source not in replay_sources_text)

        tags = attribute_diff(
            ref_status=str(ref.get("status") or "missing"),
            replay_warnings=replay.warnings,
            price_rows=int(replay.price_info.get("count", 0) or 0),
            required_tickers=len(required),
            source_mismatch=source_mismatch,
            weights_diff=weights_diff,
            trades_diff=trades_diff,
            gate_diff=gate_diff,
            strict=bool(strict),
        )
        for t in tags:
            tag_counter[t] += 1

        diff_obj = {
            "schema_version": 1,
            "cycle": cycle,
            "ref_status": ref.get("status"),
            "ref_source": ref.get("source"),
            "weights_diff": weights_diff,
            "trades_diff": trades_diff,
            "gate_diff": gate_diff,
            "attribution_tags": tags,
        }
        _write_json_atomic(cycle_dir / "diff.json", diff_obj)

        target_hash = _canonical_hash({k: round(float(v), 12) for k, v in sorted(replay.target_weights.items())})
        trades_hash = _canonical_hash(
            [
                {
                    "ticker": str(r.get("ticker")),
                    "side": str(r.get("side")),
                    "desired_trade_value": round(float(r.get("desired_trade_value", 0.0) or 0.0), 10),
                }
                for r in sorted(replay.planned_trades, key=lambda x: (str(x.get("ticker")), str(x.get("side"))))
            ]
        )

        row = {
            "cycle": cycle,
            "time_utc": time_utc,
            "price_rows": int(replay.price_info.get("count", 0) or 0),
            "num_trades": len(replay.planned_trades),
            "target_hash": target_hash,
            "trades_hash": trades_hash,
            "gate_fail": bool(replay.gate.get("gate_fail", False)),
            "warnings_count": len(replay.warnings),
            "ref_status": str(ref.get("status") or "missing"),
            "attribution_tags": "|".join(tags),
            "weights_l1": float(weights_diff.get("weights_l1", 0.0) or 0.0),
            "trades_notional_delta": float(trades_diff.get("notional_delta", 0.0) or 0.0),
            "diff_path": (f"per_cycle/{cycle}/diff.json" if compare_ref else ""),
            "decision_path": f"per_cycle/{cycle}/replay_decision.md",
        }
        per_cycle_rows.append(row)

        if replay.exit_code == 2:
            warnings.append(f"cycle_{cycle}:replay_input_unavailable")
            if strict:
                manifest = {
                    "schema_version": 1,
                    "status": "fail",
                    "reason": f"replay_failed_cycle_{cycle}",
                    "run_dir": str(run_dir),
                    "started_at_utc": started,
                    "finished_at_utc": _now_utc_iso(),
                    "strict": bool(strict),
                    "compare_ref": bool(compare_ref),
                    "warnings": warnings,
                }
                _write_json_atomic(out_dir / "replay_window_manifest.json", manifest)
                return 2, manifest

        if compare_ref and strict and ref.get("status") != "ok":
            warnings.append(f"cycle_{cycle}:reference_missing")
            manifest = {
                "schema_version": 1,
                "status": "fail",
                "reason": f"reference_missing_cycle_{cycle}",
                "run_dir": str(run_dir),
                "started_at_utc": started,
                "finished_at_utc": _now_utc_iso(),
                "strict": bool(strict),
                "compare_ref": bool(compare_ref),
                "warnings": warnings,
            }
            _write_json_atomic(out_dir / "replay_window_manifest.json", manifest)
            return 2, manifest

        if fail_on_drift and "INPUT_DRIFT" in tags:
            drift_fail = True

        if replay.warnings:
            warnings.extend([f"cycle_{cycle}:{w}" for w in replay.warnings])

    # deterministic row order
    per_cycle_rows.sort(key=lambda r: int(r.get("cycle") or 0))

    summary_csv = out_dir / "replay_window_summary.csv"
    _write_csv(
        summary_csv,
        per_cycle_rows,
        [
            "cycle",
            "time_utc",
            "price_rows",
            "num_trades",
            "target_hash",
            "trades_hash",
            "gate_fail",
            "warnings_count",
            "ref_status",
            "attribution_tags",
            "weights_l1",
            "trades_notional_delta",
            "diff_path",
            "decision_path",
        ],
    )

    ref_cov = (ref_ok / len(per_cycle_rows)) if per_cycle_rows else 0.0
    top_diff = sorted(per_cycle_rows, key=lambda r: (-float(r.get("weights_l1", 0.0) or 0.0), -float(r.get("trades_notional_delta", 0.0) or 0.0), int(r.get("cycle") or 0)))[:10]

    report_lines: List[str] = []
    report_lines.append("# Replay Window Report")
    report_lines.append("")
    report_lines.append(f"- Run Dir: `{run_dir}`")
    report_lines.append(f"- Cycles: `{selected_cycles[0]}..{selected_cycles[-1]}` ({len(per_cycle_rows)} cycles)")
    report_lines.append(f"- Strict: `{bool(strict)}`")
    report_lines.append(f"- Compare Ref: `{bool(compare_ref)}`")
    report_lines.append(f"- Ref Coverage: `{ref_ok}/{len(per_cycle_rows)} ({ref_cov:.1%})`")
    report_lines.append("")
    report_lines.append("## Attribution Tags (Top)")
    if tag_counter:
        for tag, cnt in tag_counter.most_common(10):
            report_lines.append(f"- {tag}: {cnt}")
    else:
        report_lines.append("- none")
    report_lines.append("")
    report_lines.append("## Top Diff Cycles")
    if top_diff:
        report_lines.append("| cycle | weights_l1 | trades_notional_delta | tags |")
        report_lines.append("|---:|---:|---:|---|")
        for r in top_diff:
            report_lines.append(
                f"| {r.get('cycle')} | {float(r.get('weights_l1',0.0)):.6f} | {float(r.get('trades_notional_delta',0.0)):.2f} | {r.get('attribution_tags','')} |"
            )
    else:
        report_lines.append("- none")
    report_lines.append("")
    if warnings:
        report_lines.append("## Warnings")
        for w in sorted(set(warnings))[:40]:
            report_lines.append(f"- {w}")
        report_lines.append("")
    report_lines.append("## Outputs")
    report_lines.append("- `replay_window_manifest.json`")
    report_lines.append("- `replay_window_summary.csv`")
    report_lines.append("- `per_cycle/<cycle>/diff.json`")
    report_lines.append("")

    report_path = out_dir / "replay_window_report.md"
    _write_text(report_path, "\n".join(report_lines))

    summary_hash = _canonical_hash(
        [
            {
                "cycle": int(r.get("cycle") or 0),
                "target_hash": r.get("target_hash"),
                "trades_hash": r.get("trades_hash"),
                "attribution_tags": r.get("attribution_tags"),
            }
            for r in per_cycle_rows
        ]
    )

    manifest = {
        "schema_version": 1,
        "status": "ok",
        "run_dir": str(run_dir),
        "started_at_utc": started,
        "finished_at_utc": _now_utc_iso(),
        "strict": bool(strict),
        "compare_ref": bool(compare_ref),
        "cycle_range": {
            "start": selected_cycles[0],
            "end": selected_cycles[-1],
            "count": len(selected_cycles),
        },
        "discover": discover_info,
        "ref_coverage": {
            "ok": ref_ok,
            "missing": ref_missing,
            "ratio": ref_cov,
        },
        "warnings": sorted(set(warnings)),
        "steps_ok": True,
        "summary_hash": summary_hash,
        "paths": {
            "summary_csv": str(summary_csv.resolve()),
            "report_md": str(report_path.resolve()),
        },
    }
    _write_json_atomic(out_dir / "replay_window_manifest.json", manifest)

    rc = 0
    if strict and warnings:
        rc = 2
    elif (not strict) and warnings:
        rc = 1
    if fail_on_drift and drift_fail:
        rc = 3

    if verbose:
        print(f"[A10] run_dir={run_dir}")
        print(f"[A10] cycles={selected_cycles[0]}..{selected_cycles[-1]} count={len(selected_cycles)}")
        print(f"[A10] ref_coverage={ref_ok}/{len(selected_cycles)}")
        print(f"[A10] summary_hash={summary_hash}")

    return rc, manifest


def write_window_outputs(*, out_dir: Path, manifest: Dict[str, Any]) -> None:
    _write_json_atomic(out_dir / "replay_window_manifest.json", manifest)
