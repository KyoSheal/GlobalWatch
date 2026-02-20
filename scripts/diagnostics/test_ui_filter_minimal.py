#!/usr/bin/env python3
"""T22: minimal UI risk-profile filter logic test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_CHOICES = {"low", "mid", "high", "ultra"}


def norm_profile(x) -> str:
    s = str(x or "").strip().lower()
    return s if s in _CHOICES else "mid"


def filter_metrics(rows, selected_filter):
    if str(selected_filter) == "All":
        return list(rows)
    return [
        row
        for row in rows
        if norm_profile(
            (row.get("payload") or {}).get("active_risk_profile")
            or row.get("active_risk_profile")
        ) == selected_filter
    ]


def filter_events(rows, selected_filter):
    if str(selected_filter) == "All":
        return list(rows)
    return [
        row
        for row in rows
        if norm_profile(
            (row.get("payload") or {}).get("active_risk_profile")
            or row.get("active_risk_profile")
        ) == selected_filter
    ]


def filter_trades(rows, selected_filter):
    if str(selected_filter) == "All":
        return list(rows)
    return [
        row
        for row in rows
        if norm_profile(row.get("risk_profile")) == selected_filter
    ]


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    metrics_rows = [
        {"id": "m_missing", "payload": {"foo": 1}},  # -> mid
        {"id": "m_high", "active_risk_profile": "high"},
        {"id": "m_invalid", "active_risk_profile": "manual"},  # -> mid
    ]
    events_rows = [
        {"id": "e_ultra", "payload": {"active_risk_profile": "ultra"}},
        {"id": "e_missing", "payload": {"foo": 2}},  # -> mid
        {"id": "e_invalid", "active_risk_profile": "???"},  # -> mid
    ]
    trades_rows = [
        {"id": "t_missing"},  # -> mid
        {"id": "t_low", "risk_profile": "low"},
        {"id": "t_invalid", "risk_profile": "abc"},  # -> mid
    ]

    # All returns full set.
    if len(filter_metrics(metrics_rows, "All")) != len(metrics_rows):
        return _fail("metrics All filter did not return full set")
    if len(filter_events(events_rows, "All")) != len(events_rows):
        return _fail("events All filter did not return full set")
    if len(filter_trades(trades_rows, "All")) != len(trades_rows):
        return _fail("trades All filter did not return full set")

    # high returns only the explicit high row.
    m_high = filter_metrics(metrics_rows, "high")
    if [r.get("id") for r in m_high] != ["m_high"]:
        return _fail(f"metrics high filter mismatch: {[r.get('id') for r in m_high]}")
    e_high = filter_events(events_rows, "high")
    if e_high:
        return _fail(f"events high filter should be empty, got ids={[r.get('id') for r in e_high]}")
    t_high = filter_trades(trades_rows, "high")
    if t_high:
        return _fail(f"trades high filter should be empty, got ids={[r.get('id') for r in t_high]}")

    # mid returns normalized missing/invalid rows.
    m_mid_ids = [r.get("id") for r in filter_metrics(metrics_rows, "mid")]
    if set(m_mid_ids) != {"m_missing", "m_invalid"}:
        return _fail(f"metrics mid filter mismatch: {m_mid_ids}")
    e_mid_ids = [r.get("id") for r in filter_events(events_rows, "mid")]
    if set(e_mid_ids) != {"e_missing", "e_invalid"}:
        return _fail(f"events mid filter mismatch: {e_mid_ids}")
    t_mid_ids = [r.get("id") for r in filter_trades(trades_rows, "mid")]
    if set(t_mid_ids) != {"t_missing", "t_invalid"}:
        return _fail(f"trades mid filter mismatch: {t_mid_ids}")

    print("[PASS] ui_filter_minimal")
    print(
        f"[INFO] metrics_mid={m_mid_ids} events_mid={e_mid_ids} trades_mid={t_mid_ids}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

