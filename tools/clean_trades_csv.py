"""Clean paper_trades.csv by removing likely test/smoke contamination rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import tempfile
from typing import Dict, List


def _norm(v) -> str:
    return str(v or "").strip()


def _parse_cycle(v):
    text = _norm(v)
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def clean_csv(
    input_path: str,
    output_path: str,
    allowed_envs: List[str],
    account_id: str | None,
    session_id: str | None,
    max_cycle: int | None,
    ticker_blacklist: List[str],
    dry_run: bool = False,
) -> Dict[str, int]:
    stats: Dict[str, int] = {
        "total": 0,
        "kept": 0,
        "dropped_env": 0,
        "dropped_account": 0,
        "dropped_session": 0,
        "dropped_cycle": 0,
        "dropped_ticker": 0,
    }
    allowed_env_set = {x.lower() for x in allowed_envs if _norm(x)}
    ticker_black_set = {x.upper() for x in ticker_blacklist if _norm(x)}

    rows = []
    fieldnames = None
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            stats["total"] += 1
            if not isinstance(row, dict):
                continue

            ticker = _norm(row.get("ticker")).upper()
            if ticker and ticker in ticker_black_set:
                stats["dropped_ticker"] += 1
                continue

            row_env = _norm(row.get("env")).lower()
            if row_env and allowed_env_set and row_env not in allowed_env_set:
                stats["dropped_env"] += 1
                continue

            if account_id:
                row_acc = _norm(row.get("account_id"))
                if row_acc and row_acc != account_id:
                    stats["dropped_account"] += 1
                    continue

            if session_id:
                row_sid = _norm(row.get("session_id"))
                if row_sid and row_sid != session_id:
                    stats["dropped_session"] += 1
                    continue

            if max_cycle is not None:
                c = _parse_cycle(row.get("cycle"))
                if c is not None and c > max_cycle:
                    stats["dropped_cycle"] += 1
                    continue

            rows.append(row)
            stats["kept"] += 1

    if dry_run:
        return stats

    folder = os.path.dirname(output_path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_clean_trades_", suffix=".csv", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames or [])
            if fieldnames:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean contaminated trades csv rows.")
    parser.add_argument("--input", default=os.path.join("outputs", "paper_trades.csv"), help="Input trades csv path.")
    parser.add_argument("--output", default=None, help="Output csv path. Default: overwrite input.")
    parser.add_argument("--allowed-envs", default="live", help="Comma separated env whitelist.")
    parser.add_argument("--account-id", default=None, help="Only keep rows for this account_id when field exists.")
    parser.add_argument("--session-id", default=None, help="Only keep rows for this session_id when field exists.")
    parser.add_argument("--max-cycle", type=int, default=None, help="Drop rows with cycle > max-cycle.")
    parser.add_argument("--ticker-blacklist", default="AAA,TEST,SMOKE", help="Comma separated ticker blacklist.")
    parser.add_argument("--dry-run", action="store_true", help="Only print stats, no file write.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backup when overwriting input.")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output or args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if output_path == input_path and not args.no_backup and not args.dry_run:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{input_path}.{ts}.bak"
        shutil.copy2(input_path, backup_path)
        print(f"[backup] {backup_path}")

    stats = clean_csv(
        input_path=input_path,
        output_path=output_path,
        allowed_envs=[x.strip() for x in str(args.allowed_envs).split(",") if x.strip()],
        account_id=_norm(args.account_id) or None,
        session_id=_norm(args.session_id) or None,
        max_cycle=args.max_cycle,
        ticker_blacklist=[x.strip() for x in str(args.ticker_blacklist).split(",") if x.strip()],
        dry_run=bool(args.dry_run),
    )
    print(stats)
    if args.dry_run:
        print("[dry-run] no file written")
    else:
        print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

