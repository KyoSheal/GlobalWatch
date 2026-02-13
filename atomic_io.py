"""Atomic file IO helpers for concurrent readers/writers."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from typing import Any, Optional


def _ensure_parent_dir(path: str) -> str:
    target = os.path.abspath(str(path))
    folder = os.path.dirname(target) or "."
    os.makedirs(folder, exist_ok=True)
    return target


def _replace_with_retry(
    tmp_path: str,
    target_path: str,
    *,
    max_attempts: int = 20,
    sleep_ms: int = 50,
) -> None:
    delay = max(1, int(sleep_ms)) / 1000.0
    attempts = max(1, int(max_attempts))
    last_exc: Optional[BaseException] = None
    for idx in range(attempts):
        try:
            os.replace(tmp_path, target_path)
            return
        except PermissionError as exc:
            last_exc = exc
            if idx >= attempts - 1:
                raise
            time.sleep(delay * float(idx + 1))
    if last_exc is not None:
        raise last_exc


def atomic_write_text(path: str, content: str) -> None:
    """Atomically replace a UTF-8 text file."""
    target = _ensure_parent_dir(path)
    base = os.path.basename(target)
    folder = os.path.dirname(target) or "."
    prefix = f".{base}.{uuid.uuid4().hex}."
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(content or ""))
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def atomic_write_json(path: str, obj: Any, *, indent: int = 2) -> None:
    """Atomically replace a JSON file."""
    payload = json.dumps(obj, ensure_ascii=False, indent=indent, allow_nan=False)
    atomic_write_text(path, payload)


def atomic_write_jsonl(path: str, rows: list[dict]) -> None:
    """Atomically rewrite a JSONL file."""
    data_rows = rows if isinstance(rows, list) else []
    lines = [json.dumps(row, ensure_ascii=False, allow_nan=False) for row in data_rows]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    atomic_write_text(path, content)


def safe_read_json(path: str, retries: int = 3, sleep_ms: int = 30) -> dict | None:
    """Safely read a JSON object with transient parse retries."""
    attempts = max(1, int(retries))
    delay = max(0, int(sleep_ms)) / 1000.0
    for idx in range(attempts):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else None
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, PermissionError, OSError):
            if idx >= attempts - 1:
                return None
            time.sleep(delay)
    return None
