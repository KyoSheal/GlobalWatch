"""Batch quote prefetch + TTL cache service for cycle-level price reuse."""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional


CacheRow = Dict[str, Any]


class PriceService:
    """Price cache with batch yfinance prefetch and lightweight stats."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 45,
        missing_ttl_seconds: int = 5,
        get_yfinance_module: Optional[Callable[[], Any]] = None,
        symbol_mapper: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds or 45))
        self.missing_ttl_seconds = max(1, int(missing_ttl_seconds or 5))
        self._get_yfinance_module = get_yfinance_module
        self._symbol_mapper = symbol_mapper or (lambda t: t)
        self._cache: Dict[str, CacheRow] = {}
        self._lock = threading.Lock()
        self._prefetch_lock = threading.Lock()

    # ---------- cache helpers ----------
    def normalize_ticker(self, ticker: Any) -> str:
        return str(ticker or "").strip().upper()

    def should_refresh(self, ticker: Any, now_utc: Optional[datetime] = None) -> bool:
        key = self.normalize_ticker(ticker)
        if not key:
            return False
        now = now_utc or datetime.now(timezone.utc)
        with self._lock:
            row = self._cache.get(key)
        if not isinstance(row, dict):
            return True
        fetched_at = row.get("fetched_at")
        if not isinstance(fetched_at, datetime):
            return True
        row_source = str(row.get("source", "") or "").strip().lower()
        ttl = self.missing_ttl_seconds if row_source == "missing" else self.ttl_seconds
        age_sec = max(0.0, (now - fetched_at).total_seconds())
        return bool(age_sec > float(ttl))

    def get_cached(self, ticker: Any, now_utc: Optional[datetime] = None) -> Optional[CacheRow]:
        key = self.normalize_ticker(ticker)
        if not key:
            return None
        if self.should_refresh(key, now_utc=now_utc):
            return None
        with self._lock:
            row = self._cache.get(key)
            return dict(row) if isinstance(row, dict) else None

    def update_cache(self, ticker: Any, row: CacheRow) -> None:
        key = self.normalize_ticker(ticker)
        if not key:
            return
        payload = dict(row or {})
        fetched_at = payload.get("fetched_at")
        if not isinstance(fetched_at, datetime):
            payload["fetched_at"] = datetime.now(timezone.utc)
        with self._lock:
            self._cache[key] = payload

    @staticmethod
    def _coerce_price(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            price = float(value)
            if price != price:  # NaN
                return None
            return price
        except Exception:
            return None

    def _is_usable_cached_row(self, row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        return self._coerce_price(row.get("price")) is not None

    # ---------- provider helpers ----------
    def _get_yf(self) -> Any:
        if callable(self._get_yfinance_module):
            try:
                mod = self._get_yfinance_module()
                if mod is not None:
                    return mod
            except Exception:
                return None
        try:
            import yfinance as yf_mod  # lazy import

            return yf_mod
        except Exception:
            return None

    def _to_symbol(self, ticker: str) -> str:
        try:
            mapped = self._symbol_mapper(ticker)
            return str(mapped or ticker).strip()
        except Exception:
            return str(ticker).strip()

    # ---------- dataframe parsing ----------
    @staticmethod
    def _extract_close_series(df: Any, symbol: str, pd_mod: Any) -> Any:
        if df is None or getattr(df, "empty", True):
            return None

        columns = getattr(df, "columns", None)
        if columns is None:
            return None

        # Multi-index output (most common for yf.download with group_by="ticker")
        if isinstance(columns, pd_mod.MultiIndex):
            # Pattern 1: first level is ticker, second level is OHLCV
            level0 = set(columns.get_level_values(0))
            if symbol in level0:
                try:
                    sub = df[symbol]
                    if "Close" in sub.columns:
                        return sub["Close"]
                except Exception:
                    pass
            # Pattern 2: first level is OHLCV, second level is ticker
            level1 = set(columns.get_level_values(1))
            if symbol in level1:
                try:
                    close = df.xs("Close", axis=1, level=0)
                    if symbol in close.columns:
                        return close[symbol]
                except Exception:
                    pass
            return None

        # Single-index output (single symbol or simplified output)
        if "Close" in columns:
            try:
                return df["Close"]
            except Exception:
                return None
        return None

    @staticmethod
    def _series_last_valid(close_series: Any) -> tuple[Optional[float], Optional[datetime], bool]:
        if close_series is None:
            return None, None, False
        try:
            valid = close_series.dropna()
            if valid.empty:
                return None, None, False
            price = float(valid.iloc[-1])
            idx = valid.index[-1]
        except Exception:
            return None, None, False

        ts = None
        tz_ok = False
        if isinstance(idx, datetime):
            ts = idx
            tz_ok = bool(ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None)
        return price, ts, tz_ok

    def _build_row(
        self,
        *,
        price: Optional[float],
        price_ts: Optional[datetime],
        fetched_at: datetime,
        source: str,
        bar_interval: Optional[str],
        tz_ok: bool,
        notes: str = "",
        raw_price_ts: Any = None,
        raw_tz: Any = None,
    ) -> CacheRow:
        return {
            "price": float(price) if price is not None else None,
            "price_ts": price_ts,
            "fetched_at": fetched_at,
            "source": str(source),
            "bar_interval": bar_interval,
            "tz_ok": bool(tz_ok),
            "notes": str(notes or ""),
            "raw_price_ts": raw_price_ts,
            "raw_tz": raw_tz,
        }

    def _batch_download(
        self,
        yf_mod: Any,
        symbols: list[str],
        *,
        interval: str,
        period: str,
    ) -> tuple[Dict[str, CacheRow], list[str], Optional[str], str]:
        try:
            import pandas as pd_mod  # lazy import
        except Exception:
            pd_mod = None
        if pd_mod is None:
            missing = [str(s) for s in symbols]
            return {}, missing, "ImportError", "pandas_missing"

        now_utc = datetime.now(timezone.utc)
        out: Dict[str, CacheRow] = {}
        missing: list[str] = []

        try:
            data = yf_mod.download(
                tickers=symbols,
                period=period,
                interval=interval,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            for symbol in symbols:
                missing.append(symbol)
            return out, missing, e.__class__.__name__, "download_error"

        for symbol in symbols:
            close_series = self._extract_close_series(data, symbol, pd_mod)
            price, price_ts, tz_ok = self._series_last_valid(close_series)
            if price is None:
                missing.append(symbol)
                continue
            out[symbol] = self._build_row(
                price=price,
                price_ts=price_ts,
                fetched_at=now_utc,
                source=f"yfinance_download_{interval}",
                bar_interval=interval,
                tz_ok=tz_ok,
                raw_price_ts=str(price_ts) if price_ts is not None else None,
                raw_tz=str(getattr(price_ts, "tzinfo", "")) if price_ts is not None else None,
            )

        return out, missing, None, "ok"

    # ---------- public batch API ----------
    def prefetch(
        self,
        tickers: Iterable[str],
        *,
        interval: str = "5m",
        period: str = "1d",
        max_chunk: int = 50,
        allow_1m_fallback: bool = True,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        now_utc = datetime.now(timezone.utc)
        max_chunk = max(1, int(max_chunk or 50))

        normalized: list[str] = []
        for t in list(tickers or []):
            key = self.normalize_ticker(t)
            if key and key not in normalized:
                normalized.append(key)

        stats: Dict[str, Any] = {
            "status": "ok",
            "tickers_in": len(normalized),
            "tickers_fetched": 0,
            "missing": [],
            "batch_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "symbols_unique": 0,
            "missing_after_pass1": 0,
            "fetched_by_1m": 0,
            "tz_ok_false": 0,
            "error_type": None,
            "provider": "yfinance",
            "elapsed_ms": 0,
        }

        if not normalized:
            stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000.0)
            return stats

        if not self._prefetch_lock.acquire(blocking=False):
            stats["status"] = "inflight_skip"
            for t in normalized:
                row = self.get_cached(t, now_utc=now_utc)
                if self._is_usable_cached_row(row):
                    stats["cache_hits"] += 1
                else:
                    stats["cache_misses"] += 1
            stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000.0)
            return stats

        try:
            # Cache gate.
            to_fetch: list[str] = []
            for t in normalized:
                row = self.get_cached(t, now_utc=now_utc)
                if self._is_usable_cached_row(row):
                    stats["cache_hits"] += 1
                else:
                    stats["cache_misses"] += 1
                    to_fetch.append(t)

            if not to_fetch:
                stats["tickers_fetched"] = len(normalized)
                stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000.0)
                return stats

            yf_mod = self._get_yf()
            if yf_mod is None:
                stats["status"] = "provider_unavailable"
                stats["missing"] = list(to_fetch)
                stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000.0)
                return stats

            # Build ticker<->symbol maps.
            ticker_to_symbol: Dict[str, str] = {}
            symbol_to_tickers: Dict[str, list[str]] = {}
            symbols: list[str] = []
            for t in to_fetch:
                sym = self._to_symbol(t)
                ticker_to_symbol[t] = sym
                if sym not in symbol_to_tickers:
                    symbol_to_tickers[sym] = []
                    symbols.append(sym)
                symbol_to_tickers[sym].append(t)
            stats["symbols_unique"] = int(len(symbols))

            symbol_rows: Dict[str, CacheRow] = {}
            symbol_missing: list[str] = []
            had_download_error = False

            # Pass 1: requested interval.
            for i in range(0, len(symbols), max_chunk):
                chunk = symbols[i : i + max_chunk]
                stats["batch_calls"] += 1
                rows, missing, error_type, batch_status = self._batch_download(
                    yf_mod,
                    chunk,
                    interval=interval,
                    period=period,
                )
                symbol_rows.update(rows)
                symbol_missing.extend(missing)
                if batch_status == "pandas_missing":
                    stats["status"] = "pandas_missing"
                    stats["error_type"] = error_type
                    stats["missing"] = list(to_fetch)
                    stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000.0)
                    return stats
                if batch_status == "download_error":
                    had_download_error = True
                    if stats.get("error_type") is None:
                        stats["error_type"] = error_type

            missing_after_pass1 = [s for s in symbol_missing if s not in symbol_rows]
            stats["missing_after_pass1"] = int(len(missing_after_pass1))

            # Pass 2: optional 1m fallback for unresolved symbols.
            if allow_1m_fallback and interval != "1m":
                fallback_symbols = list(missing_after_pass1)
                cap = max_chunk * 2
                if len(fallback_symbols) > cap:
                    fallback_symbols = fallback_symbols[:cap]
                if fallback_symbols:
                    for i in range(0, len(fallback_symbols), max_chunk):
                        chunk = fallback_symbols[i : i + max_chunk]
                        stats["batch_calls"] += 1
                        rows, _missing, error_type, batch_status = self._batch_download(
                            yf_mod,
                            chunk,
                            interval="1m",
                            period=period,
                        )
                        if batch_status == "download_error":
                            had_download_error = True
                            if stats.get("error_type") is None:
                                stats["error_type"] = error_type
                        before = len(symbol_rows)
                        for sym, row in rows.items():
                            row["source"] = "yfinance_download_1m"
                            row["bar_interval"] = "1m"
                            symbol_rows[sym] = row
                        stats["fetched_by_1m"] += max(0, len(symbol_rows) - before)

            # Fan-out rows from symbols to requested tickers.
            final_missing: list[str] = []
            for t in to_fetch:
                sym = ticker_to_symbol.get(t, t)
                row = symbol_rows.get(sym)
                if isinstance(row, dict) and self._coerce_price(row.get("price")) is not None:
                    if not bool(row.get("tz_ok", False)):
                        stats["tz_ok_false"] += 1
                    self.update_cache(t, row)
                else:
                    final_missing.append(t)
                    self.update_cache(
                        t,
                        self._build_row(
                            price=None,
                            price_ts=None,
                            fetched_at=now_utc,
                            source="missing",
                            bar_interval=None,
                            tz_ok=False,
                            notes=f"missing_after_batch interval={interval}",
                        ),
                    )

            stats["tickers_fetched"] = len(normalized) - len(final_missing)
            stats["missing"] = final_missing
            if had_download_error and stats["tickers_fetched"] == 0:
                stats["status"] = "download_error"
            stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000.0)
            return stats
        finally:
            self._prefetch_lock.release()
