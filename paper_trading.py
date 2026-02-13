"""Paper trading engine (simulation only)."""

import json
import io
import os
import sys
import time
import uuid
import hashlib
import argparse
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # NOTE: comment omitted (was garbled/non-ASCII).
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from atomic_io import (
    atomic_write_json as io_atomic_write_json,
    atomic_write_jsonl as io_atomic_write_jsonl,
    atomic_write_text as io_atomic_write_text,
)
from price_service import PriceService

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import daily_reporter
    DAILY_REPORTER_AVAILABLE = True
except Exception as e:
    daily_reporter = None  # type: ignore
    DAILY_REPORTER_AVAILABLE = False
    print(f"[WARN] daily_reporter unavailable: {e}")

try:
    from market_session import get_market_session_state, is_market_open_for_trading
    MARKET_SESSION_AVAILABLE = True
except Exception as e:
    MARKET_SESSION_AVAILABLE = False
    print(f"[WARN] market_session unavailable: {e}")

    def get_market_session_state(now_dt, tz_market="America/New_York", open_time_et="09:30", close_time_et="16:00", open_grace_min=15, close_grace_min=10):
        now_fallback = datetime.now(timezone.utc)
        return {
            "state": "OPEN",
            "now_et": now_fallback.isoformat(),
            "now_utc": now_fallback.isoformat(),
            "trading_date_et": now_fallback.date().isoformat(),
            "last_completed_trading_date_et": now_fallback.date().isoformat(),
            "open_time_et": str(open_time_et),
            "close_time_et": str(close_time_et),
            "is_weekend": False,
            "is_holiday": False,
            "open_grace_passed": True,
            "close_grace_passed": False,
            "open_grace_min": int(open_grace_min),
            "close_grace_min": int(close_grace_min),
        }

    def is_market_open_for_trading(session_dict):
        return True

# ChromaDB for macro signals
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[WARN] ChromaDB not available - macro integration disabled")

# NOTE: comment omitted (was garbled/non-ASCII).
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# NOTE: comment omitted (was garbled/non-ASCII).
REAL_BROKER_KEYWORDS = ['alpaca', 'interactive_brokers', 'ib_insync', 'robinhood', 'td_ameritrade']
for keyword in REAL_BROKER_KEYWORDS:
    try:
        __import__(keyword)
        raise RuntimeError(f"[SAFETY] Violation: detected real broker library '{keyword}'. Paper trading is simulation only.")
    except ImportError:
        pass  # Good, no real broker library


class MacroSignalAdapter:
    """class MacroSignalAdapter: docstring omitted (was garbled/non-ASCII)."""
    
    def __init__(self, config):
        """def __init__: docstring omitted (was garbled/non-ASCII)."""
        self.config = config
        self.macro_config = config.get('macro_integration', {})
        self.enabled = self.macro_config.get('enabled', False) and CHROMADB_AVAILABLE
        self.quality_window = int(self.macro_config.get('quality_window', 50))
        if self.quality_window <= 0:
            self.quality_window = 50
        self.theme_accuracy_history = {}
        self.source_accuracy_history = {}
        self.quality_seen_signal_ids = set()
        
        if not self.enabled:
            print("[MACRO] Macro integration disabled")
            return
        
        try:
            chroma_path = self.macro_config.get('chroma_path', './memory_db')
            collection_name = self.macro_config.get('collection', 'trading_signals')
            
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.signals_collection = self.chroma_client.get_collection(name=collection_name)
            
            print(f"[MACRO] Connected to ChromaDB: {chroma_path}/{collection_name}")
        except Exception as e:
            print(f"[MACRO] Failed to connect to ChromaDB: {e}")
            self.enabled = False

    def _extract_source_key(self, metadata):
        """def _extract_source_key: docstring omitted (was garbled/non-ASCII)."""
        source = (
            metadata.get('source')
            or metadata.get('source_name')
            or metadata.get('publisher')
            or metadata.get('channel')
            or metadata.get('origin')
            or 'unknown'
        )
        return str(source).strip().lower()

    def _to_float_optional(self, value):
        """def _to_float_optional: docstring omitted (was garbled/non-ASCII)."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_correct_flag(self, value):
        """def _parse_correct_flag: docstring omitted (was garbled/non-ASCII)."""
        if value is None:
            return None

        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (int, float)):
            return float(np.clip(value, 0.0, 1.0))

        if isinstance(value, str):
            text = value.strip().lower()
            if text in ('true', 't', 'yes', 'y', '1'):
                return 1.0
            if text in ('false', 'f', 'no', 'n', '0'):
                return 0.0
            try:
                return float(np.clip(float(text), 0.0, 1.0))
            except ValueError:
                return None

        return None

    def _append_rolling_accuracy(self, history_map, key, correct_value):
        """def _append_rolling_accuracy: docstring omitted (was garbled/non-ASCII)."""
        if correct_value is None:
            return

        values = history_map.setdefault(key, [])
        values.append(float(correct_value))

        if len(values) > self.quality_window:
            del values[:-self.quality_window]

    def _update_quality_calibration(self, signals):
        """def _update_quality_calibration: docstring omitted (was garbled/non-ASCII)."""
        summary = {
            'verified_count': 0,
            'with_correct_1d': 0,
            'with_correct_4h': 0,
            'with_return_1d': 0,
            'new_quality_updates': 0
        }

        for signal in signals:
            metadata = signal.get('metadata', {})
            status = str(metadata.get('status', 'UNKNOWN')).upper()
            correct_1d = self._parse_correct_flag(metadata.get('correct_1d'))
            correct_4h = self._parse_correct_flag(metadata.get('correct_4h'))
            return_1d = self._to_float_optional(metadata.get('return_1d'))

            if status == 'VERIFIED':
                summary['verified_count'] += 1
            if correct_1d is not None:
                summary['with_correct_1d'] += 1
            if correct_4h is not None:
                summary['with_correct_4h'] += 1
            if return_1d is not None:
                summary['with_return_1d'] += 1

            signal_id = signal.get('id')
            if signal_id is not None and signal_id in self.quality_seen_signal_ids:
                continue

            if signal_id is not None:
                self.quality_seen_signal_ids.add(signal_id)

            if correct_1d is None:
                continue

            theme_key = str(metadata.get('theme', 'unknown')).strip().lower()
            source_key = self._extract_source_key(metadata)

            self._append_rolling_accuracy(self.theme_accuracy_history, theme_key, correct_1d)
            self._append_rolling_accuracy(self.source_accuracy_history, source_key, correct_1d)
            summary['new_quality_updates'] += 1

        return summary

    def _get_accuracy_factor(self, theme, source):
        """def _get_accuracy_factor: docstring omitted (was garbled/non-ASCII)."""
        theme_key = str(theme or 'unknown').strip().lower()
        source_key = str(source or 'unknown').strip().lower()

        if source_key in self.source_accuracy_history and self.source_accuracy_history[source_key]:
            acc = float(np.mean(self.source_accuracy_history[source_key]))
            scope = f"source:{source_key}"
        elif theme_key in self.theme_accuracy_history and self.theme_accuracy_history[theme_key]:
            acc = float(np.mean(self.theme_accuracy_history[theme_key]))
            scope = f"theme:{theme_key}"
        else:
            acc = 0.5
            scope = "default"

        accuracy_factor = float(np.clip(0.7 + 0.6 * (acc - 0.5), 0.7, 1.3))
        return accuracy_factor, acc, scope

    def _parse_topic_outcome(self, metadata):
        """Infer topic outcome (+1/-1/0) from metadata if available."""
        correct_1d = self._parse_correct_flag(metadata.get('correct_1d'))
        if correct_1d is not None:
            if correct_1d > 0.5:
                return 1
            if correct_1d < 0.5:
                return -1
            return 0

        return_1d = self._to_float_optional(metadata.get('return_1d'))
        topic_score = self._to_float_optional(metadata.get('topic_score', metadata.get('topic_score_raw')))
        if return_1d is None or topic_score is None:
            return None
        if abs(topic_score) <= 1e-12 or abs(return_1d) <= 1e-12:
            return 0
        return 1 if (return_1d * topic_score) > 0 else -1

    def _compute_topic_accuracy(self, outcomes):
        """Compute topic accuracy from recent outcomes."""
        window = int(self.macro_config.get('topic_memory_window', 50))
        window = max(1, window)
        sample = list(outcomes or [])[-window:]
        informative = [x for x in sample if x in (-1, 1)]
        if not informative:
            return 0.5, 0
        accuracy = float(sum(1 for x in informative if x == 1) / len(informative))
        return accuracy, len(informative)

    def _topic_accuracy_to_weight(self, accuracy):
        """Map accuracy bands to adaptive tilt multiplier."""
        acc = float(accuracy)
        if acc < 0.40:
            return 0.75
        if acc > 0.60:
            return 1.25
        return 1.00
    
    def fetch_recent_signals(self, n=50):
        """def fetch_recent_signals: docstring omitted (was garbled/non-ASCII)."""
        if not self.enabled:
            return []
        
        try:
            # NOTE: comment omitted (was garbled/non-ASCII).
            results = self.signals_collection.get(
                include=['metadatas', 'documents']
            )
            
            if not results['ids']:
                print("[MACRO] No signals found in database")
                return []
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            signals = []
            for i, metadata in enumerate(results['metadatas']):
                status = metadata.get('status', 'UNKNOWN')
                
                if status in ['PENDING', 'VERIFIED']:
                    signals.append({
                        'id': results['ids'][i],
                        'metadata': metadata,
                        'document': results['documents'][i] if i < len(results['documents']) else ''
                    })
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            signals.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            recent_signals = signals[:n]
            
            print(f"[MACRO] Fetched {len(recent_signals)} recent signals (from {len(signals)} valid)")
            
            return recent_signals
            
        except Exception as e:
            print(f"[MACRO] Error fetching signals: {e}")
            return []
    
    def compute_signal_weight(self, signal_timestamp):
        """def compute_signal_weight: docstring omitted (was garbled/non-ASCII)."""
        try:
            # NOTE: comment omitted (was garbled/non-ASCII).
            signal_time = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
            now = datetime.now(signal_time.tzinfo) if signal_time.tzinfo else datetime.now()
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            age_hours = (now - signal_time).total_seconds() / 3600
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            decay_lambda = self.macro_config.get('decay_lambda_per_hour', 0.15)
            weight = np.exp(-decay_lambda * age_hours)
            
            return weight, age_hours
            
        except Exception as e:
            print(f"[MACRO] Error computing weight: {e}")
            return 0.0, 0.0
    
    def analyze_signals(self):
        """def analyze_signals: docstring omitted (was garbled/non-ASCII)."""
        if not self.enabled:
            return 0.0, [], {}, {}
        
        print(f"\n[MACRO] Analyzing macro signals from GlobalWatch...")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        all_signals = self.fetch_recent_signals(n=200)
        
        if not all_signals:
            print("[MACRO] No signals to analyze")
            return 0.0, [], {}, {}

        # NOTE: comment omitted (was garbled/non-ASCII).
        quality_summary = self._update_quality_calibration(all_signals)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        confirm_k, confirm_n = self.macro_config.get('confirm_k_of_n', [2, 3])
        signal_max_age_hours = self.macro_config.get('signal_max_age_hours', 48)
        decay_lambda = self.macro_config.get('decay_lambda_per_hour', 0.15)
        
        now = datetime.now()
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        valid_signals = []
        for signal in all_signals:
            metadata = signal['metadata']
            timestamp_str = metadata.get('timestamp', '')
            
            try:
                signal_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                # NOTE: comment omitted (was garbled/non-ASCII).
                if signal_time.tzinfo:
                    now_aware = datetime.now(signal_time.tzinfo)
                else:
                    now_aware = now
                    signal_time = signal_time.replace(tzinfo=None)
                
                age_hours = (now_aware - signal_time).total_seconds() / 3600
                
                if age_hours > signal_max_age_hours:
                    continue  # NOTE: comment omitted (was garbled/non-ASCII).
                
                valid_signals.append({
                    'metadata': metadata,
                    'document': signal.get('document', ''),
                    'timestamp': timestamp_str,
                    'age_hours': age_hours,
                    'status': str(metadata.get('status', 'UNKNOWN')).upper(),
                    'correct_1d': self._parse_correct_flag(metadata.get('correct_1d')),
                    'correct_4h': self._parse_correct_flag(metadata.get('correct_4h')),
                    'return_1d': self._to_float_optional(metadata.get('return_1d')),
                    'source_key': self._extract_source_key(metadata)
                })
                
            except Exception as e:
                # NOTE: comment omitted (was garbled/non-ASCII).
                continue
        
        print(f"[MACRO] Filtered {len(valid_signals)}/{len(all_signals)} signals within {signal_max_age_hours}h window")
        
        if not valid_signals:
            print("[MACRO] No valid signals after age filtering")
            return 0.0, [], {}, {}
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        theme_groups = {}
        for sig in valid_signals:
            theme = sig['metadata'].get('theme', 'unknown')
            if theme not in theme_groups:
                theme_groups[theme] = []
            theme_groups[theme].append(sig)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        for theme in theme_groups:
            theme_groups[theme].sort(key=lambda x: x['timestamp'], reverse=True)
            theme_groups[theme] = theme_groups[theme][:confirm_n]
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        confirmed_topics = []
        
        print(f"\n[MACRO] Theme Confirmation (k={confirm_k}, n={confirm_n}, max_age={signal_max_age_hours}h):")
        print(f"{'Theme':<20} {'Bull':>5} {'Bear':>5} {'Neut':>5} {'Status':<20}")
        print("-" * 70)
        
        for theme, signals_list in sorted(theme_groups.items(), key=lambda x: len(x[1]), reverse=True):
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            bullish_items = []
            bearish_items = []
            
            for sig in signals_list:
                metadata = sig['metadata']
                direction = metadata.get('direction', 'neutral').lower()
                confidence = metadata.get('confidence', 50.0) / 100.0
                age_hours = sig['age_hours']
                theme_key = metadata.get('theme', theme)
                source_key = sig.get('source_key', self._extract_source_key(metadata))
                accuracy_factor, rolling_acc, accuracy_scope = self._get_accuracy_factor(theme_key, source_key)
                confidence_effective = confidence * accuracy_factor
                
                if 'bullish' in direction or 'long' in direction:
                    bullish_count += 1
                    bullish_items.append({
                        'confidence': confidence,
                        'confidence_effective': confidence_effective,
                        'accuracy_factor': accuracy_factor,
                        'rolling_accuracy': rolling_acc,
                        'accuracy_scope': accuracy_scope,
                        'age_hours': age_hours,
                        'timestamp': sig['timestamp'],
                        'document': sig['document'],
                        'status': sig.get('status', 'UNKNOWN'),
                        'correct_1d': sig.get('correct_1d'),
                        'correct_4h': sig.get('correct_4h'),
                        'return_1d': sig.get('return_1d')
                    })
                elif 'bearish' in direction or 'short' in direction:
                    bearish_count += 1
                    bearish_items.append({
                        'confidence': confidence,
                        'confidence_effective': confidence_effective,
                        'accuracy_factor': accuracy_factor,
                        'rolling_accuracy': rolling_acc,
                        'accuracy_scope': accuracy_scope,
                        'age_hours': age_hours,
                        'timestamp': sig['timestamp'],
                        'document': sig['document'],
                        'status': sig.get('status', 'UNKNOWN'),
                        'correct_1d': sig.get('correct_1d'),
                        'correct_4h': sig.get('correct_4h'),
                        'return_1d': sig.get('return_1d')
                    })
                else:
                    neutral_count += 1
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            confirmed_direction = None
            confirmed_items = []
            
            if bullish_count >= confirm_k and bullish_count > bearish_count:
                confirmed_direction = 'bullish'
                confirmed_items = bullish_items
                status = f"[OK] BULLISH ({bullish_count}/{len(signals_list)})"
            elif bearish_count >= confirm_k and bearish_count > bullish_count:
                confirmed_direction = 'bearish'
                confirmed_items = bearish_items
                status = f"[OK] BEARISH ({bearish_count}/{len(signals_list)})"
            else:
                # NOTE: comment omitted (was garbled/non-ASCII).
                max_count = max(bullish_count, bearish_count)
                needed = confirm_k - max_count
                if needed > 0:
                    status = f"[WAIT] NEED {needed} MORE"
                else:
                    status = f"[WARN] CONFLICTED ({bullish_count}v{bearish_count})"
            
            print(f"{theme:<20} {bullish_count:>5} {bearish_count:>5} {neutral_count:>5} {status:<20}")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if confirmed_direction:
                strength = 0.0
                confidence_raw_sum = 0.0
                confidence_effective_sum = 0.0
                accuracy_factor_sum = 0.0
                
                for item in confirmed_items:
                    time_weight = np.exp(-decay_lambda * item['age_hours'])
                    strength += item['confidence_effective'] * time_weight
                    confidence_raw_sum += item['confidence']
                    confidence_effective_sum += item['confidence_effective']
                    accuracy_factor_sum += item['accuracy_factor']
                
                confidence_raw_avg = confidence_raw_sum / len(confirmed_items) if confirmed_items else 0.0
                confidence_effective = confidence_effective_sum / len(confirmed_items) if confirmed_items else 0.0
                accuracy_factor_avg = accuracy_factor_sum / len(confirmed_items) if confirmed_items else 1.0
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                top_sources = []
                for item in confirmed_items[:3]:
                    doc_preview = item['document'][:100] if item['document'] else 'N/A'
                    top_sources.append(doc_preview)
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                newest_timestamp = confirmed_items[0]['timestamp'] if confirmed_items else ''
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                confirmed_topics.append({
                    'theme': theme,
                    'direction': confirmed_direction,
                    'strength': strength,
                    'confidence_raw': confidence_raw_avg,
                    'confidence_effective': confidence_effective,
                    'accuracy_factor': accuracy_factor_avg,
                    'top_sources': top_sources,
                    'newest_timestamp': newest_timestamp,
                    'count': len(confirmed_items)
                })
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                print(f"  [DEBUG] {theme}: direction={confirmed_direction}, "
                      f"count={bullish_count if confirmed_direction=='bullish' else bearish_count}/"
                      f"{bearish_count if confirmed_direction=='bullish' else bullish_count}, "
                      f"strength={strength:.3f}, conf_raw={confidence_raw_avg:.3f}, "
                      f"conf_eff={confidence_effective:.3f}, acc_factor={accuracy_factor_avg:.3f}, "
                      f"newest={newest_timestamp[:19]}")
        
        print("-" * 70)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        risk_off_themes = ['risk_off', 'recession', 'rates_up', 'credit_stress', 'inflation_risk', 
                           'geopolitical_risk', 'market_crash', 'volatility_spike']
        risk_on_themes = ['risk_on', 'soft_landing', 'growth_acceleration', 'dovish_fed', 
                          'earnings_beat', 'tech_rally']
        
        risk_score = 0.0
        
        for topic in confirmed_topics:
            theme_lower = topic['theme'].lower()
            strength = topic['strength']
            direction = topic['direction']
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            is_risk_off = any(keyword in theme_lower for keyword in risk_off_themes)
            # NOTE: comment omitted (was garbled/non-ASCII).
            is_risk_on = any(keyword in theme_lower for keyword in risk_on_themes)
            
            if is_risk_off:
                if direction == 'bearish':
                    # NOTE: comment omitted (was garbled/non-ASCII).
                    risk_score += min(strength * 2.0, 3.0)  # NOTE: comment omitted (was garbled/non-ASCII).
                elif direction == 'bullish':
                    # NOTE: comment omitted (was garbled/non-ASCII).
                    risk_score += min(strength * 1.5, 2.5)
            
            elif is_risk_on:
                if direction == 'bullish':
                    # NOTE: comment omitted (was garbled/non-ASCII).
                    risk_score -= min(strength * 1.0, 2.0)
                elif direction == 'bearish':
                    # NOTE: comment omitted (was garbled/non-ASCII).
                    risk_score += min(strength * 1.0, 2.0)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        risk_score = max(0.0, min(risk_score, 10.0))
        
        print(f"\n[MACRO] Risk Score: {risk_score:.1f}/10.0 (from {len(confirmed_topics)} confirmed topics)")
        print(f"[MACRO] Confirmed Topics: {len(confirmed_topics)}")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        macro_tilts = self._generate_tilts(confirmed_topics)
        topic_tilts, applied_sector_topics = self._generate_topic_tilts_from_signals(valid_signals)
        if topic_tilts:
            tilt_max_delta = float(self.macro_config.get('tilt_max_delta', 0.02))
            for ticker, tilt in topic_tilts.items():
                macro_tilts[ticker] = float(np.clip(macro_tilts.get(ticker, 0.0) + tilt, -tilt_max_delta, tilt_max_delta))
        
        if macro_tilts:
            print(f"[MACRO] Asset Tilts:")
            for ticker, tilt in macro_tilts.items():
                print(f"  {ticker}: {tilt:+.2%}")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        signal_summary = {
            'total_signals_fetched': len(all_signals),
            'valid_signals_in_window': len(valid_signals),
            'themes_analyzed': len(theme_groups),
            'confirmed_topics': len(confirmed_topics),
            'llm_topic_signals_applied': len(applied_sector_topics),
            'llm_topic_sectors': [x.get('sector') for x in applied_sector_topics],
            'risk_score': risk_score,
            'quality_verified_count': quality_summary.get('verified_count', 0),
            'quality_with_correct_1d': quality_summary.get('with_correct_1d', 0),
            'quality_with_correct_4h': quality_summary.get('with_correct_4h', 0),
            'quality_with_return_1d': quality_summary.get('with_return_1d', 0),
            'quality_new_updates': quality_summary.get('new_quality_updates', 0),
            'quality_window': self.quality_window
        }
        
        return risk_score, confirmed_topics, macro_tilts, signal_summary
    
    def _generate_tilts(self, confirmed_topics):
        """def _generate_tilts: docstring omitted (was garbled/non-ASCII)."""
        macro_mapping = self.config.get('macro_mapping', {})
        tilt_max_delta = self.macro_config.get('tilt_max_delta', 0.02)
        
        tilts = {}
        
        for topic in confirmed_topics:
            theme = topic['theme'].lower()
            direction = topic['direction']
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            for rule_name, rule_config in macro_mapping.items():
                # NOTE: comment omitted (was garbled/non-ASCII).
                if rule_name.lower() in theme or theme in rule_name.lower():
                    
                    # NOTE: comment omitted (was garbled/non-ASCII).
                    if 'tilt' in rule_config:
                        for ticker, tilt_value in rule_config['tilt'].items():
                            # NOTE: comment omitted (was garbled/non-ASCII).
                            if direction == 'bearish':
                                tilt_value = -abs(tilt_value)  # NOTE: comment omitted (was garbled/non-ASCII).
                            
                            # NOTE: comment omitted (was garbled/non-ASCII).
                            current_tilt = tilts.get(ticker, 0.0)
                            new_tilt = current_tilt + tilt_value
                            
                            # NOTE: comment omitted (was garbled/non-ASCII).
                            tilts[ticker] = max(-tilt_max_delta, min(new_tilt, tilt_max_delta))
        
        return tilts

    def _generate_topic_tilts_from_signals(self, valid_signals):
        """Convert optional LLM topic sentiment signals into ticker tilts."""
        if not bool(self.macro_config.get('enable_llm_topic_signals', True)):
            return {}, []

        topic_map_cfg = self.macro_config.get('topic_sector_ticker_map', {})
        if not isinstance(topic_map_cfg, dict) or not topic_map_cfg:
            return {}, []
        normalized_topic_map = {}
        for key, tickers in topic_map_cfg.items():
            key_norm = str(key).strip().lower()
            if not key_norm:
                continue
            if isinstance(tickers, list):
                normalized_topic_map[key_norm] = [str(t).upper() for t in tickers if str(t).strip()]
        if not normalized_topic_map:
            return {}, []

        confidence_threshold = float(self.macro_config.get('llm_topic_confidence_threshold', 0.6))
        score_threshold = float(self.macro_config.get('llm_topic_score_threshold', 0.5))
        tilt_scale = float(self.macro_config.get('llm_topic_tilt_scale', 0.02))
        tilt_max_delta = float(self.macro_config.get('tilt_max_delta', 0.02))

        sector_scores = {}
        sector_weights = {}
        sector_details = {}
        sector_outcomes = {}

        for sig in valid_signals:
            metadata = sig.get('metadata', {}) or {}
            signal_type = str(metadata.get('signal_type', '')).strip().lower()
            if signal_type != 'topic_sentiment':
                continue
            sector = str(metadata.get('topic_sector') or metadata.get('sector') or '').strip().lower()
            if not sector:
                continue
            outcome = self._parse_topic_outcome(metadata)
            if outcome is None:
                continue
            sector_outcomes.setdefault(sector, []).append(int(outcome))

        for sig in valid_signals:
            metadata = sig.get('metadata', {}) or {}
            signal_type = str(metadata.get('signal_type', '')).strip().lower()
            if signal_type != 'topic_sentiment':
                continue

            sector = str(metadata.get('topic_sector') or metadata.get('sector') or '').strip().lower()
            if not sector:
                continue
            if sector not in normalized_topic_map:
                continue

            try:
                topic_score = float(metadata.get('topic_score', metadata.get('topic_score_raw', 0.0)))
            except (TypeError, ValueError):
                continue

            confidence_raw = metadata.get('topic_confidence', metadata.get('confidence', 0.0))
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence > 1.0:
                confidence = confidence / 100.0

            if abs(topic_score) < score_threshold or confidence < confidence_threshold:
                continue

            recurrence_raw = metadata.get('topic_recurrence', metadata.get('recurrence', 1.0))
            try:
                recurrence = max(1.0, float(recurrence_raw))
            except (TypeError, ValueError):
                recurrence = 1.0

            meta_adaptive = self._to_float_optional(metadata.get('topic_adaptive_weight'))
            meta_accuracy = self._to_float_optional(metadata.get('topic_accuracy'))
            meta_samples = self._to_float_optional(metadata.get('topic_accuracy_samples'))
            if meta_adaptive is not None and meta_adaptive > 0:
                adaptive_weight = float(np.clip(meta_adaptive, 0.5, 1.5))
                adaptive_accuracy = float(np.clip(meta_accuracy, 0.0, 1.0)) if meta_accuracy is not None else 0.5
                adaptive_samples = int(max(0, meta_samples)) if meta_samples is not None else 0
                adaptive_source = "globalwatch_memory"
            else:
                adaptive_accuracy, adaptive_samples = self._compute_topic_accuracy(sector_outcomes.get(sector, []))
                adaptive_weight = self._topic_accuracy_to_weight(adaptive_accuracy)
                adaptive_source = "local_fallback"

            weight = confidence * min(1.0, recurrence / 3.0) * adaptive_weight
            sector_scores[sector] = sector_scores.get(sector, 0.0) + topic_score * weight
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
            sector_details[sector] = {
                'score': topic_score,
                'confidence': confidence,
                'recurrence': recurrence,
                'adaptive_weight': adaptive_weight,
                'adaptive_accuracy': adaptive_accuracy,
                'adaptive_samples': adaptive_samples,
                'adaptive_source': adaptive_source
            }

        sector_avg_scores = {}
        for sector, weighted_sum in sector_scores.items():
            denom = max(1e-9, sector_weights.get(sector, 0.0))
            sector_avg_scores[sector] = float(weighted_sum / denom)

        topic_tilts = {}
        applied_sector_topics = []
        for sector, avg_score in sorted(sector_avg_scores.items(), key=lambda x: abs(x[1]), reverse=True):
            mapped_tickers = normalized_topic_map.get(sector, [])
            if not mapped_tickers:
                continue

            direction = "Overweight" if avg_score > 0 else "Underweight"
            signed_tilt = float(np.clip(avg_score * tilt_scale, -tilt_max_delta, tilt_max_delta))
            if abs(signed_tilt) <= 1e-12:
                continue

            print(f"[GLOBALWATCH] {direction} {sector} due to LLM {avg_score:+.2f}")
            detail = sector_details.get(sector, {})
            print(
                f"[GLOBALWATCH MEMORY] {sector}: acc={detail.get('adaptive_accuracy', 0.5):.2f}, "
                f"samples={int(detail.get('adaptive_samples', 0))}, "
                f"weight={detail.get('adaptive_weight', 1.0):.2f}, "
                f"source={detail.get('adaptive_source', 'na')}"
            )
            applied_sector_topics.append({
                'sector': sector,
                'score': float(avg_score),
                'tilt': signed_tilt,
                'tickers': list(mapped_tickers),
                'adaptive_weight': float(detail.get('adaptive_weight', 1.0)),
                'adaptive_accuracy': float(detail.get('adaptive_accuracy', 0.5)),
                'adaptive_samples': int(detail.get('adaptive_samples', 0)),
                'adaptive_source': str(detail.get('adaptive_source', 'na'))
            })

            for ticker in mapped_tickers:
                if not ticker:
                    continue
                ticker_u = str(ticker).upper()
                topic_tilts[ticker_u] = float(np.clip(topic_tilts.get(ticker_u, 0.0) + signed_tilt, -tilt_max_delta, tilt_max_delta))

        return topic_tilts, applied_sector_topics


class PaperTradingEngine:
    """class PaperTradingEngine: docstring omitted (was garbled/non-ASCII)."""
    
    def __init__(self, config_path='paper_config.json'):
        """def __init__: docstring omitted (was garbled/non-ASCII)."""
        self.config = self.load_config(config_path)
        self.validate_config()
        self.config_hash = self._compute_config_hash(self.config)
        reporting_cfg = self.config.get('reporting', {})
        self.account_id = str(reporting_cfg.get('account_id', 'paper_main') or 'paper_main').strip()
        self.runtime_env = str(reporting_cfg.get('env', 'live') or 'live').strip().lower()
        configured_session_id = str(os.environ.get('GW_SESSION_ID', '') or '').strip()
        if configured_session_id:
            self.session_id = configured_session_id
        else:
            self.session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.cash = self.config['initial_cash_usd']
        self.initial_cash = self.cash
        self.positions = {}  # {ticker: quantity}
        self.cost_basis = {}  # NOTE: comment omitted (was garbled/non-ASCII).
        self.equity_curve = []  # [(timestamp, equity, cash, positions_value)]
        self.trades_log = []  # NOTE: comment omitted (was garbled/non-ASCII).
        self.portfolio_snapshots = []  # NOTE: comment omitted (was garbled/non-ASCII).
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.start_time = None
        self.end_time = None
        self.current_cycle = 0
        self.peak_equity = self.cash
        self.status = "READY"  # READY/RUNNING/COMPLETED
        self.last_rebalance_time = None  # backward-compatible alias of last successful rebalance time
        self.last_rebalance_attempt_time = None
        self.last_rebalance_success_time = None
        self.current_regime = {}
        self.current_macro = {}
        self.current_stale_info = {}
        self.current_price_debug = {}
        self.current_market_session = {}
        self.current_rebalance_gate = {"allowed": True, "reason": "", "session_state": "UNKNOWN"}
        self.current_rebalance_skipped_reason = ""
        self._debug_session_override = None
        self._debug_now_override = None
        self._debug_now_override_warned = False
        self.current_turnover_info = {}
        self.current_exit_info = {}
        self.current_risk_check_info = {}
        self.current_vol_targeting_info = {'enabled': False, 'status': 'disabled'}
        self.current_cost_est_info = {
            'enabled': False,
            'total': 0.0,
            'fee': 0.0,
            'slippage': 0.0,
            'impact': 0.0,
            'num_trades': 0
        }
        self.current_news_overlay_info = {
            'enabled': False,
            'status': 'disabled',
            'mode': 'risk_only',
            'alpha': 0.08,
            'applied_cash_delta': 0.0,
            'worst_l2': None,
            'worst_delta': 0.0,
            'used_signals': 0,
            'ticker_deltas': {},
            'l2_deltas': {},
        }
        self._industry_chroma_client = None
        self._industry_collection_cache = {}
        self.current_planner_info = {
            'enabled': False,
            'status': 'disabled',
            'turnover_limit': 0.0,
            'turnover_used_forced': 0.0,
            'turnover_used_normal': 0.0,
            'turnover_used_total': 0.0,
            'num_forced': 0,
            'num_normal': 0,
            'num_dropped': 0,
            'num_adv_clipped': 0,
            'num_adv_dropped': 0,
            'adv_limit_enabled': False,
            'adv_limit_frac': 0.0,
            'normal_sorted_by': 'notional',
            'lambda_cost': 1.0,
            'benefit_mode': 'delta_weight',
            'dropped': [],
            'scaled': []
        }
        self.current_holding_blocks = []
        self.forced_until_time = None  # NOTE: comment omitted (was garbled/non-ASCII).
        self.forced_regime_reason = ""
        self.scoreboard_history = []  # 2w scoreboard records
        self.last_diagnostic_hint = ""
        self.current_weights_reused = False
        self.current_macro_reused = False
        self.score_history_by_ticker = {}
        self.hot_momentum_streaks = {}
        self.daily_report_stale_tracker = {
            'streak': 0,
            'ratio_threshold': float(self.config.get('reporting', {}).get('daily_report_stale_ratio_threshold', 0.8)),
            'threshold': int(self.config.get('reporting', {}).get('daily_report_stale_streak_threshold', 3))
        }
        self.latest_daily_report_date = None
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.macro_risk_score_history = []  # NOTE: comment omitted (was garbled/non-ASCII).
        self.macro_smoothing_window = self.config.get('macro_integration', {}).get('smoothing_window', 3)
        self.macro_smoothing_method = self.config.get('macro_integration', {}).get('smoothing_method', 'median')  # 'median' or 'ewma'
        self.macro_ewma_alpha = self.config.get('macro_integration', {}).get('ewma_alpha', 0.4)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.last_macro_cash_target = self.config['objectives']['min_cash_pct']  # NOTE: comment omitted (was garbled/non-ASCII).
        self.macro_cooldown_cycles = self.config.get('macro_integration', {}).get('cooldown_cycles', 2)
        self.macro_cooldown_remaining = 0  # NOTE: comment omitted (was garbled/non-ASCII).
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.price_cache = {}  # {ticker: (price, timestamp)}
        self.price_cache_duration = 60  # NOTE: comment omitted (was garbled/non-ASCII).
        self.current_price_fetch_stats = {}

        # Signal/Macro refresh decoupling state
        execution_config = self.config.get('execution', {})
        price_ttl_seconds = execution_config.get('price_ttl_seconds', 45)
        self.price_batch_chunk_size = max(1, int(execution_config.get('price_batch_chunk_size', 50) or 50))
        self.price_batch_allow_1m_fallback = bool(execution_config.get('price_batch_allow_1m_fallback', True))
        self.price_service = PriceService(
            get_yfinance_module=self._get_yfinance_module,
            symbol_mapper=self._normalize_market_ticker,
            ttl_seconds=price_ttl_seconds,
        )
        self.signal_refresh_minutes = execution_config.get('signal_refresh_minutes', 1440)
        self.macro_refresh_minutes = execution_config.get('macro_refresh_minutes', 60)
        self.min_holding_cycles = int(execution_config.get('min_holding_cycles', 4))
        self.position_entry_cycle = {}
        self.last_signal_time = None
        self.last_macro_time = None
        self.cached_target_weights = {}
        self.cached_macro = {
            'macro_risk_score_raw': 0.0,
            'macro_risk_score_smoothed': 0.0,
            'confirmed_topics': [],
            'macro_tilts': {},
            'macro_tilts_ignored': {},
            'signal_summary': {}
        }
        self.market_data_fetcher = None
        self.price_fetcher = None
        self.returns_cache = {}
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.macro_adapter = MacroSignalAdapter(self.config)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.resume_from_checkpoint()
        self.rebuild_position_entry_cycles()
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        os.makedirs('outputs', exist_ok=True)

        # NOTE: comment omitted (was garbled/non-ASCII).
        self.load_scoreboard_history()
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        np.random.seed(self.config['safety']['random_seed'])
        
        print("[OK] Paper Trading Engine initialized")
        print(f"   Initial Cash: ${self.cash:,.2f}")
        print(f"   Duration: {self.config['duration_hours']} hours")
        print(f"   Rebalance Interval: {self.config['rebalance_minutes']} minutes")
        print(f"   Universe: {len(self.config['universe'])} assets")
        print(f"   CWD: {os.getcwd()}")
        print(f"   Live Snapshot Path: {self.config.get('reporting', {}).get('snapshot_live_path', 'outputs/snapshot_live.json')}")
        print(f"   Trade History Path: {self.config.get('reporting', {}).get('trade_history_path', 'outputs/trade_history.jsonl')}")
    
    def load_config(self, config_path):
        """def load_config: docstring omitted (was garbled/non-ASCII)."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Backward-compatible defaults for decoupled refresh controls
        execution_config = config.setdefault('execution', {})
        execution_config.setdefault('signal_refresh_minutes', 1440)
        execution_config.setdefault('macro_refresh_minutes', 60)
        execution_config.setdefault('rebalance_cooldown_minutes', 0)
        execution_config.setdefault('rebalance_attempt_cooldown_minutes', execution_config.get('rebalance_cooldown_minutes', 0))
        execution_config.setdefault('max_stale_ratio', 0.3)
        execution_config.setdefault('circuit_breaker_forced_days', 1)
        execution_config.setdefault('fill_gap_max', 0.03)
        execution_config.setdefault('fill_gap_max_iters', 2)
        execution_config.setdefault('allow_buy_benchmarks', False)
        execution_config.setdefault('cross_section_top_n', 10)
        execution_config.setdefault('correlation_lookback_days', 60)
        execution_config.setdefault('correlation_threshold', 0.80)
        execution_config.setdefault('volatility_floor', 0.08)
        execution_config.setdefault('min_holding_cycles', 4)
        execution_config.setdefault('allow_high_conviction_override', True)
        execution_config.setdefault('high_conviction_zscore_threshold', 2.5)
        execution_config.setdefault('high_conviction_lead_threshold', 0.20)
        execution_config.setdefault('high_conviction_cash_surplus_buffer', 0.05)
        execution_config.setdefault('enable_high_conviction_weighting', True)
        execution_config.setdefault('max_high_conviction_weight', 0.40)
        execution_config.setdefault('high_conviction_weight_zscore_threshold', 2.5)
        execution_config.setdefault('high_conviction_weight_ratio_threshold', 2.0)
        execution_config.setdefault('enable_short_term_momentum', True)
        execution_config.setdefault('short_momentum_lookback_days', 10)
        execution_config.setdefault('enable_exit_signals', True)
        execution_config.setdefault('exit_signal_lookback_days', 20)
        execution_config.setdefault('exit_signal_action', 'reduce')
        execution_config.setdefault('exit_signal_reduce_factor', 0.5)
        execution_config.setdefault('exit_signal_min_trigger_count', 1)
        execution_config.setdefault('exit_signal_gap_down_pct', 0.03)
        execution_config.setdefault('exit_signal_volume_spike_ratio', 2.0)
        execution_config.setdefault('exit_signal_consecutive_down_days', 3)
        execution_config.setdefault('exit_signal_long_upper_shadow_ratio', 2.0)
        execution_config.setdefault('exit_on_gap_volume', True)
        execution_config.setdefault('exit_gap_down_pct', 0.04)
        execution_config.setdefault('exit_gap_volume_zscore', 2.5)
        execution_config.setdefault('exit_gap_volume_window', 30)
        execution_config.setdefault('enable_score_smoothing', True)
        execution_config.setdefault('score_smoothing_window', 3)
        execution_config.setdefault('max_portfolio_volatility', 0.25)
        execution_config.setdefault('enable_diversity_check', True)
        execution_config.setdefault('max_herfindahl_index', 0.35)
        execution_config.setdefault('portfolio_vol_min_coverage', 0.70)
        execution_config.setdefault('price_ttl_seconds', 45)
        execution_config.setdefault('price_batch_chunk_size', 50)
        execution_config.setdefault('price_batch_allow_1m_fallback', True)
        execution_config.setdefault('max_weight_boost_for_hot', 0.05)
        execution_config.setdefault('hot_zscore_threshold', 1.5)
        execution_config.setdefault('hot_momentum_top_k', 3)
        execution_config.setdefault('hot_persistence_cycles', 2)
        momentum_weights_cfg = execution_config.setdefault('momentum_weights', {})
        momentum_weights_cfg.setdefault('short', 0.4)
        momentum_weights_cfg.setdefault('medium', 0.6)
        stale_policy = execution_config.setdefault('price_stale_policy', {})
        stale_policy.setdefault('allow_buy', ['LIVE', 'RECENT'])
        stale_policy.setdefault('allow_sell', ['LIVE', 'RECENT', 'STALE'])
        regime_config = config.setdefault('regime_filter', {})
        regime_config.setdefault('cash_risk_on', 0.10)
        regime_config.setdefault('cash_neutral', 0.15)
        regime_config.setdefault('cash_risk_off', 0.25)
        macro_config = config.setdefault('macro_integration', {})
        macro_config.setdefault('macro_allow_new_positions', ['TLT', 'GLD'])
        macro_config.setdefault('enable_llm_topic_signals', True)
        macro_config.setdefault('llm_topic_confidence_threshold', 0.6)
        macro_config.setdefault('llm_topic_score_threshold', 0.5)
        macro_config.setdefault('llm_topic_tilt_scale', 0.02)
        macro_config.setdefault('topic_memory_window', 50)
        macro_config.setdefault('topic_sector_ticker_map', {
            'semiconductors': ['NVDA', 'SOXX', 'XLK'],
            'technology': ['XLK', 'MSFT', 'AAPL'],
            'utilities': ['NEE', 'DUK'],
            'energy': ['XOM', 'CVX', 'XLE'],
            'defense': ['LMT', 'RTX'],
            'healthcare': ['XLV', 'JNJ', 'MRK'],
            'financials': ['XLF', 'JPM', 'V']
        })
        config.setdefault('industry_map', {})
        config.setdefault('ticker_tags', {})
        news_overlay_cfg = config.setdefault('news_overlay', {})
        news_overlay_cfg.setdefault('enabled', False)
        news_overlay_cfg.setdefault('industry_collection', 'industry_signals')
        news_overlay_cfg.setdefault('max_age_hours', 48)
        news_overlay_cfg.setdefault('alpha', 0.08)
        news_overlay_cfg.setdefault('mode', 'risk_only')
        news_overlay_cfg.setdefault('min_confidence', 0.55)
        news_overlay_cfg.setdefault('max_abs_delta', 0.10)
        news_overlay_cfg.setdefault('enable_confidence_scaling', True)
        risk_model_config = config.setdefault('risk_model', {})
        risk_model_config.setdefault('enable_cov_diagnostics', True)
        risk_model_config.setdefault('returns_period', '6mo')
        risk_model_config.setdefault('returns_interval', '1d')
        risk_model_config.setdefault('returns_lookback_days', 60)
        risk_model_config.setdefault('min_obs', 30)
        risk_model_config.setdefault('drop_threshold', 0.5)
        risk_model_config.setdefault('shrinkage_alpha', 0.15)
        risk_model_config.setdefault('annualization_factor', 252)
        risk_model_config.setdefault('max_pair_corr_pairs', 3)
        risk_model_config.setdefault('debug_log', False)
        risk_model_config.setdefault('fallback_to_diag_on_error', True)
        risk_model_config.setdefault('use_cov_vol_for_gate', False)
        risk_model_config.setdefault('rc_limit', 0.35)
        risk_model_config.setdefault('min_cov_gate_coverage', 0.6)
        risk_model_config.setdefault('cov_gate_fallback_to_weighted', True)
        reporting_config = config.setdefault('reporting', {})
        reporting_config.setdefault('scoreboard_path', 'outputs/scoreboard.jsonl')
        reporting_config.setdefault('snapshot_live_path', 'outputs/snapshot_live.json')
        reporting_config.setdefault('trade_history_path', 'outputs/trade_history.jsonl')
        reporting_config.setdefault('account_id', 'paper_main')
        reporting_config.setdefault('env', 'live')
        reporting_config.setdefault('daily_report_tz', 'America/Vancouver')
        reporting_config.setdefault('daily_report_dirs', [
            'outputs/Daily Report',
            r'C:\Users\kyosh\Desktop\Project\News\outputs\Daily Report'
        ])
        reporting_config.setdefault('daily_report_stale_ratio_threshold', 0.8)
        reporting_config.setdefault('daily_report_stale_streak_threshold', 3)
        reporting_config.setdefault('max_price_debug_items', 50)
        
        return config

    def _compute_config_hash(self, config_obj):
        """Compute a stable short hash for the loaded config."""
        try:
            serialized = json.dumps(config_obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:12]
        except Exception:
            return "unknown"
    
    def validate_config(self):
        """def validate_config: docstring omitted (was garbled/non-ASCII)."""
        assert self.config['paper_mode'] == True, "paper_mode must be True"
        assert self.config['safety']['no_real_broker'] == True, "no_real_broker must be True"
        assert self.config['safety']['simulation_only'] == True, "simulation_only must be True"
        assert self.config.get('execution', {}).get('signal_refresh_minutes', 1440) > 0, "execution.signal_refresh_minutes must be > 0"
        assert self.config.get('execution', {}).get('macro_refresh_minutes', 60) > 0, "execution.macro_refresh_minutes must be > 0"
        assert self.config.get('execution', {}).get('max_stale_ratio', 0.3) >= 0, "execution.max_stale_ratio must be >= 0"
        assert float(self.config.get('execution', {}).get('rebalance_cooldown_minutes', 0)) >= 0, "execution.rebalance_cooldown_minutes must be >= 0"
        assert float(self.config.get('execution', {}).get('rebalance_attempt_cooldown_minutes', self.config.get('execution', {}).get('rebalance_cooldown_minutes', 0))) >= 0, "execution.rebalance_attempt_cooldown_minutes must be >= 0"
        assert self.config.get('execution', {}).get('circuit_breaker_forced_days', 1) > 0, "execution.circuit_breaker_forced_days must be > 0"
        assert self.config.get('execution', {}).get('fill_gap_max', 0.03) >= 0, "execution.fill_gap_max must be >= 0"
        assert int(self.config.get('execution', {}).get('fill_gap_max_iters', 2)) >= 1, "execution.fill_gap_max_iters must be >= 1"
        assert isinstance(self.config.get('execution', {}).get('allow_buy_benchmarks', False), bool), "execution.allow_buy_benchmarks must be bool"
        assert int(self.config.get('execution', {}).get('cross_section_top_n', 10)) >= 1, "execution.cross_section_top_n must be >= 1"
        assert int(self.config.get('execution', {}).get('correlation_lookback_days', 60)) >= 20, "execution.correlation_lookback_days must be >= 20"
        corr_threshold = float(self.config.get('execution', {}).get('correlation_threshold', 0.80))
        assert 0.0 <= corr_threshold <= 1.0, "execution.correlation_threshold must be in [0,1]"
        assert float(self.config.get('execution', {}).get('volatility_floor', 0.08)) > 0, "execution.volatility_floor must be > 0"
        assert int(self.config.get('execution', {}).get('min_holding_cycles', 4)) >= 0, "execution.min_holding_cycles must be >= 0"
        assert isinstance(self.config.get('execution', {}).get('allow_high_conviction_override', True), bool), "execution.allow_high_conviction_override must be bool"
        assert float(self.config.get('execution', {}).get('high_conviction_zscore_threshold', 2.5)) > 0, "execution.high_conviction_zscore_threshold must be > 0"
        assert float(self.config.get('execution', {}).get('high_conviction_lead_threshold', 0.20)) >= 0, "execution.high_conviction_lead_threshold must be >= 0"
        assert float(self.config.get('execution', {}).get('high_conviction_cash_surplus_buffer', 0.05)) >= 0, "execution.high_conviction_cash_surplus_buffer must be >= 0"
        assert isinstance(self.config.get('execution', {}).get('enable_high_conviction_weighting', True), bool), "execution.enable_high_conviction_weighting must be bool"
        max_hc_weight = float(self.config.get('execution', {}).get('max_high_conviction_weight', 0.40))
        assert 0 < max_hc_weight <= 1.0, "execution.max_high_conviction_weight must be in (0,1]"
        assert float(self.config.get('execution', {}).get('high_conviction_weight_zscore_threshold', 2.5)) > 0, "execution.high_conviction_weight_zscore_threshold must be > 0"
        assert float(self.config.get('execution', {}).get('high_conviction_weight_ratio_threshold', 2.0)) >= 1.0, "execution.high_conviction_weight_ratio_threshold must be >= 1"
        assert isinstance(self.config.get('execution', {}).get('enable_short_term_momentum', True), bool), "execution.enable_short_term_momentum must be bool"
        assert int(self.config.get('execution', {}).get('short_momentum_lookback_days', 10)) >= 2, "execution.short_momentum_lookback_days must be >= 2"
        assert isinstance(self.config.get('execution', {}).get('enable_exit_signals', True), bool), "execution.enable_exit_signals must be bool"
        assert int(self.config.get('execution', {}).get('exit_signal_lookback_days', 20)) >= 5, "execution.exit_signal_lookback_days must be >= 5"
        assert str(self.config.get('execution', {}).get('exit_signal_action', 'reduce')).lower() in ('reduce', 'exit'), "execution.exit_signal_action must be reduce|exit"
        reduce_factor = float(self.config.get('execution', {}).get('exit_signal_reduce_factor', 0.5))
        assert 0.0 <= reduce_factor <= 1.0, "execution.exit_signal_reduce_factor must be in [0,1]"
        assert int(self.config.get('execution', {}).get('exit_signal_min_trigger_count', 1)) >= 1, "execution.exit_signal_min_trigger_count must be >= 1"
        assert float(self.config.get('execution', {}).get('exit_signal_gap_down_pct', 0.03)) > 0, "execution.exit_signal_gap_down_pct must be > 0"
        assert float(self.config.get('execution', {}).get('exit_signal_volume_spike_ratio', 2.0)) >= 1.0, "execution.exit_signal_volume_spike_ratio must be >= 1"
        assert int(self.config.get('execution', {}).get('exit_signal_consecutive_down_days', 3)) >= 2, "execution.exit_signal_consecutive_down_days must be >= 2"
        assert float(self.config.get('execution', {}).get('exit_signal_long_upper_shadow_ratio', 2.0)) >= 1.0, "execution.exit_signal_long_upper_shadow_ratio must be >= 1"
        assert isinstance(self.config.get('execution', {}).get('exit_on_gap_volume', True), bool), "execution.exit_on_gap_volume must be bool"
        assert float(self.config.get('execution', {}).get('exit_gap_down_pct', 0.04)) > 0, "execution.exit_gap_down_pct must be > 0"
        assert float(self.config.get('execution', {}).get('exit_gap_volume_zscore', 2.5)) >= 0.0, "execution.exit_gap_volume_zscore must be >= 0"
        assert int(self.config.get('execution', {}).get('exit_gap_volume_window', 30)) >= 10, "execution.exit_gap_volume_window must be >= 10"
        assert isinstance(self.config.get('execution', {}).get('enable_score_smoothing', True), bool), "execution.enable_score_smoothing must be bool"
        assert int(self.config.get('execution', {}).get('score_smoothing_window', 3)) >= 1, "execution.score_smoothing_window must be >= 1"
        assert float(self.config.get('execution', {}).get('max_portfolio_volatility', 0.25)) > 0, "execution.max_portfolio_volatility must be > 0"
        assert isinstance(self.config.get('execution', {}).get('enable_diversity_check', True), bool), "execution.enable_diversity_check must be bool"
        assert 0.0 < float(self.config.get('execution', {}).get('max_herfindahl_index', 0.35)) <= 1.0, "execution.max_herfindahl_index must be in (0,1]"
        assert 0.0 <= float(self.config.get('execution', {}).get('portfolio_vol_min_coverage', 0.70)) <= 1.0, "execution.portfolio_vol_min_coverage must be in [0,1]"
        assert float(self.config.get('execution', {}).get('max_weight_boost_for_hot', 0.05)) >= 0.0, "execution.max_weight_boost_for_hot must be >= 0"
        assert float(self.config.get('execution', {}).get('hot_zscore_threshold', 1.5)) >= 0.0, "execution.hot_zscore_threshold must be >= 0"
        assert int(self.config.get('execution', {}).get('hot_momentum_top_k', 3)) >= 1, "execution.hot_momentum_top_k must be >= 1"
        assert int(self.config.get('execution', {}).get('hot_persistence_cycles', 2)) >= 1, "execution.hot_persistence_cycles must be >= 1"
        momentum_weights_cfg = self.config.get('execution', {}).get('momentum_weights', {})
        assert isinstance(momentum_weights_cfg, dict), "execution.momentum_weights must be an object"
        short_w = float(momentum_weights_cfg.get('short', 0.4))
        medium_w = float(momentum_weights_cfg.get('medium', 0.6))
        assert short_w >= 0 and medium_w >= 0, "execution.momentum_weights.short/medium must be >= 0"
        assert (short_w + medium_w) > 0, "execution.momentum_weights short+medium must be > 0"
        assert self.config.get('execution', {}).get('price_stale_policy', {}).get('allow_buy'), "execution.price_stale_policy.allow_buy must not be empty"
        assert self.config.get('execution', {}).get('price_stale_policy', {}).get('allow_sell'), "execution.price_stale_policy.allow_sell must not be empty"
        assert int(self.config.get('reporting', {}).get('max_price_debug_items', 50)) >= 1, "reporting.max_price_debug_items must be >= 1"
        macro_cfg = self.config.get('macro_integration', {})
        assert isinstance(macro_cfg.get('enable_llm_topic_signals', True), bool), "macro_integration.enable_llm_topic_signals must be bool"
        assert 0.0 <= float(macro_cfg.get('llm_topic_confidence_threshold', 0.6)) <= 1.0, "macro_integration.llm_topic_confidence_threshold must be in [0,1]"
        assert float(macro_cfg.get('llm_topic_score_threshold', 0.5)) >= 0.0, "macro_integration.llm_topic_score_threshold must be >= 0"
        assert float(macro_cfg.get('llm_topic_tilt_scale', 0.02)) >= 0.0, "macro_integration.llm_topic_tilt_scale must be >= 0"
        assert int(macro_cfg.get('topic_memory_window', 50)) >= 1, "macro_integration.topic_memory_window must be >= 1"
        assert isinstance(macro_cfg.get('topic_sector_ticker_map', {}), dict), "macro_integration.topic_sector_ticker_map must be an object"
        assert isinstance(self.config.get('industry_map', {}), dict), "industry_map must be an object"
        assert isinstance(self.config.get('ticker_tags', {}), dict), "ticker_tags must be an object"
        news_overlay_cfg = self.config.get('news_overlay', {})
        assert isinstance(news_overlay_cfg, dict), "news_overlay must be an object"
        assert isinstance(news_overlay_cfg.get('enabled', False), bool), "news_overlay.enabled must be bool"
        assert isinstance(news_overlay_cfg.get('industry_collection', 'industry_signals'), str), "news_overlay.industry_collection must be string"
        assert float(news_overlay_cfg.get('max_age_hours', 48)) >= 0.0, "news_overlay.max_age_hours must be >= 0"
        assert 0.0 <= float(news_overlay_cfg.get('alpha', 0.08)) <= 1.0, "news_overlay.alpha must be in [0,1]"
        assert str(news_overlay_cfg.get('mode', 'risk_only')).lower() in ('risk_only', 'symmetric'), "news_overlay.mode must be risk_only|symmetric"
        assert 0.0 <= float(news_overlay_cfg.get('min_confidence', 0.55)) <= 1.0, "news_overlay.min_confidence must be in [0,1]"
        assert 0.0 <= float(news_overlay_cfg.get('max_abs_delta', 0.10)) <= 1.0, "news_overlay.max_abs_delta must be in [0,1]"
        assert isinstance(news_overlay_cfg.get('enable_confidence_scaling', True), bool), "news_overlay.enable_confidence_scaling must be bool"
        
        print("[OK] Safety checks passed: SIMULATION ONLY mode confirmed")
    
    def resume_from_checkpoint(self):
        """def resume_from_checkpoint: docstring omitted (was garbled/non-ASCII)."""
        snapshots_path = self.config['reporting']['portfolio_snapshots_path']
        trades_path = self.config['reporting']['trades_log_path']
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        if not os.path.exists(snapshots_path):
            print("[INFO] No checkpoint found - starting fresh")
            return
        
        try:
            print("\n" + "="*60)
            print("[CHECKPOINT] Detected - attempting to resume")
            print("="*60)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            with open(snapshots_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    print("[WARN] Checkpoint file is empty - starting fresh")
                    return
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                for line in lines:
                    snapshot = json.loads(line.strip())
                    self.portfolio_snapshots.append(snapshot)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            last_snapshot = self.portfolio_snapshots[-1]
            
            self.cash = last_snapshot['cash']
            self.current_cycle = last_snapshot['cycle'] + 1  # NOTE: comment omitted (was garbled/non-ASCII).
            self.status = "RESUMED"
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            self.positions = {}
            for ticker, pos in last_snapshot['positions'].items():
                self.positions[ticker] = pos['quantity']
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            for snapshot in self.portfolio_snapshots:
                timestamp = datetime.fromisoformat(snapshot['timestamp'])
                self.equity_curve.append((
                    timestamp,
                    snapshot['total_equity'],
                    snapshot['cash'],
                    snapshot['positions_value']
                ))
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            self.peak_equity = max(s['total_equity'] for s in self.portfolio_snapshots)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if os.path.exists(trades_path):
                trades_df = pd.read_csv(trades_path)
                self.trades_log = trades_df.to_dict('records')
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                self.rebuild_cost_basis()
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            print(f"[OK] Successfully resumed from checkpoint")
            print(f"   Last cycle: {last_snapshot['cycle']}")
            print(f"   Last update: {last_snapshot['timestamp']}")
            print(f"   Cash: ${self.cash:,.2f}")
            print(f"   Positions: {len(self.positions)} holdings")
            print(f"   Total equity: ${last_snapshot['total_equity']:,.2f}")
            print(f"   Return: {last_snapshot['total_return']:.2%}")
            print(f"   Historical snapshots: {len(self.portfolio_snapshots)}")
            print(f"   Historical trades: {len(self.trades_log)}")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if self.positions:
                print(f"\n   Current Holdings:")
                for ticker, qty in sorted(self.positions.items()):
                    cost = self.cost_basis.get(ticker, 0)
                    print(f"     {ticker}: {qty} shares (avg cost: ${cost:.2f})")
            
            print("="*60 + "\n")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            response = self.prompt_checkpoint_choice()
            if response == 'n':
                print("Starting fresh as requested...")
                self.clear_checkpoint()
                return
            print("Resuming from checkpoint as requested...")
            self.write_live_snapshot(last_snapshot, source="resume_checkpoint")
            self.generate_live_summary()
            self.save_trade_history_jsonl()
            
        except RuntimeError:
            # NOTE: comment omitted (was garbled/non-ASCII).
            raise
        except Exception as e:
            print(f"[WARN] Failed to resume from checkpoint: {e}")
            print("   Starting fresh...")

    def prompt_checkpoint_choice(self):
        """def prompt_checkpoint_choice: docstring omitted (was garbled/non-ASCII)."""
        env_choice = os.environ.get('GW_CHECKPOINT_ACTION', '').strip().lower()
        if env_choice in ('y', 'yes', 'resume', 'continue'):
            return 'y'
        if env_choice in ('n', 'no', 'fresh', 'restart'):
            return 'n'

        while True:
            try:
                response = input("Continue from checkpoint? (y/n): ").strip().lower()
            except EOFError as e:
                raise RuntimeError(
                    "Checkpoint choice required. Run in interactive terminal and input y/n, "
                    "or set env GW_CHECKPOINT_ACTION=resume|fresh."
                ) from e
            except KeyboardInterrupt as e:
                raise RuntimeError("Checkpoint choice cancelled by user.") from e

            if response in ('y', 'yes'):
                return 'y'
            if response in ('n', 'no'):
                return 'n'
            print("Please type 'y' to resume or 'n' to start fresh.")

    def load_scoreboard_history(self):
        """def load_scoreboard_history: docstring omitted (was garbled/non-ASCII)."""
        scoreboard_path = self.config.get('reporting', {}).get('scoreboard_path', 'outputs/scoreboard.jsonl')
        self.scoreboard_history = []
        self.last_diagnostic_hint = ""

        if not os.path.exists(scoreboard_path):
            return

        try:
            with open(scoreboard_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self.scoreboard_history.append(rec)
                    except Exception:
                        continue

            if self.scoreboard_history:
                self.last_diagnostic_hint = str(self.scoreboard_history[-1].get('diagnostic_hint', '') or '')
        except Exception as e:
            print(f"[SCOREBOARD] Failed to load history: {e}")
            self.scoreboard_history = []
            self.last_diagnostic_hint = ""

    def append_scoreboard_record(self):
        """def append_scoreboard_record: docstring omitted (was garbled/non-ASCII)."""
        if not self.portfolio_snapshots:
            return None

        scoreboard_path = self.config.get('reporting', {}).get('scoreboard_path', 'outputs/scoreboard.jsonl')
        window_n = int(self.config.get('benchmarks', {}).get('evaluation_days', 10))
        window_n = max(2, window_n)
        window = self.portfolio_snapshots[-window_n:]
        latest = window[-1]

        start_equity = float(window[0].get('total_equity', latest.get('total_equity', self.initial_cash)))
        end_equity = float(latest.get('total_equity', start_equity))
        strategy_return_2w = ((end_equity - start_equity) / start_equity) if start_equity > 0 else 0.0

        bench_avg_return_2w = float(latest.get('bench_avg_return', 0.0))
        excess_return_2w = float(strategy_return_2w - bench_avg_return_2w)
        win_flag_2w = bool(excess_return_2w > 0)

        turnover_sum_2w = float(sum(float(s.get('turnover_notional_post', 0.0) or 0.0) for s in window))

        cash_ratios = []
        for s in window:
            equity = float(s.get('total_equity', 0.0) or 0.0)
            cash = float(s.get('cash', 0.0) or 0.0)
            cash_ratios.append((cash / equity) if equity > 0 else 1.0)
        avg_cash_2w = float(np.mean(cash_ratios)) if cash_ratios else 0.0

        macro_active_ratio_2w = float(np.mean([0.0 if s.get('macro_reused', False) else 1.0 for s in window])) if window else 0.0
        risk_off_ratio_2w = float(np.mean([1.0 if s.get('regime_state') in ('risk_off', 'risk_off_forced') else 0.0 for s in window])) if window else 0.0
        avg_equity_2w = float(np.mean([float(s.get('total_equity', 0.0) or 0.0) for s in window])) if window else 0.0

        diagnostic_hint = ""
        temp_rec = {
            'win_flag_2w': win_flag_2w
        }
        tail = (self.scoreboard_history + [temp_rec])[-3:]
        if len(tail) == 3 and all(not bool(x.get('win_flag_2w', True)) for x in tail):
            turnover_ratio = (turnover_sum_2w / max(avg_equity_2w, 1.0))
            if turnover_ratio >= 1.5:
                diagnostic_hint = "turnover_too_high"
            elif avg_cash_2w >= 0.40:
                diagnostic_hint = "too_defensive"
            elif macro_active_ratio_2w >= 0.60:
                diagnostic_hint = "macro_too_noisy"
            elif risk_off_ratio_2w >= 0.60:
                diagnostic_hint = "regime_filter_too_strict"
            else:
                diagnostic_hint = "underperforming_no_clear_driver"

        record = {
            'timestamp': latest.get('timestamp', self._now().isoformat()),
            'strategy_return_2w': float(strategy_return_2w),
            'bench_avg_return_2w': float(bench_avg_return_2w),
            'excess_return_2w': float(excess_return_2w),
            'win_flag_2w': bool(win_flag_2w),
            'turnover_sum_2w': float(turnover_sum_2w),
            'avg_cash_2w': float(avg_cash_2w),
            'macro_active_ratio_2w': float(macro_active_ratio_2w),
            'diagnostic_hint': diagnostic_hint
        }

        try:
            with open(scoreboard_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.scoreboard_history.append(record)
            self.last_diagnostic_hint = diagnostic_hint
        except Exception as e:
            print(f"[SCOREBOARD] Failed to append record: {e}")

        return record

    def _format_topic_summary(self, confirmed_topics):
        """Build a compact topic summary string for live dashboard use."""
        if not isinstance(confirmed_topics, list) or not confirmed_topics:
            return ""
        parts = []
        for topic in confirmed_topics[:5]:
            if not isinstance(topic, dict):
                continue
            theme = str(topic.get('theme', '')).strip()
            direction = str(topic.get('direction', '')).strip()
            if not theme:
                continue
            if direction:
                parts.append(f"{theme}:{direction}")
            else:
                parts.append(theme)
        return "; ".join(parts)

    def _extract_snapshot_relevant_tickers(self, snapshot):
        """Collect relevant tickers for snapshot price-debug coverage."""
        out = []
        seen = set()

        def _add(ticker):
            t = str(ticker).upper().strip()
            if not t or t == 'CASH' or t in seen:
                return
            seen.add(t)
            out.append(t)

        if isinstance(self.positions, dict):
            for ticker in self.positions.keys():
                _add(ticker)

        positions_detail = snapshot.get('positions')
        if isinstance(positions_detail, dict):
            for ticker in positions_detail.keys():
                _add(ticker)

        for key in ('target_weights',):
            maybe_map = snapshot.get(key)
            if isinstance(maybe_map, dict):
                for ticker in maybe_map.keys():
                    _add(ticker)

        for key in ('planned_trades', 'candidate_trades', 'ranked_candidates'):
            maybe_list = snapshot.get(key)
            if not isinstance(maybe_list, list):
                continue
            for row in maybe_list:
                if isinstance(row, dict):
                    _add(row.get('ticker', ''))
                else:
                    _add(row)

        return out

    def build_live_snapshot(self, snapshot):
        """Centralized builder for all live snapshot writes."""
        payload = self._build_live_snapshot_payload(snapshot)

        reporting_cfg = self.config.get('reporting', {}) if isinstance(self.config, dict) else {}
        cap_raw = reporting_cfg.get('max_price_debug_items', 50)
        try:
            cap = max(1, int(cap_raw))
        except Exception:
            cap = 50

        relevant_tickers = self._extract_snapshot_relevant_tickers(snapshot if isinstance(snapshot, dict) else {})
        incoming_debug = payload.get('price_debug') if isinstance(payload, dict) else None
        if not isinstance(incoming_debug, dict):
            incoming_debug = {}

        collected_debug = self._collect_price_debug(
            relevant_tickers=relevant_tickers,
            planned_trades=(snapshot or {}).get('planned_trades') if isinstance(snapshot, dict) else None,
            price_debug_cache=incoming_debug,
            cap=cap,
        )
        payload['price_debug'] = collected_debug
        self.current_price_debug = dict(collected_debug)
        return payload

    def _build_live_snapshot_payload(self, snapshot):
        """Build a compact, UI-friendly live snapshot payload."""
        total_equity = float(snapshot.get('total_equity', 0.0) or 0.0)
        cash = float(snapshot.get('cash', 0.0) or 0.0)
        positions_value = float(snapshot.get('positions_value', max(0.0, total_equity - cash)) or 0.0)
        positions_detail = snapshot.get('positions', {})

        position_weights = {}
        if isinstance(positions_detail, dict) and total_equity > 0:
            for ticker, pos in positions_detail.items():
                t = str(ticker).upper().strip()
                if not t:
                    continue
                if isinstance(pos, dict):
                    value = float(pos.get('value', 0.0) or 0.0)
                else:
                    value = float(pos or 0.0)
                if value > 0:
                    position_weights[t] = float(value / total_equity)

        regime_state_raw = str(snapshot.get('regime_state', 'neutral'))
        regime_state_upper = regime_state_raw.upper()
        if regime_state_upper == "RISK_OFF_FORCED":
            regime_state_upper = "RISK_OFF_FORCED"

        confirmed_topics = self.current_macro.get('confirmed_topics', []) if isinstance(self.current_macro, dict) else []
        macro_tilts = self.current_macro.get('applied_tilts', self.current_macro.get('macro_tilts', {})) if isinstance(self.current_macro, dict) else {}
        signal_summary = self.current_macro.get('signal_summary', {}) if isinstance(self.current_macro, dict) else {}
        topic_summary = self._format_topic_summary(confirmed_topics)
        macro_summary = ""
        if isinstance(signal_summary, dict) and signal_summary:
            confirmed_count = int(signal_summary.get('confirmed_topics', len(confirmed_topics)))
            valid_signals = int(signal_summary.get('valid_signals_in_window', 0))
            macro_summary = f"confirmed_topics={confirmed_count}, valid_signals={valid_signals}"
        elif topic_summary:
            macro_summary = f"confirmed_topics={len(confirmed_topics)}"

        equity_history = []
        for ts, equity, _cash, _posv in self.equity_curve[-200:]:
            time_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
            equity_history.append({
                'time': time_str,
                'equity': float(equity)
            })

        cov_diag = {"enabled": True, "status": "error", "error": "cov_diag_uninitialized"}
        try:
            diag_weights = {}
            if total_equity > 0 and isinstance(positions_detail, dict):
                for ticker, pos in positions_detail.items():
                    ticker_upper = str(ticker).strip().upper()
                    if not ticker_upper or ticker_upper == 'CASH':
                        continue
                    try:
                        if isinstance(pos, dict):
                            pos_value = float(pos.get('value', 0.0) or 0.0)
                        else:
                            pos_value = float(pos or 0.0)
                    except Exception:
                        continue
                    if pos_value > 0:
                        diag_weights[ticker_upper] = float(pos_value / total_equity)

            cycle_id = int(snapshot.get('cycle', self.current_cycle))
            weight_signature = tuple(
                sorted(
                    (str(k), round(float(v), 12))
                    for k, v in diag_weights.items()
                    if np.isfinite(float(v))
                )
            )
            cache_key = ("cov_diag_snapshot", cycle_id, weight_signature)
            cached_entry = self.returns_cache.get(cache_key)
            cached_result = None
            if isinstance(cached_entry, dict):
                cached_meta = cached_entry.get("meta", {})
                if isinstance(cached_meta, dict):
                    maybe_result = cached_meta.get("result")
                    if isinstance(maybe_result, dict):
                        cached_result = maybe_result

            if isinstance(cached_result, dict):
                cov_diag = cached_result
            else:
                cov_diag = self.compute_cov_risk_diagnostics(diag_weights)
                self.returns_cache[cache_key] = {
                    "ts": datetime.now(),
                    "returns": pd.DataFrame(),
                    "meta": {
                        "kind": "cov_diag",
                        "cycle": cycle_id,
                        "result": cov_diag
                    }
                }
                for existing_key in list(self.returns_cache.keys()):
                    if existing_key == cache_key:
                        continue
                    if isinstance(existing_key, tuple) and len(existing_key) > 0 and existing_key[0] == "cov_diag_snapshot":
                        self.returns_cache.pop(existing_key, None)
        except Exception as e:
            cov_diag = {"enabled": True, "status": "error", "error": str(e)}

        try:
            if not isinstance(self.current_risk_check_info, dict):
                self.current_risk_check_info = {}
            self.current_risk_check_info["cov_risk_diag"] = cov_diag
        except Exception:
            pass

        vt_meta = None
        if isinstance(snapshot.get('vol_targeting'), dict):
            vt_meta = dict(snapshot.get('vol_targeting'))
        elif isinstance(self.current_vol_targeting_info, dict):
            vt_meta = dict(self.current_vol_targeting_info)
        elif isinstance(self.current_risk_check_info, dict) and isinstance(self.current_risk_check_info.get('vol_targeting'), dict):
            vt_meta = dict(self.current_risk_check_info.get('vol_targeting'))
        else:
            vt_meta = {'enabled': False, 'status': 'unavailable'}
        cost_est_meta = snapshot.get('cost_est')
        if isinstance(cost_est_meta, dict):
            cost_est_meta = dict(cost_est_meta)
        elif isinstance(self.current_cost_est_info, dict):
            cost_est_meta = dict(self.current_cost_est_info)
        else:
            cost_est_meta = {
                'enabled': False,
                'total': 0.0,
                'fee': 0.0,
                'slippage': 0.0,
                'impact': 0.0,
                'num_trades': 0
            }
        planner_meta = snapshot.get('trade_planner')
        if isinstance(planner_meta, dict):
            planner_meta = dict(planner_meta)
        elif isinstance(self.current_planner_info, dict):
            planner_meta = dict(self.current_planner_info)
        else:
            planner_meta = {
                'enabled': False,
                'status': 'disabled',
                'turnover_limit': 0.0,
                'turnover_used_forced': 0.0,
                'turnover_used_normal': 0.0,
                'turnover_used_total': 0.0,
                'num_forced': 0,
                'num_normal': 0,
                'num_dropped': 0,
                'num_adv_clipped': 0,
                'num_adv_dropped': 0,
                'adv_limit_enabled': False,
                'adv_limit_frac': 0.0,
                'normal_sorted_by': 'notional',
                'lambda_cost': 1.0,
                'benefit_mode': 'delta_weight',
                'normal_score_stats': {'count': 0},
                'dropped': [],
                'scaled': []
            }
        planner_turnover_used = float(planner_meta.get('turnover_used_forced', 0.0) or 0.0) + float(planner_meta.get('turnover_used_normal', 0.0) or 0.0)
        last_attempt_time = self.last_rebalance_attempt_time.isoformat() if isinstance(self.last_rebalance_attempt_time, datetime) else self.last_rebalance_attempt_time
        success_ref = self.last_rebalance_success_time if self.last_rebalance_success_time is not None else self.last_rebalance_time
        last_success_time = success_ref.isoformat() if isinstance(success_ref, datetime) else success_ref

        payload = {
            'timestamp': snapshot.get('timestamp', self._now().isoformat()),
            'account_id': self.account_id,
            'session_id': self.session_id,
            'config_hash': snapshot.get('config_hash', self.config_hash),
            'env': self.runtime_env,
            'cycle': int(snapshot.get('cycle', self.current_cycle)),
            'status': snapshot.get('status', self.status),
            'total_equity': total_equity,
            'cash': cash,
            'positions_value': positions_value,
            'drawdown': float(snapshot.get('drawdown', 0.0) or 0.0),
            'return': float(snapshot.get('total_return', 0.0) or 0.0),
            'positions': position_weights,
            'positions_detail': positions_detail,
            'risk_config': {
                'cash_target': float(snapshot.get('cash_target', 0.0) or 0.0),
                'max_weight': float(snapshot.get('dynamic_max_weight', self.config.get('objectives', {}).get('max_weight_per_asset', 0.25)) or 0.0),
                'risk_state': regime_state_upper,
                'regime_trend_score': float(snapshot.get('trend_score', 0.0) or 0.0),
                'cash_override_reason': snapshot.get('forced_regime_reason', '') or None
            },
            'macro_summary': macro_summary,
            'topic_summary': topic_summary,
            'macro_signal_score': float(self.current_macro.get('macro_risk_score_smoothed', self.current_macro.get('macro_risk_score', 0.0)) if isinstance(self.current_macro, dict) else 0.0),
            'theme_confidence': {},
            'last_macro': {
                'macro_risk_score_raw': float(snapshot.get('macro_risk_score_raw', 0.0) or 0.0),
                'macro_risk_score_smoothed': float(snapshot.get('macro_risk_score', 0.0) or 0.0),
                'confirmed_topics': confirmed_topics,
                'macro_tilts': macro_tilts,
                'topic_summary': topic_summary,
                'summary': macro_summary
            },
            'market_session': snapshot.get('market_session', dict(self.current_market_session) if isinstance(self.current_market_session, dict) else {}),
            'rebalance_gate': snapshot.get('rebalance_gate', dict(self.current_rebalance_gate) if isinstance(self.current_rebalance_gate, dict) else {}),
            'rebalance_skipped_reason': snapshot.get('rebalance_skipped_reason', self.current_rebalance_skipped_reason),
            'price_debug': snapshot.get('price_debug', dict(self.current_price_debug) if isinstance(self.current_price_debug, dict) else {}),
            'price_fetch_stats': snapshot.get(
                'price_fetch_stats',
                dict(self.current_price_fetch_stats) if isinstance(self.current_price_fetch_stats, dict) else {}
            ),
            'last_rebalance_attempt_time': snapshot.get('last_rebalance_attempt_time', last_attempt_time),
            'last_rebalance_success_time': snapshot.get('last_rebalance_success_time', last_success_time),
            'stale_count': snapshot.get('stale_count', self.current_stale_info.get('stale_count', 0)),
            'stale_ratio': snapshot.get('stale_ratio', self.current_stale_info.get('stale_ratio', 0.0)),
            'stale_candidate_count': snapshot.get('stale_candidate_count', self.current_stale_info.get('stale_candidate_count', 0)),
            'stale_ratio_candidates': snapshot.get('stale_ratio_candidates', self.current_stale_info.get('stale_ratio_candidates', 0.0)),
            'stale_candidate_count_policy_pass': snapshot.get(
                'stale_candidate_count_policy_pass',
                self.current_stale_info.get('stale_candidate_count_policy_pass', self.current_stale_info.get('stale_candidate_count', 0))
            ),
            'stale_ratio_candidates_policy_pass': snapshot.get(
                'stale_ratio_candidates_policy_pass',
                self.current_stale_info.get('stale_ratio_candidates_policy_pass', self.current_stale_info.get('stale_ratio_candidates', 0.0))
            ),
            'stale_candidates_policy_pass': snapshot.get(
                'stale_candidates_policy_pass',
                self.current_stale_info.get('stale_candidates_policy_pass', {
                    'stale': self.current_stale_info.get('stale_candidate_count', 0),
                    'total': self.current_stale_info.get('stale_candidate_count', 0),
                })
            ),
            'rebalance_trigger': snapshot.get('stale_decision_trace', ''),
            'last_trade_time': self.trades_log[-1].get('timestamp') if self.trades_log else None,
            'equity_history': equity_history,
            'gate_vol_method': snapshot.get('gate_vol_method', self.current_risk_check_info.get('gate_vol_method') if isinstance(self.current_risk_check_info, dict) else None),
            'cov_gate_reason': snapshot.get('cov_gate_reason', self.current_risk_check_info.get('cov_gate_reason') if isinstance(self.current_risk_check_info, dict) else None),
            'cov_gate_used': snapshot.get('cov_gate_used', self.current_risk_check_info.get('cov_gate_used') if isinstance(self.current_risk_check_info, dict) else None),
            'cov_gate_pass': snapshot.get('cov_gate_pass', self.current_risk_check_info.get('cov_gate_pass') if isinstance(self.current_risk_check_info, dict) else None),
            'cov_gate_coverage': snapshot.get('cov_gate_coverage', self.current_risk_check_info.get('cov_gate_coverage') if isinstance(self.current_risk_check_info, dict) else None),
            'cov_gate_vol': snapshot.get('cov_gate_vol', self.current_risk_check_info.get('cov_gate_vol') if isinstance(self.current_risk_check_info, dict) else None),
            'cov_gate_max_rc': snapshot.get('cov_gate_max_rc', self.current_risk_check_info.get('cov_gate_max_rc') if isinstance(self.current_risk_check_info, dict) else None),
            'cov_risk_diag': cov_diag,
            'portfolio_vol_cov_annualized': cov_diag.get('portfolio_vol_annualized') if isinstance(cov_diag, dict) else None,
            'max_rc_fraction_cov': cov_diag.get('max_rc_fraction') if isinstance(cov_diag, dict) else None,
            'max_rc_ticker_cov': cov_diag.get('max_rc_ticker') if isinstance(cov_diag, dict) else None,
            'avg_pairwise_corr_cov': cov_diag.get('avg_pairwise_corr') if isinstance(cov_diag, dict) else None,
            'vol_targeting': vt_meta,
            'vol_targeting_scale': vt_meta.get('scale') if isinstance(vt_meta, dict) else None,
            'vol_targeting_vol_before': vt_meta.get('vol_before') if isinstance(vt_meta, dict) else None,
            'vol_targeting_cash_after': vt_meta.get('cash_after') if isinstance(vt_meta, dict) else None,
            'cost_est': cost_est_meta,
            'trade_planner': planner_meta,
            'trade_planner_num_dropped': int(planner_meta.get('num_dropped', 0) or 0),
            'trade_planner_turnover_used': float(planner_turnover_used),
            'trade_planner_num_adv_clipped': int(planner_meta.get('num_adv_clipped', 0) or 0),
            'trade_planner_num_adv_dropped': int(planner_meta.get('num_adv_dropped', 0) or 0),
            'trade_planner_normal_score_count': int((planner_meta.get('normal_score_stats', {}) or {}).get('count', 0) or 0),
            'news_overlay_debug': snapshot.get(
                'news_overlay_debug',
                dict(self.current_news_overlay_info) if isinstance(self.current_news_overlay_info, dict) else {'enabled': False, 'status': 'unavailable'}
            ),
        }
        return payload

    def atomic_write_text(self, path, content):
        """Atomically write text content using the shared production-safe helper."""
        io_atomic_write_text(str(path), str(content or ""))

    def atomic_write_json(self, path, obj):
        """Atomically write a JSON object using the shared production-safe helper."""
        io_atomic_write_json(str(path), obj, indent=2)

    def atomic_write_jsonl(self, path, list_of_dicts):
        """Atomically rewrite a JSONL file using the shared production-safe helper."""
        rows = list_of_dicts if isinstance(list_of_dicts, list) else []
        io_atomic_write_jsonl(str(path), rows)

    def write_live_snapshot(self, snapshot, source="unknown"):
        """Write UI-friendly live snapshot to outputs/snapshot_live.json."""
        try:
            live_snapshot_path = self.config.get('reporting', {}).get('snapshot_live_path', 'outputs/snapshot_live.json')
            payload = self.build_live_snapshot(snapshot)
            price_debug_items = 0
            if isinstance(payload, dict) and isinstance(payload.get('price_debug'), dict):
                price_debug_items = len(payload.get('price_debug', {}))
            print(f"[PRICE_DEBUG_SAVE] items={price_debug_items} source={source}")
            if price_debug_items == 0 and isinstance(payload, dict):
                holdings_count = 0
                positions_obj = payload.get('positions_detail')
                if isinstance(positions_obj, dict):
                    holdings_count = len([k for k, v in positions_obj.items() if str(k).upper() != 'CASH' and (v or 0)])
                if holdings_count <= 0 and isinstance(self.positions, dict):
                    holdings_count = len([k for k, v in self.positions.items() if str(k).upper() != 'CASH' and float(v or 0) > 0])
                if holdings_count > 0:
                    print(f"[WARN] [PRICE_DEBUG_SAVE] holdings={holdings_count} but price_debug is empty")
            self.atomic_write_json(live_snapshot_path, payload)
        except Exception as e:
            print(f"[WARN] Failed to write live snapshot: {e}")

    def _build_post_rebalance_snapshot(self):
        """Build a lightweight snapshot from current in-memory state without appending history."""
        positions_value = 0.0
        positions_detail = {}
        for ticker, qty in self.positions.items():
            price, age_min, status = self.get_current_price(ticker)
            if not price:
                continue
            value = float(qty) * float(price)
            positions_value += value
            positions_detail[ticker] = {
                'quantity': qty,
                'price': float(price),
                'value': float(value),
                'age_minutes': float(age_min) if age_min is not None else None,
                'status': str(status).upper() if status is not None else None
            }

        total_equity = float(self.cash) + float(positions_value)
        total_return = (total_equity - self.initial_cash) / self.initial_cash if self.initial_cash > 0 else 0.0
        peak_base = max(float(self.peak_equity or 0.0), total_equity)
        drawdown = (peak_base - total_equity) / peak_base if peak_base > 0 else 0.0
        last_attempt_time = self.last_rebalance_attempt_time.isoformat() if isinstance(self.last_rebalance_attempt_time, datetime) else self.last_rebalance_attempt_time
        success_ref = self.last_rebalance_success_time if self.last_rebalance_success_time is not None else self.last_rebalance_time
        last_success_time = success_ref.isoformat() if isinstance(success_ref, datetime) else success_ref

        snapshot = {
            'timestamp': self._now().isoformat(),
            'account_id': self.account_id,
            'session_id': self.session_id,
            'env': self.runtime_env,
            'config_hash': self.config_hash,
            'cycle': self.current_cycle,
            'cash': float(self.cash),
            'positions_value': float(positions_value),
            'total_equity': float(total_equity),
            'total_return': float(total_return),
            'drawdown': float(drawdown),
            'positions': positions_detail,
            'status': self.status,
            'weights_reused': self.current_weights_reused,
            'macro_reused': self.current_macro_reused,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'last_macro_time': self.last_macro_time.isoformat() if self.last_macro_time else None,
            'market_session': dict(self.current_market_session) if isinstance(self.current_market_session, dict) else {},
            'rebalance_gate': dict(self.current_rebalance_gate) if isinstance(self.current_rebalance_gate, dict) else {},
            'rebalance_skipped_reason': self.current_rebalance_skipped_reason,
            'price_debug': dict(self.current_price_debug) if isinstance(self.current_price_debug, dict) else {},
            'price_fetch_stats': dict(self.current_price_fetch_stats) if isinstance(self.current_price_fetch_stats, dict) else {},
            'last_rebalance_attempt_time': last_attempt_time,
            'last_rebalance_success_time': last_success_time,
            'regime_state': self.current_regime.get('regime_state', 'neutral'),
            'trend_score': self.current_regime.get('trend_score', 0.5),
            'dynamic_min_cash': self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct']),
            'dynamic_max_weight': self.current_regime.get('dynamic_max_weight', self.config['objectives']['max_weight_per_asset']),
            'cash_target': self.current_regime.get('cash_target', self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct'])),
            'risk_caps_applied': self.current_regime.get('risk_caps_applied', False),
            'forced_until_time': self.current_regime.get('forced_until_time', self.forced_until_time.isoformat() if self.forced_until_time else None),
            'forced_regime_reason': self.current_regime.get('forced_reason', self.forced_regime_reason),
            'macro_risk_score_raw': self.current_macro.get('macro_risk_score', 0.0),
            'macro_risk_score': self.current_macro.get('macro_risk_score', 0.0),
            'macro_tilts': self.current_macro.get('macro_tilts', {}),
            'applied_tilts': self.current_macro.get('applied_tilts', {}),
            'capped_assets': self.current_macro.get('capped_assets', []),
            'turnover_notional': self.current_turnover_info.get('turnover_notional', 0.0),
            'turnover_notional_pre': self.current_turnover_info.get('turnover_notional_pre', self.current_turnover_info.get('turnover_notional', 0.0)),
            'turnover_notional_post': self.current_turnover_info.get('turnover_notional_post', 0.0),
            'turnover_limit': self.current_turnover_info.get('turnover_limit', 0.0),
            'turnover_scale': self.current_turnover_info.get('turnover_scale', 1.0),
            'turnover_capped': self.current_turnover_info.get('turnover_capped', False),
            'cost_est': dict(self.current_cost_est_info) if isinstance(self.current_cost_est_info, dict) else {'enabled': False, 'total': 0.0, 'fee': 0.0, 'slippage': 0.0, 'impact': 0.0, 'num_trades': 0},
            'trade_planner': dict(self.current_planner_info) if isinstance(self.current_planner_info, dict) else {'enabled': False, 'status': 'disabled', 'dropped': [], 'scaled': []},
            'trade_planner_num_dropped': int((self.current_planner_info or {}).get('num_dropped', 0)) if isinstance(self.current_planner_info, dict) else 0,
            'trade_planner_turnover_used': (
                float((self.current_planner_info or {}).get('turnover_used_forced', 0.0) or 0.0) +
                float((self.current_planner_info or {}).get('turnover_used_normal', 0.0) or 0.0)
            ) if isinstance(self.current_planner_info, dict) else 0.0,
            'trade_planner_num_adv_clipped': int((self.current_planner_info or {}).get('num_adv_clipped', 0)) if isinstance(self.current_planner_info, dict) else 0,
            'trade_planner_num_adv_dropped': int((self.current_planner_info or {}).get('num_adv_dropped', 0)) if isinstance(self.current_planner_info, dict) else 0,
            'news_overlay_debug': dict(self.current_news_overlay_info) if isinstance(self.current_news_overlay_info, dict) else {'enabled': False, 'status': 'unavailable'},
        }
        return snapshot

    def _write_post_rebalance_live_snapshot(self, trades_count, source="execute_rebalance"):
        """Refresh live snapshot immediately after trades are persisted."""
        try:
            snapshot = self._build_post_rebalance_snapshot()
            self.write_live_snapshot(snapshot, source=f"post_rebalance:{source}")
            # Keep text summary aligned with the same post-rebalance snapshot payload.
            self.generate_live_summary(snapshot_override=snapshot)
            live_snapshot_path = self.config.get('reporting', {}).get('snapshot_live_path', 'outputs/snapshot_live.json')
            print(f"[SNAPSHOT] Post-rebalance live snapshot written (cycle={self.current_cycle}, trades={int(trades_count)}, path={live_snapshot_path}, source={source})")
        except Exception as e:
            print(f"[WARN] Post-rebalance live snapshot refresh failed: {e}")

    def _refresh_market_session_state(self, now_dt=None):
        """Update in-memory market session + rebalance gate state."""
        reporting_cfg = self.config.get('reporting', {}) if isinstance(self.config, dict) else {}
        tz_market = str(reporting_cfg.get('market_tz', 'America/New_York') or 'America/New_York')
        open_time_et = reporting_cfg.get('market_open_time_et', '09:30')
        close_time_et = reporting_cfg.get('market_close_time_et', '16:00')
        open_grace_min = int(reporting_cfg.get('market_open_grace_min', 15) or 15)
        close_grace_min = int(reporting_cfg.get('market_close_grace_min', 10) or 10)

        now_val = self._coerce_datetime_utc(now_dt) if now_dt is not None else self._now()
        if now_val is None:
            now_val = self._now()
        session_override = getattr(self, '_debug_session_override', None)
        if callable(session_override):
            try:
                session = session_override(now_val)
            except TypeError:
                session = session_override()
            except Exception:
                session = None
        elif isinstance(session_override, dict):
            session = dict(session_override)
        else:
            session = get_market_session_state(
                now_dt=now_val,
                tz_market=tz_market,
                open_time_et=open_time_et,
                close_time_et=close_time_et,
                open_grace_min=open_grace_min,
                close_grace_min=close_grace_min,
            )
        if not isinstance(session, dict):
            session = {}

        session.setdefault('state', 'UNKNOWN')
        session.setdefault('open_grace_passed', False)
        session.setdefault('close_grace_passed', False)
        session.setdefault('now_et', now_val.isoformat())
        session.setdefault('now_utc', now_val.astimezone(timezone.utc).isoformat() if isinstance(now_val, datetime) else self._now().isoformat())
        today_str = now_val.date().isoformat()
        session.setdefault('trading_date_et', today_str)
        session.setdefault('last_completed_trading_date_et', today_str)
        session.setdefault('open_grace_min', int(open_grace_min))
        session.setdefault('close_grace_min', int(close_grace_min))
        allowed = bool(is_market_open_for_trading(session))
        state = str(session.get('state', 'UNKNOWN')).upper()
        open_grace_passed = bool(session.get('open_grace_passed', False))
        close_grace_passed = bool(session.get('close_grace_passed', False))
        is_weekend = bool(state == 'WEEKEND')
        is_holiday = bool(session.get('is_holiday', False))
        if is_holiday:
            reason_detail = 'holiday'
        elif state != 'OPEN':
            reason_detail = f"state_{state.lower()}"
        elif not open_grace_passed:
            reason_detail = 'open_grace_not_passed'
        else:
            reason_detail = 'allowed'
        reason = 'market_open' if allowed else 'market_closed_gate'
        now_utc = now_val.astimezone(timezone.utc) if isinstance(now_val, datetime) else self._now()
        now_local = now_utc.astimezone() if isinstance(now_utc, datetime) else now_utc
        now_et_parsed = self._parse_datetime_utc_safe(session.get('now_et'))
        checkpoint_raw = None
        checkpoint_age_seconds = None
        if isinstance(self.portfolio_snapshots, list) and self.portfolio_snapshots:
            last_snapshot = self.portfolio_snapshots[-1]
            if isinstance(last_snapshot, dict):
                checkpoint_raw = last_snapshot.get('timestamp')
        checkpoint_dt = self._parse_datetime_utc_safe(checkpoint_raw)
        if isinstance(checkpoint_dt, datetime) and isinstance(now_utc, datetime):
            try:
                checkpoint_age_seconds = round((now_utc - checkpoint_dt).total_seconds(), 3)
            except Exception:
                checkpoint_age_seconds = None
        within_grace = bool(state == 'OPEN' and not open_grace_passed)
        open_grace_seconds = int(max(0, int(open_grace_min)) * 60)
        close_grace_seconds = int(max(0, int(close_grace_min)) * 60)
        gate = {
            'allowed': bool(allowed),
            'reason': reason,
            'reason_detail': reason_detail,
            'session_state': state,
            'open_grace_passed': open_grace_passed,
            'close_grace_passed': close_grace_passed,
            'within_grace': within_grace,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'open_grace_seconds': open_grace_seconds,
            'close_grace_seconds': close_grace_seconds,
            'now_utc': now_utc.isoformat() if isinstance(now_utc, datetime) else str(now_utc),
            'now_local': now_local.isoformat() if isinstance(now_local, datetime) else str(now_local),
            'now_tz': str(getattr(now_local, 'tzinfo', None)),
            'now_et': str(session.get('now_et')),
            'market_open_time': str(session.get('open_time_et') or open_time_et),
            'market_close_time': str(session.get('close_time_et') or close_time_et),
            'market_tz': tz_market,
            'holiday_calendar': 'none',
            'checkpoint_ts': str(checkpoint_raw) if checkpoint_raw is not None else None,
            'checkpoint_age_seconds': checkpoint_age_seconds,
            'timebase_mismatch_seconds': (
                round((now_utc - now_et_parsed).total_seconds(), 3)
                if isinstance(now_utc, datetime) and isinstance(now_et_parsed, datetime) else None
            ),
        }

        print(
            "[GATE_DEBUG] "
            f"reason={reason} detail={reason_detail} session={state} "
            f"now={gate.get('now_local')} now_tz={gate.get('now_tz')} now_utc={gate.get('now_utc')} now_et={gate.get('now_et')} "
            f"market_open_time={gate.get('market_open_time')} market_close_time={gate.get('market_close_time')} "
            f"grace_seconds=open:{open_grace_seconds},close:{close_grace_seconds} "
            f"within_grace={within_grace} open_grace_passed={open_grace_passed} close_grace_passed={close_grace_passed} "
            f"is_weekend={is_weekend} is_holiday={is_holiday} checkpoint_ts={gate.get('checkpoint_ts')} "
            f"checkpoint_age_seconds={checkpoint_age_seconds} timebase_mismatch_seconds={gate.get('timebase_mismatch_seconds')}"
        )

        self.current_market_session = dict(session) if isinstance(session, dict) else {}
        self.current_rebalance_gate = dict(gate)
        self.current_rebalance_skipped_reason = '' if allowed else 'market_closed_gate'
        return self.current_market_session, self.current_rebalance_gate

    def _resolve_daily_report_dirs(self):
        """Resolve and deduplicate Daily Report output directories."""
        reporting_cfg = self.config.get('reporting', {})
        raw_dirs = reporting_cfg.get('daily_report_dirs', ['outputs/Daily Report'])
        if isinstance(raw_dirs, str):
            raw_dirs = [raw_dirs]
        resolved = []
        seen = set()
        for path in raw_dirs:
            if not path:
                continue
            full = os.path.abspath(str(path))
            key = full.lower()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(full)
        if not resolved:
            resolved = [os.path.abspath('outputs/Daily Report')]
        return resolved

    def _update_live_snapshot_daily_report_ref(self, report_date, report_path):
        """Attach latest Daily Report pointer into snapshot_live.json for UI."""
        try:
            snapshot_path = self.config.get('reporting', {}).get('snapshot_live_path', 'outputs/snapshot_live.json')
            if not os.path.exists(snapshot_path):
                return
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return

            payload['latest_daily_report_date'] = str(report_date)
            payload['latest_daily_report_path'] = str(report_path)
            pd_items = len(payload.get('price_debug', {})) if isinstance(payload.get('price_debug'), dict) else 0
            print(f"[PRICE_DEBUG_SAVE] items={pd_items} source=update_daily_report_ref")
            self.atomic_write_json(snapshot_path, payload)
        except Exception as e:
            print(f"[WARN] Failed to update snapshot daily report pointer: {e}")

    def _maybe_generate_daily_report(self):
        """Generate Daily Report after market close. Never raises."""
        if not DAILY_REPORTER_AVAILABLE or daily_reporter is None:
            return

        try:
            reporting_cfg = self.config.get('reporting', {})
            tz_name = str(reporting_cfg.get('daily_report_tz', 'America/Vancouver'))
            snapshot_path = reporting_cfg.get('snapshot_live_path', 'outputs/snapshot_live.json')
            trades_csv_path = reporting_cfg.get('trades_log_path', 'outputs/paper_trades.csv')
            report_dirs = self._resolve_daily_report_dirs()
            now_local = self._now()

            stale_ratio = float(
                self.current_stale_info.get(
                    'stale_ratio_candidates',
                    self.current_stale_info.get('stale_ratio', 0.0)
                ) or 0.0
            )
            observe_count = int(
                self.current_stale_info.get(
                    'stale_candidate_count',
                    self.current_stale_info.get('stale_count', 0)
                ) or 0
            )
            stale_snapshot = {
                'stale_ratio': stale_ratio,
                'observe_count': observe_count,
                'stale_info': dict(self.current_stale_info) if isinstance(self.current_stale_info, dict) else {}
            }

            session = daily_reporter.get_market_session_state(now_local, tz_market='America/New_York')
            closed, reason = daily_reporter.is_market_closed(
                now=now_local,
                tz=tz_name,
                snapshot=stale_snapshot,
                stale_tracker=self.daily_report_stale_tracker
            )

            state = str(session.get('state', '')).upper()
            trading_date = str(session.get('trading_date_et', '')).strip()
            last_completed_date = str(session.get('last_completed_trading_date_et', '')).strip()
            close_method = str(reason.get('method', '')).strip().lower() if isinstance(reason, dict) else ''
            is_time_close = bool(closed and close_method == 'time')
            is_stale_close = bool(closed and close_method == 'stale_streak' and state == 'OPEN')

            trigger_mode = 'backfill_last_completed'
            report_date = last_completed_date
            if is_time_close:
                report_date = trading_date or last_completed_date
                trigger_mode = 'post_close_time'
            elif is_stale_close:
                report_date = trading_date or last_completed_date
                trigger_mode = 'open_stale_streak'

            if not report_date:
                return

            if self.latest_daily_report_date == report_date:
                return

            if trigger_mode == 'backfill_last_completed' and state in ('PRE_OPEN', 'OPEN', 'WEEKEND', 'POST_CLOSE'):
                # Before close confirmation, only backfill the last completed trading day.
                pass
            else:
                # Defensive fallback: if state is unexpected and close is not confirmed, skip.
                if not is_time_close and not is_stale_close and state not in ('PRE_OPEN', 'OPEN', 'WEEKEND', 'POST_CLOSE'):
                    return

            report = daily_reporter.generate_daily_report(
                date=report_date,
                snapshot_path=snapshot_path,
                trades_csv_path=trades_csv_path,
                report_dirs=report_dirs,
                tz=tz_name
            )
            if not isinstance(report, dict):
                return

            if report.get('_already_exists'):
                self.latest_daily_report_date = report_date
                existing_path = report.get('_existing_path')
                if existing_path:
                    self._update_live_snapshot_daily_report_ref(report_date, existing_path)
                return

            report['market_close'] = {
                'closed': True,
                'reason': {
                    'method': trigger_mode,
                    'details': {
                        'session': session,
                        'close_check': reason,
                    },
                }
            }
            wrote_paths = daily_reporter.write_daily_report(report, report_dirs)
            if wrote_paths:
                self.latest_daily_report_date = report_date
                self._update_live_snapshot_daily_report_ref(report_date, wrote_paths[0])
                print(f"[DAILY REPORT] Generated {report_date} -> {wrote_paths[0]}")
        except Exception as e:
            print(f"[WARN] Daily report generation skipped due to error: {e}")

    def save_trade_history_jsonl(self):
        """Write trade history JSONL for Streamlit portfolio monitor."""
        trade_history_path = self.config.get('reporting', {}).get('trade_history_path', 'outputs/trade_history.jsonl')

        def _sanitize(value):
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_sanitize(v) for v in value]
            if isinstance(value, tuple):
                return [_sanitize(v) for v in value]
            if isinstance(value, (np.floating, float)):
                if not np.isfinite(value):
                    return None
                return float(value)
            if isinstance(value, (np.integer, int)):
                return int(value)
            return value

        try:
            trade_rows = []
            for trade in self.trades_log:
                trade_clean = _sanitize(trade)
                if isinstance(trade_clean, dict):
                    trade_clean.setdefault('account_id', self.account_id)
                    trade_clean.setdefault('session_id', self.session_id)
                    trade_clean.setdefault('env', self.runtime_env)
                    trade_clean.setdefault('config_hash', self.config_hash)
                trade_rows.append(trade_clean)
            self.atomic_write_jsonl(trade_history_path, trade_rows)
        except Exception as e:
            print(f"[WARN] Failed to write trade history JSONL: {e}")
    
    def rebuild_cost_basis(self):
        """def rebuild_cost_basis: docstring omitted (was garbled/non-ASCII)."""
        self.cost_basis = {}
        position_qty = {}
        
        for trade in self.trades_log:
            ticker = trade['ticker']
            side = trade['side']
            qty = trade['quantity']
            price = trade['price']
            
            if side == 'BUY':
                old_qty = position_qty.get(ticker, 0)
                old_cost = self.cost_basis.get(ticker, 0)
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                if old_qty > 0:
                    total_cost = (old_qty * old_cost) + (qty * price)
                    position_qty[ticker] = old_qty + qty
                    self.cost_basis[ticker] = total_cost / position_qty[ticker]
                else:
                    position_qty[ticker] = qty
                    self.cost_basis[ticker] = price
                    
            elif side == 'SELL':
                position_qty[ticker] = position_qty.get(ticker, 0) - qty
                if position_qty[ticker] <= 0:
                    position_qty[ticker] = 0
                    self.cost_basis[ticker] = 0
    
    def rebuild_position_entry_cycles(self):
        """Rebuild last BUY cycle per open position for holding-period checks."""
        self.position_entry_cycle = {}
        position_qty = {}

        for trade in self.trades_log:
            ticker = str(trade.get('ticker', '')).upper()
            if not ticker:
                continue

            side = str(trade.get('side', '')).upper()
            try:
                qty = int(float(trade.get('quantity', 0)))
            except (TypeError, ValueError):
                qty = 0
            if qty <= 0:
                continue

            raw_cycle = trade.get('cycle', None)
            if raw_cycle is None or raw_cycle == '':
                trade_cycle = 0
            else:
                try:
                    trade_cycle = int(raw_cycle)
                except (TypeError, ValueError):
                    trade_cycle = 0

            if side == 'BUY':
                position_qty[ticker] = position_qty.get(ticker, 0) + qty
                self.position_entry_cycle[ticker] = trade_cycle
            elif side == 'SELL':
                remaining = position_qty.get(ticker, 0) - qty
                if remaining <= 0:
                    position_qty.pop(ticker, None)
                    self.position_entry_cycle.pop(ticker, None)
                else:
                    position_qty[ticker] = remaining

        mature_cycle = max(0, int(self.current_cycle) - int(self.min_holding_cycles))
        for ticker in self.positions.keys():
            self.position_entry_cycle.setdefault(str(ticker).upper(), mature_cycle)

    def clear_checkpoint(self):
        """Clear checkpoint state and restart from fresh paper account."""
        self.cash = self.config['initial_cash_usd']
        self.positions = {}
        self.cost_basis = {}
        self.equity_curve = []
        self.trades_log = []
        self.portfolio_snapshots = []
        self.current_cycle = 0
        self.peak_equity = self.cash
        self.status = "READY"
        self.position_entry_cycle = {}
        self.current_holding_blocks = []
        self.current_price_debug = {}

    def set_market_data_fetcher(self, fetcher):
        """Set optional market data fetcher for backtests/replay injection."""
        if fetcher is not None and not callable(fetcher):
            raise ValueError("market_data_fetcher must be callable or None")
        self.market_data_fetcher = fetcher

    def set_price_fetcher(self, fetcher):
        """Set optional current price fetcher for backtests/replay injection."""
        if fetcher is not None and not callable(fetcher):
            raise ValueError("price_fetcher must be callable or None")
        self.price_fetcher = fetcher

    def _get_yfinance_module(self):
        """Lazily import yfinance so offline dry-runs do not require it at import time."""
        global yf
        if yf is None:
            try:
                import yfinance as _yf
                yf = _yf
            except Exception:
                return None
        return yf

    def _now(self):
        """Current wall-clock time with optional deterministic debug override."""
        override = getattr(self, '_debug_now_override', None)
        if callable(override):
            try:
                value = override()
                if isinstance(value, datetime):
                    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
                        if not getattr(self, '_debug_now_override_warned', False):
                            print("[WARN] _debug_now_override returned naive datetime; assuming UTC")
                            self._debug_now_override_warned = True
                        return value.replace(tzinfo=timezone.utc)
                    return value.astimezone(timezone.utc)
            except Exception:
                pass
        if isinstance(override, datetime):
            if override.tzinfo is None or override.tzinfo.utcoffset(override) is None:
                if not getattr(self, '_debug_now_override_warned', False):
                    print("[WARN] _debug_now_override is naive datetime; assuming UTC")
                    self._debug_now_override_warned = True
                return override.replace(tzinfo=timezone.utc)
            return override.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    def _coerce_datetime_utc(self, value):
        """Normalize datetime values to timezone-aware UTC for safe arithmetic."""
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _parse_datetime_utc_safe(self, value):
        """Best-effort parse of datetime-like values into timezone-aware UTC datetime."""
        if isinstance(value, datetime):
            return self._coerce_datetime_utc(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith('Z'):
                text = text[:-1] + '+00:00'
            try:
                parsed = datetime.fromisoformat(text)
            except Exception:
                return None
            return self._coerce_datetime_utc(parsed)
        return None

    def to_yahoo_symbol(self, ticker):
        """Return provider symbol for Yahoo/yfinance without unsafe ticker rewrites."""
        if ticker is None:
            return ticker
        t = str(ticker).strip()
        if not t:
            return t
        upper = t.upper()
        explicit_map = {
            'BRK.B': 'BRK-B',
            'BRK/A': 'BRK-A',
        }
        if upper in explicit_map:
            return explicit_map[upper]
        # Keep exchange suffix formats such as ".TO" unchanged for Yahoo.
        return t

    def to_safe_key(self, ticker):
        """Return a filesystem/cache-safe key without changing provider symbols."""
        if ticker is None:
            return ticker
        t = str(ticker).strip().upper()
        if not t:
            return t
        return t.replace('.', '-').replace('/', '-')

    def _normalize_market_ticker(self, ticker):
        """Backward-compatible wrapper for existing call sites."""
        return self.to_yahoo_symbol(ticker)

    def get_market_data(self, ticker, period='1mo', interval='1d'):
        """def get_market_data: docstring omitted (was garbled/non-ASCII)."""
        try:
            if ticker == 'CASH':
                return None

            if self.market_data_fetcher is not None:
                try:
                    return self.market_data_fetcher(ticker=ticker, period=period, interval=interval)
                except TypeError:
                    try:
                        return self.market_data_fetcher(ticker, period, interval)
                    except Exception as e:
                        print(f"[WARN] Injected market_data_fetcher failed for {ticker}: {e}")
                except Exception as e:
                    print(f"[WARN] Injected market_data_fetcher failed for {ticker}: {e}")

            yf_mod = self._get_yfinance_module()
            if yf_mod is None:
                print(f"[WARN] yfinance unavailable for {ticker}; skipping market data fetch")
                return None

            market_ticker = self._normalize_market_ticker(ticker)
            t = yf_mod.Ticker(market_ticker)
            hist = t.history(period=period, interval=interval)
            
            if hist.empty:
                print(f"[WARN] No data for {ticker} (provider symbol: {market_ticker}), skipping")
                return None
            
            return hist
        except Exception as e:
            print(f"[WARN] Error fetching data for {ticker}: {e}")
            return None

    def get_current_price(self, ticker, return_debug=False):
        """Get current price with backward-compatible tuple return; optionally include debug payload."""
        import pytz

        now_raw = self._now()
        if now_raw.tzinfo is None or now_raw.tzinfo.utcoffset(now_raw) is None:
            now_utc = now_raw.replace(tzinfo=timezone.utc)
        else:
            now_utc = now_raw.astimezone(timezone.utc)
        default_thresholds = {"live_max_min": 10.0, "recent_max_min": 60.0}

        def _emit_result(
            price,
            age_min,
            status,
            *,
            source='missing',
            price_ts=None,
            tz_ok=False,
            thresholds=None,
            notes=None,
            bar_interval=None,
            raw_price_ts=None,
            raw_tz=None,
            debug_status=None,
        ):
            final_status = str(status).upper() if status is not None else "STALE"
            if thresholds is None:
                thresholds = default_thresholds

            if isinstance(price_ts, datetime):
                price_ts_iso = price_ts.isoformat()
            elif isinstance(price_ts, str):
                price_ts_iso = price_ts
            else:
                price_ts_iso = None

            try:
                age_num = float(age_min) if age_min is not None and np.isfinite(float(age_min)) else None
            except Exception:
                age_num = None

            debug_payload = {
                "ticker": str(ticker).upper(),
                "now_ts": now_utc.isoformat(),
                "status": str(debug_status or ("MISSING" if price is None else final_status)).upper(),
                "age_min": age_num,
                "source": str(source),
                "price_ts": price_ts_iso,
                "tz_ok": bool(tz_ok),
                "thresholds": {
                    "live_max_min": float(thresholds.get("live_max_min", default_thresholds["live_max_min"])),
                    "recent_max_min": float(thresholds.get("recent_max_min", default_thresholds["recent_max_min"])),
                },
                "notes": str(notes) if notes else None,
            }
            if bar_interval is not None:
                debug_payload["bar_interval"] = str(bar_interval)
            if raw_price_ts is not None:
                debug_payload["raw_price_ts"] = str(raw_price_ts)
            if raw_tz is not None:
                debug_payload["raw_tz"] = str(raw_tz)

            if return_debug:
                return (price, age_min, final_status, debug_payload)
            return (price, age_min, final_status)

        def _normalize_ts(raw_ts):
            """Normalize raw timestamp into timezone-aware datetime for age calculation."""
            note_items = []
            raw_ts_str = None
            raw_tz = None
            ts_obj = None
            tz_ok = False
            if raw_ts is None:
                return ts_obj, tz_ok, note_items, raw_ts_str, raw_tz

            try:
                raw_ts_str = str(raw_ts)
                raw_tz = str(getattr(raw_ts, 'tz', None)) if hasattr(raw_ts, 'tz') else None
                if hasattr(raw_ts, 'to_pydatetime'):
                    ts_obj = raw_ts.to_pydatetime()
                elif isinstance(raw_ts, datetime):
                    ts_obj = raw_ts
                else:
                    note_items.append("price_ts_parse_error")
                    return None, False, note_items, raw_ts_str, raw_tz

                if ts_obj.tzinfo is None or ts_obj.tzinfo.utcoffset(ts_obj) is None:
                    tz_ok = False
                    note_items.append("naive_ts_detected")
                    try:
                        if ZoneInfo is not None:
                            ts_obj = ts_obj.replace(tzinfo=ZoneInfo('America/New_York'))
                        else:
                            ts_obj = pytz.timezone('US/Eastern').localize(ts_obj)
                        note_items.append("localized_assumption=America/New_York")
                    except Exception:
                        note_items.append("localized_assumption_failed")
                else:
                    tz_ok = True
                    if raw_tz is None:
                        raw_tz = str(ts_obj.tzinfo)
            except Exception:
                return None, False, ["price_ts_parse_error"], raw_ts_str, raw_tz

            return ts_obj, tz_ok, note_items, raw_ts_str, raw_tz

        def _classify_status(age_min, thresholds):
            if age_min is None:
                return "STALE"
            if age_min < thresholds["live_max_min"]:
                return "LIVE"
            if age_min < thresholds["recent_max_min"]:
                return "RECENT"
            return "STALE"

        def _cache_quote(
            price,
            *,
            source,
            bar_interval,
            price_ts,
            tz_ok,
            notes=None,
            raw_price_ts=None,
            raw_tz=None,
        ):
            service = getattr(self, "price_service", None)
            if service is None:
                return
            try:
                if price is None:
                    return
                price_f = float(price)
                if not np.isfinite(price_f):
                    return
                service.update_cache(
                    ticker,
                    {
                        "price": price_f,
                        "price_ts": price_ts if isinstance(price_ts, datetime) else None,
                        "fetched_at": now_utc,
                        "source": str(source),
                        "bar_interval": bar_interval,
                        "tz_ok": bool(tz_ok),
                        "notes": str(notes or ""),
                        "raw_price_ts": raw_price_ts,
                        "raw_tz": raw_tz,
                    },
                )
            except Exception:
                return

        if ticker == 'CASH':
            return _emit_result(
                1.0,
                0.0,
                "LIVE",
                source="manual_fallback",
                price_ts=now_utc,
                tz_ok=True,
                thresholds={"live_max_min": 1.0, "recent_max_min": 5.0},
                notes="cash_proxy",
                bar_interval=None,
            )

        fallback_notes = []
        try:
            if self.price_fetcher is not None:
                try:
                    fetch_result = self.price_fetcher(ticker=ticker)
                except TypeError:
                    fetch_result = self.price_fetcher(ticker)
                except Exception as e:
                    fetch_result = None
                    fallback_notes.append(f"price_fetcher_error={e}")
                    print(f"[WARN] Injected price_fetcher failed for {ticker}: {e}")

                if fetch_result is not None:
                    if isinstance(fetch_result, tuple) and len(fetch_result) == 4:
                        price, age_min, status, injected_debug = fetch_result
                        if return_debug and isinstance(injected_debug, dict):
                            debug_copy = dict(injected_debug)
                            debug_copy.setdefault("ticker", str(ticker).upper())
                            debug_copy.setdefault("now_ts", now_utc.isoformat())
                            debug_copy.setdefault("status", str(status).upper())
                            debug_copy.setdefault("age_min", float(age_min) if age_min is not None else None)
                            debug_copy.setdefault("source", "injected_price_fetcher")
                            debug_copy.setdefault("price_ts", None)
                            debug_copy.setdefault("tz_ok", False)
                            debug_copy.setdefault("thresholds", default_thresholds)
                            debug_copy.setdefault("notes", "injected_price_fetcher")
                            return (price, age_min, str(status).upper(), debug_copy)
                        return (price, age_min, str(status).upper())
                    if isinstance(fetch_result, tuple) and len(fetch_result) == 3:
                        price, age_min, status = fetch_result
                        return _emit_result(
                            price,
                            age_min,
                            status,
                            source="injected_price_fetcher",
                            price_ts=None,
                            tz_ok=False,
                            thresholds=default_thresholds,
                            notes="injected_price_fetcher",
                            bar_interval=None,
                        )
                    print(f"[WARN] Injected price_fetcher returned invalid payload for {ticker}; falling back")

            service = getattr(self, "price_service", None)
            if service is not None:
                cached_row = service.get_cached(ticker, now_utc=now_utc)
                if isinstance(cached_row, dict):
                    cached_price = cached_row.get("price")
                    try:
                        cached_price_f = float(cached_price) if cached_price is not None else None
                    except Exception:
                        cached_price_f = None
                    if cached_price_f is not None and np.isfinite(cached_price_f):
                        cached_price_ts = cached_row.get("price_ts")
                        cached_age = (now_utc - now_utc).total_seconds() / 60.0
                        if isinstance(cached_row.get("fetched_at"), datetime):
                            cached_age = max(
                                0.0,
                                (now_utc - cached_row["fetched_at"]).total_seconds() / 60.0,
                            )
                        cached_thresholds = {"live_max_min": 10.0, "recent_max_min": 60.0}
                        cached_status = _classify_status(cached_age, cached_thresholds)
                        return _emit_result(
                            cached_price_f,
                            cached_age,
                            cached_status,
                            source=cached_row.get("source", "cache"),
                            price_ts=cached_price_ts if isinstance(cached_price_ts, datetime) else None,
                            tz_ok=bool(cached_row.get("tz_ok", False)),
                            thresholds=cached_thresholds,
                            notes=cached_row.get("notes"),
                            bar_interval=cached_row.get("bar_interval"),
                            raw_price_ts=cached_row.get("raw_price_ts"),
                            raw_tz=cached_row.get("raw_tz"),
                        )

            now_et = now_utc.astimezone(pytz.timezone('US/Eastern'))

            yf_mod = self._get_yfinance_module()
            if yf_mod is None:
                print(f"[WARN] yfinance unavailable for {ticker}; skipping price fetch")
                return _emit_result(
                    None,
                    99999.0,
                    "STALE",
                    source="missing",
                    price_ts=None,
                    tz_ok=False,
                    thresholds=default_thresholds,
                    notes="yfinance_unavailable",
                    bar_interval=None,
                    raw_price_ts=None,
                    raw_tz=None,
                    debug_status="MISSING",
                )

            market_ticker = self._normalize_market_ticker(ticker)
            t = yf_mod.Ticker(market_ticker)

            try:
                hist = t.history(period='1d', interval='5m')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    raw_ts = hist.index[-1]
                    ts_obj, tz_ok, note_items, raw_ts_str, raw_tz = _normalize_ts(raw_ts)
                    thresholds = {"live_max_min": 10.0, "recent_max_min": 60.0}
                    if ts_obj is not None:
                        age_min = (now_utc - ts_obj.astimezone(timezone.utc)).total_seconds() / 60.0
                    else:
                        age_min = 99999.0
                    market_status = _classify_status(age_min, thresholds)
                    ts_label = ts_obj.astimezone(pytz.timezone('US/Eastern')).strftime('%H:%M ET') if isinstance(ts_obj, datetime) else "N/A"
                    print(f"[PRICE] {ticker}: ${price:.2f} (5m @ {ts_label}, {age_min:.0f}min ago) {market_status}")
                    notes_joined = ';'.join(note_items) if note_items else None
                    _cache_quote(
                        price,
                        source="yfinance_history_5m",
                        bar_interval="5m",
                        price_ts=ts_obj,
                        tz_ok=tz_ok,
                        notes=notes_joined,
                        raw_price_ts=raw_ts_str,
                        raw_tz=raw_tz,
                    )
                    return _emit_result(
                        price,
                        age_min,
                        market_status,
                        source="yfinance_history_5m",
                        price_ts=ts_obj,
                        tz_ok=tz_ok,
                        thresholds=thresholds,
                        notes=notes_joined,
                        bar_interval="5m",
                        raw_price_ts=raw_ts_str,
                        raw_tz=raw_tz,
                    )
                fallback_notes.append("empty_history_5m")
            except Exception as e:
                fallback_notes.append(f"history_5m_error={e}")
                print(f"[PRICE] {ticker}: 5m history failed - {e}")

            try:
                hist = t.history(period='1d', interval='1m')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    raw_ts = hist.index[-1]
                    ts_obj, tz_ok, note_items, raw_ts_str, raw_tz = _normalize_ts(raw_ts)
                    thresholds = {"live_max_min": 5.0, "recent_max_min": 60.0}
                    if ts_obj is not None:
                        age_min = (now_utc - ts_obj.astimezone(timezone.utc)).total_seconds() / 60.0
                    else:
                        age_min = 99999.0
                    market_status = _classify_status(age_min, thresholds)
                    ts_label = ts_obj.astimezone(pytz.timezone('US/Eastern')).strftime('%H:%M ET') if isinstance(ts_obj, datetime) else "N/A"
                    print(f"[PRICE] {ticker}: ${price:.2f} (1m @ {ts_label}, {age_min:.0f}min ago) {market_status}")
                    notes_joined = ';'.join(note_items) if note_items else None
                    _cache_quote(
                        price,
                        source="yfinance_history_1m",
                        bar_interval="1m",
                        price_ts=ts_obj,
                        tz_ok=tz_ok,
                        notes=notes_joined,
                        raw_price_ts=raw_ts_str,
                        raw_tz=raw_tz,
                    )
                    return _emit_result(
                        price,
                        age_min,
                        market_status,
                        source="yfinance_history_1m",
                        price_ts=ts_obj,
                        tz_ok=tz_ok,
                        thresholds=thresholds,
                        notes=notes_joined,
                        bar_interval="1m",
                        raw_price_ts=raw_ts_str,
                        raw_tz=raw_tz,
                    )
                fallback_notes.append("empty_history_1m")
            except Exception as e:
                fallback_notes.append(f"history_1m_error={e}")
                print(f"[PRICE] {ticker}: 1m history failed - {e}")

            try:
                info = t.info
                for price_field in ['currentPrice', 'regularMarketPrice', 'ask', 'bid']:
                    if price_field in info and info[price_field]:
                        price = float(info[price_field])
                        if price > 0:
                            print(f"[PRICE] {ticker}: ${price:.2f} (from info.{price_field}) STALE (no timestamp)")
                            _cache_quote(
                                price,
                                source="yfinance_info",
                                bar_interval=None,
                                price_ts=None,
                                tz_ok=False,
                                notes=f"price_field={price_field};no_price_timestamp",
                                raw_price_ts=None,
                                raw_tz=None,
                            )
                            return _emit_result(
                                price,
                                99999.0,
                                "STALE",
                                source="yfinance_info",
                                price_ts=None,
                                tz_ok=False,
                                thresholds=default_thresholds,
                                notes=f"price_field={price_field};no_price_timestamp",
                                bar_interval=None,
                                raw_price_ts=None,
                                raw_tz=None,
                            )
                fallback_notes.append("missing_info_price_fields")
            except Exception as e:
                fallback_notes.append(f"info_error={e}")
                print(f"[PRICE] {ticker}: info failed - {e}")

            try:
                hist = t.history(period='5d', interval='1d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    raw_ts = hist.index[-1]
                    ts_obj, tz_ok, note_items, raw_ts_str, raw_tz = _normalize_ts(raw_ts)
                    if ts_obj is not None:
                        age_min = (now_utc - ts_obj.astimezone(timezone.utc)).total_seconds() / 60.0
                    else:
                        age_min = 99999.0
                    ts_label = ts_obj.astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d') if isinstance(ts_obj, datetime) else "N/A"
                    print(f"[PRICE] {ticker}: ${price:.2f} (from daily close {ts_label}, {age_min:.0f}min ago) STALE")
                    notes_joined = ';'.join(note_items) if note_items else None
                    _cache_quote(
                        price,
                        source="cached_last_close",
                        bar_interval="1d",
                        price_ts=ts_obj,
                        tz_ok=tz_ok,
                        notes=notes_joined,
                        raw_price_ts=raw_ts_str,
                        raw_tz=raw_tz,
                    )
                    return _emit_result(
                        price,
                        age_min,
                        "STALE",
                        source="cached_last_close",
                        price_ts=ts_obj,
                        tz_ok=tz_ok,
                        thresholds=default_thresholds,
                        notes=notes_joined,
                        bar_interval="1d",
                        raw_price_ts=raw_ts_str,
                        raw_tz=raw_tz,
                    )
                fallback_notes.append("empty_history_daily")
            except Exception as e:
                fallback_notes.append(f"history_daily_error={e}")
                print(f"[PRICE] {ticker}: daily history failed - {e}")

        except Exception as e:
            fallback_notes.append(f"all_methods_error={e}")
            print(f"[ERROR] All price methods failed for {ticker}: {e}")

        final_notes = ';'.join(fallback_notes[:4]) if fallback_notes else "all_price_methods_failed"
        return _emit_result(
            None,
            99999.0,
            "STALE",
            source="missing",
            price_ts=None,
            tz_ok=False,
            thresholds=default_thresholds,
            notes=final_notes,
            bar_interval=None,
            raw_price_ts=None,
            raw_tz=None,
            debug_status="MISSING",
        )

    def _collect_price_debug(self, relevant_tickers=None, planned_trades=None, price_debug_cache=None, cap=None):
        """Collect capped price_debug for holdings + relevant tickers without changing trading decisions."""
        reporting_cfg = self.config.get('reporting', {}) if isinstance(self.config, dict) else {}
        cap_raw = cap if cap is not None else reporting_cfg.get('max_price_debug_items', 50)
        try:
            cap_val = max(1, int(cap_raw))
        except Exception:
            cap_val = 50

        price_debug_cache = dict(price_debug_cache) if isinstance(price_debug_cache, dict) else {}
        holdings = [str(t).upper() for t in self.positions.keys() if str(t).upper() != 'CASH']
        candidate_set = set(holdings)
        if isinstance(relevant_tickers, (list, tuple, set)):
            for t in relevant_tickers:
                t_u = str(t).upper().strip()
                if t_u and t_u != 'CASH':
                    candidate_set.add(t_u)

        planned_notional = {}
        if isinstance(planned_trades, list):
            for tr in planned_trades:
                if not isinstance(tr, dict):
                    continue
                t_u = str(tr.get('ticker', '')).upper().strip()
                if not t_u or t_u == 'CASH':
                    continue
                try:
                    n = abs(float(tr.get('desired_trade_value', tr.get('notional', 0.0)) or 0.0))
                except Exception:
                    n = 0.0
                if n > planned_notional.get(t_u, 0.0):
                    planned_notional[t_u] = n
                candidate_set.add(t_u)

        ordered = []
        seen = set()
        for t in holdings:
            if t not in seen:
                ordered.append(t)
                seen.add(t)
        for t, _ in sorted(planned_notional.items(), key=lambda x: x[1], reverse=True):
            if t not in seen:
                ordered.append(t)
                seen.add(t)
        for t in sorted(candidate_set):
            if t not in seen:
                ordered.append(t)
                seen.add(t)

        selected = ordered[:cap_val]
        market_state = None
        if isinstance(self.current_market_session, dict):
            market_state = self.current_market_session.get('state')

        price_debug = {}
        now_iso = self._now().isoformat()
        default_thresholds = {"live_max_min": 10.0, "recent_max_min": 60.0}
        for t_u in selected:
            dbg = price_debug_cache.get(t_u)
            if not isinstance(dbg, dict):
                try:
                    _price, _age, _status, dbg = self.get_current_price(t_u, return_debug=True)
                except Exception as e:
                    dbg = {
                        "ticker": t_u,
                        "now_ts": now_iso,
                        "status": "MISSING",
                        "age_min": None,
                        "source": "missing",
                        "price_ts": None,
                        "tz_ok": False,
                        "thresholds": dict(default_thresholds),
                        "notes": f"price_debug_collect_error={e}",
                    }
            if not isinstance(dbg, dict):
                dbg = {
                    "ticker": t_u,
                    "now_ts": now_iso,
                    "status": "MISSING",
                    "age_min": None,
                    "source": "missing",
                    "price_ts": None,
                    "tz_ok": False,
                    "thresholds": dict(default_thresholds),
                    "notes": "price_debug_missing_payload",
                }
            dbg_row = dict(dbg)
            dbg_row.setdefault("ticker", t_u)
            dbg_row.setdefault("now_ts", now_iso)
            dbg_row["status"] = str(dbg_row.get("status", "MISSING")).upper()
            dbg_row.setdefault("age_min", None)
            dbg_row.setdefault("source", "missing")
            dbg_row.setdefault("price_ts", None)
            dbg_row.setdefault("tz_ok", False)
            dbg_row.setdefault("thresholds", dict(default_thresholds))
            dbg_row.setdefault("notes", None)
            if market_state is not None:
                dbg_row["market_session_state"] = str(market_state)
            price_debug[t_u] = dbg_row
        return price_debug

    def _update_cycle_price_debug(self, candidate_tickers=None, planned_trades=None, price_debug_cache=None):
        """Build capped price_debug map for relevant tickers and store in engine state."""
        reporting_cfg = self.config.get('reporting', {}) if isinstance(self.config, dict) else {}
        cap_raw = reporting_cfg.get('max_price_debug_items', 50)
        try:
            cap = max(1, int(cap_raw))
        except Exception:
            cap = 50

        service = getattr(self, "price_service", None)
        if service is not None:
            prefetch_tickers = set()
            for ticker in self.positions.keys():
                ticker_u = str(ticker).upper().strip()
                if ticker_u and ticker_u != 'CASH':
                    prefetch_tickers.add(ticker_u)
            if isinstance(candidate_tickers, (list, tuple, set)):
                for ticker in candidate_tickers:
                    ticker_u = str(ticker).upper().strip()
                    if ticker_u and ticker_u != 'CASH':
                        prefetch_tickers.add(ticker_u)
            if isinstance(planned_trades, list):
                for trade in planned_trades:
                    if not isinstance(trade, dict):
                        continue
                    ticker_u = str(trade.get('ticker', '')).upper().strip()
                    if ticker_u and ticker_u != 'CASH':
                        prefetch_tickers.add(ticker_u)
            if prefetch_tickers:
                try:
                    self.price_service.prefetch(
                        prefetch_tickers,
                        interval="5m",
                        period="1d",
                        max_chunk=getattr(self, "price_batch_chunk_size", 50),
                        allow_1m_fallback=bool(getattr(self, "price_batch_allow_1m_fallback", True)),
                    )
                except Exception:
                    pass

        price_debug = self._collect_price_debug(
            relevant_tickers=candidate_tickers,
            planned_trades=planned_trades,
            price_debug_cache=price_debug_cache,
            cap=cap,
        )

        missing_count = 0
        stale_count = 0
        for dbg_row in price_debug.values():
            status = str((dbg_row or {}).get('status', '')).upper()
            if status == 'MISSING':
                missing_count += 1
            if status == 'STALE':
                stale_count += 1

        self.current_price_debug = price_debug
        print(f"[PRICE_DEBUG] n={len(price_debug)} cap={cap} missing={missing_count} stale={stale_count}")
        return price_debug

    def build_returns_matrix(
        self,
        tickers: list,
        lookback_days: int,
        *,
        period: str = '6mo',
        interval: str = '1d',
        min_obs: int = 30,
        drop_threshold: float = 0.5,
    ) -> tuple[pd.DataFrame, dict]:
        """Build aligned return matrix with diagnostics for future risk-model phases."""
        meta = {
            "lookback_days": int(lookback_days) if isinstance(lookback_days, (int, np.integer)) else 0,
            "min_obs": int(min_obs),
            "drop_threshold": float(drop_threshold),
            "input_tickers": [],
            "used_tickers": [],
            "missing_tickers": [],
            "dropped_tickers": [],
            "rows": 0,
            "cols": 0,
            "obs_by_ticker": {},
            "coverage_by_ticker": {},
            "overall_row_coverage": 0.0,
        }

        try:
            raw_tickers = list(tickers or [])
            seen = set()
            normalized = []
            for raw in raw_tickers:
                t = str(raw).upper().strip()
                if not t or t == 'CASH' or t in seen:
                    continue
                seen.add(t)
                normalized.append(t)

            meta["input_tickers"] = list(normalized)
            if int(lookback_days) <= 0:
                return pd.DataFrame(), meta

            lookback_days = int(lookback_days)
            min_obs = int(min_obs)
            drop_threshold = float(drop_threshold)
            threshold_obs = float(lookback_days) * (1.0 - drop_threshold)

            series_map = {}
            missing_tickers = []
            obs_by_ticker = {t: 0 for t in normalized}

            for ticker in normalized:
                hist = self.get_market_data(ticker, period=period, interval=interval)
                if hist is None or hist.empty or 'Close' not in hist.columns:
                    missing_tickers.append(ticker)
                    continue

                close = hist['Close'].astype(float).dropna()
                if close.empty:
                    missing_tickers.append(ticker)
                    continue

                daily_returns = close.pct_change().dropna()
                if daily_returns.empty:
                    missing_tickers.append(ticker)
                    continue

                series = daily_returns.tail(lookback_days)
                series_map[ticker] = series
                obs_by_ticker[ticker] = int(series.notna().sum())

            returns_df = pd.DataFrame()
            if series_map:
                returns_df = pd.concat(series_map, axis=1, join='outer')
                returns_df = returns_df.dropna(how='all')

            dropped_tickers = []
            if not returns_df.empty:
                for ticker in list(returns_df.columns):
                    obs_i = int(returns_df[ticker].notna().sum())
                    obs_by_ticker[ticker] = obs_i
                    if obs_i < min_obs or obs_i < threshold_obs:
                        dropped_tickers.append(ticker)

                if dropped_tickers:
                    returns_df = returns_df.drop(columns=dropped_tickers, errors='ignore')

            used_tickers = list(returns_df.columns) if not returns_df.empty else []

            coverage_by_ticker = {}
            denom = float(lookback_days) if lookback_days > 0 else 0.0
            for ticker in normalized:
                obs_i = int(obs_by_ticker.get(ticker, 0))
                coverage_by_ticker[ticker] = float(obs_i / denom) if denom > 0 else 0.0

            overall_row_coverage = 0.0
            if returns_df.shape[1] > 0 and len(returns_df.index) > 0:
                non_na_counts = returns_df.notna().sum(axis=1).astype(float)
                overall_row_coverage = float((non_na_counts / float(returns_df.shape[1])).mean())

            meta.update({
                "used_tickers": used_tickers,
                "missing_tickers": missing_tickers,
                "dropped_tickers": dropped_tickers,
                "rows": int(returns_df.shape[0]),
                "cols": int(returns_df.shape[1]),
                "obs_by_ticker": obs_by_ticker,
                "coverage_by_ticker": coverage_by_ticker,
                "overall_row_coverage": overall_row_coverage,
            })

            return returns_df, meta
        except Exception as e:
            print(f"[WARN] build_returns_matrix failed: {e}")
            return pd.DataFrame(), meta

    def _get_risk_model_cfg(self) -> dict:
        """Return risk-model config with safe defaults and type-normalized values."""
        defaults = {
            "enable_cov_diagnostics": True,
            "returns_period": "6mo",
            "returns_interval": "1d",
            "returns_lookback_days": 60,
            "min_obs": 30,
            "drop_threshold": 0.5,
            "shrinkage_alpha": 0.15,
            "annualization_factor": 252,
            "max_pair_corr_pairs": 3,
            "debug_log": False,
            "fallback_to_diag_on_error": True,
            "use_cov_vol_for_gate": False,
            "rc_limit": 0.35,
            "min_cov_gate_coverage": 0.6,
            "cov_gate_fallback_to_weighted": True,
            "enable_vol_targeting": False,
            "vol_target": 0.18,
            "vol_target_min_coverage": 0.6,
            "vol_target_min_scale": 0.10,
            "vol_target_max_scale": 1.00,
            "vol_target_use_cov_only": True,
        }
        try:
            raw_cfg = self.config.get("risk_model", {})
            if not isinstance(raw_cfg, dict):
                raw_cfg = {}
            cfg = dict(defaults)
            cfg.update(raw_cfg)

            cfg["enable_cov_diagnostics"] = bool(cfg.get("enable_cov_diagnostics", True))
            cfg["returns_period"] = str(cfg.get("returns_period", "6mo"))
            cfg["returns_interval"] = str(cfg.get("returns_interval", "1d"))
            cfg["returns_lookback_days"] = int(cfg.get("returns_lookback_days", 60))
            cfg["min_obs"] = int(cfg.get("min_obs", 30))
            cfg["drop_threshold"] = float(cfg.get("drop_threshold", 0.5))
            cfg["shrinkage_alpha"] = float(cfg.get("shrinkage_alpha", 0.15))
            cfg["annualization_factor"] = int(cfg.get("annualization_factor", 252))
            cfg["max_pair_corr_pairs"] = int(cfg.get("max_pair_corr_pairs", 3))
            cfg["debug_log"] = bool(cfg.get("debug_log", False))
            cfg["fallback_to_diag_on_error"] = bool(cfg.get("fallback_to_diag_on_error", True))
            cfg["use_cov_vol_for_gate"] = bool(cfg.get("use_cov_vol_for_gate", False))
            cfg["rc_limit"] = float(cfg.get("rc_limit", 0.35))
            cfg["min_cov_gate_coverage"] = float(cfg.get("min_cov_gate_coverage", 0.6))
            cfg["cov_gate_fallback_to_weighted"] = bool(cfg.get("cov_gate_fallback_to_weighted", True))
            cfg["enable_vol_targeting"] = bool(cfg.get("enable_vol_targeting", False))
            cfg["vol_target"] = float(cfg.get("vol_target", 0.18))
            cfg["vol_target_min_coverage"] = float(cfg.get("vol_target_min_coverage", 0.6))
            cfg["vol_target_min_scale"] = float(cfg.get("vol_target_min_scale", 0.10))
            cfg["vol_target_max_scale"] = float(cfg.get("vol_target_max_scale", 1.00))
            cfg["vol_target_use_cov_only"] = bool(cfg.get("vol_target_use_cov_only", True))

            cfg["drop_threshold"] = float(np.clip(cfg["drop_threshold"], 0.0, 1.0))
            cfg["shrinkage_alpha"] = float(np.clip(cfg["shrinkage_alpha"], 0.0, 1.0))
            if cfg["returns_lookback_days"] < 0:
                cfg["returns_lookback_days"] = 0
            if cfg["min_obs"] < 0:
                cfg["min_obs"] = 0
            if cfg["annualization_factor"] <= 0:
                cfg["annualization_factor"] = 252
            if cfg["max_pair_corr_pairs"] < 0:
                cfg["max_pair_corr_pairs"] = 0
            if cfg["rc_limit"] < 0:
                cfg["rc_limit"] = 0.0
            cfg["min_cov_gate_coverage"] = float(np.clip(cfg["min_cov_gate_coverage"], 0.0, 1.0))
            if cfg["vol_target"] <= 0:
                cfg["vol_target"] = 0.18
            cfg["vol_target_min_coverage"] = float(np.clip(cfg["vol_target_min_coverage"], 0.0, 1.0))
            cfg["vol_target_min_scale"] = float(np.clip(cfg["vol_target_min_scale"], 0.0, 1.0))
            cfg["vol_target_max_scale"] = float(np.clip(cfg["vol_target_max_scale"], 0.0, 1.0))
            if cfg["vol_target_max_scale"] < cfg["vol_target_min_scale"]:
                cfg["vol_target_max_scale"] = cfg["vol_target_min_scale"]
            return cfg
        except Exception:
            return dict(defaults)

    def _get_cost_model_cfg(self) -> dict:
        """Return cost-model config with safe defaults."""
        defaults = {
            "enabled": False,
            "fee_bps": 1.0,
            "slippage_bps": 2.0,
            "impact_enabled": False,
            "impact_k": 0.1,
            "adv_lookback_days": 20,
            "debug_log": False
        }
        try:
            raw_cfg = self.config.get("cost_model", {})
            if not isinstance(raw_cfg, dict):
                raw_cfg = {}
            cfg = dict(defaults)
            cfg.update(raw_cfg)

            cfg["enabled"] = bool(cfg.get("enabled", False))
            cfg["fee_bps"] = max(0.0, float(cfg.get("fee_bps", 1.0)))
            cfg["slippage_bps"] = max(0.0, float(cfg.get("slippage_bps", 2.0)))
            cfg["impact_enabled"] = bool(cfg.get("impact_enabled", False))
            cfg["impact_k"] = max(0.0, float(cfg.get("impact_k", 0.1)))
            cfg["adv_lookback_days"] = max(1, int(cfg.get("adv_lookback_days", 20)))
            cfg["debug_log"] = bool(cfg.get("debug_log", False))
            return cfg
        except Exception:
            return dict(defaults)

    def _get_planner_cfg(self) -> dict:
        """Return trade-planner config with safe defaults."""
        defaults = {
            "enable_trade_planner": False,
            "enabled": False,
            "planner_debug_log": False,
            "min_trade_notional": 5.0,
            "allow_partial_fill": True,
            "enable_adv_limit": False,
            "adv_limit_frac": 0.02,
            "adv_lookback_days": 20,
            "adv_volume_field": "Volume",
            "adv_price_field": "Close",
            "adv_apply_to_forced": True,
            "enable_cost_sensitive_ranking": False,
            "lambda_cost": 1.0,
            "benefit_mode": "delta_weight",
            "max_audit_items": 20,
        }
        try:
            raw_cfg = self.config.get("trade_planner", {})
            if not isinstance(raw_cfg, dict):
                raw_cfg = {}
            execution_cfg = self.config.get("execution", {})
            if isinstance(execution_cfg, dict):
                exec_planner_cfg = execution_cfg.get("trade_planner", {})
                if isinstance(exec_planner_cfg, dict):
                    merged = dict(exec_planner_cfg)
                    merged.update(raw_cfg)
                    raw_cfg = merged
            cfg = dict(defaults)
            cfg.update(raw_cfg)
            if "enable_trade_planner" in raw_cfg:
                enabled_raw = raw_cfg.get("enable_trade_planner")
            elif "enabled" in raw_cfg:
                enabled_raw = raw_cfg.get("enabled")
            else:
                enabled_raw = cfg.get("enable_trade_planner", False)
            cfg["enable_trade_planner"] = bool(enabled_raw)
            cfg["enabled"] = bool(enabled_raw)
            cfg["planner_debug_log"] = bool(cfg.get("planner_debug_log", False))
            cfg["min_trade_notional"] = max(0.0, float(cfg.get("min_trade_notional", 5.0)))
            cfg["allow_partial_fill"] = bool(cfg.get("allow_partial_fill", True))
            cfg["enable_adv_limit"] = bool(cfg.get("enable_adv_limit", False))
            cfg["adv_limit_frac"] = max(0.0, float(cfg.get("adv_limit_frac", 0.02)))
            cfg["adv_lookback_days"] = max(1, int(cfg.get("adv_lookback_days", 20)))
            cfg["adv_volume_field"] = str(cfg.get("adv_volume_field", "Volume") or "Volume")
            cfg["adv_price_field"] = str(cfg.get("adv_price_field", "Close") or "Close")
            cfg["adv_apply_to_forced"] = bool(cfg.get("adv_apply_to_forced", True))
            cfg["enable_cost_sensitive_ranking"] = bool(cfg.get("enable_cost_sensitive_ranking", False))
            cfg["lambda_cost"] = float(cfg.get("lambda_cost", 1.0))
            cfg["benefit_mode"] = str(cfg.get("benefit_mode", "delta_weight")).lower()
            if cfg["benefit_mode"] not in ("delta_weight", "delta_notional"):
                cfg["benefit_mode"] = "delta_weight"
            cfg["max_audit_items"] = max(1, int(cfg.get("max_audit_items", 20)))
            return cfg
        except Exception:
            return dict(defaults)

    def estimate_adv_notional(self, ticker: str, *, lookback_days: int) -> float | None:
        """Estimate ADV notional from recent daily Volume*Close."""
        try:
            ticker_u = str(ticker).upper().strip()
            if not ticker_u or ticker_u == "CASH":
                return None
            planner_cfg = self._get_planner_cfg()
            volume_field = str(planner_cfg.get("adv_volume_field", "Volume") or "Volume")
            price_field = str(planner_cfg.get("adv_price_field", "Close") or "Close")
            lookback_days = max(1, int(lookback_days))
            hist = self.get_market_data(ticker_u, period='3mo', interval='1d')
            if hist is None or hist.empty:
                return None
            if volume_field not in hist.columns or price_field not in hist.columns:
                return None
            vol = pd.to_numeric(hist[volume_field], errors='coerce').tail(lookback_days)
            px = pd.to_numeric(hist[price_field], errors='coerce').tail(lookback_days)
            notional_series = (vol * px).replace([np.inf, -np.inf], np.nan).dropna()
            if notional_series.empty:
                return None
            adv_notional = float(notional_series.mean())
            if not np.isfinite(adv_notional) or adv_notional <= 0:
                return None
            return adv_notional
        except Exception:
            return None

    def apply_trade_planner(self, trades: list, equity: float, turnover_limit: float, *, reason_tag: str = "planner") -> tuple[list, dict]:
        """Prioritize forced trades under turnover budget without changing target generation."""
        cfg = self._get_planner_cfg()
        meta = {
            "enabled": bool(cfg.get("enable_trade_planner", False)),
            "status": "disabled",
            "reason_tag": reason_tag,
            "turnover_limit": float(turnover_limit if turnover_limit is not None else 0.0),
            "turnover_used_forced": 0.0,
            "turnover_used_normal": 0.0,
            "turnover_used_total": 0.0,
            "num_forced": 0,
            "num_normal": 0,
            "num_dropped": 0,
            "dropped": [],
            "scaled": [],
            "adv_limit_enabled": bool(cfg.get("enable_adv_limit", False)),
            "adv_limit_frac": float(cfg.get("adv_limit_frac", 0.02)),
            "num_adv_clipped": 0,
            "num_adv_dropped": 0,
            "normal_sorted_by": "score" if bool(cfg.get("enable_cost_sensitive_ranking", False)) else "notional",
            "lambda_cost": float(cfg.get("lambda_cost", 1.0)),
            "benefit_mode": str(cfg.get("benefit_mode", "delta_weight")),
            "normal_score_stats": {
                "count": 0,
                "benefit_min": None,
                "benefit_median": None,
                "benefit_p95": None,
                "benefit_max": None,
                "cost_weight_min": None,
                "cost_weight_median": None,
                "cost_weight_p95": None,
                "cost_weight_max": None,
                "score_min": None,
                "score_median": None,
                "score_p95": None,
                "score_max": None
            },
        }
        if not bool(cfg.get("enable_trade_planner", False)):
            return trades, meta

        try:
            budget_notional = float(turnover_limit or 0.0)
            if budget_notional <= 1.0 and float(equity or 0.0) > 0:
                budget_notional = float(equity) * budget_notional
            budget_notional = max(0.0, budget_notional)
            meta["turnover_limit"] = float(budget_notional)

            min_trade_notional = float(cfg.get("min_trade_notional", 5.0))
            allow_partial_fill = bool(cfg.get("allow_partial_fill", True))
            planner_debug_log = bool(cfg.get("planner_debug_log", False))
            enable_adv_limit = bool(cfg.get("enable_adv_limit", False))
            adv_limit_frac = float(cfg.get("adv_limit_frac", 0.02))
            adv_lookback_days = int(cfg.get("adv_lookback_days", 20))
            adv_apply_to_forced = bool(cfg.get("adv_apply_to_forced", True))
            enable_cost_sensitive_ranking = bool(cfg.get("enable_cost_sensitive_ranking", False))
            lambda_cost = float(cfg.get("lambda_cost", 1.0))
            benefit_mode = str(cfg.get("benefit_mode", "delta_weight")).lower()
            max_audit_items = int(cfg.get("max_audit_items", 20))
            cost_cfg = self._get_cost_model_cfg()
            impact_enabled = bool(cost_cfg.get("impact_enabled", False))
            if benefit_mode == "delta_notional":
                meta["benefit_mode"] = "delta_notional_weighted"

            def _extract_notional(trade_obj):
                try:
                    if isinstance(trade_obj, dict):
                        if "desired_trade_value" in trade_obj:
                            return abs(float(trade_obj.get("desired_trade_value", 0.0) or 0.0))
                        if "notional" in trade_obj:
                            return abs(float(trade_obj.get("notional", 0.0) or 0.0))
                        if "price" in trade_obj and "quantity" in trade_obj:
                            return abs(float(trade_obj.get("price", 0.0) or 0.0) * float(trade_obj.get("quantity", 0.0) or 0.0))
                except Exception:
                    return 0.0
                return 0.0

            def _is_forced_trade(trade_obj):
                if not isinstance(trade_obj, dict):
                    return False
                if bool(trade_obj.get("is_forced", False)):
                    return True
                if str(trade_obj.get("priority", "")).lower() == "forced":
                    return True

                side_u = str(trade_obj.get("side", "")).upper()
                if side_u != "SELL":
                    return False

                force_reason = str(trade_obj.get("force_reason", "")).lower()
                if force_reason in ("exit_signal", "risk_off", "risk_off_forced", "circuit_breaker", "stale_sell"):
                    return True

                reason_blob = " ".join([
                    str(trade_obj.get("reason", "")),
                    str(trade_obj.get("decision_trace", "")),
                    str(trade_obj.get("regime_state", "")),
                    str(trade_obj.get("price_status", "")),
                ]).lower()
                if ("exit_signal" in reason_blob) or ("circuit_breaker" in reason_blob) or ("risk_off" in reason_blob):
                    return True
                if str(trade_obj.get("status", "")).upper() == "STALE" and "sell" in reason_blob:
                    return True
                return False

            def _scale_trade(trade_obj, scale):
                scaled = dict(trade_obj) if isinstance(trade_obj, dict) else trade_obj
                old_notional = _extract_notional(scaled)
                if old_notional <= 0:
                    return None, 0.0

                scaled_desired_notional = float(max(0.0, old_notional * scale))
                if isinstance(scaled, dict) and "desired_trade_value" in scaled:
                    scaled["desired_trade_value"] = scaled_desired_notional

                if isinstance(scaled, dict) and "quantity" in scaled:
                    try:
                        old_qty = int(float(scaled.get("quantity", 0) or 0))
                    except Exception:
                        old_qty = 0
                    if old_qty > 0:
                        new_qty = int(np.floor(old_qty * scale))
                        if new_qty <= 0:
                            return None, 0.0
                        scaled["quantity"] = int(new_qty)
                        try:
                            price_v = float(scaled.get("price", 0.0) or 0.0)
                            if price_v > 0:
                                scaled_desired_notional = float(new_qty * price_v)
                                scaled["notional"] = scaled_desired_notional
                        except Exception:
                            pass

                if isinstance(scaled, dict):
                    if "desired_trade_value" in scaled:
                        scaled["desired_trade_value"] = float(scaled_desired_notional)
                    scaled["notional"] = float(scaled_desired_notional)

                new_notional = _extract_notional(scaled)
                if new_notional <= 0:
                    return None, 0.0
                if isinstance(scaled, dict) and "notional" in scaled:
                    scaled["notional"] = float(new_notional)
                return scaled, float(new_notional)

            def _compute_benefit(trade_obj, notional_v):
                eq = float(equity or 0.0)
                if benefit_mode == "delta_notional":
                    if eq > 0:
                        return float(abs(notional_v) / eq)
                    return 0.0
                try:
                    if isinstance(trade_obj, dict) and trade_obj.get("delta_weight") is not None:
                        return float(abs(float(trade_obj.get("delta_weight", 0.0) or 0.0)))
                except Exception:
                    pass
                if eq > 0:
                    return float(abs(notional_v) / eq)
                return 0.0

            def _compute_cost_weight(cost_dollars):
                eq = float(equity or 0.0)
                if eq <= 0:
                    return 0.0
                try:
                    return float(float(cost_dollars or 0.0) / eq)
                except Exception:
                    return 0.0

            prepared = []
            dropped = []
            scaled = []
            dropped_map = {}
            scaled_map = {}
            num_adv_clipped = 0
            num_adv_dropped = 0
            for idx, trade in enumerate(trades or []):
                trade_copy = dict(trade) if isinstance(trade, dict) else trade
                ticker = str(trade_copy.get("ticker", "")).upper().strip() if isinstance(trade_copy, dict) else ""
                side = str(trade_copy.get("side", "")).upper().strip() if isinstance(trade_copy, dict) else ""
                notional = _extract_notional(trade_copy)
                trade_id = f"{ticker}:{side}:{round(float(notional or 0.0), 2)}:{idx}"
                if isinstance(trade_copy, dict):
                    trade_copy["_planner_trade_id"] = trade_id
                    trade_copy["_orig_notional"] = float(notional)
                if notional < min_trade_notional:
                    if isinstance(trade_copy, dict):
                        drop_item = {
                            "ticker": str(trade_copy.get("ticker", "")),
                            "side": str(trade_copy.get("side", "")),
                            "reason": "min_notional",
                            "adv_clipped": bool(trade_copy.get("adv_clipped", False)),
                            "adv_participation": trade_copy.get("adv_participation"),
                            "planner_score": trade_copy.get("planner_score"),
                            "trade_id": trade_id,
                            "old_notional": float(notional)
                        }
                        dropped.append(drop_item)
                        dropped_map[trade_id] = drop_item
                    continue
                forced = _is_forced_trade(trade_copy)
                if isinstance(trade_copy, dict):
                    trade_copy["is_forced"] = bool(forced)
                    trade_copy["priority"] = "forced" if forced else "normal"

                adv_notional = None
                if (enable_adv_limit or impact_enabled) and ticker and ticker != "CASH":
                    adv_notional = self.estimate_adv_notional(ticker, lookback_days=adv_lookback_days)

                cost_est = self.estimate_trade_cost(ticker, side, notional, adv_notional=adv_notional)
                planner_cost_dollars = float(cost_est.get("total", 0.0) or 0.0)
                planner_cost_weight = _compute_cost_weight(planner_cost_dollars)

                adv_clipped = False
                adv_max_notional = None
                adv_scale_entry = None
                if enable_adv_limit and adv_notional is not None and adv_notional > 0:
                    adv_max_notional = float(max(0.0, adv_limit_frac * adv_notional))
                    if notional > adv_max_notional + 1e-12:
                        if forced and (not adv_apply_to_forced):
                            pass
                        else:
                            if (not forced) and (not allow_partial_fill):
                                num_adv_dropped += 1
                                drop_item = {
                                    "ticker": ticker,
                                    "side": side,
                                    "reason": "adv_limit",
                                    "adv_clipped": False,
                                    "adv_participation": float(notional / adv_notional) if adv_notional else None,
                                    "planner_score": None,
                                    "trade_id": trade_id,
                                    "old_notional": float(trade_copy.get("_orig_notional", notional))
                                }
                                dropped.append(drop_item)
                                dropped_map[trade_id] = drop_item
                                continue
                            adv_scale = float(adv_max_notional / notional) if notional > 0 else 0.0
                            scaled_trade, new_notional = _scale_trade(trade_copy, adv_scale)
                            if scaled_trade is None or new_notional < min_trade_notional:
                                num_adv_dropped += 1
                                drop_reason = "adv_limit_forced_qty0" if forced and side == "SELL" else "adv_limit"
                                drop_item = {
                                    "ticker": ticker,
                                    "side": side,
                                    "reason": drop_reason,
                                    "adv_clipped": False,
                                    "adv_participation": float(notional / adv_notional) if adv_notional else None,
                                    "planner_score": None,
                                    "trade_id": trade_id,
                                    "old_notional": float(trade_copy.get("_orig_notional", notional))
                                }
                                dropped.append(drop_item)
                                dropped_map[trade_id] = drop_item
                                continue
                            trade_copy = scaled_trade
                            if isinstance(trade_copy, dict):
                                trade_copy["_planner_trade_id"] = trade_id
                                trade_copy["_orig_notional"] = float(trade_copy.get("_orig_notional", notional))
                            notional = float(new_notional)
                            adv_clipped = True
                            num_adv_clipped += 1
                            adv_scale_entry = {
                                "ticker": ticker,
                                "side": side,
                                "scale": float(adv_scale),
                                "old_notional": float(_extract_notional(trade)),
                                "new_notional": float(new_notional),
                                "reason": "adv_limit",
                                "adv_clipped": True,
                                "adv_participation": float(new_notional / adv_notional) if adv_notional else None,
                                "planner_score": None,
                                "trade_id": trade_id
                            }
                            cost_est = self.estimate_trade_cost(ticker, side, notional, adv_notional=adv_notional)
                            planner_cost_dollars = float(cost_est.get("total", 0.0) or 0.0)
                            planner_cost_weight = _compute_cost_weight(planner_cost_dollars)

                adv_participation = cost_est.get("participation")
                if adv_participation is None and adv_notional is not None and adv_notional > 0:
                    adv_participation = float(notional / adv_notional)

                planner_benefit = _compute_benefit(trade_copy, notional)
                planner_score = float(planner_benefit - (lambda_cost * planner_cost_weight))

                if isinstance(trade_copy, dict):
                    trade_copy["adv_notional"] = float(adv_notional) if adv_notional is not None else None
                    trade_copy["adv_limit_frac"] = float(adv_limit_frac)
                    trade_copy["adv_max_notional"] = float(adv_max_notional) if adv_max_notional is not None else None
                    trade_copy["adv_clipped"] = bool(adv_clipped)
                    trade_copy["adv_participation"] = float(adv_participation) if adv_participation is not None else None
                    trade_copy["planner_cost_est"] = cost_est
                    trade_copy["planner_cost"] = float(planner_cost_dollars)
                    trade_copy["planner_cost_dollars"] = float(planner_cost_dollars)
                    trade_copy["planner_cost_weight"] = float(planner_cost_weight)
                    trade_copy["planner_benefit"] = float(planner_benefit)
                    trade_copy["planner_score"] = float(planner_score)
                    trade_copy["notional"] = float(notional)
                if isinstance(adv_scale_entry, dict):
                    adv_scale_entry["planner_score"] = float(planner_score)
                    adv_scale_entry["planner_benefit"] = float(planner_benefit)
                    adv_scale_entry["planner_cost_dollars"] = float(planner_cost_dollars)
                    adv_scale_entry["planner_cost_weight"] = float(planner_cost_weight)
                    scaled.append(adv_scale_entry)
                    scaled_map[trade_id] = adv_scale_entry

                prepared.append({
                    "idx": idx,
                    "trade_id": trade_id,
                    "trade": trade_copy,
                    "notional": float(notional),
                    "forced": bool(forced),
                    "score": float(planner_score),
                    "benefit": float(planner_benefit),
                    "cost": float(planner_cost_dollars),
                    "cost_weight": float(planner_cost_weight),
                })

            forced_items = [x for x in prepared if x["forced"]]
            normal_items = [x for x in prepared if not x["forced"]]
            forced_items.sort(key=lambda x: (-x["notional"], x["idx"]))
            if enable_cost_sensitive_ranking:
                normal_items.sort(key=lambda x: (-x["score"], -x["notional"], -x["benefit"], x["idx"]))
            else:
                normal_items.sort(key=lambda x: (-x["notional"], x["idx"]))
            meta["num_forced"] = len(forced_items)
            meta["num_normal"] = len(normal_items)
            normal_benefits = [float(x.get("benefit", 0.0)) for x in normal_items if np.isfinite(float(x.get("benefit", 0.0)))]
            normal_cost_weights = [float(x.get("cost_weight", 0.0)) for x in normal_items if np.isfinite(float(x.get("cost_weight", 0.0)))]
            normal_scores = [float(x.get("score", 0.0)) for x in normal_items if np.isfinite(float(x.get("score", 0.0)))]

            def _summary_stats(values):
                if not values:
                    return None, None, None, None
                arr = np.array(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return None, None, None, None
                return float(np.min(arr)), float(np.median(arr)), float(np.percentile(arr, 95)), float(np.max(arr))

            b_min, b_med, b_p95, b_max = _summary_stats(normal_benefits)
            c_min, c_med, c_p95, c_max = _summary_stats(normal_cost_weights)
            s_min, s_med, s_p95, s_max = _summary_stats(normal_scores)
            meta["normal_score_stats"] = {
                "count": int(len(normal_items)),
                "benefit_min": b_min,
                "benefit_median": b_med,
                "benefit_p95": b_p95,
                "benefit_max": b_max,
                "cost_weight_min": c_min,
                "cost_weight_median": c_med,
                "cost_weight_p95": c_p95,
                "cost_weight_max": c_max,
                "score_min": s_min,
                "score_median": s_med,
                "score_p95": s_p95,
                "score_max": s_max,
            }

            remaining_budget = float(budget_notional)
            planned_entries = []

            def _consume(group_items, is_forced_group):
                nonlocal remaining_budget
                used = 0.0
                for item in group_items:
                    tr = item["trade"]
                    trade_id = item.get("trade_id")
                    notional = item["notional"]
                    ticker = str(tr.get("ticker", "")) if isinstance(tr, dict) else ""
                    side = str(tr.get("side", "")) if isinstance(tr, dict) else ""
                    score = tr.get("planner_score") if isinstance(tr, dict) else None
                    adv_clipped = bool(tr.get("adv_clipped", False)) if isinstance(tr, dict) else False
                    adv_participation = tr.get("adv_participation") if isinstance(tr, dict) else None
                    old_notional = float(tr.get("_orig_notional", notional) or notional) if isinstance(tr, dict) else float(notional)

                    if remaining_budget <= 1e-12:
                        drop_item = {
                            "ticker": ticker,
                            "side": side,
                            "reason": "over_budget_forced" if is_forced_group else "over_budget",
                            "adv_clipped": adv_clipped,
                            "adv_participation": adv_participation,
                            "planner_score": score,
                            "planner_benefit": tr.get("planner_benefit") if isinstance(tr, dict) else None,
                            "planner_cost_dollars": tr.get("planner_cost_dollars") if isinstance(tr, dict) else None,
                            "planner_cost_weight": tr.get("planner_cost_weight") if isinstance(tr, dict) else None,
                            "trade_id": trade_id,
                            "old_notional": old_notional
                        }
                        dropped.append(drop_item)
                        if isinstance(trade_id, str):
                            dropped_map[trade_id] = drop_item
                        continue

                    if notional <= remaining_budget + 1e-12:
                        planned_entries.append(tr)
                        remaining_budget -= notional
                        used += notional
                        continue

                    if not allow_partial_fill:
                        drop_item = {
                            "ticker": ticker,
                            "side": side,
                            "reason": "over_budget_forced" if is_forced_group else "over_budget",
                            "adv_clipped": adv_clipped,
                            "adv_participation": adv_participation,
                            "planner_score": score,
                            "planner_benefit": tr.get("planner_benefit") if isinstance(tr, dict) else None,
                            "planner_cost_dollars": tr.get("planner_cost_dollars") if isinstance(tr, dict) else None,
                            "planner_cost_weight": tr.get("planner_cost_weight") if isinstance(tr, dict) else None,
                            "trade_id": trade_id,
                            "old_notional": old_notional
                        }
                        dropped.append(drop_item)
                        if isinstance(trade_id, str):
                            dropped_map[trade_id] = drop_item
                        # Budget overflow: stop allocating further trades this cycle.
                        remaining_budget = 0.0
                        continue

                    scale = float(remaining_budget / notional) if notional > 0 else 0.0
                    scaled_trade, new_notional = _scale_trade(tr, scale)
                    if scaled_trade is None or new_notional < min_trade_notional:
                        drop_item = {
                            "ticker": ticker,
                            "side": side,
                            "reason": "over_budget_forced" if is_forced_group else "over_budget",
                            "adv_clipped": adv_clipped,
                            "adv_participation": adv_participation,
                            "planner_score": score,
                            "planner_benefit": tr.get("planner_benefit") if isinstance(tr, dict) else None,
                            "planner_cost_dollars": tr.get("planner_cost_dollars") if isinstance(tr, dict) else None,
                            "planner_cost_weight": tr.get("planner_cost_weight") if isinstance(tr, dict) else None,
                            "trade_id": trade_id,
                            "old_notional": old_notional
                        }
                        dropped.append(drop_item)
                        if isinstance(trade_id, str):
                            dropped_map[trade_id] = drop_item
                        # Overflow trade cannot be filled: halt and drop all remaining.
                        remaining_budget = 0.0
                        continue

                    scaled_cost_est = self.estimate_trade_cost(
                        str(scaled_trade.get("ticker", "")),
                        str(scaled_trade.get("side", "")),
                        float(new_notional),
                        adv_notional=scaled_trade.get("adv_notional")
                    ) if isinstance(scaled_trade, dict) else {}
                    scaled_cost = float(scaled_cost_est.get("total", 0.0) or 0.0) if isinstance(scaled_cost_est, dict) else 0.0
                    scaled_cost_weight = _compute_cost_weight(scaled_cost)
                    scaled_benefit = _compute_benefit(scaled_trade if isinstance(scaled_trade, dict) else tr, new_notional)
                    scaled_score = float(scaled_benefit - (lambda_cost * scaled_cost_weight))
                    if isinstance(scaled_trade, dict):
                        scaled_trade["_planner_trade_id"] = trade_id
                        scaled_trade["_orig_notional"] = old_notional
                        scaled_trade["planner_cost_est"] = scaled_cost_est if isinstance(scaled_cost_est, dict) else {}
                        scaled_trade["planner_cost"] = float(scaled_cost)
                        scaled_trade["planner_cost_dollars"] = float(scaled_cost)
                        scaled_trade["planner_cost_weight"] = float(scaled_cost_weight)
                        scaled_trade["planner_benefit"] = float(scaled_benefit)
                        scaled_trade["planner_score"] = float(scaled_score)
                        if scaled_trade.get("adv_notional") not in (None, 0):
                            try:
                                scaled_trade["adv_participation"] = float(new_notional / float(scaled_trade.get("adv_notional")))
                            except Exception:
                                pass

                    planned_entries.append(scaled_trade)
                    scale_item = {
                        "ticker": ticker,
                        "side": side,
                        "scale": float(scale),
                        "old_notional": old_notional,
                        "new_notional": float(new_notional),
                        "reason": "budget_scale",
                        "adv_clipped": bool(scaled_trade.get("adv_clipped", False)) if isinstance(scaled_trade, dict) else False,
                        "adv_participation": scaled_trade.get("adv_participation") if isinstance(scaled_trade, dict) else None,
                        "planner_score": scaled_score,
                        "planner_benefit": float(scaled_benefit),
                        "planner_cost_dollars": float(scaled_cost),
                        "planner_cost_weight": float(scaled_cost_weight),
                        "trade_id": trade_id
                    }
                    scaled.append(scale_item)
                    if isinstance(trade_id, str):
                        scaled_map[trade_id] = scale_item
                    # Scale one last trade to remaining budget, then stop allocation.
                    remaining_budget = 0.0
                    used += new_notional
                return used

            used_forced = _consume(forced_items, True)
            used_normal = _consume(normal_items, False)

            meta["turnover_used_forced"] = float(used_forced)
            meta["turnover_used_normal"] = float(used_normal)
            meta["turnover_used_total"] = float(used_forced + used_normal)
            kept_ids = set()
            for planned in planned_entries:
                if isinstance(planned, dict):
                    planned_id = planned.get("_planner_trade_id")
                    if isinstance(planned_id, str):
                        kept_ids.add(planned_id)

            dropped_dedup = []
            dropped_seen = set()
            for tid, item in dropped_map.items():
                if tid in kept_ids:
                    continue
                if tid in dropped_seen:
                    continue
                dropped_seen.add(tid)
                dropped_dedup.append(item)
            for item in dropped:
                tid = item.get("trade_id")
                if isinstance(tid, str):
                    if tid in kept_ids or tid in dropped_seen:
                        continue
                    dropped_seen.add(tid)
                dropped_dedup.append(item)

            scaled_dedup = []
            scaled_seen = set()
            for tid, item in scaled_map.items():
                if tid not in kept_ids:
                    continue
                if tid in scaled_seen:
                    continue
                scaled_seen.add(tid)
                scaled_dedup.append(item)
            for item in scaled:
                tid = item.get("trade_id")
                if isinstance(tid, str):
                    if tid not in kept_ids or tid in scaled_seen:
                        continue
                    scaled_seen.add(tid)
                scaled_dedup.append(item)

            meta["dropped"] = dropped_dedup[:max_audit_items]
            meta["scaled"] = scaled_dedup[:max_audit_items]
            meta["num_dropped"] = len(dropped_dedup)
            meta["num_adv_clipped"] = int(num_adv_clipped)
            meta["num_adv_dropped"] = int(num_adv_dropped)
            meta["status"] = "ok"

            if planner_debug_log:
                print(
                    f"[PLANNER] forced={meta['num_forced']} normal={meta['num_normal']} "
                    f"dropped={meta['num_dropped']} adv_clipped={meta['num_adv_clipped']} "
                    f"used={meta['turnover_used_total']:.2f}/{meta['turnover_limit']:.2f}"
                )

            return planned_entries, meta
        except Exception as e:
            meta["status"] = "error"
            meta["error"] = str(e)
            return trades, meta

    def estimate_trade_cost(self, ticker: str, side: str, notional: float, *, adv_notional: float | None = None) -> dict:
        """Estimate transaction cost components without affecting execution."""
        cfg = self._get_cost_model_cfg()
        result = {
            "enabled": bool(cfg.get("enabled", False)),
            "status": "ok",
            "fee": 0.0,
            "slippage": 0.0,
            "impact": 0.0,
            "total": 0.0,
            "fee_bps": float(cfg.get("fee_bps", 0.0)),
            "slippage_bps": float(cfg.get("slippage_bps", 0.0)),
            "impact_enabled": bool(cfg.get("impact_enabled", False)),
            "impact_k": float(cfg.get("impact_k", 0.0)),
            "notional": 0.0,
            "adv_notional": None,
            "participation": None,
            "ticker": str(ticker).upper().strip() if ticker is not None else "",
            "side": str(side).upper().strip() if side is not None else "",
        }
        try:
            n = abs(float(notional or 0.0))
            if not np.isfinite(n) or n <= 0:
                result["status"] = "invalid_notional"
                return result

            result["notional"] = float(n)
            if not bool(cfg.get("enabled", False)):
                result["status"] = "disabled"
                return result

            fee = n * (float(cfg.get("fee_bps", 0.0)) / 10000.0)
            slippage = n * (float(cfg.get("slippage_bps", 0.0)) / 10000.0)
            impact = 0.0
            participation = None
            adv_val = None

            if bool(cfg.get("impact_enabled", False)) and adv_notional is not None:
                try:
                    adv_val = float(adv_notional)
                    if np.isfinite(adv_val) and adv_val > 0:
                        participation = float(n / adv_val)
                        impact = n * float(cfg.get("impact_k", 0.0)) * participation
                except Exception:
                    adv_val = None
                    participation = None
                    impact = 0.0

            total = fee + slippage + impact
            result.update({
                "fee": float(fee),
                "slippage": float(slippage),
                "impact": float(impact),
                "total": float(total),
                "adv_notional": float(adv_val) if adv_val is not None else None,
                "participation": float(participation) if participation is not None else None,
                "status": "ok"
            })

            if bool(cfg.get("debug_log", False)):
                print(
                    f"[COST EST] {result['side']} {result['ticker']} notional=${n:,.2f} "
                    f"fee=${fee:.2f} slippage=${slippage:.2f} impact=${impact:.2f} total=${total:.2f}"
                )
            return result
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["total"] = 0.0
            return result

    def _estimate_covariance_diag_shrink(self, returns_df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict]:
        """Estimate covariance with diagonal shrinkage for stability."""
        meta = {
            "method": "diag_shrink",
            "alpha": float(np.clip(alpha, 0.0, 1.0)),
            "rows": 0,
            "cols": 0,
        }
        try:
            if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
                return pd.DataFrame(), meta
            if returns_df.shape[1] < 1 or returns_df.shape[0] < 2:
                meta["rows"] = int(returns_df.shape[0])
                meta["cols"] = int(returns_df.shape[1])
                return pd.DataFrame(), meta

            sample_cov = returns_df.cov()
            meta["rows"] = int(returns_df.shape[0])
            meta["cols"] = int(sample_cov.shape[1]) if isinstance(sample_cov, pd.DataFrame) else 0
            if sample_cov.empty:
                return pd.DataFrame(), meta

            diag_cov = pd.DataFrame(
                np.diag(np.diag(sample_cov.values)),
                index=sample_cov.index,
                columns=sample_cov.columns,
            )
            a = float(np.clip(alpha, 0.0, 1.0))
            shrunk_cov = (1.0 - a) * sample_cov + a * diag_cov
            return shrunk_cov, meta
        except Exception as e:
            meta["error"] = str(e)
            return pd.DataFrame(), meta

    def _compute_portfolio_vol_and_rc(self, cov: pd.DataFrame, weights: dict, annualization_factor: int) -> dict:
        """Compute annualized portfolio volatility and risk contributions."""
        result = {
            "portfolio_var": 0.0,
            "portfolio_vol": 0.0,
            "marginal_contrib": {},
            "rc_fraction": {},
            "max_rc_ticker": None,
            "max_rc_fraction": 0.0,
        }
        try:
            if not isinstance(cov, pd.DataFrame) or cov.empty:
                return result
            if not isinstance(weights, dict) or not weights:
                return result

            tickers = []
            for ticker in cov.columns:
                w = float(weights.get(ticker, 0.0) or 0.0)
                if abs(w) > 1e-12:
                    tickers.append(ticker)

            if not tickers:
                return result

            sub_cov = cov.loc[tickers, tickers]
            w_vec = np.array([float(weights.get(t, 0.0) or 0.0) for t in tickers], dtype=float)
            cov_values = sub_cov.values.astype(float)
            annualization_factor = int(annualization_factor) if int(annualization_factor) > 0 else 252

            port_var_daily = float(np.dot(w_vec, np.dot(cov_values, w_vec)))
            if not np.isfinite(port_var_daily):
                port_var_daily = 0.0

            result["portfolio_var"] = max(0.0, float(port_var_daily))
            result["portfolio_vol"] = float(np.sqrt(max(0.0, port_var_daily) * float(annualization_factor)))

            if port_var_daily <= 0:
                result["marginal_contrib"] = {t: 0.0 for t in tickers}
                result["rc_fraction"] = {t: 0.0 for t in tickers}
                return result

            marginal = np.dot(cov_values, w_vec)
            rc_raw = w_vec * marginal
            rc_fraction = rc_raw / float(port_var_daily)

            marginal_dict = {}
            rc_fraction_dict = {}
            for idx, ticker in enumerate(tickers):
                marginal_dict[ticker] = float(marginal[idx])
                rc_fraction_dict[ticker] = float(rc_fraction[idx])

            result["marginal_contrib"] = marginal_dict
            result["rc_fraction"] = rc_fraction_dict

            if rc_fraction_dict:
                max_ticker = max(rc_fraction_dict, key=lambda k: rc_fraction_dict[k])
                result["max_rc_ticker"] = str(max_ticker)
                result["max_rc_fraction"] = float(rc_fraction_dict[max_ticker])

            return result
        except Exception as e:
            return {"error": str(e)}

    def compute_cov_risk_diagnostics(self, weights: dict, tickers: list | None = None) -> dict:
        """Compute covariance diagnostics without altering any trading decisions."""
        try:
            cfg = self._get_risk_model_cfg()
            if not cfg.get("enable_cov_diagnostics", True):
                return {"enabled": False}

            if not isinstance(weights, dict):
                weights = {}

            if tickers is None:
                raw_universe = list(weights.keys())
            else:
                raw_universe = list(tickers)

            filtered_tickers = []
            seen = set()
            for raw in raw_universe:
                t = str(raw).upper().strip()
                if not t or t == "CASH" or t in seen:
                    continue
                w = float(weights.get(t, 0.0) or 0.0)
                if abs(w) <= 1e-12:
                    continue
                seen.add(t)
                filtered_tickers.append(t)

            returns_df, returns_meta = self.build_returns_matrix(
                filtered_tickers,
                int(cfg.get("returns_lookback_days", 60)),
                period=str(cfg.get("returns_period", "6mo")),
                interval=str(cfg.get("returns_interval", "1d")),
                min_obs=int(cfg.get("min_obs", 30)),
                drop_threshold=float(cfg.get("drop_threshold", 0.5)),
            )

            if returns_df.empty or returns_df.shape[1] < 1:
                return {
                    "enabled": True,
                    "status": "no_data",
                    "returns_meta": returns_meta,
                }

            cov, cov_meta = self._estimate_covariance_diag_shrink(
                returns_df,
                float(cfg.get("shrinkage_alpha", 0.15)),
            )

            if cov.empty:
                if bool(cfg.get("fallback_to_diag_on_error", True)):
                    try:
                        var_series = returns_df.var()
                        cov = pd.DataFrame(
                            np.diag(var_series.values.astype(float)),
                            index=var_series.index,
                            columns=var_series.index,
                        )
                        cov_meta = {
                            "method": "diag_fallback",
                            "alpha": 1.0,
                            "rows": int(returns_df.shape[0]),
                            "cols": int(returns_df.shape[1]),
                            "fallback": True,
                        }
                    except Exception as e:
                        return {
                            "enabled": True,
                            "status": "cov_failed",
                            "returns_meta": returns_meta,
                            "cov_meta": {"error": str(e)},
                        }
                else:
                    return {
                        "enabled": True,
                        "status": "cov_failed",
                        "returns_meta": returns_meta,
                        "cov_meta": cov_meta,
                    }

            rc_info = self._compute_portfolio_vol_and_rc(
                cov,
                weights,
                int(cfg.get("annualization_factor", 252)),
            )

            avg_corr = 0.0
            top_corr_pairs = []
            try:
                corr = returns_df.corr()
                corr_values = []
                pair_rows = []
                cols = list(corr.columns)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        cval = corr.iloc[i, j]
                        if pd.isna(cval):
                            continue
                        cval = float(cval)
                        corr_values.append(cval)
                        pair_rows.append({
                            "a": str(cols[i]),
                            "b": str(cols[j]),
                            "corr": cval,
                        })

                if corr_values:
                    avg_corr = float(np.mean(corr_values))
                if pair_rows:
                    k = int(cfg.get("max_pair_corr_pairs", 3))
                    k = max(0, k)
                    pair_rows = sorted(pair_rows, key=lambda x: abs(float(x["corr"])), reverse=True)
                    top_corr_pairs = pair_rows[:k]
            except Exception:
                avg_corr = 0.0
                top_corr_pairs = []

            result = {
                "enabled": True,
                "status": "ok",
                "method": cov_meta.get("method") if isinstance(cov_meta, dict) else None,
                "returns_meta": returns_meta,
                "cov_meta": cov_meta,
                "portfolio_vol_annualized": float(rc_info.get("portfolio_vol", 0.0)) if isinstance(rc_info, dict) else 0.0,
                "max_rc_fraction": float(rc_info.get("max_rc_fraction", 0.0)) if isinstance(rc_info, dict) else 0.0,
                "max_rc_ticker": rc_info.get("max_rc_ticker") if isinstance(rc_info, dict) else None,
                "rc_fraction": rc_info.get("rc_fraction", {}) if isinstance(rc_info, dict) else {},
                "avg_pairwise_corr": float(avg_corr),
                "top_corr_pairs": top_corr_pairs,
            }

            if bool(cfg.get("debug_log", False)):
                print(
                    f"[CovRisk] vol={result.get('portfolio_vol_annualized', 0.0):.4f}, "
                    f"max_rc={result.get('max_rc_fraction', 0.0):.4f} "
                    f"ticker={result.get('max_rc_ticker')}, "
                    f"cols={int(returns_df.shape[1])}, rows={int(returns_df.shape[0])}"
                )

            return result
        except Exception as e:
            return {
                "enabled": True,
                "status": "error",
                "error": str(e),
            }

    def apply_vol_targeting_to_targets(self, target_weights: dict, *, reason_tag: str = "vol_targeting") -> tuple[dict, dict]:
        """Scale non-cash target weights when covariance vol exceeds configured target."""
        original = dict(target_weights or {})
        cfg = self._get_risk_model_cfg()
        vt_meta = {
            "enabled": bool(cfg.get("enable_vol_targeting", False)),
            "status": "disabled",
            "reason_tag": reason_tag,
            "vol_before": None,
            "vol_target": float(cfg.get("vol_target", 0.18)),
            "scale": 1.0,
            "coverage": None,
            "cov_status": None,
            "method": None,
            "non_cash_sum_before": 0.0,
            "non_cash_sum_after": 0.0,
            "cash_before": 0.0,
            "cash_after": 0.0,
            "negative_clipped_count": 0,
        }

        try:
            if not bool(cfg.get("enable_vol_targeting", False)):
                return dict(original), vt_meta

            normalized = {}
            negative_clipped = 0
            for raw_ticker, raw_weight in original.items():
                ticker = str(raw_ticker).upper().strip()
                if not ticker:
                    continue
                try:
                    w = float(raw_weight) if raw_weight is not None else 0.0
                except Exception:
                    w = 0.0
                if not np.isfinite(w):
                    w = 0.0
                if w < 0:
                    w = 0.0
                    negative_clipped += 1
                normalized[ticker] = float(normalized.get(ticker, 0.0) + w)

            cash_w = float(normalized.get("CASH", 0.0))
            non_cash = {
                t: float(w)
                for t, w in normalized.items()
                if t != "CASH" and float(w) > 1e-12
            }
            vt_meta["negative_clipped_count"] = int(negative_clipped)
            vt_meta["cash_before"] = float(cash_w)
            vt_meta["non_cash_sum_before"] = float(sum(non_cash.values()))

            if not non_cash:
                vt_meta["status"] = "no_risky_assets"
                vt_meta["cash_after"] = float(cash_w)
                return dict(normalized), vt_meta

            cycle_id = int(self.current_cycle)
            weight_signature = tuple(
                sorted(
                    (str(k), round(float(v), 12))
                    for k, v in non_cash.items()
                    if np.isfinite(float(v))
                )
            )
            cache_key = ("vol_targeting_diag", cycle_id, weight_signature)
            cov_diag = None
            cached_entry = self.returns_cache.get(cache_key)
            if isinstance(cached_entry, dict):
                cached_meta = cached_entry.get("meta", {})
                if isinstance(cached_meta, dict):
                    maybe_result = cached_meta.get("result")
                    if isinstance(maybe_result, dict):
                        cov_diag = maybe_result
            if cov_diag is None:
                cov_diag = self.compute_cov_risk_diagnostics(non_cash, tickers=list(non_cash.keys()))
                self.returns_cache[cache_key] = {
                    "ts": datetime.now(),
                    "returns": pd.DataFrame(),
                    "meta": {"kind": "vol_targeting_diag", "cycle": cycle_id, "result": cov_diag},
                }
                for existing_key in list(self.returns_cache.keys()):
                    if existing_key == cache_key:
                        continue
                    if isinstance(existing_key, tuple) and len(existing_key) > 0 and existing_key[0] == "vol_targeting_diag":
                        self.returns_cache.pop(existing_key, None)

            if not isinstance(cov_diag, dict):
                cov_diag = {"status": "error", "error": "invalid_cov_diag"}

            cov_status = str(cov_diag.get("status", "error"))
            vt_meta["cov_status"] = cov_status
            vt_meta["method"] = cov_diag.get("method")

            returns_meta = cov_diag.get("returns_meta", {})
            if not isinstance(returns_meta, dict):
                returns_meta = {}
            coverage = float(returns_meta.get("overall_row_coverage", 0.0) or 0.0)
            vt_meta["coverage"] = float(coverage)
            min_cov = float(cfg.get("vol_target_min_coverage", 0.6))

            use_cov_only = bool(cfg.get("vol_target_use_cov_only", True))
            if cov_status.lower() != "ok" or coverage < min_cov:
                vt_meta["status"] = "cov_unavailable"
                if not use_cov_only:
                    vt_meta["status"] = "cov_unavailable_no_scale"
                vt_meta["cash_after"] = float(cash_w)
                vt_meta["non_cash_sum_after"] = float(sum(non_cash.values()))
                return dict(normalized), vt_meta

            vol_before = cov_diag.get("portfolio_vol_annualized")
            try:
                vol_before = float(vol_before)
            except Exception:
                vol_before = 0.0
            if not np.isfinite(vol_before) or vol_before <= 0:
                vt_meta["status"] = "bad_vol"
                vt_meta["vol_before"] = None
                vt_meta["cash_after"] = float(cash_w)
                vt_meta["non_cash_sum_after"] = float(sum(non_cash.values()))
                return dict(normalized), vt_meta

            vt_meta["vol_before"] = float(vol_before)
            vol_target = float(cfg.get("vol_target", 0.18))
            raw_scale = vol_target / vol_before
            scale = float(np.clip(raw_scale, float(cfg.get("vol_target_min_scale", 0.10)), float(cfg.get("vol_target_max_scale", 1.00))))
            vt_meta["scale"] = float(scale)

            if scale >= 0.999999:
                vt_meta["status"] = "already_below_target"
                vt_meta["scale"] = 1.0
                vt_meta["cash_after"] = float(cash_w)
                vt_meta["non_cash_sum_after"] = float(sum(non_cash.values()))
                return dict(normalized), vt_meta

            scaled_non_cash = {t: max(0.0, float(w) * float(scale)) for t, w in non_cash.items()}
            old_non_cash_sum = float(sum(non_cash.values()))
            new_non_cash_sum = float(sum(scaled_non_cash.values()))
            delta = old_non_cash_sum - new_non_cash_sum
            new_cash = float(cash_w + delta)

            if new_cash < 0:
                new_cash = 0.0
                risky_total = float(sum(scaled_non_cash.values()))
                if risky_total > 0:
                    scaled_non_cash = {t: float(max(0.0, w) / risky_total) for t, w in scaled_non_cash.items()}
                else:
                    vt_meta["status"] = "degenerate_total"
                    return {"CASH": 1.0}, vt_meta

            total = float(new_cash + sum(scaled_non_cash.values()))
            if total <= 0:
                vt_meta["status"] = "degenerate_total"
                return {"CASH": 1.0}, vt_meta

            final_weights = {t: float(max(0.0, w) / total) for t, w in scaled_non_cash.items()}
            final_weights["CASH"] = float(max(0.0, new_cash) / total)
            final_sum = float(sum(final_weights.values()))
            if final_sum > 0:
                final_weights = {k: float(max(0.0, v) / final_sum) for k, v in final_weights.items()}
            else:
                final_weights = {"CASH": 1.0}

            vt_meta["status"] = "scaled"
            vt_meta["non_cash_sum_after"] = float(sum(v for k, v in final_weights.items() if k != "CASH"))
            vt_meta["cash_after"] = float(final_weights.get("CASH", 0.0))
            return final_weights, vt_meta
        except Exception as e:
            vt_meta["status"] = "error"
            vt_meta["error"] = str(e)
            return dict(original), vt_meta
    
    def calculate_momentum(self, ticker, lookback_days=20):
        """def calculate_momentum: docstring omitted (was garbled/non-ASCII)."""
        try:
            hist = self.get_market_data(ticker, period='3mo', interval='1d')
            if hist is None or len(hist) < lookback_days:
                return 0.0
            
            recent_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[-lookback_days]) / hist['Close'].iloc[-lookback_days]
            return float(recent_return)
        except:
            return 0.0
    
    def calculate_volatility(self, ticker, lookback_days=20):
        """def calculate_volatility: docstring omitted (was garbled/non-ASCII)."""
        try:
            hist = self.get_market_data(ticker, period='3mo', interval='1d')
            if hist is None or len(hist) < lookback_days:
                return 0.20
            
            returns = hist['Close'].pct_change().dropna()
            vol = float(returns.tail(lookback_days).std() * np.sqrt(252))
            return vol
        except:
            return 0.20

    def _build_industry_lookup(self):
        """Build ticker->industry mapping from config.industry_map."""
        raw_map = self.config.get('industry_map', {})
        lookup = {}
        if not isinstance(raw_map, dict):
            return lookup

        if raw_map and all(isinstance(v, str) for v in raw_map.values()):
            for ticker, industry in raw_map.items():
                t = str(ticker).strip().upper()
                if t:
                    lookup[t] = str(industry)
            return lookup

        for industry, tickers in raw_map.items():
            if not isinstance(tickers, (list, tuple, set)):
                continue
            industry_name = str(industry)
            for ticker in tickers:
                t = str(ticker).strip().upper()
                if t:
                    lookup[t] = industry_name
        return lookup

    def _get_news_overlay_cfg(self):
        defaults = {
            'enabled': False,
            'industry_collection': 'industry_signals',
            'max_age_hours': 48.0,
            'alpha': 0.08,
            'mode': 'risk_only',
            'min_confidence': 0.55,
            'max_abs_delta': 0.10,
            'enable_confidence_scaling': True,
        }
        raw = self.config.get('news_overlay', {}) if isinstance(self.config, dict) else {}
        cfg = dict(defaults)
        if isinstance(raw, dict):
            cfg.update(raw)
        try:
            cfg['enabled'] = bool(cfg.get('enabled', False))
            cfg['industry_collection'] = str(cfg.get('industry_collection', 'industry_signals'))
            cfg['max_age_hours'] = max(0.0, float(cfg.get('max_age_hours', 48.0)))
            cfg['alpha'] = float(np.clip(float(cfg.get('alpha', 0.08)), 0.0, 1.0))
            cfg['mode'] = str(cfg.get('mode', 'risk_only')).lower()
            if cfg['mode'] not in ('risk_only', 'symmetric'):
                cfg['mode'] = 'risk_only'
            cfg['min_confidence'] = float(np.clip(float(cfg.get('min_confidence', 0.55)), 0.0, 1.0))
            cfg['max_abs_delta'] = float(np.clip(float(cfg.get('max_abs_delta', 0.10)), 0.0, 1.0))
            cfg['enable_confidence_scaling'] = bool(cfg.get('enable_confidence_scaling', True))
        except Exception:
            cfg = dict(defaults)
        return cfg

    def _build_ticker_tags_lookup(self):
        tag_lookup = {}
        raw_tags = self.config.get('ticker_tags', {}) if isinstance(self.config, dict) else {}
        if isinstance(raw_tags, dict):
            for ticker, row in raw_tags.items():
                t = str(ticker).strip().upper()
                if not t:
                    continue
                if not isinstance(row, dict):
                    continue
                l2_tags = sorted({str(x).strip() for x in row.get('L2', []) if str(x).strip()})
                l3_tags = sorted({str(x).strip() for x in row.get('L3', []) if str(x).strip()})
                keywords = sorted({str(x).strip().lower() for x in row.get('keywords', []) if str(x).strip()})
                tag_lookup[t] = {
                    'L2': l2_tags,
                    'L3': l3_tags,
                    'keywords': keywords,
                }

        fallback_industry = self._build_industry_lookup()
        for ticker, industry_name in fallback_industry.items():
            t = str(ticker).strip().upper()
            if not t:
                continue
            if t not in tag_lookup:
                tag_lookup[t] = {'L2': [str(industry_name)], 'L3': [], 'keywords': []}
            else:
                l2_existing = set(tag_lookup[t].get('L2', []))
                if str(industry_name) not in l2_existing:
                    tag_lookup[t]['L2'] = sorted(list(l2_existing | {str(industry_name)}))
        return tag_lookup

    def _parse_iso_datetime(self, value):
        if value is None:
            return None
        try:
            text = str(value).replace('Z', '+00:00')
            dt_obj = datetime.fromisoformat(text)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj.astimezone(timezone.utc)
        except Exception:
            return None

    def _get_industry_signals_collection(self, collection_name, chroma_path):
        if not CHROMADB_AVAILABLE:
            return None
        cache_key = (str(chroma_path), str(collection_name))
        if cache_key in self._industry_collection_cache:
            return self._industry_collection_cache[cache_key]
        try:
            if self._industry_chroma_client is None or getattr(self._industry_chroma_client, "_path_cache", None) != str(chroma_path):
                client = chromadb.PersistentClient(path=str(chroma_path))
                setattr(client, "_path_cache", str(chroma_path))
                self._industry_chroma_client = client
            coll = self._industry_chroma_client.get_or_create_collection(name=str(collection_name))
            self._industry_collection_cache[cache_key] = coll
            return coll
        except Exception as e:
            print(f"[NEWS_OVERLAY] WARN collection init failed: {e}")
            return None

    def _read_recent_industry_signals(self):
        cfg = self._get_news_overlay_cfg()
        if not bool(cfg.get('enabled', False)):
            return []
        if not CHROMADB_AVAILABLE:
            return []

        chroma_path = self.config.get('macro_integration', {}).get('chroma_path', './memory_db')
        collection_name = cfg.get('industry_collection', 'industry_signals')
        collection = self._get_industry_signals_collection(collection_name=collection_name, chroma_path=chroma_path)
        if collection is None:
            return []

        include = ['metadatas', 'documents']
        try:
            results = collection.get(include=include)
        except Exception as e:
            print(f"[NEWS_OVERLAY] WARN collection read failed: {e}")
            return []

        metadata_rows = results.get('metadatas', []) if isinstance(results, dict) else []
        document_rows = results.get('documents', []) if isinstance(results, dict) else []
        now_utc = datetime.now(timezone.utc)
        latest_by_l2 = {}

        for idx, meta in enumerate(metadata_rows):
            if not isinstance(meta, dict):
                continue
            if str(meta.get('scope', '')).lower() != 'industry':
                continue
            ts = self._parse_iso_datetime(meta.get('timestamp'))
            if ts is None:
                continue

            l2 = str(meta.get('L2', '')).strip()
            if not l2:
                continue

            confidence = float(meta.get('confidence', 0.0) or 0.0)
            risk_delta = float(meta.get('risk_delta', 0.0) or 0.0)
            age_hours = max(0.0, (now_utc - ts).total_seconds() / 3600.0)
            doc = document_rows[idx] if idx < len(document_rows) else None
            doc_text = str(doc) if isinstance(doc, str) else ''
            payload = None
            if isinstance(doc, str) and doc.strip():
                try:
                    payload = json.loads(doc)
                except Exception:
                    payload = None
            raw_fields_snapshot = {
                'metadata_direction': str(meta.get('direction', '') or ''),
                'metadata_risk_delta': float(meta.get('risk_delta', 0.0) or 0.0),
                'metadata_confidence': float(meta.get('confidence', 0.0) or 0.0),
                'metadata_horizon': str(meta.get('horizon', '') or ''),
                'doc_direction': str((payload or {}).get('direction', '') if isinstance(payload, dict) else ''),
                'doc_risk_delta': float((payload or {}).get('risk_delta', 0.0) or 0.0) if isinstance(payload, dict) else 0.0,
                'doc_confidence': float((payload or {}).get('confidence', 0.0) or 0.0) if isinstance(payload, dict) else 0.0,
                'doc_horizon': str((payload or {}).get('horizon', '') if isinstance(payload, dict) else ''),
            }

            row = {
                'L2': l2,
                'timestamp': ts.isoformat(),
                'confidence': confidence,
                'risk_delta': risk_delta,
                'horizon': str(meta.get('horizon', '1d')),
                'age_hours': age_hours,
                'payload': payload if isinstance(payload, dict) else {},
                'raw_metadata': dict(meta),
                'raw_doc_head_600': doc_text[:600],
                'raw_fields_snapshot': raw_fields_snapshot,
            }
            prev = latest_by_l2.get(l2)
            if prev is None:
                latest_by_l2[l2] = row
            else:
                prev_ts = self._parse_iso_datetime(prev.get('timestamp'))
                if prev_ts is None or ts > prev_ts:
                    latest_by_l2[l2] = row

        return list(latest_by_l2.values())

    def apply_news_overlay_to_cash_target(self, tickers, cash_target):
        cfg = self._get_news_overlay_cfg()
        info = {
            'enabled': bool(cfg.get('enabled', False)),
            'status': 'disabled',
            'mode': cfg.get('mode', 'risk_only'),
            'alpha': float(cfg.get('alpha', 0.08)),
            'enable_confidence_scaling': bool(cfg.get('enable_confidence_scaling', True)),
            'applied_cash_delta': 0.0,
            'worst_l2': None,
            'worst_delta': 0.0,
            'used_signals': 0,
            'ticker_deltas': {},
            'l2_deltas': {},
            'max_abs_delta': float(cfg.get('max_abs_delta', 0.10)),
            'min_confidence': float(cfg.get('min_confidence', 0.55)),
            'max_age_hours': float(cfg.get('max_age_hours', 48.0)),
            'included_rows_count': 0,
            'excluded_rows_count': 0,
            'excluded_by_confidence_count': 0,
            'excluded_by_age_count': 0,
            'excluded_by_confidence_sample': [],
            'l2_delta_map_sample': [],
            'chosen_cash_delta_source': None,
            'included_rows_audit': [],
            'excluded_rows_audit': [],
        }
        base_cash = float(cash_target)
        if not info['enabled']:
            return base_cash, info

        try:
            signal_rows = self._read_recent_industry_signals()
            if not signal_rows:
                info['status'] = 'no_data'
                return base_cash, info

            alpha = float(cfg.get('alpha', 0.08))
            max_abs_delta = float(cfg.get('max_abs_delta', 0.10))
            min_confidence = float(cfg.get('min_confidence', 0.55))
            max_age_hours = float(cfg.get('max_age_hours', 48.0))
            mode = str(cfg.get('mode', 'risk_only')).lower()
            enable_confidence_scaling = bool(cfg.get('enable_confidence_scaling', True))
            l2_delta_map = {}
            l2_diag_map = {}
            excluded_conf_sample = []
            excluded_conf_count = 0
            excluded_age_count = 0
            included_count = 0
            for row in signal_rows:
                l2 = str(row.get('L2', '')).strip()
                if not l2:
                    continue
                confidence = float(row.get('confidence', 0.0) or 0.0)
                age_hours = float(row.get('age_hours', 0.0) or 0.0)
                risk_delta = float(row.get('risk_delta', 0.0) or 0.0)
                raw_doc_head_600 = str(row.get('raw_doc_head_600', '') or '')[:600]
                raw_metadata = row.get('raw_metadata', {}) if isinstance(row.get('raw_metadata', {}), dict) else {}
                raw_fields_snapshot = row.get('raw_fields_snapshot', {}) if isinstance(row.get('raw_fields_snapshot', {}), dict) else {}
                transform_chain = "industry_signal:risk_delta -> raw=rd*alpha -> (optional)*confidence -> clip(max_abs_delta) -> risk_only_cap"
                if age_hours > max_age_hours:
                    excluded_age_count += 1
                    if len(info['excluded_rows_audit']) < 3:
                        info['excluded_rows_audit'].append(
                            {
                                'L2': l2,
                                'exclude_reason': 'age',
                                'age_hours': float(age_hours),
                                'risk_delta_used_in_overlay': float(risk_delta),
                                'confidence_used_in_overlay': float(confidence),
                                'transform_chain': transform_chain,
                                'raw_doc_head_600': raw_doc_head_600,
                                'raw_metadata': {
                                    'timestamp': raw_metadata.get('timestamp'),
                                    'status': raw_metadata.get('status'),
                                    'source_count': raw_metadata.get('source_count'),
                                    'scope': raw_metadata.get('scope'),
                                    'L2': raw_metadata.get('L2'),
                                },
                                'raw_fields_snapshot': raw_fields_snapshot,
                            }
                        )
                    continue
                if confidence < min_confidence:
                    excluded_conf_count += 1
                    if len(excluded_conf_sample) < 5:
                        excluded_conf_sample.append(
                            {
                                'L2': l2,
                                'confidence': float(confidence),
                                'risk_delta': float(risk_delta),
                            }
                        )
                    if len(info['excluded_rows_audit']) < 3:
                        info['excluded_rows_audit'].append(
                            {
                                'L2': l2,
                                'exclude_reason': 'min_confidence',
                                'age_hours': float(age_hours),
                                'risk_delta_used_in_overlay': float(risk_delta),
                                'confidence_used_in_overlay': float(confidence),
                                'transform_chain': transform_chain,
                                'raw_doc_head_600': raw_doc_head_600,
                                'raw_metadata': {
                                    'timestamp': raw_metadata.get('timestamp'),
                                    'status': raw_metadata.get('status'),
                                    'source_count': raw_metadata.get('source_count'),
                                    'scope': raw_metadata.get('scope'),
                                    'L2': raw_metadata.get('L2'),
                                },
                                'raw_fields_snapshot': raw_fields_snapshot,
                            }
                        )
                    continue
                raw_delta = float(risk_delta * alpha)
                if enable_confidence_scaling:
                    raw_delta = float(raw_delta * confidence)
                delta = float(np.clip(raw_delta, -max_abs_delta, max_abs_delta))
                if mode == 'risk_only' and delta > 0:
                    delta = 0.0
                l2_delta_map[l2] = delta
                l2_diag_map[l2] = {
                    'L2': l2,
                    'risk_delta': float(risk_delta),
                    'confidence': float(confidence),
                    'delta': float(delta),
                }
                if len(info['included_rows_audit']) < 3:
                    info['included_rows_audit'].append(
                        {
                            'L2': l2,
                            'age_hours': float(age_hours),
                            'risk_delta_used_in_overlay': float(risk_delta),
                            'confidence_used_in_overlay': float(confidence),
                            'raw_delta_before_clip': float(raw_delta),
                            'delta_after_clip_and_mode': float(delta),
                            'transform_chain': transform_chain,
                            'raw_doc_head_600': raw_doc_head_600,
                            'raw_metadata': {
                                'timestamp': raw_metadata.get('timestamp'),
                                'status': raw_metadata.get('status'),
                                'source_count': raw_metadata.get('source_count'),
                                'scope': raw_metadata.get('scope'),
                                'L2': raw_metadata.get('L2'),
                            },
                            'raw_fields_snapshot': raw_fields_snapshot,
                        }
                    )
                included_count += 1

            info['included_rows_count'] = int(included_count)
            info['excluded_by_confidence_count'] = int(excluded_conf_count)
            info['excluded_by_age_count'] = int(excluded_age_count)
            info['excluded_rows_count'] = int(excluded_conf_count + excluded_age_count)
            info['excluded_by_confidence_sample'] = list(excluded_conf_sample)

            if not l2_delta_map:
                info['status'] = 'filtered_empty'
                return base_cash, info

            ticker_tags = self._build_ticker_tags_lookup()
            ticker_deltas = {}
            ticker_delta_source = {}
            ordered_tickers = []
            for ticker in (tickers or []):
                t = str(ticker).strip().upper()
                if t and t != 'CASH' and t not in ordered_tickers:
                    ordered_tickers.append(t)

            for ticker in ordered_tickers:
                tags = ticker_tags.get(ticker, {})
                l2_tags = tags.get('L2', []) if isinstance(tags, dict) else []
                delta_rows = []
                for l2 in l2_tags:
                    one = l2_delta_map.get(str(l2), None)
                    if one is not None:
                        delta_rows.append((str(l2), float(one)))
                if delta_rows:
                    chosen_l2, chosen_delta = min(delta_rows, key=lambda x: x[1])
                    ticker_deltas[ticker] = float(chosen_delta)
                    ticker_delta_source[ticker] = {
                        'ticker': ticker,
                        'l2': chosen_l2,
                        'delta': float(chosen_delta),
                    }

            if not ticker_deltas:
                info['status'] = 'no_ticker_match'
                info['l2_deltas'] = dict(l2_delta_map)
                info['used_signals'] = len(l2_delta_map)
                return base_cash, info

            worst_ticker = min(ticker_deltas, key=lambda x: ticker_deltas[x])
            worst_delta = float(ticker_deltas.get(worst_ticker, 0.0))
            applied_cash_delta = float(np.clip(abs(min(0.0, worst_delta)), 0.0, max_abs_delta))
            min_cash = float(self.config.get('objectives', {}).get('min_cash_pct', 0.10))
            new_cash = float(np.clip(base_cash + applied_cash_delta, min_cash, 0.60))

            worst_l2 = None
            worst_l2_delta = 0.0
            for l2, d in l2_delta_map.items():
                if d < worst_l2_delta:
                    worst_l2_delta = float(d)
                    worst_l2 = str(l2)

            info['status'] = 'applied' if applied_cash_delta > 0 else 'neutral'
            info['applied_cash_delta'] = applied_cash_delta
            info['worst_l2'] = worst_l2
            info['worst_delta'] = worst_l2_delta
            info['used_signals'] = len(l2_delta_map)
            info['l2_deltas'] = {k: float(v) for k, v in list(l2_delta_map.items())[:20]}
            info['ticker_deltas'] = {k: float(v) for k, v in list(ticker_deltas.items())[:30]}
            info['l2_delta_map_sample'] = [dict(v) for v in list(l2_diag_map.values())[:5]]
            info['cash_target_before'] = base_cash
            info['cash_target_after'] = new_cash
            info['worst_ticker'] = worst_ticker
            info['chosen_cash_delta_source'] = ticker_delta_source.get(worst_ticker)

            chosen_src = info.get('chosen_cash_delta_source') if isinstance(info.get('chosen_cash_delta_source'), dict) else {}
            src_ticker = str(chosen_src.get('ticker', info.get('worst_ticker', 'NA')))
            src_l2 = str(chosen_src.get('l2', info.get('worst_l2', 'NA')))
            src_delta = float(chosen_src.get('delta', info.get('worst_delta', 0.0)) or 0.0)
            print(
                f"[NEWS_OVERLAY] inc={int(info.get('included_rows_count', 0))} "
                f"exc_conf={int(info.get('excluded_by_confidence_count', 0))} "
                f"exc_age={int(info.get('excluded_by_age_count', 0))} "
                f"cash={float(base_cash):.3f}->{float(new_cash):.3f} "
                f"src={src_ticker} l2={src_l2} delta={src_delta:+.3f} "
                f"cap={float(max_abs_delta):.2f}"
            )
            return new_cash, info
        except Exception as e:
            info['status'] = 'error'
            info['error'] = str(e)
            print(f"[NEWS_OVERLAY] WARN apply failed: {e}")
            return base_cash, info

    def calculate_volume_zscore(self, ticker, lookback_days=60):
        """Calculate volume Z-score versus trailing rolling mean/std."""
        try:
            hist = self.get_market_data(ticker, period='6mo', interval='1d')
            if hist is None or hist.empty or 'Volume' not in hist:
                return 0.0

            volume_series = hist['Volume'].dropna().astype(float)
            if len(volume_series) < max(20, int(lookback_days)):
                return 0.0

            trailing = volume_series.tail(int(lookback_days))
            vol_mean = float(trailing.mean())
            vol_std = float(trailing.std())
            if vol_std <= 1e-12:
                return 0.0

            latest_volume = float(volume_series.iloc[-1])
            return float((latest_volume - vol_mean) / vol_std)
        except Exception:
            return 0.0
    
    def _sync_current_macro_from_cache(self):
        """def _sync_current_macro_from_cache: docstring omitted (was garbled/non-ASCII)."""
        macro_risk_score_raw = self.cached_macro.get('macro_risk_score_raw', 0.0)
        self.current_macro = {
            'macro_risk_score': macro_risk_score_raw,
            'macro_risk_score_smoothed': self.cached_macro.get('macro_risk_score_smoothed', macro_risk_score_raw),
            'confirmed_topics': self.cached_macro.get('confirmed_topics', []),
            'macro_tilts': self.cached_macro.get('macro_tilts', {}),
            'macro_tilts_ignored': self.cached_macro.get('macro_tilts_ignored', {}),
            'signal_summary': self.cached_macro.get('signal_summary', {}),
            'applied_tilts': self.cached_macro.get('applied_tilts', self.current_macro.get('applied_tilts', {}) if isinstance(self.current_macro, dict) else {}),
            'blocked_tilts': self.cached_macro.get('blocked_tilts', self.current_macro.get('blocked_tilts', {}) if isinstance(self.current_macro, dict) else {}),
            'capped_assets': self.cached_macro.get('capped_assets', self.current_macro.get('capped_assets', []) if isinstance(self.current_macro, dict) else []),
            'max_weight_per_asset_effective': self.cached_macro.get(
                'max_weight_per_asset_effective',
                self.current_macro.get('max_weight_per_asset_effective', {}) if isinstance(self.current_macro, dict) else {}
            ),
            'allocation_diagnostics': self.cached_macro.get(
                'allocation_diagnostics',
                self.current_macro.get('allocation_diagnostics', {}) if isinstance(self.current_macro, dict) else {}
            )
        }

    def refresh_macro_cache(self, now=None):
        """def refresh_macro_cache: docstring omitted (was garbled/non-ASCII)."""
        if now is None:
            now = datetime.now()

        macro_risk_score_raw, confirmed_topics, macro_tilts_raw, signal_summary = self.macro_adapter.analyze_signals()

        # NOTE: comment omitted (was garbled/non-ASCII).
        self.macro_risk_score_history.append(macro_risk_score_raw)
        if len(self.macro_risk_score_history) > self.macro_smoothing_window:
            self.macro_risk_score_history = self.macro_risk_score_history[-self.macro_smoothing_window:]

        if self.macro_smoothing_method == 'median':
            import statistics
            macro_risk_score_smoothed = statistics.median(self.macro_risk_score_history)
        elif self.macro_smoothing_method == 'ewma':
            if len(self.macro_risk_score_history) == 1:
                macro_risk_score_smoothed = macro_risk_score_raw
            else:
                prev_smoothed = self.cached_macro.get('macro_risk_score_smoothed', macro_risk_score_raw)
                macro_risk_score_smoothed = (
                    self.macro_ewma_alpha * macro_risk_score_raw +
                    (1 - self.macro_ewma_alpha) * prev_smoothed
                )
        else:
            macro_risk_score_smoothed = macro_risk_score_raw

        print(f"[MACRO SMOOTH] Raw: {macro_risk_score_raw:.2f}, Smoothed: {macro_risk_score_smoothed:.2f} "
              f"(method: {self.macro_smoothing_method}, window: {len(self.macro_risk_score_history)})")

        # NOTE: comment omitted (was garbled/non-ASCII).
        universe_tickers = {asset['ticker'] for asset in self.config['universe']}
        macro_tilts_filtered = {}
        macro_tilts_ignored = {}

        for ticker, tilt in macro_tilts_raw.items():
            if ticker in universe_tickers:
                macro_tilts_filtered[ticker] = tilt
            else:
                macro_tilts_ignored[ticker] = tilt

        if macro_tilts_ignored:
            print(f"[MACRO TILTS] Ignored (not in universe): {', '.join([f'{t}:{v:+.2%}' for t, v in macro_tilts_ignored.items()])}")

        self.cached_macro = {
            'macro_risk_score_raw': macro_risk_score_raw,
            'macro_risk_score_smoothed': macro_risk_score_smoothed,
            'confirmed_topics': confirmed_topics,
            'macro_tilts': macro_tilts_filtered,
            'macro_tilts_ignored': macro_tilts_ignored,
            'signal_summary': signal_summary
        }
        self._sync_current_macro_from_cache()
        self.last_macro_time = now


    def _compute_cross_sectional_metrics(
        self,
        trade_universe_assets,
        lookback_days,
        vol_target,
        momentum_weight,
        vol_weight,
        top_n,
        enable_short_term_momentum=False,
        short_lookback_days=10,
        momentum_short_weight=0.4,
        momentum_medium_weight=0.6
    ):
        """Compute momentum/vol metrics and cross-sectional rank score."""
        blend_weight_sum = max(1e-12, float(momentum_short_weight) + float(momentum_medium_weight))
        short_w = float(momentum_short_weight) / blend_weight_sum
        medium_w = float(momentum_medium_weight) / blend_weight_sum
        industry_lookup = self._build_industry_lookup()
        metrics = {}
        for asset in trade_universe_assets:
            ticker = str(asset.get('ticker', ''))
            if not ticker or ticker.upper() == 'CASH':
                continue

            medium_momentum = self.calculate_momentum(ticker, lookback_days)
            short_momentum = self.calculate_momentum(ticker, short_lookback_days) if enable_short_term_momentum else medium_momentum
            blended_momentum = (medium_w * medium_momentum) + (short_w * short_momentum)
            volatility = self.calculate_volatility(ticker, lookback_days)
            volume_z = self.calculate_volume_zscore(ticker, lookback_days=60)
            base_score = momentum_weight * blended_momentum - vol_weight * (volatility - vol_target)
            industry_name = industry_lookup.get(str(ticker).upper(), "UNCLASSIFIED")
            metrics[ticker] = {
                'momentum': float(blended_momentum),
                'medium_momentum': float(medium_momentum),
                'short_momentum': float(short_momentum),
                'volatility': float(max(volatility, 1e-6)),
                'volume_z': float(volume_z),
                'industry': industry_name,
                'industry_strength': 0.0,
                'volatility_score': 0.0,
                'base_score': float(base_score),
                'rank_score': 0.0,
                'momentum_rank_pct': 0.0,
                'medium_term_z': 0.0,
                'short_term_z': 0.0,
                'blended_momentum_z': 0.0
            }

        if not metrics:
            return {}, []

        medium_values = np.array([v['medium_momentum'] for v in metrics.values()], dtype=float)
        short_values = np.array([v['short_momentum'] for v in metrics.values()], dtype=float)
        med_mu = float(np.mean(medium_values))
        med_sigma = float(np.std(medium_values))
        short_mu = float(np.mean(short_values))
        short_sigma = float(np.std(short_values))
        vol_values = np.array([v['volatility'] for v in metrics.values()], dtype=float)
        vol_mu = float(np.mean(vol_values))
        vol_sigma = float(np.std(vol_values))

        for ticker, data in metrics.items():
            medium_z = (data['medium_momentum'] - med_mu) / med_sigma if med_sigma > 1e-12 else 0.0
            short_z = (data['short_momentum'] - short_mu) / short_sigma if short_sigma > 1e-12 else 0.0
            blended_z = (medium_w * medium_z) + (short_w * short_z)
            if not enable_short_term_momentum:
                short_z = medium_z
                blended_z = medium_z
            data['medium_term_z'] = float(medium_z)
            data['short_term_z'] = float(short_z)
            data['blended_momentum_z'] = float(blended_z)
            data['volatility_score'] = float((vol_mu - data['volatility']) / vol_sigma) if vol_sigma > 1e-12 else 0.0
            if enable_short_term_momentum:
                print(f"[MOMENTUM] {ticker}: short={short_z:+.2f}, med={medium_z:+.2f}, blended={blended_z:+.2f}")

        industry_grouped = {}
        for ticker, data in metrics.items():
            industry_name = data.get('industry', 'UNCLASSIFIED')
            industry_grouped.setdefault(industry_name, []).append(float(data.get('blended_momentum_z', 0.0)))

        for ticker, data in metrics.items():
            industry_name = data.get('industry', 'UNCLASSIFIED')
            peers = industry_grouped.get(industry_name, [])
            industry_strength = float(np.mean(peers)) if peers else 0.0
            momentum_z = float(data.get('blended_momentum_z', 0.0))
            volatility_score = float(data.get('volatility_score', 0.0))
            volume_z = float(data.get('volume_z', 0.0))
            final_score = (
                0.5 * momentum_z +
                0.3 * volatility_score +
                0.2 * industry_strength +
                0.2 * volume_z
            )
            data['industry_strength'] = industry_strength
            data['final_score_raw'] = float(final_score)
            data['rank_score'] = float(final_score)
            data['base_score'] = float(final_score)

        metrics = self._apply_score_stability_controls(metrics)
        ranked_for_pct = sorted(metrics.items(), key=lambda x: x[1]['rank_score'], reverse=True)
        n = len(ranked_for_pct)
        for rank_idx, (ticker, data) in enumerate(ranked_for_pct, start=1):
            data['momentum_rank_pct'] = float((n - rank_idx + 1) / n)

        ranked_tickers = sorted(
            metrics.keys(),
            key=lambda t: (metrics[t]['rank_score'], metrics[t]['base_score'], metrics[t]['momentum']),
            reverse=True
        )
        top_n = max(1, min(int(top_n), len(ranked_tickers)))
        return metrics, ranked_tickers[:top_n]

    def _get_returns_for_correlation(self, ticker, lookback_days):
        """Fetch trailing daily return series for correlation checks."""
        hist = self.get_market_data(ticker, period='6mo', interval='1d')
        if hist is None or hist.empty or 'Close' not in hist:
            return None
        returns = hist['Close'].pct_change().dropna()
        if returns.empty:
            return None
        return returns.tail(int(max(20, lookback_days)))

    def _apply_correlation_filter(self, ranked_tickers, lookback_days, threshold):
        """Drop lower-ranked assets when pairwise corr exceeds threshold."""
        try:
            if not ranked_tickers:
                return [], [], []

            returns_map = {}
            degraded_reasons = []
            min_overlap = max(20, int(lookback_days // 2))

            for ticker in ranked_tickers:
                series = self._get_returns_for_correlation(ticker, lookback_days)
                if series is None or len(series) < min_overlap:
                    degraded_reasons.append(f"{ticker}:insufficient_history")
                    continue
                returns_map[ticker] = series

            kept = []
            decisions = []
            for ticker in ranked_tickers:
                dropped = False
                for kept_ticker in kept:
                    a = returns_map.get(ticker)
                    b = returns_map.get(kept_ticker)
                    if a is None or b is None:
                        continue

                    aligned = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner').dropna()
                    if len(aligned) < min_overlap:
                        continue

                    corr = float(aligned['a'].corr(aligned['b']))
                    if np.isnan(corr):
                        continue
                    if corr > threshold:
                        decisions.append({
                            'dropped': ticker,
                            'kept': kept_ticker,
                            'corr': corr,
                            'reason': f'corr>{threshold:.2f}'
                        })
                        dropped = True
                        break

                if not dropped:
                    kept.append(ticker)

            return kept, decisions, degraded_reasons
        except Exception as e:
            print(f"[CORR] Degraded: correlation filter failed ({e}), fallback to ranked list")
            return list(ranked_tickers), [], [f"error:{e}"]

    def _apply_score_stability_controls(self, metrics):
        """Apply optional smoothing, normalization and clipping to rank scores."""
        if not metrics:
            return metrics

        execution_cfg = self.config.get('execution', {})
        enable_smoothing = bool(execution_cfg.get('enable_score_smoothing', True))
        window = max(1, int(execution_cfg.get('score_smoothing_window', 3)))

        raw_scores = {ticker: float(data.get('rank_score', 0.0)) for ticker, data in metrics.items()}
        smoothed_scores = {}
        for ticker, raw_score in raw_scores.items():
            if enable_smoothing:
                history = self.score_history_by_ticker.setdefault(ticker, [])
                history.append(raw_score)
                if len(history) > window:
                    del history[:-window]
                smoothed_scores[ticker] = float(np.mean(history))
            else:
                smoothed_scores[ticker] = raw_score

        active_tickers = set(metrics.keys())
        for ticker in list(self.score_history_by_ticker.keys()):
            if ticker not in active_tickers and len(self.score_history_by_ticker.get(ticker, [])) == 0:
                self.score_history_by_ticker.pop(ticker, None)

        smoothed_values = np.array(list(smoothed_scores.values()), dtype=float)
        mu = float(np.mean(smoothed_values)) if len(smoothed_values) > 0 else 0.0
        sigma = float(np.std(smoothed_values)) if len(smoothed_values) > 0 else 0.0

        clipped_count = 0
        for ticker, data in metrics.items():
            raw_score = raw_scores.get(ticker, 0.0)
            smooth_score = smoothed_scores.get(ticker, raw_score)
            normalized_score = (smooth_score - mu) / sigma if sigma > 1e-12 else 0.0
            capped_score = float(np.clip(normalized_score, -3.0, 3.0))
            if abs(capped_score - normalized_score) > 1e-12:
                clipped_count += 1
            data['raw_rank_score'] = float(raw_score)
            data['smoothed_rank_score'] = float(smooth_score)
            data['normalized_rank_score'] = float(normalized_score)
            data['rank_score'] = capped_score

        print(f"[SCORE STABILITY] smoothing={enable_smoothing} window={window} mu={mu:+.3f} sigma={sigma:.3f} clipped={clipped_count}")
        return metrics

    def _get_asset_volatility_optional(self, ticker, lookback_days):
        """Return annualized volatility or None if data is insufficient."""
        try:
            hist = self.get_market_data(ticker, period='3mo', interval='1d')
            if hist is None or hist.empty or len(hist) < int(max(5, lookback_days)):
                return None
            returns = hist['Close'].pct_change().dropna()
            if len(returns) < int(max(5, lookback_days // 2)):
                return None
            return float(returns.tail(int(lookback_days)).std() * np.sqrt(252))
        except Exception:
            return None

    def _evaluate_portfolio_risk_gate(self, target_weights):
        """Check portfolio-level volatility/diversification before execution."""
        execution_cfg = self.config.get('execution', {})
        strategy_cfg = self.config.get('strategy', {})
        risk_model_cfg = self._get_risk_model_cfg()

        lookback_days = int(strategy_cfg.get('lookback_days', 40))
        max_portfolio_volatility = float(execution_cfg.get('max_portfolio_volatility', 0.25))
        min_coverage = float(execution_cfg.get('portfolio_vol_min_coverage', 0.70))
        enable_diversity_check = bool(execution_cfg.get('enable_diversity_check', True))
        max_hhi = float(execution_cfg.get('max_herfindahl_index', 0.35))

        use_cov_vol_for_gate = bool(risk_model_cfg.get('use_cov_vol_for_gate', False))
        rc_limit = float(risk_model_cfg.get('rc_limit', 0.35))
        min_cov_gate_coverage = float(risk_model_cfg.get('min_cov_gate_coverage', 0.6))
        cov_gate_fallback_to_weighted = bool(risk_model_cfg.get('cov_gate_fallback_to_weighted', True))

        invested_weights = {
            str(t).upper(): max(0.0, float(w))
            for t, w in (target_weights or {}).items()
            if str(t).upper() != 'CASH' and float(w) > 1e-12
        }

        known_weight = 0.0
        weighted_volatility = 0.0
        vol_map = {}
        for ticker, weight in invested_weights.items():
            vol = self._get_asset_volatility_optional(ticker, lookback_days)
            if vol is None:
                continue
            vol_map[ticker] = float(vol)
            known_weight += weight
            weighted_volatility += weight * vol

        invested_budget = float(sum(invested_weights.values()))
        hhi = 0.0
        if invested_budget > 1e-12:
            normalized = [w / invested_budget for w in invested_weights.values()]
            hhi = float(sum(w * w for w in normalized))

        # Build current portfolio weights for covariance diagnostics (read-only).
        current_position_weights = {}
        try:
            snapshot_used = False
            if self.portfolio_snapshots:
                last_snapshot = self.portfolio_snapshots[-1]
                snapshot_equity = float(last_snapshot.get('total_equity', 0.0) or 0.0)
                snapshot_positions = last_snapshot.get('positions', {})
                if snapshot_equity > 0 and isinstance(snapshot_positions, dict):
                    for ticker, pos in snapshot_positions.items():
                        ticker_upper = str(ticker).strip().upper()
                        if not ticker_upper or ticker_upper == 'CASH':
                            continue
                        try:
                            pos_value = float(pos.get('value', 0.0) if isinstance(pos, dict) else 0.0)
                        except Exception:
                            pos_value = 0.0
                        if pos_value > 0:
                            current_position_weights[ticker_upper] = float(pos_value / snapshot_equity)
                            snapshot_used = True

            if not snapshot_used and self.positions:
                position_values = {}
                positions_value = 0.0
                for ticker, qty in self.positions.items():
                    if qty is None:
                        continue
                    try:
                        qty_val = float(qty)
                    except Exception:
                        continue
                    if qty_val <= 0:
                        continue
                    price, _, _ = self.get_current_price(ticker)
                    if price is None:
                        continue
                    value = float(qty_val * float(price))
                    if value <= 0:
                        continue
                    ticker_upper = str(ticker).strip().upper()
                    if not ticker_upper or ticker_upper == 'CASH':
                        continue
                    position_values[ticker_upper] = float(position_values.get(ticker_upper, 0.0) + value)
                    positions_value += value

                total_equity_current = float(self.cash) + float(positions_value)
                if total_equity_current > 0:
                    for ticker, value in position_values.items():
                        current_position_weights[ticker] = float(value / total_equity_current)
        except Exception:
            current_position_weights = {}

        cov_diag = {"enabled": True, "status": "error", "error": "cov_diag_uninitialized"}
        cov_gate_coverage = None
        cov_gate_vol = None
        cov_gate_max_rc = None
        cov_gate_pass = None
        cov_gate_reason = "not_evaluated"
        cov_gate_used = False
        gate_vol_method = "weighted_fallback"

        try:
            cycle_id = int(self.current_cycle)
            weight_signature = tuple(
                sorted(
                    (str(k), round(float(v), 12))
                    for k, v in current_position_weights.items()
                    if np.isfinite(float(v))
                )
            )
            cache_key = ("cov_diag_snapshot", cycle_id, weight_signature)
            cached_result = None
            cached_entry = self.returns_cache.get(cache_key)
            if isinstance(cached_entry, dict):
                cached_meta = cached_entry.get('meta', {})
                if isinstance(cached_meta, dict):
                    maybe_result = cached_meta.get('result')
                    if isinstance(maybe_result, dict):
                        cached_result = maybe_result

            if isinstance(cached_result, dict):
                cov_diag = cached_result
            else:
                cov_diag = self.compute_cov_risk_diagnostics(current_position_weights)
                self.returns_cache[cache_key] = {
                    "ts": datetime.now(),
                    "returns": pd.DataFrame(),
                    "meta": {
                        "kind": "cov_diag",
                        "cycle": cycle_id,
                        "result": cov_diag
                    }
                }
                for existing_key in list(self.returns_cache.keys()):
                    if existing_key == cache_key:
                        continue
                    if isinstance(existing_key, tuple) and len(existing_key) > 0 and existing_key[0] == "cov_diag_snapshot":
                        self.returns_cache.pop(existing_key, None)
        except Exception as e:
            cov_diag = {"enabled": True, "status": "error", "error": str(e)}

        cov_status = str(cov_diag.get('status', '')).lower() if isinstance(cov_diag, dict) else 'error'
        returns_meta = cov_diag.get('returns_meta', {}) if isinstance(cov_diag, dict) else {}
        if not isinstance(returns_meta, dict):
            returns_meta = {}

        try:
            cov_gate_coverage = float(returns_meta.get('overall_row_coverage', 0.0))
        except Exception:
            cov_gate_coverage = 0.0

        try:
            cov_cols = int(returns_meta.get('cols', 0) or 0)
        except Exception:
            cov_cols = 0

        try:
            cov_gate_vol = float(cov_diag.get('portfolio_vol_annualized')) if cov_diag.get('portfolio_vol_annualized') is not None else None
        except Exception:
            cov_gate_vol = None

        try:
            cov_gate_max_rc = float(cov_diag.get('max_rc_fraction')) if cov_diag.get('max_rc_fraction') is not None else None
        except Exception:
            cov_gate_max_rc = None

        coverage_ok = bool(cov_status == 'ok' and cov_cols > 0 and cov_gate_coverage is not None and cov_gate_coverage >= min_cov_gate_coverage)
        vol_ok = bool(cov_gate_vol is not None and cov_gate_vol <= max_portfolio_volatility)
        if rc_limit > 0:
            rc_ok = bool(cov_gate_max_rc is not None and cov_gate_max_rc <= rc_limit)
        else:
            rc_ok = True

        abort_reason = ""
        abort_flag = False
        volatility_confident = known_weight >= min_coverage

        if use_cov_vol_for_gate:
            cov_gate_used = True
            if coverage_ok:
                gate_vol_method = "cov"
                cov_gate_pass = bool(vol_ok and rc_ok)
                if cov_gate_pass:
                    cov_gate_reason = "ok"
                else:
                    if not vol_ok and not rc_ok:
                        cov_gate_reason = "vol_and_rc_limit"
                        abort_reason = "portfolio_cov_vol_and_rc"
                    elif not vol_ok:
                        cov_gate_reason = "vol_limit"
                        abort_reason = "portfolio_cov_volatility"
                    else:
                        cov_gate_reason = "rc_limit"
                        abort_reason = "portfolio_cov_rc_limit"
                    abort_flag = True
            else:
                unavailable_reason = cov_status if cov_status else "unavailable"
                if cov_status == 'ok' and cov_cols <= 0:
                    unavailable_reason = "no_data"
                elif cov_status == 'ok' and (cov_gate_coverage is None or cov_gate_coverage < min_cov_gate_coverage):
                    unavailable_reason = "low_coverage"

                if cov_gate_fallback_to_weighted:
                    gate_vol_method = "weighted_fallback"
                    cov_gate_reason = f"fallback_to_weighted:{unavailable_reason}"
                    if volatility_confident and weighted_volatility > max_portfolio_volatility:
                        abort_flag = True
                        abort_reason = "portfolio_volatility"
                else:
                    gate_vol_method = "cov"
                    cov_gate_pass = False
                    cov_gate_reason = f"cov_unavailable:{unavailable_reason}"
                    abort_flag = True
                    abort_reason = "cov_unavailable"
        else:
            gate_vol_method = "weighted_fallback"
            cov_gate_reason = "disabled"
            if volatility_confident and weighted_volatility > max_portfolio_volatility:
                abort_flag = True
                abort_reason = "portfolio_volatility"

        if (not abort_flag) and enable_diversity_check and invested_budget > 1e-12 and hhi > max_hhi:
            abort_flag = True
            abort_reason = "diversity_hhi"

        return {
            'abort': bool(abort_flag),
            'abort_reason': abort_reason,
            'weighted_volatility': float(weighted_volatility),
            'max_portfolio_volatility': float(max_portfolio_volatility),
            'volatility_known_weight': float(known_weight),
            'volatility_confident': bool(volatility_confident),
            'min_coverage': float(min_coverage),
            'enable_diversity_check': bool(enable_diversity_check),
            'herfindahl_index': float(hhi),
            'max_herfindahl_index': float(max_hhi),
            'invested_budget': float(invested_budget),
            'asset_volatility_map': vol_map,
            'cov_risk_diag': cov_diag,
            'gate_vol_method': gate_vol_method,
            'cov_gate_used': bool(cov_gate_used),
            'cov_gate_coverage': float(cov_gate_coverage) if cov_gate_coverage is not None else None,
            'cov_gate_vol': float(cov_gate_vol) if cov_gate_vol is not None else None,
            'cov_gate_max_rc': float(cov_gate_max_rc) if cov_gate_max_rc is not None else None,
            'cov_gate_pass': cov_gate_pass,
            'cov_gate_reason': cov_gate_reason,
            'rc_limit': float(rc_limit),
            'min_cov_gate_coverage': float(min_cov_gate_coverage),
            'use_cov_vol_for_gate': bool(use_cov_vol_for_gate),
            'cov_gate_fallback_to_weighted': bool(cov_gate_fallback_to_weighted),
        }

    def _estimate_current_cash_ratio(self):
        """Estimate current portfolio cash ratio for high-conviction checks."""
        if self.portfolio_snapshots:
            last = self.portfolio_snapshots[-1]
            equity = float(last.get('total_equity', 0.0))
            cash = float(last.get('cash', self.cash))
            if equity > 1e-9:
                return float(np.clip(cash / equity, 0.0, 1.0))

        if self.equity_curve:
            _, equity, cash, _ = self.equity_curve[-1]
            equity = float(equity)
            cash = float(cash)
            if equity > 1e-9:
                return float(np.clip(cash / equity, 0.0, 1.0))

        positions_value = 0.0
        for ticker, qty in self.positions.items():
            if qty <= 0:
                continue
            price, _, _ = self.get_current_price(ticker)
            if price is None or price <= 0:
                continue
            positions_value += qty * price

        total_equity = self.cash + positions_value
        if total_equity <= 1e-9:
            return 1.0
        return float(np.clip(self.cash / total_equity, 0.0, 1.0))

    def _apply_high_conviction_cash_override(self, selected_assets, asset_metrics, cash_target, regime_state):
        """Optionally reduce cash target to min cash when conviction is very strong."""
        execution_cfg = self.config.get('execution', {})
        min_cash_pct = float(self.config['objectives']['min_cash_pct'])
        allow_override = bool(execution_cfg.get('allow_high_conviction_override', True))
        zscore_threshold = float(execution_cfg.get('high_conviction_zscore_threshold', 2.5))
        lead_threshold = float(execution_cfg.get('high_conviction_lead_threshold', 0.20))
        cash_surplus_buffer = float(execution_cfg.get('high_conviction_cash_surplus_buffer', 0.05))

        info = {
            'enabled': allow_override,
            'applied': False,
            'regime_state': regime_state,
            'top_ticker': None,
            'top_rank_score': 0.0,
            'lead_ratio': 0.0,
            'cash_ratio': 0.0,
            'required_cash_ratio': min_cash_pct + cash_surplus_buffer,
            'reason': 'not_triggered'
        }

        if not allow_override:
            info['reason'] = 'disabled_by_config'
            return float(cash_target), info
        if regime_state == 'risk_off_forced':
            info['reason'] = 'risk_off_forced_active'
            return float(cash_target), info
        if not selected_assets:
            info['reason'] = 'no_selected_assets'
            return float(cash_target), info
        if cash_target <= min_cash_pct + 1e-12:
            info['reason'] = 'already_at_floor'
            return float(cash_target), info

        ranked = sorted(
            selected_assets,
            key=lambda t: float(asset_metrics.get(t, {}).get('rank_score', 0.0)),
            reverse=True
        )
        top_ticker = ranked[0]
        top_rank_score = float(asset_metrics.get(top_ticker, {}).get('rank_score', 0.0))
        second_rank_score = float(asset_metrics.get(ranked[1], {}).get('rank_score', 0.0)) if len(ranked) > 1 else None

        if second_rank_score is None:
            lead_ratio = float('inf') if top_rank_score > 0 else 0.0
        elif abs(second_rank_score) <= 1e-12:
            lead_ratio = float('inf') if top_rank_score > 0 else 0.0
        else:
            lead_ratio = (top_rank_score - second_rank_score) / abs(second_rank_score)

        current_cash_ratio = self._estimate_current_cash_ratio()
        has_cash_surplus = current_cash_ratio >= (min_cash_pct + cash_surplus_buffer)
        strong_zscore = top_rank_score > zscore_threshold
        strong_lead = lead_ratio >= lead_threshold

        info.update({
            'top_ticker': top_ticker,
            'top_rank_score': top_rank_score,
            'lead_ratio': float(lead_ratio if np.isfinite(lead_ratio) else 9.99),
            'cash_ratio': current_cash_ratio
        })

        if has_cash_surplus and (strong_zscore or strong_lead):
            lead_text = f"{lead_ratio:.2%}" if np.isfinite(lead_ratio) else "INF"
            print(f"[CASH ADJUST] High-conviction override applied: cash_target reduced to {min_cash_pct:.2%} "
                  f"(top={top_ticker}, z={top_rank_score:.2f}, lead={lead_text}, cash_ratio={current_cash_ratio:.2%})")
            info['applied'] = True
            info['reason'] = 'high_conviction'
            return float(min_cash_pct), info

        if not has_cash_surplus:
            info['reason'] = 'insufficient_cash_surplus'
        elif not (strong_zscore or strong_lead):
            info['reason'] = 'signal_not_strong_enough'

        return float(cash_target), info

    def _select_high_conviction_weight_boost(self, selected_assets, asset_metrics, regime_state):
        """Select at most one exceptional asset for temporary position-cap boost."""
        execution_cfg = self.config.get('execution', {})
        enabled = bool(execution_cfg.get('enable_high_conviction_weighting', True))
        max_weight = float(execution_cfg.get('max_high_conviction_weight', 0.40))
        zscore_threshold = float(execution_cfg.get('high_conviction_weight_zscore_threshold', 2.5))
        ratio_threshold = float(execution_cfg.get('high_conviction_weight_ratio_threshold', 2.0))

        info = {
            'enabled': enabled,
            'applied': False,
            'boosted_ticker': None,
            'max_weight': max_weight,
            'top_score': 0.0,
            'second_score': 0.0,
            'score_ratio': 0.0,
            'reason': 'not_triggered'
        }

        if not enabled:
            info['reason'] = 'disabled_by_config'
            return None, info
        if regime_state == 'risk_off_forced':
            info['reason'] = 'risk_off_forced_active'
            return None, info
        if not selected_assets:
            info['reason'] = 'no_selected_assets'
            return None, info

        ranked = sorted(
            selected_assets,
            key=lambda t: float(asset_metrics.get(t, {}).get('rank_score', 0.0)),
            reverse=True
        )
        top_ticker = ranked[0]
        top_score = float(asset_metrics.get(top_ticker, {}).get('rank_score', 0.0))
        second_score = float(asset_metrics.get(ranked[1], {}).get('rank_score', 0.0)) if len(ranked) > 1 else 0.0

        if top_score <= 0:
            info.update({'top_score': top_score, 'second_score': second_score, 'reason': 'top_score_not_positive'})
            return None, info

        strong_zscore = top_score > zscore_threshold
        score_ratio = 0.0
        strong_ratio = False
        if second_score > 1e-12:
            score_ratio = top_score / second_score
            strong_ratio = score_ratio >= ratio_threshold

        info.update({
            'boosted_ticker': top_ticker,
            'top_score': top_score,
            'second_score': second_score,
            'score_ratio': score_ratio
        })

        if strong_zscore or strong_ratio:
            info['applied'] = True
            info['reason'] = 'high_conviction'
            return top_ticker, info

        info['reason'] = 'signal_not_strong_enough'
        return None, info

    def _select_hot_stock_boosts(self, selected_assets, asset_metrics, regime_state):
        """Select hot stocks eligible for adaptive cap boost."""
        execution_cfg = self.config.get('execution', {})
        boost_amount = float(execution_cfg.get('max_weight_boost_for_hot', 0.05))
        zscore_threshold = float(execution_cfg.get('hot_zscore_threshold', 1.5))
        persistence_cycles = int(execution_cfg.get('hot_persistence_cycles', 2))
        top_k = int(execution_cfg.get('hot_momentum_top_k', 3))

        info = {
            'enabled': boost_amount > 0 and top_k > 0,
            'boost_amount': float(max(0.0, boost_amount)),
            'zscore_threshold': float(zscore_threshold),
            'persistence_cycles': int(max(1, persistence_cycles)),
            'top_k': int(max(1, top_k)),
            'momentum_top': [],
            'boosted_assets': []
        }
        if not info['enabled']:
            return [], info
        if regime_state == 'risk_off_forced':
            info['reason'] = 'risk_off_forced_active'
            return [], info

        ranked_by_momentum = sorted(
            [t for t in selected_assets if t in asset_metrics],
            key=lambda t: float(asset_metrics.get(t, {}).get('momentum', 0.0)),
            reverse=True
        )
        top_assets = ranked_by_momentum[:info['top_k']]
        info['momentum_top'] = list(top_assets)

        top_set = set(top_assets)
        updated_streaks = {}
        for ticker in top_set:
            updated_streaks[ticker] = int(self.hot_momentum_streaks.get(ticker, 0)) + 1
        for ticker, prev_streak in self.hot_momentum_streaks.items():
            if ticker not in top_set and prev_streak > 0:
                updated_streaks[ticker] = 0
        self.hot_momentum_streaks = updated_streaks

        boosted = []
        for ticker in top_assets:
            metric = asset_metrics.get(ticker, {})
            zscore = float(metric.get('blended_momentum_z', metric.get('rank_score', 0.0)))
            streak = int(self.hot_momentum_streaks.get(ticker, 0))
            if zscore > zscore_threshold and streak >= info['persistence_cycles']:
                boosted.append({
                    'ticker': ticker,
                    'zscore': zscore,
                    'streak': streak,
                    'momentum': float(metric.get('momentum', 0.0))
                })

        info['boosted_assets'] = list(boosted)
        return [x['ticker'] for x in boosted], info

    def detect_exit_signals(self, ticker, price_series):
        """Detect simple technical breakdown signals for existing holdings."""
        execution_cfg = self.config.get('execution', {})
        lookback_days = int(execution_cfg.get('exit_signal_lookback_days', 20))
        min_trigger_count = int(execution_cfg.get('exit_signal_min_trigger_count', 1))
        gap_down_threshold = abs(float(execution_cfg.get('exit_signal_gap_down_pct', 0.03)))
        volume_spike_ratio = float(execution_cfg.get('exit_signal_volume_spike_ratio', 2.0))
        consecutive_down_days = int(execution_cfg.get('exit_signal_consecutive_down_days', 3))
        long_upper_shadow_ratio = float(execution_cfg.get('exit_signal_long_upper_shadow_ratio', 2.0))
        exit_on_gap_volume = bool(execution_cfg.get('exit_on_gap_volume', True))
        gap_volume_pct = abs(float(execution_cfg.get('exit_gap_down_pct', 0.04)))
        gap_volume_z_threshold = float(execution_cfg.get('exit_gap_volume_zscore', 2.5))
        gap_volume_window = int(execution_cfg.get('exit_gap_volume_window', 30))

        if price_series is None or price_series.empty:
            return False, "insufficient_data", False

        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(set(price_series.columns)):
            return False, "missing_ohlc", False

        window_size = max(lookback_days, consecutive_down_days + 2, 8)
        window = price_series.tail(window_size).copy()
        window = window.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if len(window) < max(6, consecutive_down_days + 1):
            return False, "insufficient_window", False

        latest = window.iloc[-1]
        prev = window.iloc[-2]
        triggers = []
        force_exit = False

        prev_close = float(prev['Close'])
        latest_open = float(latest['Open'])
        gap_return = 0.0
        if prev_close > 0:
            gap_return = (latest_open - prev_close) / prev_close
            if gap_return <= -gap_down_threshold:
                triggers.append(f"gap-down {gap_return:.1%}")

        prev_open = float(prev['Open'])
        latest_close = float(latest['Close'])
        bullish_prev = prev_close > prev_open
        bearish_now = latest_close < latest_open
        bearish_engulf = bullish_prev and bearish_now and (latest_open >= prev_close) and (latest_close <= prev_open)
        if bearish_engulf:
            triggers.append("bearish engulf")

        latest_high = float(latest['High'])
        latest_low = float(latest['Low'])
        body_size = max(abs(latest_close - latest_open), 1e-6)
        upper_shadow = max(0.0, latest_high - max(latest_open, latest_close))
        candle_range = max(latest_high - latest_low, 1e-6)
        close_location = (latest_close - latest_low) / candle_range
        if (upper_shadow / body_size) >= long_upper_shadow_ratio and close_location <= 0.35:
            triggers.append("long upper shadow")

        closes = window['Close'].tail(consecutive_down_days + 1).astype(float).tolist()
        if len(closes) >= consecutive_down_days + 1:
            down_seq = all(closes[i] < closes[i - 1] for i in range(1, len(closes)))
            if down_seq:
                triggers.append(f"{consecutive_down_days}-day decline")

        if 'Volume' in window.columns:
            vol_series = window['Volume'].dropna().astype(float)
            if len(vol_series) >= 6:
                latest_vol = float(vol_series.iloc[-1])
                base_vol = float(vol_series.iloc[-6:-1].mean())
                if base_vol > 0 and latest_vol >= base_vol * volume_spike_ratio:
                    triggers.append(f"vol spike x{latest_vol / base_vol:.1f}")

            if exit_on_gap_volume and prev_close > 0 and len(vol_series) >= max(gap_volume_window + 1, 10):
                latest_vol = float(vol_series.iloc[-1])
                baseline = vol_series.iloc[-(gap_volume_window + 1):-1]
                vol_mu = float(baseline.mean()) if len(baseline) > 0 else 0.0
                vol_sigma = float(baseline.std()) if len(baseline) > 0 else 0.0
                volume_z = (latest_vol - vol_mu) / vol_sigma if vol_sigma > 1e-12 else 0.0
                if gap_return <= -gap_volume_pct and volume_z >= gap_volume_z_threshold:
                    force_exit = True
                    triggers.append(f"gap-volume shock ({gap_return:.1%}, z={volume_z:.2f})")

        if force_exit:
            reason = " + ".join(triggers[:4])
            return True, reason, True

        if len(triggers) >= max(1, min_trigger_count):
            reason = " + ".join(triggers[:3])
            return True, reason, False

        return False, "", False

    def calculate_target_weights(self):
        """def calculate_target_weights: docstring omitted (was garbled/non-ASCII)."""

        def _apply_caps_and_normalize(weights_map, cap_map, invested_budget, fill_gap_max, fill_gap_max_iters, score_map):
            """def _apply_caps_and_normalize: docstring omitted (was garbled/non-ASCII)."""
            weights = {k: max(0.0, float(v)) for k, v in weights_map.items() if float(v) > 0}
            capped = set()
            diagnostics = {
                'invested_budget': float(max(0.0, invested_budget)),
                'total_before_caps': float(sum(weights.values())),
                'total_after_caps': 0.0,
                'downscaled': False,
                'downscale_factor': 1.0,
                'remaining_gap': 0.0,
                'fill_gap_max': float(max(0.0, fill_gap_max)),
                'fill_applied': False,
                'fill_amount': 0.0,
                'fill_reason': 'no_gap',
                'fill_remaining_end': 0.0,
                'capped_assets': []
            }

            for ticker in list(weights.keys()):
                cap = float(cap_map.get(ticker, 1.0))
                if weights[ticker] > cap:
                    weights[ticker] = cap
                    capped.add(ticker)

            total = sum(weights.values())
            diagnostics['total_after_caps'] = float(total)
            diagnostics['capped_assets'] = sorted(capped)
            if total <= 0 or invested_budget <= 0:
                diagnostics['remaining_gap'] = float(max(0.0, invested_budget))
                diagnostics['fill_reason'] = 'no_weights'
                diagnostics['fill_remaining_end'] = diagnostics['remaining_gap']
                return {}, diagnostics

            if total > invested_budget:
                scale = invested_budget / total
                for ticker in list(weights.keys()):
                    weights[ticker] *= scale
                diagnostics['downscaled'] = True
                diagnostics['downscale_factor'] = float(scale)

            # Small gap fill: only if gap is small, and never force-buy large gaps
            total = sum(weights.values())
            remaining = max(0.0, invested_budget - total)
            diagnostics['remaining_gap'] = float(remaining)

            if remaining <= 1e-10:
                diagnostics['fill_reason'] = 'no_gap'
            elif remaining > fill_gap_max:
                diagnostics['fill_reason'] = 'gap_too_large'
            else:
                fill_amount = 0.0
                fill_reason = 'filled'
                max_iters = max(1, int(fill_gap_max_iters))
                for _ in range(max_iters):
                    total_now = sum(weights.values())
                    remaining_now = max(0.0, invested_budget - total_now)
                    if remaining_now <= 1e-10:
                        break

                    headroom = {
                        k: max(0.0, float(cap_map.get(k, 1.0)) - weights.get(k, 0.0))
                        for k in cap_map.keys()
                    }
                    headroom = {k: h for k, h in headroom.items() if h > 1e-10}
                    if not headroom:
                        fill_reason = 'no_headroom'
                        break

                    # NOTE: comment omitted (was garbled/non-ASCII).
                    base_sum = sum(weights.get(k, 0.0) for k in headroom.keys())
                    if base_sum > 1e-12:
                        ratios = {k: weights.get(k, 0.0) / base_sum for k in headroom.keys()}
                    else:
                        score_sum = sum(max(0.0, float(score_map.get(k, 0.0))) for k in headroom.keys())
                        if score_sum > 1e-12:
                            ratios = {k: max(0.0, float(score_map.get(k, 0.0))) / score_sum for k in headroom.keys()}
                        else:
                            eq = 1.0 / len(headroom)
                            ratios = {k: eq for k in headroom.keys()}

                    proposal = {k: remaining_now * ratios[k] for k in headroom.keys()}
                    alloc = {k: min(proposal[k], headroom[k]) for k in headroom.keys()}
                    used = sum(alloc.values())
                    if used <= 1e-10:
                        fill_reason = 'no_allocatable_headroom'
                        break

                    for k, a in alloc.items():
                        if a > 0:
                            weights[k] = weights.get(k, 0.0) + a
                            if abs(float(cap_map.get(k, 1.0)) - weights[k]) <= 1e-10:
                                capped.add(k)

                    fill_amount += used

                diagnostics['fill_applied'] = fill_amount > 1e-10
                diagnostics['fill_amount'] = float(fill_amount)
                diagnostics['fill_reason'] = fill_reason
                diagnostics['capped_assets'] = sorted(capped)

            # Safety: never exceed invested_budget
            final_total = sum(weights.values())
            if final_total > invested_budget + 1e-9:
                safety_scale = invested_budget / final_total if final_total > 0 else 1.0
                for ticker in list(weights.keys()):
                    weights[ticker] *= safety_scale
                final_total = sum(weights.values())
                diagnostics['downscaled'] = True
                diagnostics['downscale_factor'] = float(diagnostics['downscale_factor'] * safety_scale)

            diagnostics['remaining_gap'] = float(max(0.0, invested_budget - final_total))
            diagnostics['fill_remaining_end'] = diagnostics['remaining_gap']
            diagnostics['capped_assets'] = sorted(capped)

            return {k: v for k, v in weights.items() if v > 1e-10}, diagnostics

        # NOTE: comment omitted (was garbled/non-ASCII).
        regime_state, trend_score, regime_details, base_cash_from_regime, base_max_weight = self.compute_regime_state()
        self.current_regime = {
            'regime_state': regime_state,
            'trend_score': trend_score,
            'regime_details': regime_details,
            'dynamic_min_cash': base_cash_from_regime,
            'dynamic_max_weight': base_max_weight,
            'risk_caps_applied': regime_state in ('risk_off', 'risk_off_forced'),
            'forced_until_time': self.forced_until_time.isoformat() if self.forced_until_time else None
        }

        # NOTE: comment omitted (was garbled/non-ASCII).
        macro_cfg = self.config.get('macro_integration', {})
        execution_cfg = self.config.get('execution', {})
        macro_mapping = self.config.get('macro_mapping', {})
        min_cash_pct = float(self.config['objectives']['min_cash_pct'])
        tilt_max_delta = float(macro_cfg.get('tilt_max_delta', 0.02))
        allow_buy_benchmarks = bool(execution_cfg.get('allow_buy_benchmarks', False))
        macro_risk_score_raw = float(self.cached_macro.get('macro_risk_score_raw', 0.0))
        macro_risk_score_smoothed = float(self.cached_macro.get('macro_risk_score_smoothed', macro_risk_score_raw))
        macro_tilts_filtered = dict(self.cached_macro.get('macro_tilts', {}))
        confirmed_topics = self.cached_macro.get('confirmed_topics', [])
        self._sync_current_macro_from_cache()

        # NOTE: comment omitted (was garbled/non-ASCII).
        macro_cash_slope = float(macro_cfg.get('macro_cash_slope', 0.02))
        macro_cash_from_risk = macro_cash_slope * macro_risk_score_smoothed
        macro_cash_from_topics = 0.0
        macro_cash_topic_details = []

        for topic in confirmed_topics:
            theme = str(topic.get('theme', 'unknown')).lower()
            for rule_name, rule_config in macro_mapping.items():
                if rule_name.lower() in theme or theme in rule_name.lower():
                    cash_add = rule_config.get('cash_add')
                    if cash_add is None:
                        continue
                    try:
                        cash_add = float(cash_add)
                    except (TypeError, ValueError):
                        continue
                    macro_cash_from_topics += cash_add
                    macro_cash_topic_details.append(f"{rule_name}:{cash_add:+.2%}")

        base_cash_from_regime = max(float(base_cash_from_regime), min_cash_pct)
        cash_target_unclipped = base_cash_from_regime + macro_cash_from_risk + macro_cash_from_topics
        cash_target = float(np.clip(cash_target_unclipped, base_cash_from_regime, 0.60))
        self.last_macro_cash_target = cash_target

        print(f"\n[MACRO PATH1] cash_target = base({base_cash_from_regime:.2%}) + "
              f"slope*risk({macro_cash_from_risk:+.2%}) + topic_cash_add({macro_cash_from_topics:+.2%}) -> "
              f"clip[{base_cash_from_regime:.2%},60.00%] = {cash_target:.2%}")
        if macro_cash_topic_details:
            print(f"[MACRO PATH1] topic cash_add details: {', '.join(macro_cash_topic_details)}")

        # NOTE: comment omitted (was garbled/non-ASCII).
        macro_allow_new_positions = {str(x).upper() for x in macro_cfg.get('macro_allow_new_positions', ['TLT', 'GLD'])}
        defensive_tilt_assets = set(macro_allow_new_positions) | {'CASH', 'TLT', 'GLD'}
        risk_off_mode = regime_state in ('risk_off', 'risk_off_forced')

        # NOTE: comment omitted (was garbled/non-ASCII).
        benchmark_tickers = {str(t).upper() for t in self.config.get('benchmarks', {}).get('tickers', [])}
        trade_universe_assets = []
        excluded_benchmark_assets = []
        for asset in self.config.get('universe', []):
            ticker = str(asset.get('ticker', ''))
            if not ticker or ticker.upper() == 'CASH':
                continue
            ticker_u = ticker.upper()
            if allow_buy_benchmarks or (ticker_u not in benchmark_tickers) or (ticker_u in defensive_tilt_assets):
                trade_universe_assets.append(asset)
            else:
                excluded_benchmark_assets.append(ticker)

        trade_ticker_by_upper = {str(a.get('ticker', '')).upper(): str(a.get('ticker', '')) for a in trade_universe_assets}
        trade_universe_tickers = set(trade_ticker_by_upper.keys())
        if excluded_benchmark_assets and not allow_buy_benchmarks:
            print(f"[UNIVERSE] Excluding benchmark tickers from trading: {', '.join(sorted(excluded_benchmark_assets))}")

        applied_tilts = {}
        blocked_tilts = {}
        blocked_tilts_not_trade_universe = {}
        for ticker, tilt in macro_tilts_filtered.items():
            ticker_u = str(ticker).upper()
            if ticker_u != 'CASH' and ticker_u not in trade_universe_tickers:
                try:
                    blocked_tilts_not_trade_universe[ticker] = float(tilt)
                except (TypeError, ValueError):
                    blocked_tilts_not_trade_universe[ticker] = tilt
                continue
            try:
                tilt_delta = float(tilt)
            except (TypeError, ValueError):
                continue
            tilt_delta = float(np.clip(tilt_delta, -tilt_max_delta, tilt_max_delta))
            canonical_ticker = trade_ticker_by_upper.get(ticker_u, str(ticker))

            if risk_off_mode and ticker_u not in defensive_tilt_assets:
                blocked_tilts[canonical_ticker] = tilt_delta
                continue
            applied_tilts[canonical_ticker] = tilt_delta

        if blocked_tilts:
            print(f"[MACRO PATH2] blocked offensive tilts in {regime_state}: "
                  f"{', '.join([f'{k}:{v:+.2%}' for k, v in blocked_tilts.items()])}")
        if blocked_tilts_not_trade_universe:
            print(f"[MACRO PATH2] blocked tilts not in trade_universe: "
                  f"{', '.join([f'{k}:{v}' for k, v in blocked_tilts_not_trade_universe.items()])}")

        cash_tilt = applied_tilts.get('CASH', 0.0)
        if abs(cash_tilt) > 1e-12:
            cash_target_before_cash_tilt = cash_target
            cash_target = float(np.clip(cash_target + cash_tilt, base_cash_from_regime, 0.60))
            print(f"[MACRO PATH2] CASH tilt {cash_tilt:+.2%} -> cash_target {cash_target_before_cash_tilt:.2%} -> {cash_target:.2%}")

        overlay_tickers = sorted(list(trade_universe_tickers))
        cash_target, news_overlay_info = self.apply_news_overlay_to_cash_target(
            overlay_tickers,
            cash_target=cash_target,
        )
        self.current_news_overlay_info = dict(news_overlay_info) if isinstance(news_overlay_info, dict) else {
            'enabled': False,
            'status': 'unavailable',
        }

        self.current_regime['dynamic_min_cash'] = cash_target
        self.current_regime['cash_target'] = cash_target
        self.current_regime['cash_target_components'] = {
            'base_cash_from_regime': base_cash_from_regime,
            'macro_cash_slope': macro_cash_slope,
            'macro_risk_score_smoothed': macro_risk_score_smoothed,
            'macro_cash_from_risk': macro_cash_from_risk,
            'macro_cash_from_topics': macro_cash_from_topics,
            'news_overlay_cash_delta': float((self.current_news_overlay_info or {}).get('applied_cash_delta', 0.0) or 0.0),
        }
        self.current_regime['news_overlay'] = dict(self.current_news_overlay_info) if isinstance(self.current_news_overlay_info, dict) else {}

        # NOTE: comment omitted (was garbled/non-ASCII).
        strategy = self.config['strategy']
        lookback = int(strategy['lookback_days'])
        vol_target = float(strategy['vol_target'])
        momentum_weight = float(strategy['momentum_weight'])
        vol_weight = float(strategy['vol_weight'])
        fill_gap_max = float(execution_cfg.get('fill_gap_max', 0.03))
        fill_gap_max_iters = int(execution_cfg.get('fill_gap_max_iters', 2))
        top_n = int(execution_cfg.get('cross_section_top_n', 10))
        corr_lookback_days = int(execution_cfg.get('correlation_lookback_days', 60))
        corr_threshold = float(execution_cfg.get('correlation_threshold', 0.80))
        vol_floor = max(1e-4, float(execution_cfg.get('volatility_floor', 0.08)))
        enable_short_term_momentum = bool(execution_cfg.get('enable_short_term_momentum', True))
        short_momentum_lookback_days = int(execution_cfg.get('short_momentum_lookback_days', 10))
        momentum_weights_cfg = execution_cfg.get('momentum_weights', {}) if isinstance(execution_cfg.get('momentum_weights', {}), dict) else {}
        momentum_short_weight = float(momentum_weights_cfg.get('short', 0.4))
        momentum_medium_weight = float(momentum_weights_cfg.get('medium', 0.6))

        asset_metrics, top_ranked = self._compute_cross_sectional_metrics(
            trade_universe_assets,
            lookback,
            vol_target,
            momentum_weight,
            vol_weight,
            top_n,
            enable_short_term_momentum=enable_short_term_momentum,
            short_lookback_days=short_momentum_lookback_days,
            momentum_short_weight=momentum_short_weight,
            momentum_medium_weight=momentum_medium_weight
        )

        print(f"\n[RANKING] Top {len(top_ranked)} assets (cross-sectional):")
        print(f"{'Ticker':<8} {'Industry':<12} {'ShortMom':>9} {'MedMom':>9} {'Vol':>8} {'VolZ':>7} {'IndStr':>8} {'Score':>9}")
        print('-' * 86)
        for ticker in top_ranked:
            m = asset_metrics[ticker]
            industry_name = str(m.get('industry', 'UNCLASSIFIED'))[:12]
            print(
                f"{ticker:<8} {industry_name:<12} {m['short_momentum']:>8.2%} {m['medium_momentum']:>8.2%} "
                f"{m['volatility']:>7.2%} {m.get('volume_z', 0.0):>+6.2f} {m.get('industry_strength', 0.0):>+7.2f} {m['rank_score']:>8.4f}"
            )
        print('-' * 86)

        corr_selected, corr_decisions, corr_degraded = self._apply_correlation_filter(
            top_ranked,
            corr_lookback_days,
            corr_threshold
        )
        selected_assets = corr_selected
        print(f"[CORR] Selected {len(selected_assets)}/{len(top_ranked)} after correlation filter (threshold={corr_threshold:.2f})")
        if corr_decisions:
            for d in corr_decisions:
                print(f"[CORR DROP] {d['dropped']} -> keep {d['kept']} (corr={d['corr']:.2f}, {d['reason']})")
        if corr_degraded:
            print(f"[CORR DEGRADED] insufficient data for: {', '.join(corr_degraded[:8])}{' ...' if len(corr_degraded) > 8 else ''}")

        cash_target, high_conviction_info = self._apply_high_conviction_cash_override(
            selected_assets=selected_assets,
            asset_metrics=asset_metrics,
            cash_target=cash_target,
            regime_state=regime_state
        )
        self.current_regime['high_conviction_override'] = dict(high_conviction_info)
        boosted_ticker, weight_boost_info = self._select_high_conviction_weight_boost(
            selected_assets=selected_assets,
            asset_metrics=asset_metrics,
            regime_state=regime_state
        )
        self.current_regime['high_conviction_weighting'] = dict(weight_boost_info)
        hot_boosted_tickers, hot_boost_info = self._select_hot_stock_boosts(
            selected_assets=selected_assets,
            asset_metrics=asset_metrics,
            regime_state=regime_state
        )
        self.current_regime['hot_stock_boost'] = dict(hot_boost_info)

        rank_signal_map = {t: max(0.0, float(asset_metrics.get(t, {}).get('rank_score', 0.0))) for t in selected_assets}
        if sum(rank_signal_map.values()) <= 1e-12:
            rank_signal_map = {
                t: max(0.0, float(asset_metrics.get(t, {}).get('momentum_rank_pct', 0.0)) - 0.5)
                for t in selected_assets
            }
        if sum(rank_signal_map.values()) <= 1e-12:
            rank_signal_map = {
                t: max(0.0, float(asset_metrics.get(t, {}).get('base_score', 0.0)))
                for t in selected_assets
            }

        if sum(rank_signal_map.values()) > 1e-12:
            rank_only_weights = {
                t: rank_signal_map[t] / sum(rank_signal_map.values())
                for t in selected_assets
                if rank_signal_map.get(t, 0.0) > 0
            }
        else:
            rank_only_weights = {}

        vol_scaled_signal = {}
        for ticker, base_sig in rank_only_weights.items():
            vol = float(asset_metrics.get(ticker, {}).get('volatility', vol_floor))
            vol_scaled_signal[ticker] = base_sig / max(vol, vol_floor)

        signal_sum = sum(vol_scaled_signal.values())
        if signal_sum > 1e-12:
            raw_weights = {t: v / signal_sum for t, v in vol_scaled_signal.items() if v > 0}
        else:
            raw_weights = {}

        print(f"[VOL SCALE] rank->vol adjusted weights:")
        print(f"{'Ticker':<8} {'RankW':>10} {'Vol':>10} {'ScaledW':>10}")
        print('-' * 44)
        for ticker in sorted(raw_weights.keys(), key=lambda x: raw_weights[x], reverse=True):
            rank_w = rank_only_weights.get(ticker, 0.0)
            vol = float(asset_metrics.get(ticker, {}).get('volatility', vol_floor))
            print(f"{ticker:<8} {rank_w:>9.2%} {vol:>9.2%} {raw_weights[ticker]:>9.2%}")
        print('-' * 44)

        # NOTE: comment omitted (was garbled/non-ASCII).
        invested_budget_raw = max(0.0, 1.0 - cash_target)
        invested_budget = min(0.90, invested_budget_raw)
        if invested_budget < invested_budget_raw - 1e-12:
            print(f"[EXPOSURE CAP] Invested budget clipped: {invested_budget_raw:.2%} -> {invested_budget:.2%} (hard cap 90%)")
        scaled_weights = {k: v * invested_budget for k, v in raw_weights.items()}

        # NOTE: comment omitted (was garbled/non-ASCII).
        universe_tickers = {str(asset['ticker']).upper() for asset in trade_universe_assets}
        if applied_tilts:
            print(f"[MACRO PATH2] Applying tilts:")
        for ticker, tilt_delta in applied_tilts.items():
            ticker_u = str(ticker).upper()
            if ticker_u == 'CASH':
                continue
            if ticker_u not in universe_tickers:
                continue

            if ticker in scaled_weights:
                old_weight = scaled_weights[ticker]
                scaled_weights[ticker] = max(0.0, old_weight + tilt_delta)
                print(f"  {ticker}: {old_weight:.2%} -> {scaled_weights[ticker]:.2%} (tilt {tilt_delta:+.2%})")
            elif tilt_delta > 0:
                if risk_off_mode and ticker.upper() not in macro_allow_new_positions:
                    blocked_tilts[ticker] = tilt_delta
                    continue
                scaled_weights[ticker] = tilt_delta
                print(f"  {ticker}: NEW position {tilt_delta:.2%}")

        # NOTE: comment omitted (was garbled/non-ASCII).
        max_weight_effective = {}
        for asset in trade_universe_assets:
            ticker = asset['ticker']
            if ticker == 'CASH':
                continue
            tilt_delta = float(np.clip(applied_tilts.get(ticker, 0.0), -tilt_max_delta, tilt_max_delta))
            max_weight_effective[ticker] = float(np.clip(base_max_weight + tilt_delta, 0.0, 1.0))

        if boosted_ticker and boosted_ticker in max_weight_effective:
            max_hc_weight = float(self.config.get('execution', {}).get('max_high_conviction_weight', 0.40))
            old_cap = float(max_weight_effective.get(boosted_ticker, base_max_weight))
            new_cap = float(np.clip(max(old_cap, max_hc_weight), 0.0, 1.0))
            max_weight_effective[boosted_ticker] = new_cap
            top_score = float(weight_boost_info.get('top_score', 0.0))
            second_score = float(weight_boost_info.get('second_score', 0.0))
            print(f"[WEIGHT BOOST] {boosted_ticker}: score={top_score:.2f} >> 2nd best={second_score:.2f} -> allowed {new_cap:.0%} weight")

        hot_boost_delta = float(max(0.0, execution_cfg.get('max_weight_boost_for_hot', 0.05)))
        hot_boost_applied = []
        for ticker in hot_boosted_tickers:
            if ticker not in max_weight_effective:
                continue
            old_cap = float(max_weight_effective.get(ticker, base_max_weight))
            new_cap = float(np.clip(old_cap + hot_boost_delta, 0.0, 1.0))
            max_weight_effective[ticker] = new_cap
            detail = next((x for x in hot_boost_info.get('boosted_assets', []) if x.get('ticker') == ticker), {})
            zscore = float(detail.get('zscore', 0.0))
            streak = int(detail.get('streak', 0))
            print(f"[HOT BOOST] {ticker}: z={zscore:.2f}, streak={streak} -> cap {old_cap:.2%} -> {new_cap:.2%}")
            hot_boost_applied.append(ticker)

        # NOTE: comment omitted (was garbled/non-ASCII).
        score_map = {k: max(0.0, float(rank_signal_map.get(k, 0.0))) for k in scaled_weights.keys()}
        adjusted_weights, alloc_diag = _apply_caps_and_normalize(
            scaled_weights,
            max_weight_effective,
            invested_budget,
            fill_gap_max,
            fill_gap_max_iters,
            score_map
        )
        alloc_diag['cross_section_top_n'] = int(top_n)
        alloc_diag['ranked_candidates'] = list(top_ranked)
        alloc_diag['corr_selected'] = list(selected_assets)
        alloc_diag['corr_dropped'] = [d['dropped'] for d in corr_decisions]
        alloc_diag['corr_threshold'] = float(corr_threshold)
        alloc_diag['enable_short_term_momentum'] = bool(enable_short_term_momentum)
        alloc_diag['short_momentum_lookback_days'] = int(short_momentum_lookback_days)
        alloc_diag['momentum_weights'] = {
            'short': float(momentum_short_weight),
            'medium': float(momentum_medium_weight)
        }
        alloc_diag['high_conviction_weight_boost'] = dict(weight_boost_info)
        alloc_diag['boosted_ticker'] = boosted_ticker
        alloc_diag['invested_budget_raw'] = float(invested_budget_raw)
        alloc_diag['portfolio_exposure_cap'] = 0.90
        alloc_diag['hot_boost_enabled'] = bool(hot_boost_info.get('enabled', False))
        alloc_diag['hot_boost_top_momentum'] = list(hot_boost_info.get('momentum_top', []))
        alloc_diag['hot_boost_assets'] = list(hot_boost_applied)
        alloc_diag['hot_boost_details'] = list(hot_boost_info.get('boosted_assets', []))
        alloc_diag['hot_boost_amount'] = float(hot_boost_delta)
        capped_assets = alloc_diag.get('capped_assets', [])

        cash_weight = max(0.0, 1.0 - sum(adjusted_weights.values()))
        adjusted_weights['CASH'] = cash_weight

        adjusted_weights, vt_meta = self.apply_vol_targeting_to_targets(
            adjusted_weights,
            reason_tag="target_weights_final"
        )
        self.current_vol_targeting_info = dict(vt_meta) if isinstance(vt_meta, dict) else {'enabled': False, 'status': 'invalid_meta'}
        if not isinstance(self.current_risk_check_info, dict):
            self.current_risk_check_info = {}
        self.current_risk_check_info['vol_targeting'] = dict(self.current_vol_targeting_info)
        alloc_diag['vol_targeting'] = dict(self.current_vol_targeting_info)
        if self.current_vol_targeting_info.get('enabled', False):
            vt_status = str(self.current_vol_targeting_info.get('status', ''))
            vt_scale = float(self.current_vol_targeting_info.get('scale', 1.0) or 1.0)
            vt_vol_before = self.current_vol_targeting_info.get('vol_before')
            vt_vol_target = self.current_vol_targeting_info.get('vol_target')
            print(f"[VOL TARGET] status={vt_status} scale={vt_scale:.4f} vol_before={vt_vol_before} target={vt_vol_target}")

        # NOTE: comment omitted (was garbled/non-ASCII).
        self.current_macro['applied_tilts'] = dict(applied_tilts)
        self.current_macro['blocked_tilts'] = dict(blocked_tilts)
        self.current_macro['blocked_tilts_not_trade_universe'] = dict(blocked_tilts_not_trade_universe)
        self.current_macro['capped_assets'] = list(capped_assets)
        self.current_macro['max_weight_per_asset_effective'] = dict(max_weight_effective)
        self.current_macro['cash_target'] = cash_target
        self.current_macro['allocation_diagnostics'] = dict(alloc_diag)

        self.current_regime['dynamic_min_cash'] = cash_target
        self.current_regime['dynamic_max_weight'] = base_max_weight
        self.current_regime['cash_target'] = cash_target
        self.current_regime['max_weight_per_asset_effective'] = dict(max_weight_effective)
        self.current_regime['capped_assets'] = list(capped_assets)
        self.current_regime['allocation_diagnostics'] = dict(alloc_diag)

        print(f"[ALLOC] budget={alloc_diag['invested_budget']:.2%}, before_caps={alloc_diag['total_before_caps']:.2%}, "
              f"after_caps={alloc_diag['total_after_caps']:.2%}, downscaled={alloc_diag['downscaled']}, "
              f"scale={alloc_diag['downscale_factor']:.6f}, gap={alloc_diag['remaining_gap']:.2%}, "
              f"fill_applied={alloc_diag['fill_applied']}, fill_amount={alloc_diag['fill_amount']:.2%}, "
              f"fill_reason={alloc_diag['fill_reason']}")

        return adjusted_weights

    def execute_rebalance(self, target_weights):
        """def execute_rebalance: docstring omitted (was garbled/non-ASCII)."""
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        if self.positions:
            test_ticker = list(self.positions.keys())[0]
            test_price, test_age, test_status = self.get_current_price(test_ticker)
            print(f"[SELF-CHECK] get_current_price('{test_ticker}') = (price={test_price}, age={test_age}min, status={test_status})")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        trade_context = self._build_trade_context()
        alloc_trace = [
            f"alloc_budget_{trade_context.get('invested_budget', 0.0):.2%}",
            f"alloc_before_caps_{trade_context.get('total_before_caps', 0.0):.2%}",
            f"alloc_after_caps_{trade_context.get('total_after_caps', 0.0):.2%}",
            f"alloc_downscaled_{str(bool(trade_context.get('downscaled', False))).lower()}",
            f"alloc_scale_{float(trade_context.get('downscale_factor', 1.0)):.6f}",
            f"alloc_gap_{trade_context.get('remaining_gap', 0.0):.2%}",
            f"alloc_fill_gap_max_{trade_context.get('fill_gap_max', 0.0):.2%}",
            f"alloc_fill_applied_{str(bool(trade_context.get('fill_applied', False))).lower()}",
            f"alloc_fill_amount_{trade_context.get('fill_amount', 0.0):.2%}",
            f"alloc_fill_reason_{str(trade_context.get('fill_reason', 'na'))}",
            f"alloc_capped_{','.join(trade_context.get('capped_assets', [])) if trade_context.get('capped_assets') else 'none'}"
        ]
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        execution_config = self.config.get('execution', {})
        cooldown_minutes = float(execution_config.get('rebalance_cooldown_minutes', 0) or 0)
        attempt_cooldown_minutes = float(
            execution_config.get('rebalance_attempt_cooldown_minutes', cooldown_minutes) or cooldown_minutes
        )
        min_holding_cycles = int(execution_config.get('min_holding_cycles', 4))
        enable_exit_signals = bool(execution_config.get('enable_exit_signals', True))
        exit_signal_action = str(execution_config.get('exit_signal_action', 'reduce')).lower()
        if exit_signal_action not in ('reduce', 'exit'):
            exit_signal_action = 'reduce'
        exit_signal_reduce_factor = float(np.clip(execution_config.get('exit_signal_reduce_factor', 0.5), 0.0, 1.0))
        self.current_holding_blocks = []
        self.current_exit_info = {
            'enabled': enable_exit_signals,
            'signals': [],
            'triggered_count': 0
        }
        self.current_risk_check_info = {
            'checked': False,
            'abort': False,
            'vol_targeting': dict(self.current_vol_targeting_info) if isinstance(self.current_vol_targeting_info, dict) else {'enabled': False, 'status': 'unavailable'}
        }
        planner_cfg = self._get_planner_cfg()
        self.current_planner_info = {
            'enabled': bool(planner_cfg.get('enable_trade_planner', False)),
            'status': 'disabled' if not bool(planner_cfg.get('enable_trade_planner', False)) else 'pending',
            'turnover_limit': 0.0,
            'turnover_used_forced': 0.0,
            'turnover_used_normal': 0.0,
            'turnover_used_total': 0.0,
            'num_forced': 0,
            'num_normal': 0,
            'num_dropped': 0,
            'num_adv_clipped': 0,
            'num_adv_dropped': 0,
            'adv_limit_enabled': bool(planner_cfg.get('enable_adv_limit', False)),
            'adv_limit_frac': float(planner_cfg.get('adv_limit_frac', 0.02)),
            'normal_sorted_by': 'score' if bool(planner_cfg.get('enable_cost_sensitive_ranking', False)) else 'notional',
            'lambda_cost': float(planner_cfg.get('lambda_cost', 1.0)),
            'benefit_mode': str(planner_cfg.get('benefit_mode', 'delta_weight')),
            'dropped': [],
            'scaled': []
        }
        cost_cfg = self._get_cost_model_cfg()
        self.current_cost_est_info = {
            'enabled': bool(cost_cfg.get('enabled', False)),
            'total': 0.0,
            'fee': 0.0,
            'slippage': 0.0,
            'impact': 0.0,
            'num_trades': 0
        }

        now_rebalance = self._now()
        session, gate = self._refresh_market_session_state(now_rebalance)
        session_state = str((session or {}).get('state', 'UNKNOWN')).upper() if isinstance(session, dict) else 'UNKNOWN'
        open_grace_passed = bool((session or {}).get('open_grace_passed', False)) if isinstance(session, dict) else False
        if not bool(gate.get('allowed', False)):
            self.current_stale_info = {
                'stale_count': 0,
                'stale_ratio': 0.0,
                'price_stale_skip': False,
                'price_stale_abort': False,
                'stale_candidate_count': 0,
                'stale_ratio_candidates': 0.0,
                'stale_candidate_count_policy_pass': 0,
                'stale_ratio_candidates_policy_pass': 0.0,
                'stale_candidates_policy_pass': {'stale': 0, 'total': 0},
                'decision_trace': f"market_closed_gate_{session_state}"
            }
            self.current_turnover_info = {
                'turnover_notional': 0.0,
                'turnover_notional_pre': 0.0,
                'turnover_notional_post': 0.0,
                'turnover_limit': 0.0,
                'turnover_scale': 1.0,
                'turnover_capped': False
            }
            self.current_planner_info = {
                'enabled': bool(planner_cfg.get('enable_trade_planner', False)),
                'status': 'skipped_market_closed_gate',
                'turnover_limit': 0.0,
                'turnover_used_forced': 0.0,
                'turnover_used_normal': 0.0,
                'turnover_used_total': 0.0,
                'num_forced': 0,
                'num_normal': 0,
                'num_dropped': 0,
                'dropped': [],
                'scaled': []
            }
            if isinstance(self.current_risk_check_info, dict):
                self.current_risk_check_info['trade_planner'] = dict(self.current_planner_info)
            self._update_cycle_price_debug(candidate_tickers=list(self.positions.keys()), planned_trades=[], price_debug_cache={})
            gate_detail = str((gate or {}).get('reason_detail', 'unknown'))
            print(f"[GATE] Rebalance skipped: market_closed_gate (session={session_state}, detail={gate_detail})")
            self._write_post_rebalance_live_snapshot(0, source="execute_rebalance_market_closed_gate")
            return []

        last_attempt_ref = self._coerce_datetime_utc(self.last_rebalance_attempt_time)
        if isinstance(last_attempt_ref, datetime) and self.last_rebalance_attempt_time is not last_attempt_ref:
            self.last_rebalance_attempt_time = last_attempt_ref
        if attempt_cooldown_minutes > 0 and last_attempt_ref is not None:
            time_since_attempt = (now_rebalance - last_attempt_ref).total_seconds() / 60
            if time_since_attempt < attempt_cooldown_minutes:
                remaining = attempt_cooldown_minutes - time_since_attempt
                self.current_rebalance_skipped_reason = 'attempt_cooldown'
                if not isinstance(self.current_stale_info, dict):
                    self.current_stale_info = {}
                self.current_stale_info['price_stale_abort'] = False
                self.current_stale_info['decision_trace'] = 'attempt_cooldown'
                self._update_cycle_price_debug(candidate_tickers=list(self.positions.keys()), planned_trades=[], price_debug_cache={})
                print(f"[ATTEMPT COOLDOWN] Skipping rebalance - {remaining:.1f} minutes remaining")
                self._write_post_rebalance_live_snapshot(0, source="execute_rebalance_attempt_cooldown")
                return []

        # Count this as an attempt even if later aborted by stale/risk filters.
        self.last_rebalance_attempt_time = now_rebalance

        last_success_raw = self.last_rebalance_success_time if self.last_rebalance_success_time is not None else self.last_rebalance_time
        last_success_ref = self._coerce_datetime_utc(last_success_raw)
        if self.last_rebalance_success_time is not None and isinstance(last_success_ref, datetime):
            self.last_rebalance_success_time = last_success_ref
        elif self.last_rebalance_time is not None and isinstance(last_success_ref, datetime):
            self.last_rebalance_time = last_success_ref
        if cooldown_minutes > 0 and last_success_ref is not None:
            time_since_last = (now_rebalance - last_success_ref).total_seconds() / 60
            if time_since_last < cooldown_minutes:
                remaining = cooldown_minutes - time_since_last
                self._update_cycle_price_debug(candidate_tickers=list(self.positions.keys()), planned_trades=[], price_debug_cache={})
                print(f"[COOLDOWN] Skipping rebalance - {remaining:.1f} minutes remaining")
                return []
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        stale_price_skip_minutes = execution_config.get('stale_price_skip_minutes', 60)
        max_stale_ratio = execution_config.get('max_stale_ratio', 0.3)
        stale_policy_cfg = execution_config.get('price_stale_policy', {})
        allow_buy_status = {s.upper() for s in stale_policy_cfg.get('allow_buy', ['LIVE', 'RECENT'])}
        allow_sell_status = {s.upper() for s in stale_policy_cfg.get('allow_sell', ['LIVE', 'RECENT', 'STALE'])}
        self.current_price_fetch_stats = {}

        prefetch_tickers = set()
        for ticker in self.positions.keys():
            ticker_u = str(ticker).upper().strip()
            if ticker_u and ticker_u != 'CASH':
                prefetch_tickers.add(ticker_u)
        for ticker in target_weights.keys():
            ticker_u = str(ticker).upper().strip()
            if ticker_u and ticker_u != 'CASH':
                prefetch_tickers.add(ticker_u)
        if getattr(self, "price_service", None) is not None and prefetch_tickers:
            try:
                stats = self.price_service.prefetch(
                    prefetch_tickers,
                    interval="5m",
                    period="1d",
                    max_chunk=getattr(self, "price_batch_chunk_size", 50),
                    allow_1m_fallback=bool(getattr(self, "price_batch_allow_1m_fallback", True)),
                )
                self.current_price_fetch_stats = dict(stats) if isinstance(stats, dict) else {}
                if isinstance(stats, dict):
                    print(
                        "[PRICE_FETCH] "
                        f"n={int(stats.get('tickers_in', len(prefetch_tickers)))} "
                        f"batch_calls={int(stats.get('batch_calls', 0))} "
                        f"hit={int(stats.get('cache_hits', 0))} "
                        f"miss={int(stats.get('cache_misses', 0))} "
                        f"fetched={int(stats.get('tickers_fetched', 0))} "
                        f"missing={len(stats.get('missing', [])) if isinstance(stats.get('missing', []), list) else 0} "
                        f"ms={int(stats.get('elapsed_ms', 0))}"
                    )
            except Exception as e:
                self.current_price_fetch_stats = {"status": "error", "error": str(e)}
                print(f"[WARN] Price prefetch failed: {e}")
        
        price_info = {}  # {ticker: (price, data_age_minutes, market_status)}
        price_debug_cache = {}
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        for ticker in self.positions.keys():
            price, age, status, dbg = self.get_current_price(ticker, return_debug=True)
            if price is not None:
                price_info[ticker] = (price, age, status)
            if isinstance(dbg, dict):
                price_debug_cache[str(ticker).upper()] = dbg
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        for ticker in target_weights.keys():
            if ticker == 'CASH' or ticker in price_info:
                continue
            price, age, status, dbg = self.get_current_price(ticker, return_debug=True)
            if price is not None:
                price_info[ticker] = (price, age, status)
            if isinstance(dbg, dict):
                price_debug_cache[str(ticker).upper()] = dbg
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        stale_count = 0
        total_count = len(price_info)
        
        for ticker, (price, age, status) in price_info.items():
            if status == "STALE" and age > stale_price_skip_minutes:
                stale_count += 1
        
        stale_ratio = stale_count / total_count if total_count > 0 else 0
        
        print(f"\n[PRICE CHECK] Total tickers: {total_count}, STALE: {stale_count}, Ratio: {stale_ratio:.1%} | "
              f"Policy BUY={sorted(allow_buy_status)} SELL={sorted(allow_sell_status)}")
        if total_count > 0 and stale_count == total_count and 'STALE' not in allow_buy_status:
            print("[INFO] All candidate prices are STALE and BUY policy blocks STALE quotes. "
                  "Likely outside market hours; rebalance may be skipped.")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.current_stale_info = {
            'stale_count': stale_count,
            'stale_ratio': stale_ratio,
            'price_stale_skip': False,
            'price_stale_abort': False,
            'stale_candidate_count': 0,
            'stale_ratio_candidates': 0.0,
            'stale_candidate_count_policy_pass': 0,
            'stale_ratio_candidates_policy_pass': 0.0,
            'stale_candidates_policy_pass': {'stale': 0, 'total': 0},
            'decision_trace': ''
        }
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        current_values = {}
        positions_value = 0.0
        
        for ticker, qty in self.positions.items():
            if ticker not in price_info:
                continue
            price, age, status = price_info[ticker]
            value = qty * price
            current_values[ticker] = value
            positions_value += value
        
        total_equity = self.cash + positions_value

        # Evaluate exit signals for existing holdings and override target weights if triggered.
        if enable_exit_signals and total_equity > 0 and self.positions:
            adjusted_target_weights = dict(target_weights)
            regime_state_now = str(self.current_regime.get('regime_state', 'neutral'))
            for ticker, qty in self.positions.items():
                if qty <= 0:
                    continue
                if ticker not in current_values:
                    continue

                hist = self.get_market_data(ticker, period='3mo', interval='1d')
                exit_flag, exit_reason, force_exit = self.detect_exit_signals(ticker, hist)
                if not exit_flag:
                    continue

                current_weight = current_values.get(ticker, 0.0) / total_equity if total_equity > 0 else 0.0
                if current_weight <= 1e-12:
                    continue

                old_target_weight = float(adjusted_target_weights.get(ticker, 0.0))
                action = exit_signal_action
                if force_exit:
                    action = 'exit'
                if regime_state_now in ('risk_off', 'risk_off_forced'):
                    action = 'exit'

                if action == 'exit':
                    new_target_weight = 0.0
                else:
                    new_target_weight = min(old_target_weight, current_weight * exit_signal_reduce_factor)

                adjusted_target_weights[ticker] = max(0.0, float(new_target_weight))
                self.current_exit_info['signals'].append({
                    'ticker': ticker,
                    'reason': exit_reason,
                    'force_exit': bool(force_exit),
                    'action': action,
                    'current_weight': float(current_weight),
                    'old_target_weight': float(old_target_weight),
                    'new_target_weight': float(new_target_weight)
                })

                print(f"[EXIT SIGNAL] {ticker}: {exit_reason} | action={action.upper()} "
                      f"target {old_target_weight:.2%}->{new_target_weight:.2%}")

            self.current_exit_info['triggered_count'] = len(self.current_exit_info.get('signals', []))
            target_weights = adjusted_target_weights
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        target_values = {}
        for ticker, weight in target_weights.items():
            if ticker == 'CASH':
                continue
            target_values[ticker] = total_equity * weight

        exit_signal_actions = {
            str(x.get('ticker')): str(x.get('action', 'reduce')).lower()
            for x in self.current_exit_info.get('signals', [])
            if x.get('ticker')
        }
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        weight_threshold = execution_config.get('weight_threshold', 0.0)
        
        tickers_to_trade = []
        for ticker in set(list(self.positions.keys()) + list(target_values.keys())):
            if ticker == 'CASH':
                continue
            
            current_value = current_values.get(ticker, 0.0)
            target_value = target_values.get(ticker, 0.0)
            
            current_weight = current_value / total_equity if total_equity > 0 else 0
            target_weight = target_value / total_equity if total_equity > 0 else 0
            
            weight_diff = abs(target_weight - current_weight)
            
            if weight_diff < weight_threshold:
                print(f"[SKIP] {ticker} weight diff {weight_diff:.4f} < threshold {weight_threshold:.4f}")
                continue
            
            tickers_to_trade.append(ticker)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        min_notional = execution_config.get('min_trade_notional_usd', 0)
        
        planned_trades = []  # [{ticker, side, current_value, target_value, desired_trade_value, price, age, status}]
        stale_count_policy_pass = 0
        candidate_count_policy_pass = 0
        policy_skip_count = 0  # NOTE: comment omitted (was garbled/non-ASCII).
        
        for ticker in tickers_to_trade:
            current_value = current_values.get(ticker, 0.0)
            target_value = target_values.get(ticker, 0.0)
            side = 'BUY' if target_value > current_value else 'SELL'

            if side == 'SELL' and min_holding_cycles > 0:
                entry_cycle = self.position_entry_cycle.get(str(ticker).upper())
                if entry_cycle is not None:
                    held_cycles = int(self.current_cycle) - int(entry_cycle)
                    remaining_cycles = int(min_holding_cycles) - int(held_cycles)
                    if remaining_cycles > 0:
                        self.current_holding_blocks.append({
                            'ticker': ticker,
                            'remaining_cycles': int(remaining_cycles),
                            'held_cycles': int(max(0, held_cycles))
                        })
                        print(f"[HOLDING] Block SELL/REDUCE {ticker}: remaining_cycles={remaining_cycles} (held={max(0, held_cycles)})")
                        continue

            desired_trade_value = abs(target_value - current_value)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if desired_trade_value < min_notional:
                print(f"[SKIP] {ticker} trade notional ${desired_trade_value:.2f} < min ${min_notional}")
                continue
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if ticker not in price_info:
                print(f"[SKIP] {ticker} no price info")
                continue
            
            price, age, status = price_info[ticker]
            
            status = str(status).upper()

            # NOTE: comment omitted (was garbled/non-ASCII).
            if side == 'BUY' and status not in allow_buy_status:
                policy_skip_count += 1
                print(f"[SKIP] {ticker} BUY status={status} not in allow_buy={sorted(allow_buy_status)}")
                continue
            if side == 'SELL' and status not in allow_sell_status:
                policy_skip_count += 1
                print(f"[SKIP] {ticker} SELL status={status} not in allow_sell={sorted(allow_sell_status)}")
                continue

            # Candidate for stale-ratio only after policy pass.
            candidate_count_policy_pass += 1
            if status == "STALE" and age > stale_price_skip_minutes:
                stale_count_policy_pass += 1

            if status == "STALE" and side == 'SELL':
                print(f"[ALLOW] {ticker} SELL on STALE price (age: {age:.0f}min) - policy allowed")

            force_reason = None
            if side == 'SELL':
                if ticker in exit_signal_actions:
                    force_reason = 'exit_signal'
                elif trade_context.get('regime_state') in ('risk_off', 'risk_off_forced'):
                    force_reason = 'risk_off'
                elif status == 'STALE':
                    force_reason = 'stale_sell'
            is_forced = force_reason is not None
            
            planned_trades.append({
                'ticker': ticker,
                'side': side,
                'current_value': current_value,
                'target_value': target_value,
                'desired_trade_value': desired_trade_value,
                'delta_weight': (desired_trade_value / total_equity) if total_equity > 0 else 0.0,
                'price': price,
                'age': age,
                'status': status,
                'is_forced': is_forced,
                'priority': 'forced' if is_forced else 'normal',
                'force_reason': force_reason
            })

        if self.current_holding_blocks:
            blocked_str = ", ".join([f"{x['ticker']}({x['remaining_cycles']})" for x in self.current_holding_blocks])
            print(f"[HOLDING] Blocked by minimum holding period: {blocked_str}")

        self._update_cycle_price_debug(
            candidate_tickers=tickers_to_trade,
            planned_trades=planned_trades,
            price_debug_cache=price_debug_cache,
        )
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        stale_ratio_candidates = (
            stale_count_policy_pass / candidate_count_policy_pass if candidate_count_policy_pass > 0 else 0.0
        )
        stale_abort_allowed = bool(session_state == 'OPEN' and open_grace_passed)
        
        print(
            f"\n[STALE CHECK] Policy-pass candidates: {candidate_count_policy_pass}, "
            f"STALE: {stale_count_policy_pass}, Ratio: {stale_ratio_candidates:.1%} "
            f"(abort_allowed={str(stale_abort_allowed).lower()}, session={session_state})"
        )
        
        if stale_abort_allowed and candidate_count_policy_pass > 0 and stale_ratio_candidates > max_stale_ratio:
            print(f"[STALE ABORT] STALE ratio {stale_ratio_candidates:.1%} > threshold {max_stale_ratio:.1%}, aborting rebalance")
            if stale_count_policy_pass == candidate_count_policy_pass:
                print("[INFO] All candidate trades depend on STALE prices. "
                      "This typically happens when market is closed or data is delayed.")
            abort_trace = f"stale_abort_ratio_{stale_ratio_candidates:.1%}_gt_{max_stale_ratio:.1%}"
            # NOTE: comment omitted (was garbled/non-ASCII).
            self.current_stale_info = {
                'stale_count': stale_count,
                'stale_ratio': stale_ratio,
                'price_stale_skip': policy_skip_count > 0,
                'price_stale_abort': True,  # NOTE: comment omitted (was garbled/non-ASCII).
                'stale_candidate_count': candidate_count_policy_pass,
                'stale_ratio_candidates': stale_ratio_candidates,
                'stale_candidate_count_policy_pass': candidate_count_policy_pass,
                'stale_ratio_candidates_policy_pass': stale_ratio_candidates,
                'stale_candidates_policy_pass': {'stale': stale_count_policy_pass, 'total': candidate_count_policy_pass},
                'decision_trace': abort_trace
            }
            self.current_turnover_info = {
                'turnover_notional': 0.0,
                'turnover_notional_pre': 0.0,
                'turnover_notional_post': 0.0,
                'turnover_limit': total_equity * execution_config.get('max_turnover_pct_per_rebalance', 0.20),
                'turnover_scale': 1.0,
                'turnover_capped': False
            }
            self.current_planner_info = {
                'enabled': bool(planner_cfg.get('enable_trade_planner', False)),
                'status': 'skipped_stale_abort',
                'turnover_limit': total_equity * execution_config.get('max_turnover_pct_per_rebalance', 0.20),
                'turnover_used_forced': 0.0,
                'turnover_used_normal': 0.0,
                'turnover_used_total': 0.0,
                'num_forced': 0,
                'num_normal': 0,
                'num_dropped': 0,
                'dropped': [],
                'scaled': []
            }
            if isinstance(self.current_risk_check_info, dict):
                self.current_risk_check_info['trade_planner'] = dict(self.current_planner_info)
            print(f"[DECISION] {abort_trace}")
            return []
        elif candidate_count_policy_pass == 0:
            print("[STALE CHECK] Skip stale-abort: no policy-pass tradable candidates.")
        elif not stale_abort_allowed:
            print(f"[STALE CHECK] Skip stale-abort outside tradable session (state={session_state}, open_grace={open_grace_passed}).")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.current_stale_info['price_stale_skip'] = policy_skip_count > 0
        self.current_stale_info['price_stale_abort'] = False
        self.current_stale_info['stale_candidate_count'] = candidate_count_policy_pass
        self.current_stale_info['stale_ratio_candidates'] = stale_ratio_candidates
        self.current_stale_info['stale_candidate_count_policy_pass'] = candidate_count_policy_pass
        self.current_stale_info['stale_ratio_candidates_policy_pass'] = stale_ratio_candidates
        self.current_stale_info['stale_candidates_policy_pass'] = {'stale': stale_count_policy_pass, 'total': candidate_count_policy_pass}
        if candidate_count_policy_pass == 0:
            self.current_stale_info['decision_trace'] = "stale_skip_no_policy_pass_candidates"
        elif not stale_abort_allowed:
            self.current_stale_info['decision_trace'] = f"stale_skip_session_{session_state}"
        else:
            self.current_stale_info['decision_trace'] = f"stale_ok_{stale_ratio_candidates:.1%}_le_{max_stale_ratio:.1%}"

        risk_gate = self._evaluate_portfolio_risk_gate(target_weights)
        risk_gate['checked'] = True
        if isinstance(self.current_vol_targeting_info, dict):
            risk_gate['vol_targeting'] = dict(self.current_vol_targeting_info)
        self.current_risk_check_info = dict(risk_gate)

        if risk_gate.get('volatility_confident', False):
            print(f"[RISK CHECK] Portfolio volatility = {risk_gate['weighted_volatility']:.2f} "
                  f"(limit {risk_gate['max_portfolio_volatility']:.2f}, known_weight={risk_gate['volatility_known_weight']:.1%})")
        else:
            print(f"[RISK CHECK] Portfolio volatility unknown coverage {risk_gate['volatility_known_weight']:.1%} "
                  f"< required {risk_gate['min_coverage']:.1%}; skip volatility gate")
        if risk_gate.get('enable_diversity_check', False):
            print(f"[RISK CHECK] Herfindahl Index = {risk_gate['herfindahl_index']:.3f} "
                  f"(limit {risk_gate['max_herfindahl_index']:.3f})")

        if risk_gate.get('abort', False):
            reason = str(risk_gate.get('abort_reason', 'risk_gate'))
            if reason == 'portfolio_volatility':
                print(f"[RISK CHECK] Portfolio volatility = {risk_gate['weighted_volatility']:.2f} -> aborting rebalance")
            elif reason == 'diversity_hhi':
                print(f"[RISK CHECK] Herfindahl Index = {risk_gate['herfindahl_index']:.3f} -> aborting rebalance")
            else:
                print(f"[RISK CHECK] aborting rebalance: {reason}")

            self.current_turnover_info = {
                'turnover_notional': 0.0,
                'turnover_notional_pre': 0.0,
                'turnover_notional_post': 0.0,
                'turnover_limit': total_equity * execution_config.get('max_turnover_pct_per_rebalance', 0.20),
                'turnover_scale': 1.0,
                'turnover_capped': False
            }
            self.current_planner_info = {
                'enabled': bool(planner_cfg.get('enable_trade_planner', False)),
                'status': f"skipped_risk_gate_abort_{reason}",
                'turnover_limit': total_equity * execution_config.get('max_turnover_pct_per_rebalance', 0.20),
                'turnover_used_forced': 0.0,
                'turnover_used_normal': 0.0,
                'turnover_used_total': 0.0,
                'num_forced': 0,
                'num_normal': 0,
                'num_dropped': 0,
                'dropped': [],
                'scaled': []
            }
            if isinstance(self.current_risk_check_info, dict):
                self.current_risk_check_info['trade_planner'] = dict(self.current_planner_info)
            self.current_stale_info['decision_trace'] = f"{self.current_stale_info.get('decision_trace', '')}|risk_gate_abort_{reason}"
            return []
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        turnover_notional_pre = sum(abs(t['desired_trade_value']) for t in planned_trades)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        max_turnover_pct = execution_config.get('max_turnover_pct_per_rebalance', 0.20)
        turnover_limit = total_equity * max_turnover_pct
        
        planner_enabled = bool(planner_cfg.get('enable_trade_planner', False))
        turnover_scale = 1.0
        turnover_capped = False
        planned_after_planner = float(turnover_notional_pre)
        
        print(f"\n[TURNOVER] Planned(pre): ${turnover_notional_pre:,.2f}, Limit: ${turnover_limit:,.2f} ({max_turnover_pct:.1%})")
        if planner_enabled:
            try:
                planned_trades, planner_meta = self.apply_trade_planner(
                    planned_trades,
                    total_equity,
                    turnover_limit,
                    reason_tag="execute_rebalance"
                )
                self.current_planner_info = planner_meta
                if isinstance(self.current_risk_check_info, dict):
                    self.current_risk_check_info['trade_planner'] = dict(planner_meta)

                planned_after_planner = float(
                    sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades if isinstance(t, dict))
                )
                print(
                    f"[PLANNER] enabled=true used_total=${planner_meta.get('turnover_used_total', 0.0):,.2f}/"
                    f"limit=${planner_meta.get('turnover_limit', 0.0):,.2f} "
                    f"forced={planner_meta.get('num_forced', 0)} normal={planner_meta.get('num_normal', 0)} "
                    f"dropped={planner_meta.get('num_dropped', 0)} "
                    f"scaled={len(planner_meta.get('scaled', [])) if isinstance(planner_meta.get('scaled', []), list) else 0} "
                    f"status={planner_meta.get('status')}"
                )
            except Exception as e:
                self.current_planner_info = {
                    'enabled': True,
                    'status': 'error',
                    'error': str(e),
                    'turnover_limit': turnover_limit,
                    'turnover_used_forced': 0.0,
                    'turnover_used_normal': 0.0,
                    'turnover_used_total': turnover_notional_pre,
                    'num_forced': 0,
                    'num_normal': len(planned_trades),
                    'num_dropped': 0,
                    'dropped': [],
                    'scaled': []
                }
                if isinstance(self.current_risk_check_info, dict):
                    self.current_risk_check_info['trade_planner'] = dict(self.current_planner_info)
                planned_after_planner = float(
                    sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades if isinstance(t, dict))
                )
        else:
            # NOTE: comment omitted (was garbled/non-ASCII).
            if turnover_notional_pre > turnover_limit:
                turnover_scale = turnover_limit / turnover_notional_pre
                turnover_capped = True
                print(f"[TURNOVER CAP] Scaling all trades by {turnover_scale:.2%}")
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                scaled_trades = []
                for trade in planned_trades:
                    scaled_trade_value = trade['desired_trade_value'] * turnover_scale
                    
                    # NOTE: comment omitted (was garbled/non-ASCII).
                    if scaled_trade_value < min_notional:
                        print(f"[SKIP] {trade['ticker']} scaled notional ${scaled_trade_value:.2f} < min ${min_notional}")
                        continue
                    
                    trade['desired_trade_value'] = scaled_trade_value
                    scaled_trades.append(trade)
                
                planned_trades = scaled_trades
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                actual_turnover_scaled = sum(abs(t['desired_trade_value']) for t in planned_trades)
                print(f"[TURNOVER CAP] Planned(after scaling): ${actual_turnover_scaled:,.2f}")
            planned_after_planner = float(
                sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades if isinstance(t, dict))
            )

            self.current_planner_info = {
                'enabled': False,
                'status': 'disabled',
                'turnover_limit': float(turnover_limit),
                'turnover_used_forced': 0.0,
                'turnover_used_normal': float(sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades)),
                'turnover_used_total': float(sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades)),
                'num_forced': 0,
                'num_normal': len(planned_trades),
                'num_dropped': 0,
                'dropped': [],
                'scaled': []
            }
            if isinstance(self.current_risk_check_info, dict):
                self.current_risk_check_info['trade_planner'] = dict(self.current_planner_info)

        # Second min-notional filter after planner/cap to avoid wasting budget on tiny trades.
        post_planner_dropped_trades = []
        if planned_trades:
            filtered_trades = []
            for trade in planned_trades:
                if not isinstance(trade, dict):
                    continue
                desired_abs = abs(float(trade.get('desired_trade_value', 0.0) or 0.0))
                if desired_abs + 1e-12 < float(min_notional):
                    post_planner_dropped_trades.append({
                        'trade': trade,
                        'desired_abs': float(desired_abs),
                    })
                    print(f"[SKIP] {trade.get('ticker', '')} post-planner notional ${desired_abs:.2f} < min ${min_notional}")
                    continue
                filtered_trades.append(trade)
            planned_trades = filtered_trades

        if planner_enabled and isinstance(self.current_planner_info, dict) and post_planner_dropped_trades:
            planner_meta_ref = self.current_planner_info
            max_audit_items = int(max(1, planner_cfg.get('max_audit_items', 20)))
            dropped_list = planner_meta_ref.get('dropped', [])
            if not isinstance(dropped_list, list):
                dropped_list = []
            existing_keys = set()
            for item in dropped_list:
                if not isinstance(item, dict):
                    continue
                key = str(item.get('trade_id', '')).strip()
                if key:
                    existing_keys.add(key)

            drop_increase = 0
            for item in post_planner_dropped_trades:
                tr = item.get('trade', {})
                if not isinstance(tr, dict):
                    continue
                desired_abs = float(item.get('desired_abs', 0.0) or 0.0)
                trade_id = str(tr.get('_planner_trade_id', '')).strip()
                if not trade_id:
                    trade_id = (
                        f"{str(tr.get('ticker', '')).upper().strip()}:"
                        f"{str(tr.get('side', '')).upper().strip()}:"
                        f"{round(desired_abs, 2)}:post_min"
                )
                if trade_id in existing_keys:
                    continue
                existing_keys.add(trade_id)
                drop_increase += 1
                if len(dropped_list) < max_audit_items:
                    dropped_list.append({
                        'ticker': str(tr.get('ticker', '')),
                        'side': str(tr.get('side', '')),
                        'reason': 'min_notional_post_planner',
                        'adv_clipped': bool(tr.get('adv_clipped', False)),
                        'adv_participation': tr.get('adv_participation'),
                        'planner_score': tr.get('planner_score'),
                        'planner_benefit': tr.get('planner_benefit'),
                        'planner_cost_dollars': tr.get('planner_cost_dollars'),
                        'planner_cost_weight': tr.get('planner_cost_weight'),
                        'trade_id': trade_id,
                        'old_notional': desired_abs,
                    })
            planner_meta_ref['dropped'] = dropped_list[:max_audit_items]
            planner_meta_ref['num_dropped'] = int(max(0, int(planner_meta_ref.get('num_dropped', 0) or 0)) + drop_increase)
            forced_after = float(sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades if bool(t.get('is_forced', False))))
            total_after = float(sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades))
            planner_meta_ref['turnover_used_forced'] = forced_after
            planner_meta_ref['turnover_used_normal'] = max(0.0, total_after - forced_after)
            planner_meta_ref['turnover_used_total'] = total_after
            planner_meta_ref['num_forced'] = int(sum(1 for t in planned_trades if bool(t.get('is_forced', False))))
            planner_meta_ref['num_normal'] = int(sum(1 for t in planned_trades if not bool(t.get('is_forced', False))))
            if isinstance(self.current_risk_check_info, dict):
                self.current_risk_check_info['trade_planner'] = dict(planner_meta_ref)

        planned_after_exec_input = float(
            sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades if isinstance(t, dict))
        )
        turnover_scale = (planned_after_exec_input / turnover_notional_pre) if turnover_notional_pre > 0 else 1.0
        turnover_capped = planned_after_exec_input + 1e-9 < turnover_notional_pre
        if planner_enabled:
            print(
                f"[TURNOVER] planned_after_planner=${planned_after_planner:,.2f}, "
                f"planned_after_min_filter=${planned_after_exec_input:,.2f}"
            )
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.current_turnover_info = {
            'turnover_notional': turnover_notional_pre,  # backward compatibility
            'turnover_notional_pre': turnover_notional_pre,
            'turnover_notional_post': 0.0,
            'turnover_limit': turnover_limit,
            'turnover_scale': turnover_scale,
            'turnover_capped': turnover_capped
        }
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        trades = []
        turnover_notional_post = 0.0
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        for trade in [t for t in planned_trades if t['side'] == 'SELL']:
            ticker = trade['ticker']
            price = trade['price']
            desired_notional = abs(trade['desired_trade_value'])
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            current_qty = self.positions.get(ticker, 0)
            sell_qty = int(desired_notional / price)
            sell_qty = min(sell_qty, current_qty)
            
            if sell_qty <= 0:
                continue
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            proceeds = sell_qty * price
            cost = proceeds * self.config['objectives']['transaction_cost_pct']
            net_proceeds = proceeds - cost
            turnover_notional_post += proceeds
            
            self.cash += net_proceeds
            self.positions[ticker] = current_qty - sell_qty
            
            if self.positions[ticker] == 0:
                del self.positions[ticker]
                if ticker in self.cost_basis:
                    del self.cost_basis[ticker]
                self.position_entry_cycle.pop(str(ticker).upper(), None)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            decision_trace = [
                'cooldown_pass',
                'weight_threshold_pass',
                'min_notional_pass',
                f'price_{trade["status"]}_age_{trade["age"]:.0f}min'
            ]
            # NOTE: comment omitted (was garbled/non-ASCII).
            if trade['status'] == 'STALE':
                decision_trace.append('sell_allowed_on_stale')
            if ticker in exit_signal_actions:
                decision_trace.append(f'exit_signal_{exit_signal_actions[ticker]}')
            if turnover_capped:
                decision_trace.append(f'turnover_cap_scale_{turnover_scale:.2%}')
            if trade_context['regime_state'] in ('risk_off', 'risk_off_forced'):
                decision_trace.append('risk_off_de-risk')
            decision_trace.extend(alloc_trace)

            equity_reference = float(total_equity) if total_equity > 0 else 0.0
            old_position_value = float(trade.get('current_value', current_qty * price) or 0.0)
            new_position_value = max(0.0, old_position_value - proceeds)
            if equity_reference > 0:
                old_weight = float(old_position_value / equity_reference)
                new_weight = float(new_position_value / equity_reference)
            else:
                old_weight = 0.0
                new_weight = 0.0
            weight_change = float(new_weight - old_weight)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            trades.append({
                'timestamp': self._now().isoformat(),
                'account_id': self.account_id,
                'session_id': self.session_id,
                'config_hash': self.config_hash,
                'env': self.runtime_env,
                'cycle': self.current_cycle,
                'ticker': ticker,
                'side': 'SELL',
                'quantity': sell_qty,
                'price': price,
                'notional': proceeds,
                'equity_reference': equity_reference,
                'old_weight': old_weight,
                'new_weight': new_weight,
                'weight_change': weight_change,
                'cost': cost,
                'cost_est': self.estimate_trade_cost(ticker, 'SELL', proceeds, adv_notional=None),
                'cost_est_total': 0.0,
                'reason': 'rebalance',
                'regime_state': trade_context['regime_state'],
                'trend_score': trade_context['trend_score'],
                'cash_target': trade_context['cash_target'],
                'macro_risk_score': trade_context['macro_risk_score'],
                'macro_topics': trade_context['macro_topics'],
                'macro_tilts': trade_context['macro_tilts'],
                'is_forced': bool(trade.get('is_forced', False)),
                'priority': str(trade.get('priority', 'normal')),
                'force_reason': trade.get('force_reason'),
                'adv_notional': trade.get('adv_notional'),
                'adv_limit_frac': trade.get('adv_limit_frac'),
                'adv_max_notional': trade.get('adv_max_notional'),
                'adv_clipped': bool(trade.get('adv_clipped', False)),
                'adv_participation': trade.get('adv_participation'),
                'planner_score': trade.get('planner_score'),
                'planner_benefit': trade.get('planner_benefit'),
                'planner_cost': trade.get('planner_cost'),
                'planner_cost_dollars': trade.get('planner_cost_dollars'),
                'planner_cost_weight': trade.get('planner_cost_weight'),
                'decision_trace': ' | '.join(decision_trace),
                'price_age_minutes': trade['age'],
                'price_status': trade['status']
            })
            
            print(f"[TRADE] SELL {sell_qty} {ticker} @ ${price:.2f} (notional: ${proceeds:.2f}, {trade['status']})")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        for trade in [t for t in planned_trades if t['side'] == 'BUY']:
            ticker = trade['ticker']
            price = trade['price']
            desired_notional = abs(trade['desired_trade_value'])
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            buy_qty = int(desired_notional / price)
            
            if buy_qty <= 0:
                continue
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            cash_before_trade = self.cash
            required_cash = buy_qty * price
            cost = required_cash * self.config['objectives']['transaction_cost_pct']
            total_required = required_cash + cost
            
            if total_required > self.cash:
                # NOTE: comment omitted (was garbled/non-ASCII).
                buy_qty = int((self.cash * 0.99) / (price * (1 + self.config['objectives']['transaction_cost_pct'])))
                
                if buy_qty <= 0:
                    print(f"[SKIP] {ticker} insufficient cash")
                    continue
                
                required_cash = buy_qty * price
                cost = required_cash * self.config['objectives']['transaction_cost_pct']
                total_required = required_cash + cost
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            self.cash -= total_required
            turnover_notional_post += required_cash
            old_qty = self.positions.get(ticker, 0)
            old_cost = self.cost_basis.get(ticker, 0)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            self.positions[ticker] = old_qty + buy_qty
            self.position_entry_cycle[str(ticker).upper()] = int(self.current_cycle)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if old_qty > 0:
                total_cost = (old_qty * old_cost) + (buy_qty * price)
                self.cost_basis[ticker] = total_cost / (old_qty + buy_qty)
            else:
                self.cost_basis[ticker] = price
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            decision_trace = [
                'cooldown_pass',
                'weight_threshold_pass',
                'min_notional_pass',
                f'price_{trade["status"]}_age_{trade["age"]:.0f}min'
            ]
            if turnover_capped:
                decision_trace.append(f'turnover_cap_scale_{turnover_scale:.2%}')
            if ticker in trade_context.get('macro_tilts_dict', {}):
                tilt = trade_context['macro_tilts_dict'][ticker]
                decision_trace.append(f'macro_tilt_{tilt:+.2%}')
            if trade_context['regime_state'] == 'risk_on':
                decision_trace.append('risk_on_add-risk')
            if total_required >= cash_before_trade * 0.99:
                decision_trace.append('cash_limited')
            decision_trace.extend(alloc_trace)

            equity_reference = float(total_equity) if total_equity > 0 else 0.0
            old_position_value = float(trade.get('current_value', old_qty * price) or 0.0)
            new_position_value = max(0.0, old_position_value + required_cash)
            if equity_reference > 0:
                old_weight = float(old_position_value / equity_reference)
                new_weight = float(new_position_value / equity_reference)
            else:
                old_weight = 0.0
                new_weight = 0.0
            weight_change = float(new_weight - old_weight)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            trades.append({
                'timestamp': self._now().isoformat(),
                'account_id': self.account_id,
                'session_id': self.session_id,
                'config_hash': self.config_hash,
                'env': self.runtime_env,
                'cycle': self.current_cycle,
                'ticker': ticker,
                'side': 'BUY',
                'quantity': buy_qty,
                'price': price,
                'notional': required_cash,
                'equity_reference': equity_reference,
                'old_weight': old_weight,
                'new_weight': new_weight,
                'weight_change': weight_change,
                'cost': cost,
                'cost_est': self.estimate_trade_cost(ticker, 'BUY', required_cash, adv_notional=None),
                'cost_est_total': 0.0,
                'reason': 'rebalance',
                'regime_state': trade_context['regime_state'],
                'trend_score': trade_context['trend_score'],
                'cash_target': trade_context['cash_target'],
                'macro_risk_score': trade_context['macro_risk_score'],
                'macro_topics': trade_context['macro_topics'],
                'macro_tilts': trade_context['macro_tilts'],
                'is_forced': bool(trade.get('is_forced', False)),
                'priority': str(trade.get('priority', 'normal')),
                'force_reason': trade.get('force_reason'),
                'adv_notional': trade.get('adv_notional'),
                'adv_limit_frac': trade.get('adv_limit_frac'),
                'adv_max_notional': trade.get('adv_max_notional'),
                'adv_clipped': bool(trade.get('adv_clipped', False)),
                'adv_participation': trade.get('adv_participation'),
                'planner_score': trade.get('planner_score'),
                'planner_benefit': trade.get('planner_benefit'),
                'planner_cost': trade.get('planner_cost'),
                'planner_cost_dollars': trade.get('planner_cost_dollars'),
                'planner_cost_weight': trade.get('planner_cost_weight'),
                'decision_trace': ' | '.join(decision_trace),
                'price_age_minutes': trade['age'],
                'price_status': trade['status']
            })
            
            print(f"[TRADE] BUY {buy_qty} {ticker} @ ${price:.2f} (notional: ${required_cash:.2f}, {trade['status']})")

        # NOTE: comment omitted (was garbled/non-ASCII).
        self.current_turnover_info['turnover_notional_post'] = turnover_notional_post
        if turnover_capped and turnover_notional_post > turnover_limit + 1e-6:
            print(f"[WARN] turnover_notional_post ${turnover_notional_post:,.2f} > limit ${turnover_limit:,.2f}")

        planner_enabled_flag = bool((self.current_planner_info or {}).get('enabled', False)) if isinstance(self.current_planner_info, dict) else False
        planner_status_val = (self.current_planner_info or {}).get('status') if isinstance(self.current_planner_info, dict) else None
        planner_turnover_limit_val = (self.current_planner_info or {}).get('turnover_limit') if isinstance(self.current_planner_info, dict) else None
        planner_turnover_used_total_val = (self.current_planner_info or {}).get('turnover_used_total') if isinstance(self.current_planner_info, dict) else None
        planner_num_dropped_val = (self.current_planner_info or {}).get('num_dropped') if isinstance(self.current_planner_info, dict) else None

        for trade in trades:
            trade_cost_est = trade.get('cost_est') if isinstance(trade.get('cost_est'), dict) else {}
            trade_cost_total = float(trade_cost_est.get('total', 0.0) or 0.0)
            trade['cost_est_total'] = trade_cost_total
            trade['turnover_notional_pre'] = turnover_notional_pre
            trade['turnover_notional_post'] = turnover_notional_post
            trade['turnover_limit'] = turnover_limit
            trade['turnover_scale'] = turnover_scale
            trade['turnover_capped'] = turnover_capped
            trade['invested_budget'] = trade_context.get('invested_budget', 0.0)
            trade['total_before_caps'] = trade_context.get('total_before_caps', 0.0)
            trade['total_after_caps'] = trade_context.get('total_after_caps', 0.0)
            trade['downscaled'] = trade_context.get('downscaled', False)
            trade['downscale_factor'] = trade_context.get('downscale_factor', 1.0)
            trade['remaining_gap'] = trade_context.get('remaining_gap', 0.0)
            trade['fill_gap_max'] = trade_context.get('fill_gap_max', self.config.get('execution', {}).get('fill_gap_max', 0.03))
            trade['fill_applied'] = trade_context.get('fill_applied', False)
            trade['fill_amount'] = trade_context.get('fill_amount', 0.0)
            trade['fill_reason'] = trade_context.get('fill_reason', '')
            trade['fill_remaining_end'] = trade_context.get('fill_remaining_end', trade_context.get('remaining_gap', 0.0))
            trade['capped_assets'] = ';'.join(trade_context.get('capped_assets', [])) if trade_context.get('capped_assets') else ''
            trade['planner_enabled'] = bool(planner_enabled_flag)
            trade['planner_status'] = planner_status_val
            trade['planner_turnover_limit'] = planner_turnover_limit_val
            trade['planner_turnover_used_total'] = planner_turnover_used_total_val
            trade['planner_num_dropped'] = planner_num_dropped_val

        cost_summary = {
            'enabled': bool(cost_cfg.get('enabled', False)),
            'total': 0.0,
            'fee': 0.0,
            'slippage': 0.0,
            'impact': 0.0,
            'num_trades': len(trades)
        }
        for trade in trades:
            cost_est = trade.get('cost_est') if isinstance(trade.get('cost_est'), dict) else {}
            try:
                cost_summary['total'] += float(cost_est.get('total', 0.0) or 0.0)
                cost_summary['fee'] += float(cost_est.get('fee', 0.0) or 0.0)
                cost_summary['slippage'] += float(cost_est.get('slippage', 0.0) or 0.0)
                cost_summary['impact'] += float(cost_est.get('impact', 0.0) or 0.0)
            except Exception:
                continue
        self.current_cost_est_info = cost_summary
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        if trades:
            self.trades_log.extend(trades)
            self.save_trades_immediately()
            self.last_rebalance_success_time = self._now()
            self.last_rebalance_time = self.last_rebalance_success_time  # backward compatibility
            print(f"[COOLDOWN] Next rebalance allowed after {cooldown_minutes} minutes")
            self._write_post_rebalance_live_snapshot(len(trades), source="execute_rebalance")
        else:
            print(f"[INFO] No trades executed (all filtered by protections)")
        
        return trades
    
    def _build_trade_context(self):
        """def _build_trade_context: docstring omitted (was garbled/non-ASCII)."""
        # NOTE: comment omitted (was garbled/non-ASCII).
        regime_state = self.current_regime.get('regime_state', 'neutral')
        trend_score = self.current_regime.get('trend_score', 0.5)
        cash_target = self.current_regime.get('cash_target', self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct']))
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        macro_risk_score_raw = self.current_macro.get('macro_risk_score', 0.0)
        macro_risk_score_smoothed = self.current_macro.get('macro_risk_score_smoothed', 0.0)
        confirmed_topics = self.current_macro.get('confirmed_topics', [])
        macro_tilts = self.current_macro.get('applied_tilts', self.current_macro.get('macro_tilts', {}))
        alloc_diag = self.current_regime.get('allocation_diagnostics', self.current_macro.get('allocation_diagnostics', {}))
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        if confirmed_topics:
            topics_str = '; '.join([f"{t['theme']}:{t['direction']}" for t in confirmed_topics[:3]])
        else:
            topics_str = 'none'
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        if macro_tilts:
            tilts_str = '; '.join([f"{k}:{v:+.2%}" for k, v in macro_tilts.items()])
        else:
            tilts_str = 'none'
        
        return {
            'regime_state': regime_state,
            'trend_score': trend_score,
            'cash_target': cash_target,
            'macro_risk_score': macro_risk_score_smoothed,  # NOTE: comment omitted (was garbled/non-ASCII).
            'macro_risk_score_raw': macro_risk_score_raw,  # NOTE: comment omitted (was garbled/non-ASCII).
            'macro_topics': topics_str,
            'macro_tilts': tilts_str,
            'macro_tilts_dict': macro_tilts,  # NOTE: comment omitted (was garbled/non-ASCII).
            'total_before_caps': alloc_diag.get('total_before_caps', 0.0),
            'total_after_caps': alloc_diag.get('total_after_caps', 0.0),
            'downscaled': alloc_diag.get('downscaled', False),
            'downscale_factor': alloc_diag.get('downscale_factor', 1.0),
            'remaining_gap': alloc_diag.get('remaining_gap', 0.0),
            'fill_gap_max': alloc_diag.get('fill_gap_max', self.config.get('execution', {}).get('fill_gap_max', 0.03)),
            'fill_applied': alloc_diag.get('fill_applied', False),
            'fill_amount': alloc_diag.get('fill_amount', 0.0),
            'fill_reason': alloc_diag.get('fill_reason', 'na'),
            'fill_remaining_end': alloc_diag.get('fill_remaining_end', alloc_diag.get('remaining_gap', 0.0)),
            'capped_assets': alloc_diag.get('capped_assets', self.current_regime.get('capped_assets', []))
        }

    def _compute_position_score_for_derisk(self, ticker):
        """def _compute_position_score_for_derisk: docstring omitted (was garbled/non-ASCII)."""
        strategy = self.config.get('strategy', {})
        lookback = int(strategy.get('lookback_days', 40))
        vol_target = float(strategy.get('vol_target', 0.15))
        momentum_weight = float(strategy.get('momentum_weight', 0.65))
        vol_weight = float(strategy.get('vol_weight', 0.35))

        hist = self.get_market_data(ticker, period='3mo', interval='1d')
        if hist is not None and not hist.empty:
            close = hist['Close'].dropna()
        else:
            close = pd.Series(dtype=float)

        drawdown = 0.0
        if len(close) > 1:
            peak = float(close.cummax().iloc[-1])
            latest = float(close.iloc[-1])
            drawdown = ((peak - latest) / peak) if peak > 0 else 0.0

        if len(close) >= lookback + 1:
            momentum = (float(close.iloc[-1]) - float(close.iloc[-lookback])) / float(close.iloc[-lookback])
            returns = close.pct_change().dropna()
            volatility = float(returns.tail(lookback).std() * np.sqrt(252)) if not returns.empty else self.calculate_volatility(ticker, lookback)
            score = momentum_weight * momentum - vol_weight * (volatility - vol_target)
            return float(score), float(momentum), float(volatility), float(drawdown), 'momentum_vol'

        volatility = self.calculate_volatility(ticker, lookback)
        fallback_score = -(volatility + drawdown)
        return float(fallback_score), None, float(volatility), float(drawdown), 'fallback_vol_drawdown'

    def _run_circuit_breaker_derisk(self, drawdown, max_dd):
        """def _run_circuit_breaker_derisk: docstring omitted (was garbled/non-ASCII)."""
        now = datetime.now()
        execution_config = self.config.get('execution', {})
        regime_config = self.config.get('regime_filter', {})
        planner_cfg = self._get_planner_cfg()
        cost_cfg = self._get_cost_model_cfg()
        cost_summary = {
            'enabled': bool(cost_cfg.get('enabled', False)),
            'total': 0.0,
            'fee': 0.0,
            'slippage': 0.0,
            'impact': 0.0,
            'num_trades': 0
        }

        forced_days = float(execution_config.get('circuit_breaker_forced_days', 1))
        self.forced_until_time = now + timedelta(days=forced_days)
        self.forced_regime_reason = f"drawdown_{drawdown:.2%}_gt_{max_dd:.2%}"

        risk_off_cash = regime_config.get('cash_risk_off', self.config['objectives']['min_cash_pct'])
        forced_cash_target = max(self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct']), risk_off_cash)
        forced_max_weight = regime_config.get('max_weight_risk_off', self.config['objectives']['max_weight_per_asset'])

        self.current_regime.update({
            'regime_state': 'risk_off_forced',
            'trend_score': self.current_regime.get('trend_score', 0.0),
            'dynamic_min_cash': forced_cash_target,
            'dynamic_max_weight': forced_max_weight,
            'risk_caps_applied': True,
            'forced_until_time': self.forced_until_time.isoformat(),
            'forced_reason': self.forced_regime_reason
        })

        # NOTE: comment omitted (was garbled/non-ASCII).
        holdings = []
        positions_value = 0.0

        for ticker, qty in list(self.positions.items()):
            if qty <= 0:
                continue
            price, age_min, status = self.get_current_price(ticker)
            if not price or price <= 0:
                continue

            value = qty * price
            score, momentum, volatility, ticker_drawdown, score_source = self._compute_position_score_for_derisk(ticker)

            holdings.append({
                'ticker': ticker,
                'qty': qty,
                'price': price,
                'age': age_min,
                'status': status,
                'value': value,
                'score': score,
                'momentum': momentum,
                'volatility': volatility,
                'drawdown': ticker_drawdown,
                'score_source': score_source
            })
            positions_value += value

        total_equity = self.cash + positions_value
        if total_equity <= 0 or not holdings:
            self.current_turnover_info = {
                'turnover_notional': 0.0,
                'turnover_notional_pre': 0.0,
                'turnover_notional_post': 0.0,
                'turnover_limit': 0.0,
                'turnover_scale': 1.0,
                'turnover_capped': False
            }
            self.current_planner_info = {
                'enabled': bool(planner_cfg.get('enable_trade_planner', False)),
                'status': 'bypass_circuit_breaker',
                'turnover_limit': 0.0,
                'turnover_used_forced': 0.0,
                'turnover_used_normal': 0.0,
                'turnover_used_total': 0.0,
                'num_forced': 0,
                'num_normal': 0,
                'num_dropped': 0,
                'dropped': [],
                'scaled': []
            }
            self.current_cost_est_info = cost_summary
            return []

        target_cash_value = total_equity * forced_cash_target
        cash_needed_initial = max(0.0, target_cash_value - self.cash)
        if cash_needed_initial <= 0:
            self.current_turnover_info = {
                'turnover_notional': 0.0,
                'turnover_notional_pre': 0.0,
                'turnover_notional_post': 0.0,
                'turnover_limit': 0.0,
                'turnover_scale': 1.0,
                'turnover_capped': False
            }
            self.current_planner_info = {
                'enabled': bool(planner_cfg.get('enable_trade_planner', False)),
                'status': 'bypass_circuit_breaker',
                'turnover_limit': 0.0,
                'turnover_used_forced': 0.0,
                'turnover_used_normal': 0.0,
                'turnover_used_total': 0.0,
                'num_forced': 0,
                'num_normal': 0,
                'num_dropped': 0,
                'dropped': [],
                'scaled': []
            }
            self.current_cost_est_info = cost_summary
            return []

        # NOTE: comment omitted (was garbled/non-ASCII).
        max_rebalance_pct = float(self.config.get('objectives', {}).get('max_rebalance_pct', 1.0))
        max_turnover_pct = float(execution_config.get('max_turnover_pct_per_rebalance', 1.0))
        cap_pct = min(max_rebalance_pct, max_turnover_pct)
        turnover_limit = total_equity * cap_pct

        turnover_notional_pre = min(cash_needed_initial, positions_value)
        turnover_capped = turnover_notional_pre > turnover_limit
        turnover_scale = (turnover_limit / turnover_notional_pre) if turnover_capped and turnover_notional_pre > 0 else 1.0

        holdings.sort(key=lambda x: x['score'])  # NOTE: comment omitted (was garbled/non-ASCII).
        min_notional = execution_config.get('min_trade_notional_usd', 0)
        tx_cost = self.config['objectives']['transaction_cost_pct']
        remaining_cash_needed = cash_needed_initial
        remaining_turnover_budget = turnover_limit
        turnover_notional_post = 0.0

        trade_context = self._build_trade_context()
        trades = []

        for h in holdings:
            if remaining_cash_needed <= 0 or remaining_turnover_budget <= 0:
                break

            desired_notional = min(h['value'], remaining_cash_needed, remaining_turnover_budget)
            sell_qty = int(desired_notional / h['price'])
            sell_qty = min(sell_qty, h['qty'])

            if sell_qty <= 0:
                continue

            proceeds = sell_qty * h['price']
            if proceeds < min_notional:
                continue

            cost = proceeds * tx_cost
            net_proceeds = proceeds - cost

            old_qty = self.positions.get(h['ticker'], 0)
            new_qty = old_qty - sell_qty
            self.cash += net_proceeds
            turnover_notional_post += proceeds
            remaining_turnover_budget = max(0.0, remaining_turnover_budget - proceeds)
            remaining_cash_needed = max(0.0, remaining_cash_needed - net_proceeds)

            if new_qty > 0:
                self.positions[h['ticker']] = new_qty
            else:
                del self.positions[h['ticker']]
                if h['ticker'] in self.cost_basis:
                    del self.cost_basis[h['ticker']]

            decision_trace = [
                'circuit_breaker',
                f'drawdown_{drawdown:.2%}_gt_{max_dd:.2%}',
                f'forced_until_{self.forced_until_time.isoformat()}',
                f'derisk_score_{h["score"]:.4f}',
                f'score_source_{h["score_source"]}',
                f'price_{str(h["status"]).upper()}_age_{h["age"]:.0f}min',
            ]
            if turnover_capped:
                decision_trace.append(f'turnover_cap_scale_{turnover_scale:.2%}')

            equity_reference = float(total_equity) if total_equity > 0 else 0.0
            old_position_value = float(h.get('value', old_qty * h['price']) or 0.0)
            new_position_value = max(0.0, old_position_value - proceeds)
            if equity_reference > 0:
                old_weight = float(old_position_value / equity_reference)
                new_weight = float(new_position_value / equity_reference)
            else:
                old_weight = 0.0
                new_weight = 0.0
            weight_change = float(new_weight - old_weight)

            trades.append({
                'timestamp': now.isoformat(),
                'account_id': self.account_id,
                'session_id': self.session_id,
                'config_hash': self.config_hash,
                'env': self.runtime_env,
                'ticker': h['ticker'],
                'side': 'SELL',
                'quantity': sell_qty,
                'price': h['price'],
                'notional': proceeds,
                'equity_reference': equity_reference,
                'old_weight': old_weight,
                'new_weight': new_weight,
                'weight_change': weight_change,
                'cost': cost,
                'cost_est': self.estimate_trade_cost(h['ticker'], 'SELL', proceeds, adv_notional=None),
                'cost_est_total': 0.0,
                'reason': 'circuit_breaker',
                'regime_state': 'risk_off_forced',
                'trend_score': trade_context['trend_score'],
                'cash_target': forced_cash_target,
                'macro_risk_score': trade_context['macro_risk_score'],
                'macro_topics': trade_context['macro_topics'],
                'macro_tilts': trade_context['macro_tilts'],
                'is_forced': True,
                'priority': 'forced',
                'force_reason': 'circuit_breaker',
                'decision_trace': ' | '.join(decision_trace),
                'price_age_minutes': h['age'],
                'price_status': str(h['status']).upper(),
                'turnover_notional_pre': turnover_notional_pre,
                'turnover_notional_post': 0.0,
                'turnover_limit': turnover_limit,
                'turnover_scale': turnover_scale,
                'turnover_capped': turnover_capped,
            })

        self.current_turnover_info = {
            'turnover_notional': turnover_notional_pre,
            'turnover_notional_pre': turnover_notional_pre,
            'turnover_notional_post': turnover_notional_post,
            'turnover_limit': turnover_limit,
            'turnover_scale': turnover_scale,
            'turnover_capped': turnover_capped,
        }
        self.current_planner_info = {
            'enabled': bool(planner_cfg.get('enable_trade_planner', False)),
            'status': 'bypass_circuit_breaker',
            'turnover_limit': float(turnover_limit),
            'turnover_used_forced': float(turnover_notional_post),
            'turnover_used_normal': 0.0,
            'turnover_used_total': float(turnover_notional_post),
            'num_forced': len(trades),
            'num_normal': 0,
            'num_dropped': 0,
            'dropped': [],
            'scaled': []
        }

        for t in trades:
            t['turnover_notional_post'] = turnover_notional_post
            trade_cost_est = t.get('cost_est') if isinstance(t.get('cost_est'), dict) else {}
            t['cost_est_total'] = float(trade_cost_est.get('total', 0.0) or 0.0)
            t['planner_enabled'] = bool((self.current_planner_info or {}).get('enabled', False))
            t['planner_status'] = (self.current_planner_info or {}).get('status')
            t['planner_turnover_limit'] = (self.current_planner_info or {}).get('turnover_limit')
            t['planner_turnover_used_total'] = (self.current_planner_info or {}).get('turnover_used_total')
            t['planner_num_dropped'] = (self.current_planner_info or {}).get('num_dropped')
            try:
                cost_summary['total'] += float(trade_cost_est.get('total', 0.0) or 0.0)
                cost_summary['fee'] += float(trade_cost_est.get('fee', 0.0) or 0.0)
                cost_summary['slippage'] += float(trade_cost_est.get('slippage', 0.0) or 0.0)
                cost_summary['impact'] += float(trade_cost_est.get('impact', 0.0) or 0.0)
            except Exception:
                pass
        cost_summary['num_trades'] = len(trades)
        self.current_cost_est_info = cost_summary

        if trades:
            self.trades_log.extend(trades)
            self.save_trades_immediately()
            self.last_rebalance_success_time = now
            self.last_rebalance_time = self.last_rebalance_success_time  # backward compatibility
            self._write_post_rebalance_live_snapshot(len(trades), source="circuit_breaker")

        print(f"[CIRCUIT] Forced risk-off active until {self.forced_until_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[CIRCUIT] Target cash: {forced_cash_target:.1%}, sold notional: ${turnover_notional_post:,.2f}, remaining cash gap: ${remaining_cash_needed:,.2f}")

        return trades

    def check_risk_controls(self):
        """def check_risk_controls: docstring omitted (was garbled/non-ASCII)."""
        positions_value = 0.0
        for ticker, qty in self.positions.items():
            price, age_min, status = self.get_current_price(ticker)  # NOTE: comment omitted (was garbled/non-ASCII).
            if price:
                positions_value += qty * price

        total_equity = self.cash + positions_value

        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        max_dd = self.config['objectives']['max_drawdown_pct']

        if drawdown > max_dd:
            print(f"[WARN] CIRCUIT BREAKER: Drawdown {drawdown:.2%} exceeds limit {max_dd:.2%}")
            trades = self._run_circuit_breaker_derisk(drawdown, max_dd)
            if trades:
                print(f"[CIRCUIT] Executed {len(trades)} structured de-risk trades")
            else:
                print("[CIRCUIT] No de-risk trades executed (already at target cash or constrained)")
            return True

        return False
    def record_snapshot(self):
        """def record_snapshot: docstring omitted (was garbled/non-ASCII)."""
        print(f"[DEBUG] Recording snapshot at {datetime.now().strftime('%H:%M:%S')}")
        import sys; sys.stdout.flush()
        
        positions_value = 0.0
        positions_detail = {}
        
        for ticker, qty in self.positions.items():
            price, age_min, status = self.get_current_price(ticker)  # NOTE: comment omitted (was garbled/non-ASCII).
            if price:
                value = qty * price
                positions_value += value
                positions_detail[ticker] = {
                    'quantity': qty,
                    'price': price,
                    'value': value
                }
        
        print(f"[DEBUG] Snapshot complete at {datetime.now().strftime('%H:%M:%S')}")
        import sys; sys.stdout.flush()
        
        total_equity = self.cash + positions_value
        
        total_return = (total_equity - self.initial_cash) / self.initial_cash
        drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0
        last_attempt_time = self.last_rebalance_attempt_time.isoformat() if isinstance(self.last_rebalance_attempt_time, datetime) else self.last_rebalance_attempt_time
        success_ref = self.last_rebalance_success_time if self.last_rebalance_success_time is not None else self.last_rebalance_time
        last_success_time = success_ref.isoformat() if isinstance(success_ref, datetime) else success_ref
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        bench_returns = {}
        bench_avg_return = 0.0
        bench_dispersion = 0.0
        excess_return = 0.0
        win_flag = False
        
        if 'benchmarks' in self.config:
            bench_tickers = self.config['benchmarks'].get('tickers', [])
            evaluation_days = self.config['benchmarks'].get('evaluation_days', 10)
            
            if bench_tickers:
                bench_returns, bench_avg_return, bench_dispersion = self.compute_benchmark_returns(
                    bench_tickers, evaluation_days
                )
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                excess_return = total_return - bench_avg_return
                win_flag = excess_return > 0
        
        snapshot = {
            'timestamp': self._now().isoformat(),
            'account_id': self.account_id,
            'session_id': self.session_id,
            'env': self.runtime_env,
            'config_hash': self.config_hash,
            'cycle': self.current_cycle,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'total_return': total_return,
            'drawdown': drawdown,
            'positions': positions_detail,
            'status': self.status,
            'weights_reused': self.current_weights_reused,
            'macro_reused': self.current_macro_reused,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'last_macro_time': self.last_macro_time.isoformat() if self.last_macro_time else None,
            'market_session': dict(self.current_market_session) if isinstance(self.current_market_session, dict) else {},
            'rebalance_gate': dict(self.current_rebalance_gate) if isinstance(self.current_rebalance_gate, dict) else {},
            'rebalance_skipped_reason': self.current_rebalance_skipped_reason,
            'price_debug': dict(self.current_price_debug) if isinstance(self.current_price_debug, dict) else {},
            'last_rebalance_attempt_time': last_attempt_time,
            'last_rebalance_success_time': last_success_time,
            # NOTE: comment omitted (was garbled/non-ASCII).
            'bench_returns': bench_returns,
            'bench_avg_return': bench_avg_return,
            'bench_dispersion': bench_dispersion,
            'excess_return': excess_return,
            'win_flag': win_flag,
            # NOTE: comment omitted (was garbled/non-ASCII).
            'regime_state': self.current_regime.get('regime_state', 'neutral'),
            'trend_score': self.current_regime.get('trend_score', 0.5),
            'dynamic_min_cash': self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct']),
            'dynamic_max_weight': self.current_regime.get('dynamic_max_weight', self.config['objectives']['max_weight_per_asset']),
            'cash_target': self.current_regime.get('cash_target', self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct'])),
            'risk_caps_applied': self.current_regime.get('risk_caps_applied', False),
            'forced_until_time': self.current_regime.get('forced_until_time', self.forced_until_time.isoformat() if self.forced_until_time else None),
            'forced_regime_reason': self.current_regime.get('forced_reason', self.forced_regime_reason),
            # NOTE: comment omitted (was garbled/non-ASCII).
            'macro_risk_score_raw': self.current_macro.get('macro_risk_score', 0.0),
            'macro_risk_score': self.current_macro.get('macro_risk_score', 0.0),  # NOTE: comment omitted (was garbled/non-ASCII).
            'macro_tilts': self.current_macro.get('macro_tilts', {}),
            'applied_tilts': self.current_macro.get('applied_tilts', {}),
            'capped_assets': self.current_macro.get('capped_assets', []),
            'invested_budget': self.current_regime.get('allocation_diagnostics', {}).get('invested_budget', 0.0),
            'total_before_caps': self.current_regime.get('allocation_diagnostics', {}).get('total_before_caps', 0.0),
            'total_after_caps': self.current_regime.get('allocation_diagnostics', {}).get('total_after_caps', 0.0),
            'downscaled': self.current_regime.get('allocation_diagnostics', {}).get('downscaled', False),
            'downscale_factor': self.current_regime.get('allocation_diagnostics', {}).get('downscale_factor', 1.0),
            'remaining_gap': self.current_regime.get('allocation_diagnostics', {}).get('remaining_gap', 0.0),
            'fill_gap_max': self.current_regime.get('allocation_diagnostics', {}).get('fill_gap_max', self.config.get('execution', {}).get('fill_gap_max', 0.03)),
            'fill_applied': self.current_regime.get('allocation_diagnostics', {}).get('fill_applied', False),
            'fill_amount': self.current_regime.get('allocation_diagnostics', {}).get('fill_amount', 0.0),
            'fill_reason': self.current_regime.get('allocation_diagnostics', {}).get('fill_reason', ''),
            'fill_remaining_end': self.current_regime.get('allocation_diagnostics', {}).get('fill_remaining_end', 0.0),
            'macro_tilts_ignored': self.current_macro.get('macro_tilts_ignored', {}),  # NOTE: comment omitted (was garbled/non-ASCII).
            'macro_cooldown_remaining': self.macro_cooldown_remaining,  # NOTE: comment omitted (was garbled/non-ASCII).
            # NOTE: comment omitted (was garbled/non-ASCII).
            'stale_count': self.current_stale_info.get('stale_count', 0),
            'stale_ratio': self.current_stale_info.get('stale_ratio', 0.0),
            'price_stale_skip': self.current_stale_info.get('price_stale_skip', False),
            'price_stale_abort': self.current_stale_info.get('price_stale_abort', False),  # NOTE: comment omitted (was garbled/non-ASCII).
            'stale_candidate_count': self.current_stale_info.get('stale_candidate_count', 0),  # NOTE: comment omitted (was garbled/non-ASCII).
            'stale_ratio_candidates': self.current_stale_info.get('stale_ratio_candidates', 0.0),  # NOTE: comment omitted (was garbled/non-ASCII).
            'stale_candidate_count_policy_pass': self.current_stale_info.get('stale_candidate_count_policy_pass', self.current_stale_info.get('stale_candidate_count', 0)),
            'stale_ratio_candidates_policy_pass': self.current_stale_info.get('stale_ratio_candidates_policy_pass', self.current_stale_info.get('stale_ratio_candidates', 0.0)),
            'stale_candidates_policy_pass': self.current_stale_info.get('stale_candidates_policy_pass', {
                'stale': self.current_stale_info.get('stale_candidate_count', 0),
                'total': self.current_stale_info.get('stale_candidate_count', 0)
            }),
            'stale_decision_trace': self.current_stale_info.get('decision_trace', ''),
            'exit_signals_enabled': self.current_exit_info.get('enabled', False),
            'exit_signals_triggered': self.current_exit_info.get('triggered_count', 0),
            'exit_signal_tickers': [x.get('ticker') for x in self.current_exit_info.get('signals', []) if x.get('ticker')],
            'holding_block_count': len(self.current_holding_blocks),
            'holding_blocks': list(self.current_holding_blocks),
            'cross_section_top_n': self.current_regime.get('allocation_diagnostics', {}).get('cross_section_top_n', self.config.get('execution', {}).get('cross_section_top_n', 10)),
            'ranked_candidates': self.current_regime.get('allocation_diagnostics', {}).get('ranked_candidates', []),
            'corr_selected': self.current_regime.get('allocation_diagnostics', {}).get('corr_selected', []),
            'corr_dropped': self.current_regime.get('allocation_diagnostics', {}).get('corr_dropped', []),
            # NOTE: comment omitted (was garbled/non-ASCII).
            'turnover_notional': self.current_turnover_info.get('turnover_notional', 0.0),
            'turnover_notional_pre': self.current_turnover_info.get('turnover_notional_pre', self.current_turnover_info.get('turnover_notional', 0.0)),
            'turnover_notional_post': self.current_turnover_info.get('turnover_notional_post', 0.0),
            'turnover_limit': self.current_turnover_info.get('turnover_limit', 0.0),
            'turnover_scale': self.current_turnover_info.get('turnover_scale', 1.0),
            'turnover_capped': self.current_turnover_info.get('turnover_capped', False),
            'risk_check_checked': self.current_risk_check_info.get('checked', False),
            'risk_check_abort': self.current_risk_check_info.get('abort', False),
            'risk_check_reason': self.current_risk_check_info.get('abort_reason', ''),
            'portfolio_weighted_volatility': self.current_risk_check_info.get('weighted_volatility', 0.0),
            'portfolio_volatility_limit': self.current_risk_check_info.get('max_portfolio_volatility', self.config.get('execution', {}).get('max_portfolio_volatility', 0.25)),
            'portfolio_volatility_known_weight': self.current_risk_check_info.get('volatility_known_weight', 0.0),
            'portfolio_volatility_confident': self.current_risk_check_info.get('volatility_confident', False),
            'portfolio_vol_min_coverage': self.current_risk_check_info.get('min_coverage', self.config.get('execution', {}).get('portfolio_vol_min_coverage', 0.70)),
            'diversity_check_enabled': self.current_risk_check_info.get('enable_diversity_check', self.config.get('execution', {}).get('enable_diversity_check', True)),
            'herfindahl_index': self.current_risk_check_info.get('herfindahl_index', 0.0),
            'herfindahl_limit': self.current_risk_check_info.get('max_herfindahl_index', self.config.get('execution', {}).get('max_herfindahl_index', 0.35)),
            'gate_vol_method': self.current_risk_check_info.get('gate_vol_method', ''),
            'cov_gate_used': self.current_risk_check_info.get('cov_gate_used', False),
            'cov_gate_coverage': self.current_risk_check_info.get('cov_gate_coverage', None),
            'cov_gate_vol': self.current_risk_check_info.get('cov_gate_vol', None),
            'cov_gate_max_rc': self.current_risk_check_info.get('cov_gate_max_rc', None),
            'cov_gate_pass': self.current_risk_check_info.get('cov_gate_pass', None),
            'cov_gate_reason': self.current_risk_check_info.get('cov_gate_reason', ''),
            'cov_risk_diag': self.current_risk_check_info.get('cov_risk_diag', {}),
            'vol_targeting': self.current_vol_targeting_info if isinstance(self.current_vol_targeting_info, dict) else {'enabled': False, 'status': 'unavailable'},
            'vol_targeting_scale': (self.current_vol_targeting_info.get('scale') if isinstance(self.current_vol_targeting_info, dict) else None),
            'vol_targeting_vol_before': (self.current_vol_targeting_info.get('vol_before') if isinstance(self.current_vol_targeting_info, dict) else None),
            'vol_targeting_cash_after': (self.current_vol_targeting_info.get('cash_after') if isinstance(self.current_vol_targeting_info, dict) else None),
            'cost_est': dict(self.current_cost_est_info) if isinstance(self.current_cost_est_info, dict) else {
                'enabled': False,
                'total': 0.0,
                'fee': 0.0,
                'slippage': 0.0,
                'impact': 0.0,
                'num_trades': 0
            },
            'trade_planner': dict(self.current_planner_info) if isinstance(self.current_planner_info, dict) else {
                'enabled': False,
                'status': 'disabled',
                'turnover_limit': 0.0,
                'turnover_used_forced': 0.0,
                'turnover_used_normal': 0.0,
                'turnover_used_total': 0.0,
                'num_forced': 0,
                'num_normal': 0,
                'num_dropped': 0,
                'num_adv_clipped': 0,
                'num_adv_dropped': 0,
                'adv_limit_enabled': False,
                'adv_limit_frac': 0.0,
                'normal_sorted_by': 'notional',
                'lambda_cost': 1.0,
                'benefit_mode': 'delta_weight',
                'normal_score_stats': {'count': 0},
                'dropped': [],
                'scaled': []
            },
            'trade_planner_num_dropped': int((self.current_planner_info or {}).get('num_dropped', 0)) if isinstance(self.current_planner_info, dict) else 0,
            'trade_planner_turnover_used': (
                float((self.current_planner_info or {}).get('turnover_used_forced', 0.0) or 0.0) +
                float((self.current_planner_info or {}).get('turnover_used_normal', 0.0) or 0.0)
            ) if isinstance(self.current_planner_info, dict) else 0.0,
            'trade_planner_num_adv_clipped': int((self.current_planner_info or {}).get('num_adv_clipped', 0)) if isinstance(self.current_planner_info, dict) else 0,
            'trade_planner_num_adv_dropped': int((self.current_planner_info or {}).get('num_adv_dropped', 0)) if isinstance(self.current_planner_info, dict) else 0,
            'trade_planner_normal_score_count': int((((self.current_planner_info or {}).get('normal_score_stats', {}) or {}).get('count', 0))) if isinstance(self.current_planner_info, dict) else 0,
            'news_overlay_debug': dict(self.current_news_overlay_info) if isinstance(self.current_news_overlay_info, dict) else {'enabled': False, 'status': 'unavailable'},
            'diagnostic_hint': self.last_diagnostic_hint
        }
        
        self.portfolio_snapshots.append(snapshot)
        self.equity_curve.append((self._now(), total_equity, self.cash, positions_value))

        # NOTE: comment omitted (was garbled/non-ASCII).
        scoreboard_record = self.append_scoreboard_record()
        if scoreboard_record:
            snapshot['strategy_return_2w'] = scoreboard_record.get('strategy_return_2w', 0.0)
            snapshot['bench_avg_return_2w'] = scoreboard_record.get('bench_avg_return_2w', 0.0)
            snapshot['excess_return_2w'] = scoreboard_record.get('excess_return_2w', 0.0)
            snapshot['win_flag_2w'] = scoreboard_record.get('win_flag_2w', False)
            snapshot['turnover_sum_2w'] = scoreboard_record.get('turnover_sum_2w', 0.0)
            snapshot['avg_cash_2w'] = scoreboard_record.get('avg_cash_2w', 0.0)
            snapshot['macro_active_ratio_2w'] = scoreboard_record.get('macro_active_ratio_2w', 0.0)
            snapshot['diagnostic_hint'] = scoreboard_record.get('diagnostic_hint', '')

        # Write compact live snapshot for Streamlit monitor.
        self.write_live_snapshot(snapshot, source="record_snapshot")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.generate_live_summary()
        
        return snapshot

    def save_trades_immediately(self):
        """def save_trades_immediately: docstring omitted (was garbled/non-ASCII)."""
        trades_path = self.config['reporting']['trades_log_path']
        if self.trades_log:
            trades_df = pd.DataFrame(self.trades_log)
            trades_df.to_csv(trades_path, index=False)
            self.save_trade_history_jsonl()
        print(f"[OK] Trades updated: {trades_path}")
        import sys; sys.stdout.flush()  # NOTE: comment omitted (was garbled/non-ASCII).

    def generate_live_summary(self, snapshot_override=None):
        """def generate_live_summary: docstring omitted (was garbled/non-ASCII)."""
        if isinstance(snapshot_override, dict):
            final_snapshot = snapshot_override
        elif self.portfolio_snapshots:
            final_snapshot = self.portfolio_snapshots[-1]
        else:
            return

        summary_path = self.config['reporting']['summary_report_path'].replace('.txt', '_live.txt')
        
        with io.StringIO() as f:
            f.write("="*60 + "\n")
            f.write("GlobalWatch Paper Trading LIVE Summary\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Current Status: {self.status}\n")
            f.write(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cycle: {self.current_cycle}\n\n")
            
            f.write(f"Performance:\n")
            f.write(f"  Initial Cash: ${self.initial_cash:,.2f}\n")
            f.write(f"  Current Equity: ${final_snapshot['total_equity']:,.2f}\n")
            f.write(f"  Current Return: {final_snapshot['total_return']:.2%}\n")
            f.write(f"  Current Drawdown: {final_snapshot['drawdown']:.2%}\n\n")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if final_snapshot.get('regime_state'):
                f.write(f"Market Regime:\n")
                f.write(f"  State: {final_snapshot['regime_state'].upper()}")
                
                if final_snapshot.get('risk_caps_applied'):
                    f.write(" [RISK CAPS ACTIVE]\n")
                else:
                    f.write("\n")
                
                f.write(f"  Trend Score: {final_snapshot['trend_score']:.1%}\n")
                f.write(f"  Dynamic Min Cash: {final_snapshot['dynamic_min_cash']:.1%}\n")
                f.write(f"  Dynamic Max Weight: {final_snapshot['dynamic_max_weight']:.1%}\n\n")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if final_snapshot.get('macro_risk_score', 0) > 0:
                f.write(f"Macro Signals (GlobalWatch):\n")
                f.write(f"  Risk Score: {final_snapshot['macro_risk_score']:.1f}/10.0\n")
                f.write(f"  Confirmed Topics: {final_snapshot.get('confirmed_topics_count', 0)}\n")
                
                if final_snapshot.get('macro_tilts'):
                    f.write(f"  Active Tilts:\n")
                    for ticker, tilt in final_snapshot['macro_tilts'].items():
                        f.write(f"    {ticker}: {tilt:+.2%}\n")
                f.write("\n")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if final_snapshot.get('bench_returns'):
                f.write(f"Benchmark Comparison:\n")
                f.write(f"  Strategy Return: {final_snapshot['total_return']:.2%}\n")
                f.write(f"  Benchmark Avg Return: {final_snapshot['bench_avg_return']:.2%}\n")
                f.write(f"  Excess Return: {final_snapshot['excess_return']:.2%}")
                
                if final_snapshot['win_flag']:
                    f.write(" [OUTPERFORM]\n")
                else:
                    f.write(" [UNDERPERFORM]\n")
                
                f.write(f"  Benchmark Dispersion: {final_snapshot['bench_dispersion']:.2%}\n\n")
                
                f.write(f"  Individual Benchmarks:\n")
                for ticker, ret in sorted(final_snapshot['bench_returns'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"    {ticker}: {ret:.2%}\n")
                f.write("\n")
            
            f.write(f"Current Portfolio:\n")
            f.write(f"  Cash: ${final_snapshot['cash']:,.2f} ({final_snapshot['cash']/final_snapshot['total_equity']:.1%})\n")
            f.write(f"  Positions Value: ${final_snapshot['positions_value']:,.2f}\n\n")
            
            if final_snapshot['positions']:
                f.write(f"  Current Holdings:\n")
                for ticker, pos in sorted(final_snapshot['positions'].items(), key=lambda x: x[1]['value'], reverse=True):
                    weight = pos['value'] / final_snapshot['total_equity']
                    f.write(f"    {ticker}: {pos['quantity']} shares @ ${pos['price']:.2f} = ${pos['value']:,.2f} ({weight:.1%})\n")
            
            f.write(f"\nTotal Trades So Far: {len(self.trades_log)}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("[LIVE] Updates every cycle\n")
            f.write("[SIMULATION ONLY] NO REAL MONEY\n")
            f.write("="*60 + "\n")
            summary_content = f.getvalue()
        self.atomic_write_text(summary_path, summary_content)
        
        print(f"[OK] Live summary updated: {summary_path}")
        import sys; sys.stdout.flush()  # NOTE: comment omitted (was garbled/non-ASCII).

    def get_cost_basis(self, ticker):
        """def get_cost_basis: docstring omitted (was garbled/non-ASCII)."""
        return self.cost_basis.get(ticker, None)
    
    def compute_benchmark_returns(self, tickers, evaluation_days=10):
        """def compute_benchmark_returns: docstring omitted (was garbled/non-ASCII)."""
        bench_returns = {}
        
        for ticker in tickers:
            try:
                # NOTE: comment omitted (was garbled/non-ASCII).
                hist = self.get_market_data(ticker, period='1mo', interval='1d')
                
                if hist is None or len(hist) < evaluation_days + 1:
                    print(f"[BENCHMARK] {ticker}: insufficient data (need {evaluation_days+1} days)")
                    continue
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                latest_close = hist['Close'].iloc[-1]
                past_close = hist['Close'].iloc[-(evaluation_days + 1)]
                
                ret = (latest_close - past_close) / past_close
                bench_returns[ticker] = float(ret)
                
                print(f"[BENCHMARK] {ticker}: {ret:.2%} over {evaluation_days} days")
                
            except Exception as e:
                print(f"[BENCHMARK] {ticker}: error - {e}")
                continue
        
        if not bench_returns:
            print("[BENCHMARK] No valid benchmark data")
            return {}, 0.0, 0.0
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        returns_list = list(bench_returns.values())
        bench_avg_return = float(np.mean(returns_list))
        bench_dispersion = float(np.std(returns_list))
        
        print(f"[BENCHMARK] Average: {bench_avg_return:.2%}, Dispersion: {bench_dispersion:.2%}")
        
        return bench_returns, bench_avg_return, bench_dispersion
    
    def compute_regime_state(self):
        """def compute_regime_state: docstring omitted (was garbled/non-ASCII)."""
        regime_config = self.config.get('regime_filter', {})
        min_cash_pct = float(self.config['objectives']['min_cash_pct'])

        # NOTE: comment omitted (was garbled/non-ASCII).
        if self.forced_until_time is not None:
            now = datetime.now()
            if now < self.forced_until_time:
                dynamic_min_cash = max(float(regime_config.get('cash_risk_off', min_cash_pct)), min_cash_pct)
                dynamic_max_weight = regime_config.get('max_weight_risk_off', self.config['objectives']['max_weight_per_asset'])
                regime_details = {
                    'forced_until_time': self.forced_until_time.isoformat(),
                    'forced_reason': self.forced_regime_reason
                }
                print(f"\n[REGIME] FORCE MODE: RISK_OFF_FORCED until {self.forced_until_time.strftime('%Y-%m-%d %H:%M:%S')}")
                return 'risk_off_forced', 0.0, regime_details, dynamic_min_cash, dynamic_max_weight
            else:
                print(f"\n[REGIME] FORCE MODE EXPIRED at {self.forced_until_time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.forced_until_time = None
                self.forced_regime_reason = ""

        if not self.config.get('regime_filter', {}).get('enabled', False):
            # NOTE: comment omitted (was garbled/non-ASCII).
            return 'neutral', 0.5, {}, self.config['objectives']['min_cash_pct'], self.config['objectives']['max_weight_per_asset']

        regime_config = self.config['regime_filter']
        ma_window = regime_config.get('ma_window', 50)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        bench_tickers = self.config.get('benchmarks', {}).get('tickers', ['QQQ', 'SPY', 'VTI', 'DIA'])
        
        print(f"\n[REGIME] Computing market regime using MA{ma_window}...")
        
        regime_details = {}
        above_ma_count = 0
        valid_count = 0
        
        for ticker in bench_tickers:
            try:
                # NOTE: comment omitted (was garbled/non-ASCII).
                hist = self.get_market_data(ticker, period='3mo', interval='1d')
                
                if hist is None or len(hist) < ma_window:
                    print(f"[REGIME] {ticker}: insufficient data for MA{ma_window}")
                    continue
                
                # NOTE: comment omitted (was garbled/non-ASCII).
                ma50 = hist['Close'].rolling(window=ma_window).mean()
                latest_close = float(hist['Close'].iloc[-1])
                latest_ma50 = float(ma50.iloc[-1])
                
                above_ma = latest_close > latest_ma50
                
                regime_details[ticker] = {
                    'close': latest_close,
                    'ma50': latest_ma50,
                    'above_ma': above_ma
                }
                
                valid_count += 1
                if above_ma:
                    above_ma_count += 1
                
                status = "ABOVE" if above_ma else "BELOW"
                print(f"[REGIME] {ticker}: ${latest_close:.2f} vs MA50 ${latest_ma50:.2f} {status}")
                
            except Exception as e:
                print(f"[REGIME] {ticker}: error - {e}")
                continue
        
        if valid_count == 0:
            print("[REGIME] No valid data, defaulting to neutral")
            return 'neutral', 0.5, {}, min_cash_pct, self.config['objectives']['max_weight_per_asset']
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        trend_score = above_ma_count / valid_count
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        risk_on_threshold = regime_config.get('trend_score_risk_on', 0.75)
        risk_off_threshold = regime_config.get('trend_score_risk_off', 0.50)
        
        if trend_score >= risk_on_threshold:
            regime_state = 'risk_on'
        elif trend_score <= risk_off_threshold:
            regime_state = 'risk_off'
        else:
            regime_state = 'neutral'
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        dynamic_min_cash = max(float(regime_config.get(f'cash_{regime_state}', min_cash_pct)), min_cash_pct)
        
        if regime_state == 'risk_off':
            dynamic_max_weight = regime_config.get('max_weight_risk_off', 0.20)
        else:
            dynamic_max_weight = self.config['objectives']['max_weight_per_asset']
        
        print(f"\n[REGIME] Trend Score: {trend_score:.2%} ({above_ma_count}/{valid_count} above MA{ma_window})")
        print(f"[REGIME] Market State: {regime_state.upper()}")
        print(f"[REGIME] Dynamic Min Cash: {dynamic_min_cash:.1%} (was {self.config['objectives']['min_cash_pct']:.1%})")
        print(f"[REGIME] Dynamic Max Weight: {dynamic_max_weight:.1%} (was {self.config['objectives']['max_weight_per_asset']:.1%})")
        
        return regime_state, trend_score, regime_details, dynamic_min_cash, dynamic_max_weight

    def run_cycle(self):
        """def run_cycle: docstring omitted (was garbled/non-ASCII)."""
        now = self._now()
        now_local = now.astimezone()
        print(f"\n{'='*60}")
        print(f"Cycle {self.current_cycle} - {now_local.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        self._refresh_market_session_state(now)

        # Macro refresh decoupling
        last_macro_ref = self._coerce_datetime_utc(self.last_macro_time)
        if isinstance(last_macro_ref, datetime) and self.last_macro_time is not last_macro_ref:
            self.last_macro_time = last_macro_ref
        if last_macro_ref is None:
            self.refresh_macro_cache(now=now)
            self.current_macro_reused = False
            print(f"[MACRO REFRESH] First refresh completed")
        else:
            macro_elapsed = (now - last_macro_ref).total_seconds() / 60
            if macro_elapsed >= self.macro_refresh_minutes:
                self.refresh_macro_cache(now=now)
                self.current_macro_reused = False
                print(f"[MACRO REFRESH] Refreshed after {macro_elapsed:.1f} minutes")
            else:
                self.current_macro_reused = True
                self._sync_current_macro_from_cache()
                print(f"[MACRO REFRESH] Reused cached macro ({macro_elapsed:.1f}m < {self.macro_refresh_minutes}m)")

        # Signal refresh decoupling (target weights)
        last_signal_ref = self._coerce_datetime_utc(self.last_signal_time)
        if isinstance(last_signal_ref, datetime) and self.last_signal_time is not last_signal_ref:
            self.last_signal_time = last_signal_ref
        if not self.cached_target_weights or last_signal_ref is None:
            target_weights = self.calculate_target_weights()
            self.cached_target_weights = dict(target_weights)
            self.last_signal_time = now
            self.current_weights_reused = False
            print("[SIGNAL REFRESH] First target weight calculation completed")
        else:
            signal_elapsed = (now - last_signal_ref).total_seconds() / 60
            if signal_elapsed >= self.signal_refresh_minutes:
                target_weights = self.calculate_target_weights()
                self.cached_target_weights = dict(target_weights)
                self.last_signal_time = now
                self.current_weights_reused = False
                print(f"[SIGNAL REFRESH] Recalculated target weights after {signal_elapsed:.1f} minutes")
            else:
                target_weights = dict(self.cached_target_weights)
                self.current_weights_reused = True
                print(f"[SIGNAL REFRESH] Reusing target weights ({signal_elapsed:.1f}m < {self.signal_refresh_minutes}m)")

        snapshot = self.record_snapshot()

        print(f"Cash: ${snapshot['cash']:,.2f}")
        print(f"Positions Value: ${snapshot['positions_value']:,.2f}")
        print(f"Total Equity: ${snapshot['total_equity']:,.2f}")
        print(f"Return: {snapshot['total_return']:.2%}")
        print(f"Drawdown: {snapshot['drawdown']:.2%}")
        print(f"Status: {snapshot['status']}")

        if snapshot.get('regime_state'):
            regime_flag = "RISK-ON" if snapshot['regime_state'] == 'risk_on' else "NEUTRAL" if snapshot['regime_state'] == 'neutral' else "RISK-OFF"
            risk_caps = " [RISK CAPS]" if snapshot.get('risk_caps_applied') else ""
            print(f"Market Regime: {regime_flag} (trend: {snapshot['trend_score']:.1%}){risk_caps}")

        if snapshot['positions']:
            print(f"\nCurrent Holdings:")
            print(f"{'Ticker':<8} {'Qty':>6} {'Price':>10} {'Value':>12} {'Weight':>8} {'P&L':>10}")
            print("-" * 60)

            for ticker, pos in sorted(snapshot['positions'].items(), key=lambda x: x[1]['value'], reverse=True):
                qty = pos['quantity']
                current_price = pos['price']
                value = pos['value']
                weight = value / snapshot['total_equity'] * 100

                cost_basis = self.get_cost_basis(ticker)
                if cost_basis:
                    pnl = (current_price - cost_basis) / cost_basis * 100
                    pnl_str = f"{pnl:+.2f}%"
                    pnl_color = "UP" if pnl > 0 else "DOWN" if pnl < 0 else "FLAT"
                else:
                    pnl_str = "N/A"
                    pnl_color = "NA"

                print(f"{ticker:<8} {qty:>6} ${current_price:>9.2f} ${value:>11,.2f} {weight:>7.1f}% {pnl_color} {pnl_str:>8}")

            print("-" * 60)

        if self.check_risk_controls():
            print("[WARN] Risk control triggered (risk_off_forced active), skipping normal rebalance")
            self.current_cycle += 1
            return

        print("\nTarget Weights:")
        for ticker, weight in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.01:
                print(f"  {ticker}: {weight:.2%}")

        print("\nExecuting rebalance...")
        trades = self.execute_rebalance(target_weights)

        if trades:
            print(f"Executed {len(trades)} trades:")
            for trade in trades:
                print(f"  {trade['side']} {trade['quantity']} {trade['ticker']} @ ${trade['price']:.2f} (cost: ${trade['cost']:.2f})")
        else:
            if self.current_rebalance_skipped_reason == 'market_closed_gate':
                session_state = str((self.current_rebalance_gate or {}).get('session_state', 'UNKNOWN')).upper() if isinstance(self.current_rebalance_gate, dict) else 'UNKNOWN'
                print(f"No trades executed (market_closed_gate, session={session_state})")
            elif self.current_rebalance_skipped_reason == 'attempt_cooldown':
                print("No trades executed (attempt_cooldown)")
            else:
                print("No trades executed (portfolio already balanced)")

        self.current_cycle += 1

    def run(self):
        """def run: docstring omitted (was garbled/non-ASCII)."""
        print("\n" + "="*60)
        print("Starting Paper Trading Simulation")
        print("="*60)
        print("WARNING: SIMULATION ONLY - NO REAL MONEY")
        print("WARNING: NO BROKER CONNECTION")
        print("="*60 + "\n")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        print("="*60)
        print("ENGINE VERSION FINGERPRINT")
        print("="*60)
        print(f"ENGINE_VERSION: v3.2.2-2026-02-13")
        print(f"HAS_MACRO_SMOOTH: {hasattr(self, 'macro_risk_score_history')}")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        try:
            test_result = self.get_current_price("QQQ")
            is_tuple = isinstance(test_result, tuple) and len(test_result) == 3
            print(f"PRICE_API_RETURNS_TUPLE: {is_tuple}")
            if is_tuple:
                print(f"  Sample: get_current_price('QQQ') = (price={test_result[0]}, age={test_result[1]}, status='{test_result[2]}')")
        except Exception as e:
            print(f"PRICE_API_RETURNS_TUPLE: False (Error: {e})")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        print(f"HAS_STALE_PRICE_SKIP: {hasattr(self, 'current_stale_info')}")
        print(f"HAS_TURNOVER_CAP: {hasattr(self, 'current_turnover_info')}")
        print(f"HAS_MACRO_COOLDOWN: {hasattr(self, 'macro_cooldown_remaining')}")
        print(f"HAS_REGIME_FILTER: {'regime_filter' in self.config}")
        print(f"HAS_MACRO_INTEGRATION: {self.macro_adapter.enabled if hasattr(self, 'macro_adapter') else False}")
        print("="*60 + "\n")
        
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=self.config['duration_hours'])
        self.status = "RUNNING"
        
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {self.config['duration_hours']} hours")
        print(f"Rebalance Interval: {self.config['rebalance_minutes']} minutes")
        
        try:
            while datetime.now() < self.end_time:
                self.run_cycle()
                self._maybe_generate_daily_report()
                
                sleep_seconds = self.config['rebalance_minutes'] * 60
                
                if datetime.now() + timedelta(seconds=sleep_seconds) >= self.end_time:
                    print(f"\n[INFO] Approaching end time, running final cycle...")
                    break
                
                print(f"\nSleeping for {self.config['rebalance_minutes']} minutes...")
                print(f"   Next cycle at: {(datetime.now() + timedelta(seconds=sleep_seconds)).strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"[DEBUG] About to sleep at {datetime.now().strftime('%H:%M:%S')}")
                print(f"[DEBUG] Sleep duration: {sleep_seconds} seconds")
                import sys; sys.stdout.flush()  # NOTE: comment omitted (was garbled/non-ASCII).
                time.sleep(sleep_seconds)
                print(f"[DEBUG] Woke up at {datetime.now().strftime('%H:%M:%S')}")
                import sys; sys.stdout.flush()  # NOTE: comment omitted (was garbled/non-ASCII).
            
            print(f"\n{'='*60}")
            print("Final Snapshot")
            print(f"{'='*60}")
            self.run_cycle()
            self._maybe_generate_daily_report()
            
            self.status = "COMPLETED"
            
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user")
            self.status = "INTERRUPTED"
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            self.status = "ERROR"
        finally:
            self.save_results()

    def save_results(self):
        """def save_results: docstring omitted (was garbled/non-ASCII)."""
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        trades_path = self.config['reporting']['trades_log_path']
        if self.trades_log:
            trades_df = pd.DataFrame(self.trades_log)
            trades_df.to_csv(trades_path, index=False)
            print(f"[OK] Trades log saved: {trades_path}")
        else:
            # NOTE: comment omitted (was garbled/non-ASCII).
            pd.DataFrame(columns=['timestamp', 'ticker', 'side', 'quantity', 'price', 'cost', 'reason']).to_csv(trades_path, index=False)
            print(f"[OK] Trades log saved (empty): {trades_path}")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        snapshots_path = self.config['reporting']['portfolio_snapshots_path']
        snapshots_content = ''.join(f"{json.dumps(snapshot)}\n" for snapshot in self.portfolio_snapshots)
        self.atomic_write_text(snapshots_path, snapshots_content)
        print(f"[OK] Portfolio snapshots saved: {snapshots_path}")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.generate_equity_curve()
        self.generate_summary_report()
    
    def generate_equity_curve(self):
        """def generate_equity_curve: docstring omitted (was garbled/non-ASCII)."""
        if not self.equity_curve:
            print("[WARN] No equity curve data to plot")
            return
        
        timestamps = [ec[0] for ec in self.equity_curve]
        equity = [ec[1] for ec in self.equity_curve]
        cash = [ec[2] for ec in self.equity_curve]
        positions = [ec[3] for ec in self.equity_curve]
        
        drawdowns = []
        peak = equity[0]
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            drawdowns.append(dd)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(timestamps, equity, label='Total Equity', linewidth=2, color='blue')
        ax1.plot(timestamps, cash, label='Cash', linewidth=1, linestyle='--', color='green')
        ax1.plot(timestamps, positions, label='Positions Value', linewidth=1, linestyle='--', color='orange')
        ax1.axhline(y=self.initial_cash, color='red', linestyle=':', label='Initial Cash')
        ax1.set_ylabel('Value (USD)', fontsize=12)
        ax1.set_title('Paper Trading Equity Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(timestamps, 0, drawdowns, color='red', alpha=0.3)
        ax2.plot(timestamps, drawdowns, color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_title('Drawdown', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        curve_path = self.config['reporting']['equity_curve_path']
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Equity curve saved: {curve_path}")
        
        plt.close()
    
    def generate_summary_report(self):
        """def generate_summary_report: docstring omitted (was garbled/non-ASCII)."""
        if not self.portfolio_snapshots:
            print("[WARN] No snapshots to generate report")
            return
        
        final_snapshot = self.portfolio_snapshots[-1]
        
        total_return = final_snapshot['total_return']
        max_drawdown = max(s['drawdown'] for s in self.portfolio_snapshots)
        
        returns = []
        for i in range(1, len(self.portfolio_snapshots)):
            prev_equity = self.portfolio_snapshots[i-1]['total_equity']
            curr_equity = self.portfolio_snapshots[i]['total_equity']
            ret = (curr_equity - prev_equity) / prev_equity
            returns.append(ret)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        report_path = self.config['reporting']['summary_report_path']
        
        with io.StringIO() as f:
            f.write("="*60 + "\n")
            f.write("GlobalWatch Paper Trading Summary Report\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Simulation Period:\n")
            f.write(f"  Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Duration: {self.config['duration_hours']} hours\n")
            f.write(f"  Cycles: {self.current_cycle}\n")
            f.write(f"  Status: {self.status}\n\n")
            
            f.write(f"Performance:\n")
            f.write(f"  Initial Cash: ${self.initial_cash:,.2f}\n")
            f.write(f"  Final Equity: ${final_snapshot['total_equity']:,.2f}\n")
            f.write(f"  Total Return: {total_return:.2%}\n")
            f.write(f"  Max Drawdown: {max_drawdown:.2%}\n")
            f.write(f"  Sharpe Ratio: {sharpe:.2f}\n\n")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if final_snapshot.get('regime_state'):
                f.write(f"Final Market Regime:\n")
                f.write(f"  State: {final_snapshot['regime_state'].upper()}")
                
                if final_snapshot.get('risk_caps_applied'):
                    f.write(" [RISK CAPS ACTIVE]\n")
                else:
                    f.write("\n")
                
                f.write(f"  Trend Score: {final_snapshot['trend_score']:.1%}\n")
                f.write(f"  Dynamic Min Cash: {final_snapshot['dynamic_min_cash']:.1%}\n")
                f.write(f"  Dynamic Max Weight: {final_snapshot['dynamic_max_weight']:.1%}\n\n")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if final_snapshot.get('macro_risk_score', 0) > 0:
                f.write(f"Final Macro Signals (GlobalWatch):\n")
                f.write(f"  Risk Score: {final_snapshot['macro_risk_score']:.1f}/10.0\n")
                f.write(f"  Confirmed Topics: {final_snapshot.get('confirmed_topics_count', 0)}\n")
                
                if final_snapshot.get('macro_tilts'):
                    f.write(f"  Active Tilts:\n")
                    for ticker, tilt in final_snapshot['macro_tilts'].items():
                        f.write(f"    {ticker}: {tilt:+.2%}\n")
                f.write("\n")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            if final_snapshot.get('bench_returns'):
                f.write(f"Benchmark Comparison:\n")
                f.write(f"  Strategy Return: {total_return:.2%}\n")
                f.write(f"  Benchmark Avg Return: {final_snapshot['bench_avg_return']:.2%}\n")
                f.write(f"  Excess Return: {final_snapshot['excess_return']:.2%}")
                
                if final_snapshot['win_flag']:
                    f.write(" [OUTPERFORM]\n")
                else:
                    f.write(" [UNDERPERFORM]\n")
                
                f.write(f"  Benchmark Dispersion: {final_snapshot['bench_dispersion']:.2%}\n\n")
                
                f.write(f"  Individual Benchmarks:\n")
                for ticker, ret in sorted(final_snapshot['bench_returns'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"    {ticker}: {ret:.2%}\n")
                f.write("\n")
            
            f.write(f"Final Portfolio:\n")
            f.write(f"  Cash: ${final_snapshot['cash']:,.2f} ({final_snapshot['cash']/final_snapshot['total_equity']:.1%})\n")
            f.write(f"  Positions Value: ${final_snapshot['positions_value']:,.2f}\n\n")
            
            if final_snapshot['positions']:
                f.write(f"  Holdings:\n")
                for ticker, pos in sorted(final_snapshot['positions'].items(), key=lambda x: x[1]['value'], reverse=True):
                    weight = pos['value'] / final_snapshot['total_equity']
                    f.write(f"    {ticker}: {pos['quantity']} shares @ ${pos['price']:.2f} = ${pos['value']:,.2f} ({weight:.1%})\n")
            
            f.write(f"\nTrading Activity:\n")
            f.write(f"  Total Trades: {len(self.trades_log)}\n")
            
            if self.trades_log:
                total_cost = sum(t['cost'] for t in self.trades_log)
                f.write(f"  Total Transaction Costs: ${total_cost:,.2f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("[SIMULATION ONLY] NO REAL MONEY\n")
            f.write("[DISCLAIMER] Past performance does not guarantee future results\n")
            f.write("="*60 + "\n")
            report_content = f.getvalue()
        self.atomic_write_text(report_path, report_content)
        
        print(f"[OK] Summary report saved: {report_path}")
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Initial Cash: ${self.initial_cash:,.2f}")
        print(f"Final Equity: ${final_snapshot['total_equity']:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        if final_snapshot.get('bench_returns'):
            print(f"\nBenchmark Comparison:")
            print(f"  Strategy: {total_return:.2%}")
            print(f"  Benchmark Avg: {final_snapshot['bench_avg_return']:.2%}")
            print(f"  Excess Return: {final_snapshot['excess_return']:.2%}", end="")
            if final_snapshot['win_flag']:
                print(" [OK] OUTPERFORM")
            else:
                print(" [WARN] UNDERPERFORM")
        
        print(f"\nTotal Trades: {len(self.trades_log)}")
        print(f"Status: {self.status}")
        print(f"{'='*60}\n")


def debug_run_planner_once(config_path: str = "paper_config.json", turnover_limit: float | None = None):
    """Run a deterministic planner dry-run without market data dependencies."""
    cfg = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f) if f else {}
    except Exception as e:
        print(f"[DEBUG PLANNER] Failed to load config {config_path}: {e}")
        cfg = {}

    trade_planner_cfg = cfg.get("trade_planner", {}) if isinstance(cfg, dict) else {}
    execution_cfg = cfg.get("execution", {}) if isinstance(cfg, dict) else {}
    if not isinstance(trade_planner_cfg, dict):
        trade_planner_cfg = {}
    if not isinstance(execution_cfg, dict):
        execution_cfg = {}

    equity = float(trade_planner_cfg.get("debug_equity", cfg.get("initial_cash_usd", 30000.0) if isinstance(cfg, dict) else 30000.0) or 30000.0)
    max_turnover_pct = float(execution_cfg.get("max_turnover_pct_per_rebalance", 0.20) or 0.20)
    budget = float(turnover_limit if turnover_limit is not None else (equity * max_turnover_pct))
    budget = max(0.0, budget)

    allow_partial_fill = bool(trade_planner_cfg.get("allow_partial_fill", True))
    min_trade_notional = float(trade_planner_cfg.get("min_trade_notional", 5.0) or 5.0)
    enable_cost_sensitive_ranking = bool(trade_planner_cfg.get("enable_cost_sensitive_ranking", False))

    synthetic = [
        {"ticker": "AAA", "side": "SELL", "desired_trade_value": 260.0, "is_forced": True, "priority": "forced", "planner_score": 0.60},
        {"ticker": "BBB", "side": "SELL", "desired_trade_value": 180.0, "is_forced": True, "priority": "forced", "planner_score": 0.55},
        {"ticker": "CCC", "side": "BUY", "desired_trade_value": 240.0, "is_forced": False, "priority": "normal", "planner_score": 0.90},
        {"ticker": "DDD", "side": "BUY", "desired_trade_value": 170.0, "is_forced": False, "priority": "normal", "planner_score": 0.85},
        {"ticker": "EEE", "side": "BUY", "desired_trade_value": 140.0, "is_forced": False, "priority": "normal", "planner_score": 0.40},
        {"ticker": "FFF", "side": "SELL", "desired_trade_value": 120.0, "is_forced": False, "priority": "normal", "planner_score": 0.75},
        {"ticker": "GGG", "side": "BUY", "desired_trade_value": 110.0, "is_forced": False, "priority": "normal", "planner_score": 0.70},
        {"ticker": "HHH", "side": "SELL", "desired_trade_value": 95.0, "is_forced": False, "priority": "normal", "planner_score": 0.65},
        {"ticker": "III", "side": "BUY", "desired_trade_value": 80.0, "is_forced": False, "priority": "normal", "planner_score": 0.50},
        {"ticker": "JJJ", "side": "BUY", "desired_trade_value": 60.0, "is_forced": False, "priority": "normal", "planner_score": 0.45},
    ]
    for t in synthetic:
        t["delta_weight"] = (float(t["desired_trade_value"]) / equity) if equity > 0 else 0.0

    pre_notional = float(sum(abs(float(t.get("desired_trade_value", 0.0) or 0.0)) for t in synthetic))
    forced = [dict(t) for t in synthetic if bool(t.get("is_forced", False)) or str(t.get("priority", "")).lower() == "forced"]
    normal = [dict(t) for t in synthetic if t not in forced]

    forced.sort(key=lambda x: (-abs(float(x.get("desired_trade_value", 0.0) or 0.0)), str(x.get("ticker", ""))))
    if enable_cost_sensitive_ranking:
        normal.sort(
            key=lambda x: (
                -float(x.get("planner_score", 0.0) or 0.0),
                -abs(float(x.get("desired_trade_value", 0.0) or 0.0)),
                str(x.get("ticker", "")),
            )
        )
    else:
        normal.sort(key=lambda x: (-abs(float(x.get("desired_trade_value", 0.0) or 0.0)), str(x.get("ticker", ""))))

    queue = forced + normal
    chosen = []
    dropped = []
    scaled = []
    remaining = float(budget)

    for item in queue:
        notional = abs(float(item.get("desired_trade_value", 0.0) or 0.0))
        if remaining <= 1e-12:
            dropped.append({"ticker": item.get("ticker"), "side": item.get("side"), "reason": "over_budget"})
            continue
        if notional <= remaining + 1e-12:
            chosen.append(dict(item))
            remaining -= notional
            continue

        if allow_partial_fill and remaining >= min_trade_notional:
            scale = float(remaining / notional) if notional > 0 else 0.0
            partial = dict(item)
            partial["desired_trade_value"] = float(remaining)
            chosen.append(partial)
            scaled.append(
                {
                    "ticker": item.get("ticker"),
                    "side": item.get("side"),
                    "old_notional": float(notional),
                    "new_notional": float(remaining),
                    "scale": float(scale),
                    "reason": "budget_scale",
                }
            )
            remaining = 0.0
        else:
            reason = "over_budget_partial_below_min" if allow_partial_fill else "over_budget"
            dropped.append({"ticker": item.get("ticker"), "side": item.get("side"), "reason": reason})
            remaining = 0.0

    used_total = float(sum(abs(float(t.get("desired_trade_value", 0.0) or 0.0)) for t in chosen))
    print("[DEBUG PLANNER] deterministic dry-run (no market data)")
    print(
        f"[DEBUG PLANNER] pre_notional=${pre_notional:,.2f} limit=${budget:,.2f} "
        f"used=${used_total:,.2f} keep={len(chosen)} dropped={len(dropped)} scaled={len(scaled)}"
    )
    print(
        f"[DEBUG PLANNER] rank_mode={'planner_score' if enable_cost_sensitive_ranking else 'notional'} "
        f"allow_partial_fill={str(bool(allow_partial_fill)).lower()} min_trade_notional={min_trade_notional:.2f}"
    )
    print("[DEBUG PLANNER] chosen:")
    for t in chosen:
        print(
            "  - "
            + json.dumps(
                {
                    "ticker": t.get("ticker"),
                    "side": t.get("side"),
                    "notional": round(float(t.get("desired_trade_value", 0.0) or 0.0), 4),
                    "forced": bool(t.get("is_forced", False)),
                    "planner_score": t.get("planner_score"),
                },
                ensure_ascii=False,
            )
        )
    print("[DEBUG PLANNER] dropped:")
    for d in dropped:
        print("  - " + json.dumps(d, ensure_ascii=False))
    print("[DEBUG PLANNER] scaled:")
    for s in scaled:
        print("  - " + json.dumps(s, ensure_ascii=False))

    return {
        "pre_notional": pre_notional,
        "limit": budget,
        "used_total": used_total,
        "chosen": chosen,
        "dropped": dropped,
        "scaled": scaled,
    }


def debug_run_system_s1_s5(config_path: str | None = None, outdir: str = "outputs/gw_dryrun", turnover_limit: float | None = None) -> int:
    """Offline deterministic acceptance dry-run for S1..S5."""
    pass_count = 0
    fail_count = 0

    def _check(check_id: str, condition: bool, message: str):
        nonlocal pass_count, fail_count
        if bool(condition):
            pass_count += 1
            print(f"[PASS] {check_id} {message}")
        else:
            fail_count += 1
            print(f"[FAIL] {check_id} {message}")

    outdir_abs = os.path.abspath(str(outdir or "outputs/gw_dryrun"))
    if os.path.exists(outdir_abs):
        shutil.rmtree(outdir_abs, ignore_errors=True)
    os.makedirs(outdir_abs, exist_ok=True)

    base_path = str(config_path or "paper_config.json")
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    cfg.setdefault("paper_mode", True)
    cfg.setdefault("safety", {})
    cfg["safety"].setdefault("no_real_broker", True)
    cfg["safety"].setdefault("simulation_only", True)
    cfg["safety"].setdefault("random_seed", 42)
    cfg.setdefault("initial_cash_usd", 30000.0)
    cfg.setdefault("duration_hours", 1)
    cfg.setdefault("rebalance_minutes", 1)
    cfg.setdefault("universe", ["AAA", "BBB", "CCC"])
    cfg.setdefault("strategy", {"lookback_days": 40})
    cfg.setdefault("benchmarks", {"tickers": [], "evaluation_days": 10})
    cfg["benchmarks"]["tickers"] = []

    obj = cfg.setdefault("objectives", {})
    obj.setdefault("min_cash_pct", 0.05)
    obj.setdefault("max_weight_per_asset", 0.5)
    obj.setdefault("max_drawdown_pct", 0.50)
    obj.setdefault("transaction_cost_pct", 0.001)

    execution_cfg = cfg.setdefault("execution", {})
    execution_cfg["rebalance_cooldown_minutes"] = 0
    execution_cfg["rebalance_attempt_cooldown_minutes"] = 10
    execution_cfg["max_stale_ratio"] = 0.3
    execution_cfg["stale_price_skip_minutes"] = 60
    execution_cfg["min_trade_notional_usd"] = 1
    execution_cfg["weight_threshold"] = 0.0
    execution_cfg["enable_exit_signals"] = False
    execution_cfg["max_turnover_pct_per_rebalance"] = 1.0
    execution_cfg["max_portfolio_volatility"] = 999.0
    execution_cfg["portfolio_vol_min_coverage"] = 1.0
    execution_cfg["enable_diversity_check"] = False
    if turnover_limit is not None:
        try:
            eq = float(cfg.get("initial_cash_usd", 30000.0) or 30000.0)
            execution_cfg["max_turnover_pct_per_rebalance"] = max(0.0, float(turnover_limit) / max(eq, 1.0))
        except Exception:
            pass
    stale_policy = execution_cfg.setdefault("price_stale_policy", {})
    stale_policy["allow_buy"] = ["LIVE", "RECENT"]
    stale_policy["allow_sell"] = ["LIVE", "RECENT", "STALE"]

    macro_cfg = cfg.setdefault("macro_integration", {})
    macro_cfg["enabled"] = False
    macro_cfg["enable_llm_topic_signals"] = False

    risk_cfg = cfg.setdefault("risk_model", {})
    risk_cfg["enable_cov_diagnostics"] = False
    risk_cfg["use_cov_vol_for_gate"] = False

    reporting_cfg = cfg.setdefault("reporting", {})
    reporting_cfg["snapshot_live_path"] = os.path.join(outdir_abs, "snapshot_live.json")
    reporting_cfg["trade_history_path"] = os.path.join(outdir_abs, "trade_history.jsonl")
    reporting_cfg["trades_log_path"] = os.path.join(outdir_abs, "paper_trades.csv")
    reporting_cfg["portfolio_snapshots_path"] = os.path.join(outdir_abs, "portfolio_snapshots.jsonl")
    reporting_cfg["summary_report_path"] = os.path.join(outdir_abs, "paper_summary.txt")
    reporting_cfg["equity_curve_path"] = os.path.join(outdir_abs, "equity_curve.png")
    reporting_cfg["scoreboard_path"] = os.path.join(outdir_abs, "scoreboard.jsonl")
    reporting_cfg["daily_report_dirs"] = [os.path.join(outdir_abs, "Daily Report")]
    reporting_cfg["max_price_debug_items"] = 50
    reporting_cfg["account_id"] = "paper_main"
    reporting_cfg["env"] = "live"

    dryrun_config_path = os.path.join(outdir_abs, "dryrun_config.json")
    io_atomic_write_json(dryrun_config_path, cfg, indent=2)

    old_checkpoint_env = os.environ.get("GW_CHECKPOINT_ACTION")
    old_session_env = os.environ.get("GW_SESSION_ID")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    os.environ["GW_SESSION_ID"] = "DRYRUN-S1S5"
    try:
        engine = PaperTradingEngine(dryrun_config_path)
    finally:
        if old_checkpoint_env is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint_env
        if old_session_env is None:
            os.environ.pop("GW_SESSION_ID", None)
        else:
            os.environ["GW_SESSION_ID"] = old_session_env

    engine.set_market_data_fetcher(lambda *args, **kwargs: None)

    def _risk_gate_stub(target_weights):
        invested_budget = 0.0
        if isinstance(target_weights, dict):
            invested_budget = float(sum(float(v) for k, v in target_weights.items() if str(k).upper() != "CASH"))
        return {
            "abort": False,
            "abort_reason": "",
            "weighted_volatility": 0.0,
            "max_portfolio_volatility": 999.0,
            "volatility_known_weight": 0.0,
            "volatility_confident": False,
            "min_coverage": 1.0,
            "enable_diversity_check": False,
            "herfindahl_index": 0.0,
            "max_herfindahl_index": 1.0,
            "invested_budget": invested_budget,
            "asset_volatility_map": {},
            "cov_risk_diag": {"enabled": False, "status": "disabled"},
            "gate_vol_method": "weighted_fallback",
            "cov_gate_used": False,
            "cov_gate_coverage": None,
            "cov_gate_vol": None,
            "cov_gate_max_rc": None,
            "cov_gate_pass": None,
            "cov_gate_reason": "disabled",
            "rc_limit": 1.0,
            "min_cov_gate_coverage": 1.0,
            "use_cov_vol_for_gate": False,
            "cov_gate_fallback_to_weighted": True,
        }

    engine._evaluate_portfolio_risk_gate = _risk_gate_stub

    class StubPriceProvider:
        def __init__(self):
            self.mode = "live_buy"
            self.base_now = datetime(2026, 2, 10, 15, 30, tzinfo=timezone.utc)

        def set_mode(self, mode):
            self.mode = str(mode or "live_buy")

        def _mk_debug(self, ticker, status, age_min, source, tz_ok=True, notes=None):
            age_val = float(age_min) if age_min is not None else 0.0
            price_ts = (self.base_now - timedelta(minutes=age_val)).isoformat()
            return {
                "ticker": str(ticker).upper(),
                "now_ts": self.base_now.isoformat(),
                "status": str(status).upper(),
                "age_min": float(age_min) if age_min is not None else None,
                "source": str(source),
                "price_ts": price_ts,
                "tz_ok": bool(tz_ok),
                "thresholds": {"live_max_min": 10.0, "recent_max_min": 60.0},
                "notes": notes,
                "bar_interval": "1m",
                "raw_price_ts": price_ts,
                "raw_tz": "UTC",
            }

        def __call__(self, ticker=None, **kwargs):
            t = str((ticker if ticker is not None else kwargs.get("ticker", ""))).upper()
            if not t:
                t = "UNKNOWN"

            if self.mode in ("stale_buy", "stale_sell"):
                price = {"AAA": 100.0, "BBB": 110.0, "CCC": 120.0}.get(t, 90.0)
                notes = "naive_ts_detected;localized_assumption=America/New_York" if t == "AAA" else None
                tz_ok = False if t == "AAA" else True
                dbg = self._mk_debug(t, "STALE", 180.0, "stub_stale", tz_ok=tz_ok, notes=notes)
                return (price, 180.0, "STALE", dbg)

            if self.mode == "live_buy":
                price = {"AAA": 100.0, "BBB": 110.0, "CCC": 120.0}.get(t, 95.0)
                dbg = self._mk_debug(t, "LIVE", 1.0, "stub_live", tz_ok=True, notes=None)
                return (price, 1.0, "LIVE", dbg)

            dbg = self._mk_debug(t, "MISSING", None, "stub_missing", tz_ok=False, notes="missing")
            return (None, 99999.0, "MISSING", dbg)

    provider = StubPriceProvider()
    engine.set_price_fetcher(provider)

    now_holder = {"value": datetime(2026, 2, 10, 8, 0, 0)}
    engine._debug_now_override = lambda: now_holder["value"]

    def _make_session(state, open_grace_passed=False, close_grace_passed=False):
        now_local = now_holder["value"]
        if now_local.tzinfo is None or now_local.tzinfo.utcoffset(now_local) is None:
            now_local = now_local.replace(tzinfo=timezone.utc)
        now_utc = now_local.astimezone(timezone.utc)
        market_tz = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone.utc
        now_et = now_utc.astimezone(market_tz)
        return {
            "state": str(state),
            "now_et": now_et.isoformat(),
            "now_utc": now_utc.isoformat(),
            "trading_date_et": "2026-02-10",
            "last_completed_trading_date_et": "2026-02-09",
            "open_grace_passed": bool(open_grace_passed),
            "close_grace_passed": bool(close_grace_passed),
            "open_grace_min": 15,
            "close_grace_min": 10,
        }

    def _run_case(*, session, mode, positions, cash, target_weights):
        engine._debug_session_override = dict(session)
        provider.set_mode(mode)
        engine.positions = {str(k).upper(): int(v) for k, v in (positions or {}).items() if int(v) > 0}
        engine.cash = float(cash)
        for t in list(engine.positions.keys()):
            engine.cost_basis.setdefault(t, 100.0)
        trades = engine.execute_rebalance(dict(target_weights))
        snapshot = engine.record_snapshot()
        engine.save_trade_history_jsonl()
        return trades, snapshot

    # Session timezone sanity probe (UTC -> ET conversion correctness).
    session_probe = get_market_session_state(
        datetime(2026, 2, 10, 12, 0, 0, tzinfo=timezone.utc),
        tz_market="America/New_York",
        open_time_et="09:30",
        close_time_et="16:00",
        open_grace_min=15,
        close_grace_min=10,
    )
    probe_now_et = None
    probe_now_utc = None
    try:
        probe_now_et = datetime.fromisoformat(str((session_probe or {}).get("now_et", "")))
        probe_now_utc = datetime.fromisoformat(str((session_probe or {}).get("now_utc", "")))
    except Exception:
        probe_now_et = None
        probe_now_utc = None
    _check(
        "TZ-A",
        isinstance(probe_now_et, datetime) and isinstance(probe_now_utc, datetime) and probe_now_utc.hour == 12 and probe_now_et.hour == 7,
        "market_session converts UTC time to ET correctly (12:00 UTC -> 07:00 ET)",
    )

    # Unit-style gate check: 09:35 ET with open_grace=5min should be tradable.
    engine._debug_session_override = None
    engine.config.setdefault("reporting", {})["market_open_grace_min"] = 5
    now_holder["value"] = datetime(2026, 2, 10, 14, 35, 0, tzinfo=timezone.utc)  # 09:35 ET (winter)
    live_session_935, live_gate_935 = engine._refresh_market_session_state(engine._now())
    _check("GATE-1A", str((live_session_935 or {}).get("state", "")).upper() == "OPEN", "09:35 ET is OPEN session")
    _check("GATE-1B", bool((live_gate_935 or {}).get("allowed", False)), "09:35 ET after grace is not blocked by market gate")
    _check("GATE-1C", str((live_gate_935 or {}).get("reason_detail", "")) == "allowed", "gate reason_detail is explicit when tradable")

    # CASE-1 PRE_OPEN
    now_holder["value"] = datetime(2026, 2, 10, 5, 0, 0)
    attempt_before_c1 = engine.last_rebalance_attempt_time
    trades1, snap1 = _run_case(
        session=_make_session("PRE_OPEN", open_grace_passed=False),
        mode="live_buy",
        positions={"AAA": 10},
        cash=30000.0,
        target_weights={"AAA": 0.5, "CASH": 0.5},
    )
    _check("CASE1-A", trades1 == [], "PRE_OPEN does not trade")
    _check("CASE1-B", snap1.get("rebalance_skipped_reason") == "market_closed_gate", "PRE_OPEN skip reason is market_closed_gate")
    _check("CASE1-C", "stale_abort" not in str(snap1.get("stale_decision_trace", "")), "PRE_OPEN does not stale-abort")
    _check("CASE1-D", engine.last_rebalance_attempt_time == attempt_before_c1, "PRE_OPEN does not update attempt timestamp")
    _check("CASE1-E", isinstance(snap1.get("price_debug"), dict) and len(snap1.get("price_debug", {})) > 0, "PRE_OPEN with holdings still writes non-empty price_debug")
    case1_live_price_debug_ok = False
    try:
        with open(reporting_cfg["snapshot_live_path"], "r", encoding="utf-8") as f:
            case1_live_payload = json.load(f)
        case1_live_price_debug_ok = isinstance(case1_live_payload.get("price_debug"), dict) and len(case1_live_payload.get("price_debug", {})) > 0
    except Exception:
        case1_live_price_debug_ok = False
    _check("CASE1-F", case1_live_price_debug_ok, "PRE_OPEN snapshot_live.json persists non-empty price_debug when holdings > 0")

    # CASE-2 OPEN + no grace
    now_holder["value"] = datetime(2026, 2, 10, 9, 35, 0)
    attempt_before_c2 = engine.last_rebalance_attempt_time
    trades2, snap2 = _run_case(
        session=_make_session("OPEN", open_grace_passed=False),
        mode="live_buy",
        positions={},
        cash=30000.0,
        target_weights={"AAA": 0.5, "CASH": 0.5},
    )
    _check("CASE2-A", trades2 == [], "OPEN pre-grace does not trade")
    _check("CASE2-B", snap2.get("rebalance_skipped_reason") == "market_closed_gate", "OPEN pre-grace skip reason is market_closed_gate")
    _check("CASE2-C", "stale_abort" not in str(snap2.get("stale_decision_trace", "")), "OPEN pre-grace does not stale-abort")
    _check("CASE2-D", engine.last_rebalance_attempt_time == attempt_before_c2, "OPEN pre-grace does not update attempt timestamp")

    # CASE-3 OPEN + grace, BUY candidates all STALE but policy blocks them.
    now_holder["value"] = datetime(2026, 2, 10, 10, 30, 0)
    trades3, snap3 = _run_case(
        session=_make_session("OPEN", open_grace_passed=True),
        mode="stale_buy",
        positions={},
        cash=30000.0,
        target_weights={"AAA": 0.34, "BBB": 0.33, "CCC": 0.33, "CASH": 0.0},
    )
    policy_total_c3 = int(snap3.get("stale_candidate_count_policy_pass", (snap3.get("stale_candidates_policy_pass", {}) or {}).get("total", 0)) or 0)
    policy_ratio_c3 = float(snap3.get("stale_ratio_candidates_policy_pass", 0.0) or 0.0)
    _check("CASE3-A", trades3 == [], "STALE BUY candidates are skipped by policy")
    _check("CASE3-B", (policy_total_c3 == 0 or abs(policy_ratio_c3) <= 1e-9), "policy-pass stale denominator is zero (or ratio zero)")
    _check("CASE3-C", "stale_abort" not in str(snap3.get("stale_decision_trace", "")), "no stale-abort when no policy-pass candidates")
    _check("CASE3-D", isinstance(snap3.get("stale_candidates_policy_pass"), dict), "snapshot contains stale_candidates_policy_pass")

    # CASE-4 OPEN + grace, SELL candidates all STALE and policy allows STALE -> stale-abort.
    now_holder["value"] = datetime(2026, 2, 10, 10, 50, 0)
    trades4, snap4 = _run_case(
        session=_make_session("OPEN", open_grace_passed=True),
        mode="stale_sell",
        positions={"AAA": 50, "BBB": 40, "CCC": 30},
        cash=1000.0,
        target_weights={"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
    )
    attempt_after_c4 = engine.last_rebalance_attempt_time
    _check("CASE4-A", trades4 == [], "stale-abort returns no trades")
    _check("CASE4-B", "stale_abort" in str(snap4.get("stale_decision_trace", "")), "stale-abort trigger recorded")
    _check("CASE4-C", abs(float(snap4.get("stale_ratio_candidates_policy_pass", 0.0) or 0.0) - 1.0) <= 1e-9, "policy-pass stale ratio equals 1.0")

    # CASE-5 immediate retry should be blocked by attempt cooldown.
    now_holder["value"] = datetime(2026, 2, 10, 10, 50, 0)
    trades5, snap5 = _run_case(
        session=_make_session("OPEN", open_grace_passed=True),
        mode="stale_sell",
        positions={"AAA": 50, "BBB": 40, "CCC": 30},
        cash=1000.0,
        target_weights={"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0},
    )
    _check("CASE5-A", trades5 == [], "attempt cooldown returns no trades")
    _check("CASE5-B", snap5.get("rebalance_skipped_reason") == "attempt_cooldown", "attempt cooldown skip reason recorded")
    _check("CASE5-C", engine.last_rebalance_attempt_time == attempt_after_c4, "attempt timestamp unchanged when blocked by attempt cooldown")
    _check("CASE5-D", "stale_abort" not in str(snap5.get("stale_decision_trace", "")), "attempt cooldown path is not stale-abort")

    # CASE-6 optional executed trade (for trade-level identity fields).
    now_holder["value"] = datetime(2026, 2, 10, 11, 5, 0)
    trades6, _snap6 = _run_case(
        session=_make_session("OPEN", open_grace_passed=True),
        mode="live_buy",
        positions={},
        cash=30000.0,
        target_weights={"AAA": 0.5, "CASH": 0.5},
    )
    _check("CASE6-A", isinstance(trades6, list), "execution case ran deterministically")

    # S3 checks.
    for idx, snap in enumerate([snap1, snap2, snap3, snap4, snap5], start=1):
        _check(f"S3-{idx}A", isinstance(snap.get("price_debug"), dict), f"case-{idx} snapshot has price_debug dict")
    sample_dbg = None
    if isinstance(snap3.get("price_debug"), dict):
        sample_dbg = snap3.get("price_debug", {}).get("AAA")
        if sample_dbg is None and snap3.get("price_debug"):
            sample_dbg = next(iter(snap3.get("price_debug", {}).values()))
    required_fields = ["ticker", "now_ts", "status", "age_min", "source", "price_ts", "tz_ok", "thresholds"]
    _check("S3-B", isinstance(sample_dbg, dict) and all(k in sample_dbg for k in required_fields), "price_debug contains required fields")
    now_ts = str((sample_dbg or {}).get("now_ts", ""))
    price_ts = str((sample_dbg or {}).get("price_ts", ""))
    _check("S3-C", (("+" in now_ts or now_ts.endswith("Z")) and ("+" in price_ts or price_ts.endswith("Z"))), "price_debug timestamps are timezone-aware ISO")
    _check("S3-D", isinstance(sample_dbg, dict) and (sample_dbg.get("tz_ok") is False) and ("naive_ts_detected" in str(sample_dbg.get("notes", ""))), "tz_ok=false branch captured")
    age_diff_ok = False
    try:
        sample_now_dt = datetime.fromisoformat(now_ts.replace("Z", "+00:00"))
        sample_price_dt = datetime.fromisoformat(price_ts.replace("Z", "+00:00"))
        sample_age = float((sample_dbg or {}).get("age_min", 0.0) or 0.0)
        age_diff = (sample_now_dt.astimezone(timezone.utc) - sample_price_dt.astimezone(timezone.utc)).total_seconds() / 60.0
        age_diff_ok = abs(age_diff - sample_age) <= 0.1
    except Exception:
        age_diff_ok = False
    _check("S3-E", age_diff_ok, "age_min matches now_ts - price_ts in UTC")

    # S5 checks.
    snapshot_path = reporting_cfg["snapshot_live_path"]
    trade_history_path = reporting_cfg["trade_history_path"]
    _check("S5-A", os.path.exists(snapshot_path), "snapshot_live.json exists")
    _check("S5-B", os.path.exists(trade_history_path), "trade_history.jsonl exists")

    parsed_snapshot = None
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            parsed_snapshot = json.load(f)
        snapshot_parse_ok = True
    except Exception:
        snapshot_parse_ok = False
    _check("S5-C", snapshot_parse_ok, "snapshot_live.json parseable via json.load")

    parsed_trade_rows = []
    trade_parse_ok = True
    try:
        with open(trade_history_path, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                parsed_trade_rows.append(json.loads(line_s))
    except Exception:
        trade_parse_ok = False
    _check("S5-D", trade_parse_ok, "trade_history.jsonl parseable line-by-line")

    _check(
        "S5-E",
        isinstance(parsed_snapshot, dict) and bool(parsed_snapshot.get("session_id")) and bool(parsed_snapshot.get("config_hash")),
        "snapshot includes session_id and config_hash",
    )
    if parsed_trade_rows:
        first_trade = parsed_trade_rows[0]
        _check(
            "S5-F",
            isinstance(first_trade, dict) and bool(first_trade.get("session_id")) and bool(first_trade.get("config_hash")),
            "trade rows include session_id and config_hash",
        )

    print(f"DRYRUN_SUMMARY pass={pass_count} fail={fail_count}")
    return 1 if fail_count > 0 else 0


def debug_run_news_overlay_phase2(config_path: str = "paper_config.json", outdir: str = "outputs/news_overlay_phase2_dryrun") -> int:
    """Deterministic dry-run for Phase 2 news overlay consumption checks."""
    pass_count = 0
    fail_count = 0
    case_rows = []

    def _check(check_id: str, condition: bool, message: str):
        nonlocal pass_count, fail_count
        if bool(condition):
            pass_count += 1
            print(f"[PASS] {check_id} {message}")
        else:
            fail_count += 1
            print(f"[FAIL] {check_id} {message}")

    def _merge_news_cfg(base_cfg: dict, override_cfg: dict) -> dict:
        merged = dict(base_cfg or {})
        merged.update(override_cfg or {})
        return {
            "enabled": bool(merged.get("enabled", False)),
            "industry_collection": str(merged.get("industry_collection", "industry_signals")),
            "max_age_hours": max(0.0, float(merged.get("max_age_hours", 48.0) or 48.0)),
            "alpha": float(np.clip(float(merged.get("alpha", 0.08) or 0.08), 0.0, 1.0)),
            "mode": str(merged.get("mode", "risk_only")).lower(),
            "min_confidence": float(np.clip(float(merged.get("min_confidence", 0.55) or 0.55), 0.0, 1.0)),
            "max_abs_delta": float(np.clip(float(merged.get("max_abs_delta", 0.10) or 0.10), 0.0, 1.0)),
            "enable_confidence_scaling": bool(merged.get("enable_confidence_scaling", True)),
        }

    def _filter_signal_rows(raw_rows: list, cfg_row: dict):
        now_utc = datetime.now(timezone.utc)
        max_age_hours = float(cfg_row.get("max_age_hours", 48.0))
        min_confidence = float(cfg_row.get("min_confidence", 0.55))
        latest_by_l2 = {}
        filtered_low_conf = 0
        filtered_age = 0
        malformed = 0
        for row in (raw_rows or []):
            if not isinstance(row, dict):
                malformed += 1
                continue
            l2 = str(row.get("L2", "")).strip()
            if not l2:
                malformed += 1
                continue
            try:
                confidence = float(row.get("confidence", 0.0) or 0.0)
            except Exception:
                malformed += 1
                continue
            if confidence < min_confidence:
                filtered_low_conf += 1
                continue

            ts_val = row.get("timestamp", now_utc.isoformat())
            ts = None
            try:
                ts = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts = ts.astimezone(timezone.utc)
            except Exception:
                ts = now_utc
            age_hours = max(0.0, (now_utc - ts).total_seconds() / 3600.0)
            if age_hours > max_age_hours:
                filtered_age += 1
                continue

            one = {
                "L2": l2,
                "timestamp": ts.isoformat(),
                "confidence": confidence,
                "risk_delta": float(row.get("risk_delta", 0.0) or 0.0),
                "horizon": str(row.get("horizon", "1d")),
                "payload": row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {},
            }
            prev = latest_by_l2.get(l2)
            if prev is None:
                latest_by_l2[l2] = one
            else:
                try:
                    prev_ts = datetime.fromisoformat(str(prev.get("timestamp", "")).replace("Z", "+00:00"))
                    if prev_ts.tzinfo is None:
                        prev_ts = prev_ts.replace(tzinfo=timezone.utc)
                    if ts > prev_ts.astimezone(timezone.utc):
                        latest_by_l2[l2] = one
                except Exception:
                    latest_by_l2[l2] = one
        kept_rows = list(latest_by_l2.values())
        return kept_rows, {
            "raw_count": len(raw_rows or []),
            "kept_count": len(kept_rows),
            "filtered_low_confidence": filtered_low_conf,
            "filtered_age": filtered_age,
            "malformed": malformed,
        }

    def _calc_clip_events(filtered_rows: list, cfg_row: dict):
        alpha = float(cfg_row.get("alpha", 0.08))
        max_abs_delta = float(cfg_row.get("max_abs_delta", 0.10))
        mode = str(cfg_row.get("mode", "risk_only")).lower()
        enable_confidence_scaling = bool(cfg_row.get("enable_confidence_scaling", True))
        events = []
        for row in (filtered_rows or []):
            l2 = str(row.get("L2", "")).strip()
            risk_delta = float(row.get("risk_delta", 0.0) or 0.0)
            raw_delta = risk_delta * alpha
            if enable_confidence_scaling:
                raw_delta = raw_delta * float(row.get("confidence", 0.0) or 0.0)
            clipped = float(np.clip(raw_delta, -max_abs_delta, max_abs_delta))
            if mode == "risk_only" and clipped > 0:
                clipped = 0.0
            clipped_flag = abs(clipped - raw_delta) > 1e-12
            events.append(
                {
                    "L2": l2,
                    "risk_delta": float(risk_delta),
                    "raw_delta": float(raw_delta),
                    "delta_after_clip": float(clipped),
                    "clipped": bool(clipped_flag),
                }
            )
        return events

    outdir_abs = os.path.abspath(str(outdir or "outputs/news_overlay_phase2_dryrun"))
    os.makedirs(outdir_abs, exist_ok=True)

    old_checkpoint_env = os.environ.get("GW_CHECKPOINT_ACTION")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    try:
        engine = PaperTradingEngine(config_path)
    finally:
        if old_checkpoint_env is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint_env

    base_cfg = engine._get_news_overlay_cfg()
    base_cash_target = 0.20
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    tag_lookup = {
        "AAA": {"L2": ["technology"], "L3": [], "keywords": ["semiconductor"]},
        "BBB": {"L2": ["energy"], "L3": [], "keywords": ["oil"]},
        "CCC": {"L2": ["financials"], "L3": [], "keywords": ["bank"]},
        "DDD": {"L2": ["utilities"], "L3": [], "keywords": ["utility"]},
    }

    now_utc = datetime.now(timezone.utc)
    old_ts = (now_utc - timedelta(hours=72)).isoformat()
    fresh_ts = (now_utc - timedelta(hours=2)).isoformat()

    cases = [
        {
            "id": "A",
            "name": "overlay_disabled",
            "cfg": {"enabled": False},
            "signals": [
                {"L2": "technology", "confidence": 0.99, "risk_delta": -0.50, "timestamp": fresh_ts},
            ],
            "expect": {"unchanged": True},
        },
        {
            "id": "B",
            "name": "confidence_filter",
            "cfg": {"enabled": True},
            "signals": [
                {"L2": "technology", "confidence": max(0.0, float(base_cfg.get("min_confidence", 0.55)) - 0.05), "risk_delta": -0.50, "timestamp": fresh_ts},
            ],
            "expect": {"unchanged": True, "low_conf_filtered_min": 1},
        },
        {
            "id": "C",
            "name": "mild_negative_delta_applies",
            "cfg": {"enabled": True, "mode": "risk_only"},
            "signals": [
                {"L2": "technology", "confidence": 0.90, "risk_delta": -0.02, "timestamp": fresh_ts},
            ],
            "expect": {"delta_positive_min": 1e-8},
        },
        {
            "id": "D",
            "name": "risk_only_blocks_aggressive_direction",
            "cfg": {"enabled": True, "mode": "risk_only"},
            "signals": [
                {"L2": "technology", "confidence": 0.90, "risk_delta": +0.20, "timestamp": fresh_ts},
            ],
            "expect": {"non_decreasing_cash": True},
        },
        {
            "id": "E",
            "name": "clip_large_delta",
            "cfg": {"enabled": True, "mode": "risk_only"},
            "signals": [
                {"L2": "technology", "confidence": 0.95, "risk_delta": -5.0, "timestamp": fresh_ts},
            ],
            "expect": {"clipped_min": 1, "delta_positive_min": 1e-8},
        },
        {
            "id": "F",
            "name": "mixed_multi_l2_aggregation",
            "cfg": {"enabled": True, "mode": "risk_only"},
            "signals": [
                {"L2": "technology", "confidence": 0.90, "risk_delta": -0.60, "timestamp": fresh_ts},
                {"L2": "energy", "confidence": 0.70, "risk_delta": -0.15, "timestamp": fresh_ts},
                {"L2": "financials", "confidence": 0.80, "risk_delta": +0.60, "timestamp": fresh_ts},
                {"L2": "utilities", "confidence": 0.40, "risk_delta": -0.80, "timestamp": fresh_ts},
                {"L2": "utilities", "confidence": 0.80, "risk_delta": -0.20, "timestamp": old_ts},
            ],
            "expect": {"non_decreasing_cash": True, "low_conf_filtered_min": 1, "age_filtered_min": 1},
        },
    ]

    original_news_cfg = dict(engine.config.get("news_overlay", {})) if isinstance(engine.config.get("news_overlay", {}), dict) else {}
    original_read_recent = engine._read_recent_industry_signals
    original_tag_lookup = engine._build_ticker_tags_lookup

    try:
        engine._build_ticker_tags_lookup = lambda: dict(tag_lookup)

        for case in cases:
            case_id = str(case.get("id"))
            case_name = str(case.get("name"))
            cfg_row = _merge_news_cfg(base_cfg, case.get("cfg", {}))
            engine.config["news_overlay"] = dict(cfg_row)
            filtered_rows, filter_stats = _filter_signal_rows(case.get("signals", []), cfg_row)
            clip_events = _calc_clip_events(filtered_rows, cfg_row)
            clip_count = sum(1 for x in clip_events if bool(x.get("clipped", False)))
            engine._read_recent_industry_signals = (lambda rows=filtered_rows: list(rows))

            before = float(base_cash_target)
            after, info = engine.apply_news_overlay_to_cash_target(tickers, before)
            observed_delta = float(after - before)
            theoretical_upper = float(cfg_row.get("max_abs_delta", 0.10))
            bound_ok = abs(observed_delta) <= theoretical_upper + 1e-12
            risk_only_ok = True
            if str(cfg_row.get("mode", "risk_only")).lower() == "risk_only":
                risk_only_ok = observed_delta >= -1e-12

            _check(f"CASE{case_id}-A", bound_ok, f"delta bound respected (|{observed_delta:.6f}| <= {theoretical_upper:.6f})")
            _check(f"CASE{case_id}-B", risk_only_ok, "risk_only does not make cash target more aggressive")

            expected = case.get("expect", {})
            if bool(expected.get("unchanged", False)):
                _check(f"CASE{case_id}-C", abs(observed_delta) <= 1e-12, "cash_target unchanged as expected")
            if "delta_positive_min" in expected:
                _check(f"CASE{case_id}-C", observed_delta >= float(expected.get("delta_positive_min", 0.0)), "cash_target increased by expected minimum delta")
            if bool(expected.get("non_decreasing_cash", False)):
                _check(f"CASE{case_id}-D", after >= before - 1e-12, "cash_target_after >= cash_target_before")
            if "clipped_min" in expected:
                _check(f"CASE{case_id}-E", clip_count >= int(expected.get("clipped_min", 1)), "clip triggered as expected")
            if "low_conf_filtered_min" in expected:
                _check(
                    f"CASE{case_id}-F",
                    int(filter_stats.get("filtered_low_confidence", 0)) >= int(expected.get("low_conf_filtered_min", 0)),
                    "low-confidence filtering applied",
                )
            if "age_filtered_min" in expected:
                _check(
                    f"CASE{case_id}-G",
                    int(filter_stats.get("filtered_age", 0)) >= int(expected.get("age_filtered_min", 0)),
                    "age filtering applied",
                )

            case_payload = {
                "case_id": case_id,
                "case_name": case_name,
                "config": {
                    "enabled": bool(cfg_row.get("enabled", False)),
                    "mode": str(cfg_row.get("mode", "risk_only")),
                    "alpha": float(cfg_row.get("alpha", 0.08)),
                    "min_confidence": float(cfg_row.get("min_confidence", 0.55)),
                    "max_abs_delta": float(cfg_row.get("max_abs_delta", 0.10)),
                    "max_age_hours": float(cfg_row.get("max_age_hours", 48.0)),
                },
                "cash_target_before": before,
                "cash_target_after": float(after),
                "applied_delta": observed_delta,
                "delta_upper_bound_theoretical": theoretical_upper,  # Current implementation bound: |delta| <= max_abs_delta
                "filter_stats": filter_stats,
                "used_signals_count": len(filtered_rows),
                "clip_count": int(clip_count),
                "clip_events": clip_events[:20],
                "overlay_info": info if isinstance(info, dict) else {},
            }
            case_rows.append(case_payload)
            print(
                f"[PHASE2-DRYRUN] CASE {case_id} ({case_name}) "
                f"before={before:.6f} after={after:.6f} delta={observed_delta:+.6f} "
                f"kept={len(filtered_rows)} low_conf={filter_stats.get('filtered_low_confidence', 0)} "
                f"age_filtered={filter_stats.get('filtered_age', 0)} clipped={clip_count}"
            )
    finally:
        engine.config["news_overlay"] = dict(original_news_cfg)
        engine._read_recent_industry_signals = original_read_recent
        engine._build_ticker_tags_lookup = original_tag_lookup

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "outdir": outdir_abs,
        "checks": {"pass": pass_count, "fail": fail_count},
        "cases": case_rows,
    }
    summary_path = os.path.join(outdir_abs, "news_overlay_phase2_dryrun_summary.json")
    io_atomic_write_json(summary_path, summary, indent=2)
    print(f"[PHASE2-DRYRUN] Summary written: {summary_path}")
    print(f"DRYRUN_SUMMARY pass={pass_count} fail={fail_count}")
    return 1 if fail_count > 0 else 0


def debug_run_news_overlay_once(config_path: str = "paper_config.json", outdir: str = "outputs/gw_dryrun") -> int:
    """Run one deterministic news-overlay debug check without trading."""
    outdir_abs = os.path.abspath(str(outdir or "outputs/gw_dryrun"))
    os.makedirs(outdir_abs, exist_ok=True)

    old_checkpoint_env = os.environ.get("GW_CHECKPOINT_ACTION")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    try:
        engine = PaperTradingEngine(config_path)
    finally:
        if old_checkpoint_env is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint_env

    base_cfg = engine._get_news_overlay_cfg()
    probe_cfg = dict(base_cfg)
    probe_cfg["enabled"] = True
    probe_cfg["mode"] = str(base_cfg.get("mode", "risk_only")).lower()
    probe_cfg["enable_confidence_scaling"] = bool(base_cfg.get("enable_confidence_scaling", True))
    engine.config["news_overlay"] = dict(probe_cfg)

    def _to_ticker(v):
        if isinstance(v, dict):
            cand = v.get("ticker", None)
            if cand is None:
                cand = v.get("symbol", None)
            if cand is None:
                return ""
            return str(cand).strip().upper()
        return str(v).strip().upper()

    universe_tickers = []
    if isinstance(engine.config.get("universe", []), list):
        universe_tickers.extend([_to_ticker(x) for x in engine.config.get("universe", []) if _to_ticker(x)])
    tag_lookup = engine._build_ticker_tags_lookup()
    if isinstance(tag_lookup, dict):
        universe_tickers.extend([str(x).strip().upper() for x in tag_lookup.keys() if str(x).strip()])
    if isinstance(engine.config.get("industry_map", {}), dict):
        for k, v in engine.config.get("industry_map", {}).items():
            if isinstance(v, str):
                t = str(k).strip().upper()
                if t:
                    universe_tickers.append(t)
            elif isinstance(v, (list, tuple, set)):
                for t_val in v:
                    t = str(t_val).strip().upper()
                    if t:
                        universe_tickers.append(t)
    probe_tickers = []
    for t in universe_tickers:
        if t and t != "CASH" and t not in probe_tickers:
            probe_tickers.append(t)
    if not probe_tickers:
        probe_tickers = ["SPY", "QQQ", "XOM", "XLU", "XLF"]
    cash_before = 0.20

    cash_after, overlay_info = engine.apply_news_overlay_to_cash_target(
        probe_tickers,
        cash_target=cash_before,
    )

    relaxed_info = None
    try:
        included_count = int((overlay_info or {}).get("included_rows_count", 0)) if isinstance(overlay_info, dict) else 0
    except Exception:
        included_count = 0
    if included_count < 2:
        relaxed_cfg = dict(probe_cfg)
        relaxed_cfg["min_confidence"] = float(max(0.0, min(float(probe_cfg.get("min_confidence", 0.55)), 0.55)))
        engine.config["news_overlay"] = dict(relaxed_cfg)
        _, relaxed_info = engine.apply_news_overlay_to_cash_target(
            probe_tickers,
            cash_target=cash_before,
        )
        engine.config["news_overlay"] = dict(probe_cfg)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "news_overlay_cfg_used": {
            "enabled": bool(probe_cfg.get("enabled", False)),
            "mode": str(probe_cfg.get("mode", "risk_only")),
            "alpha": float(probe_cfg.get("alpha", 0.08)),
            "min_confidence": float(probe_cfg.get("min_confidence", 0.55)),
            "max_abs_delta": float(probe_cfg.get("max_abs_delta", 0.10)),
            "enable_confidence_scaling": bool(probe_cfg.get("enable_confidence_scaling", True)),
        },
        "cash_target_old": float(cash_before),
        "cash_target_new": float(cash_after),
        "overlay_info": overlay_info if isinstance(overlay_info, dict) else {},
        "probe_tickers_count": len(probe_tickers),
        "probe_tickers_sample": probe_tickers[:30],
        "included_rows_source_audit": (overlay_info or {}).get("included_rows_audit", []) if isinstance(overlay_info, dict) else [],
        "excluded_rows_source_audit": (overlay_info or {}).get("excluded_rows_audit", []) if isinstance(overlay_info, dict) else [],
        "diagnostic_relaxed_min_confidence": {
            "enabled": bool(relaxed_info is not None),
            "min_confidence_used": float(max(0.0, min(float(probe_cfg.get("min_confidence", 0.55)), 0.55))) if relaxed_info is not None else None,
            "included_rows_source_audit": (relaxed_info or {}).get("included_rows_audit", []) if isinstance(relaxed_info, dict) else [],
            "excluded_rows_source_audit": (relaxed_info or {}).get("excluded_rows_audit", []) if isinstance(relaxed_info, dict) else [],
            "included_rows_count": int((relaxed_info or {}).get("included_rows_count", 0)) if isinstance(relaxed_info, dict) else 0,
            "excluded_rows_count": int((relaxed_info or {}).get("excluded_rows_count", 0)) if isinstance(relaxed_info, dict) else 0,
        },
    }
    summary_path = os.path.join(outdir_abs, "news_overlay_once_debug.json")
    io_atomic_write_json(summary_path, payload, indent=2)

    info = payload.get("overlay_info", {}) if isinstance(payload.get("overlay_info", {}), dict) else {}
    print(f"[NEWS_OVERLAY_ONCE] Summary written: {summary_path}")
    print(
        f"[NEWS_OVERLAY_ONCE] included={int(info.get('included_rows_count', 0))} "
        f"excluded={int(info.get('excluded_rows_count', 0))} "
        f"used={int(info.get('used_signals', 0))}"
    )
    print(f"[NEWS_OVERLAY_ONCE] chosen_cash_delta_source={info.get('chosen_cash_delta_source')}")
    print(
        f"[NEWS_OVERLAY_ONCE] cash_target_old={float(payload.get('cash_target_old', 0.0)):.4f} "
        f"cash_target_new={float(payload.get('cash_target_new', 0.0)):.4f}"
    )
    return 0


def _dist_stats(values):
    vals = [float(x) for x in (values or []) if np.isfinite(float(x))]
    if not vals:
        return {"min": None, "p50": None, "p90": None, "max": None, "count": 0}
    arr = np.array(vals, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def _snap_to_choices(value, choices):
    vals = [float(x) for x in (choices or [])]
    if not vals:
        return float(value)
    return float(min(vals, key=lambda x: abs(float(value) - float(x))))


def _safe_clip(value, low, high):
    return float(np.clip(float(value), float(low), float(high)))


def calibrate_news_overlay(
    config_path: str = "paper_config.json",
    lookback_hours: float = 72.0,
    target_cash_delta: float = 0.02,
    out_path: str = "outputs/news_overlay_calibration.json",
    max_abs_delta_reco: float = 0.03,
) -> int:
    """Calibrate news overlay parameters using historical industry_signals replay."""
    warnings = []
    now_utc = datetime.now(timezone.utc)
    lookback_hours = max(1.0, float(lookback_hours or 72.0))
    target_cash_delta = max(0.0, float(target_cash_delta or 0.02))
    max_abs_delta_reco = max(1e-6, float(max_abs_delta_reco or 0.03))

    old_checkpoint_env = os.environ.get("GW_CHECKPOINT_ACTION")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    try:
        engine = PaperTradingEngine(config_path)
    finally:
        if old_checkpoint_env is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint_env

    cfg = engine._get_news_overlay_cfg()
    mode = str(cfg.get("mode", "risk_only")).lower()
    enable_confidence_scaling = bool(cfg.get("enable_confidence_scaling", True))
    min_conf_cfg = float(cfg.get("min_confidence", 0.55))
    max_age_hours = float(cfg.get("max_age_hours", 48.0))
    collection_name = str(cfg.get("industry_collection", "industry_signals"))
    chroma_path = engine.config.get("macro_integration", {}).get("chroma_path", "./memory_db")

    rows = []
    per_l2_counts = {}
    recent_24h_counts = {}
    raw_total = 0
    if not CHROMADB_AVAILABLE:
        warnings.append("chromadb_not_available")
    else:
        coll = engine._get_industry_signals_collection(collection_name=collection_name, chroma_path=chroma_path)
        if coll is None:
            warnings.append("industry_collection_unavailable")
        else:
            include = ["metadatas", "documents"]
            try:
                try:
                    res = coll.get(where={"scope": "industry"}, include=include)
                except Exception:
                    res = coll.get(include=include)
            except Exception as e:
                warnings.append(f"collection_read_failed:{e}")
                res = {}

            metas = res.get("metadatas", []) if isinstance(res, dict) else []
            docs = res.get("documents", []) if isinstance(res, dict) else []
            for idx, meta in enumerate(metas):
                if not isinstance(meta, dict):
                    continue
                raw_total += 1
                l2 = str(meta.get("L2", "")).strip()
                if not l2:
                    continue
                scope = str(meta.get("scope", "industry")).lower()
                if scope != "industry":
                    continue
                ts = engine._parse_iso_datetime(meta.get("timestamp"))
                if ts is None:
                    continue
                age_hours_from_now = max(0.0, (now_utc - ts).total_seconds() / 3600.0)
                if age_hours_from_now > lookback_hours:
                    continue

                doc = docs[idx] if idx < len(docs) else None
                payload = {}
                if isinstance(doc, str) and doc.strip():
                    try:
                        parsed = json.loads(doc)
                        if isinstance(parsed, dict):
                            payload = parsed
                    except Exception:
                        payload = {}
                status = str(meta.get("status", payload.get("status", "ok")) or "ok").lower()
                if status not in ("ok", "success", ""):
                    continue

                risk_delta = float(meta.get("risk_delta", payload.get("risk_delta", 0.0)) or 0.0)
                confidence = float(meta.get("confidence", payload.get("confidence", 0.0)) or 0.0)
                direction = str(meta.get("direction", payload.get("direction", "")) or "")
                row = {
                    "L2": l2,
                    "timestamp": ts,
                    "timestamp_iso": ts.isoformat(),
                    "age_hours": age_hours_from_now,
                    "direction": direction,
                    "risk_delta": risk_delta,
                    "confidence": confidence,
                }
                rows.append(row)
                per_l2_counts[l2] = int(per_l2_counts.get(l2, 0) + 1)
                if age_hours_from_now <= 24.0:
                    recent_24h_counts[l2] = int(recent_24h_counts.get(l2, 0) + 1)

    rows.sort(key=lambda x: x.get("timestamp"))
    rows_active = [r for r in rows if float(r.get("age_hours", 0.0) or 0.0) <= max_age_hours]

    conf_values = [float(r.get("confidence", 0.0) or 0.0) for r in rows_active]
    rd_values = [float(r.get("risk_delta", 0.0) or 0.0) for r in rows_active]
    scale_vals = [
        float(r.get("confidence", 0.0) or 0.0) if enable_confidence_scaling else 1.0
        for r in rows_active
    ]
    eff = [abs(rd_values[i]) * scale_vals[i] for i in range(len(rows_active))]
    eff_neg = [max(0.0, -rd_values[i]) * scale_vals[i] for i in range(len(rows_active))]
    eff_pos = [max(0.0, +rd_values[i]) * scale_vals[i] for i in range(len(rows_active))]
    negative_signal_ratio = float(
        (sum(1 for x in eff_neg if float(x) > 0.0) / len(rows_active)) if rows_active else 0.0
    )

    conf_p50 = float(np.percentile(np.array(conf_values, dtype=float), 50)) if conf_values else min_conf_cfg
    min_conf_choices = [0.45, 0.55, 0.60, 0.65]
    min_conf_reco = _snap_to_choices(conf_p50, min_conf_choices)
    candidate_thresholds = [0.55, 0.60, 0.65]
    coverage_candidates = {}
    for th in candidate_thresholds:
        keep = sum(1 for c in conf_values if float(c) >= float(th))
        coverage_candidates[str(th)] = float((keep / len(conf_values)) if conf_values else 0.0)
    reco_keep = sum(1 for c in conf_values if float(c) >= float(min_conf_reco))
    reco_coverage = float((reco_keep / len(conf_values)) if conf_values else 0.0)
    if reco_coverage < 0.10 and conf_values:
        warnings.append("recommended_min_confidence_keeps_less_than_10pct")

    rows_reco = [r for r in rows_active if float(r.get("confidence", 0.0) or 0.0) >= float(min_conf_reco)]
    eff_neg_reco = []
    eff_reco = []
    for r in rows_reco:
        conf_mult = float(r.get("confidence", 0.0) or 0.0) if enable_confidence_scaling else 1.0
        rd = float(r.get("risk_delta", 0.0) or 0.0)
        eff_reco.append(abs(rd) * conf_mult)
        eff_neg_reco.append(max(0.0, -rd) * conf_mult)
    eff_neg_positive = [x for x in eff_neg_reco if float(x) > 0.0]
    p90_eff_neg = float(np.percentile(np.array(eff_neg_positive, dtype=float), 90)) if eff_neg_positive else 0.0

    alpha_cap = None
    alpha_reco = None
    if len(eff_neg_positive) >= 5 and p90_eff_neg > 0.0:
        alpha_reco = float(target_cash_delta / p90_eff_neg)
        alpha_cap = float(max_abs_delta_reco / p90_eff_neg)
        upper = min(float(alpha_cap), 0.60)
        if upper < 0.05:
            alpha_final = upper
            warnings.append("alpha_upper_bound_below_min_floor")
        else:
            alpha_final = _safe_clip(alpha_reco, 0.05, upper)
    else:
        warnings.append("No meaningful negative/risk-off signals in lookback; risk_only overlay will be mostly zero.")
        eff_positive = [x for x in eff_reco if float(x) > 0.0]
        p90_eff = float(np.percentile(np.array(eff_positive, dtype=float), 90)) if eff_positive else 0.0
        alpha_reco_abs = float(target_cash_delta / p90_eff) if p90_eff > 0.0 else float(cfg.get("alpha", 0.08))
        alpha_final = _safe_clip(alpha_reco_abs, 0.05, 0.60)
        warnings.append("fallback_alpha_based_on_abs_distribution_only")
        warnings.append("recommend_step5_improve_underweight_generation_or_add_macro_risk_off_prior")

    # Replay simulation over timestamps using latest-per-L2 at each timepoint.
    replay_points = []
    if rows_reco:
        timepoints = sorted({r.get("timestamp") for r in rows_reco if isinstance(r.get("timestamp"), datetime)})
        l2_set = sorted({str(r.get("L2", "")).strip() for r in rows_reco if str(r.get("L2", "")).strip()})
        for t_point in timepoints:
            latest_by_l2 = {}
            for row in rows_reco:
                l2 = str(row.get("L2", "")).strip()
                if not l2:
                    continue
                ts = row.get("timestamp")
                if not isinstance(ts, datetime):
                    continue
                if ts > t_point:
                    continue
                prev = latest_by_l2.get(l2)
                if prev is None or ts > prev.get("timestamp"):
                    latest_by_l2[l2] = row

            l2_deltas = {}
            for l2 in l2_set:
                row = latest_by_l2.get(l2)
                if row is None:
                    continue
                age_h = max(0.0, (t_point - row.get("timestamp")).total_seconds() / 3600.0)
                if age_h > max_age_hours:
                    continue
                rd = float(row.get("risk_delta", 0.0) or 0.0)
                conf = float(row.get("confidence", 0.0) or 0.0)
                raw = float(rd * alpha_final)
                if enable_confidence_scaling:
                    raw = float(raw * conf)
                delta = float(np.clip(raw, -max_abs_delta_reco, max_abs_delta_reco))
                if mode == "risk_only" and delta > 0:
                    delta = 0.0
                l2_deltas[l2] = delta

            if l2_deltas:
                worst_l2_delta = float(min(l2_deltas.values()))
                if mode == "risk_only":
                    cash_delta = float(abs(min(0.0, worst_l2_delta)))
                else:
                    cash_delta = float(abs(worst_l2_delta))
            else:
                worst_l2_delta = 0.0
                cash_delta = 0.0
            replay_points.append(
                {
                    "timestamp": t_point.isoformat(),
                    "worst_l2_delta": float(worst_l2_delta),
                    "cash_delta_simulated": float(cash_delta),
                }
            )

    replay_cash = [float(p.get("cash_delta_simulated", 0.0) or 0.0) for p in replay_points]
    replay_nonzero_ratio = float(
        (sum(1 for x in replay_cash if x > 1e-12) / len(replay_cash)) if replay_cash else 0.0
    )
    replay_capped_ratio = float(
        (sum(1 for x in replay_cash if abs(x - max_abs_delta_reco) <= 1e-12) / len(replay_cash)) if replay_cash else 0.0
    )

    output = {
        "generated_at_utc": now_utc.isoformat(),
        "config_path": str(config_path),
        "inputs": {
            "lookback_hours": float(lookback_hours),
            "target_cash_delta": float(target_cash_delta),
            "mode": mode,
            "max_age_hours": float(max_age_hours),
            "enable_confidence_scaling": bool(enable_confidence_scaling),
            "collection": collection_name,
            "chroma_path": str(chroma_path),
        },
        "sample_counts": {
            "raw_total_rows": int(raw_total),
            "lookback_rows_status_ok": int(len(rows)),
            "active_rows_after_max_age": int(len(rows_active)),
            "recent_24h_rows": int(sum(recent_24h_counts.values())),
            "count_by_l2": dict(sorted(per_l2_counts.items())),
            "count_by_l2_recent_24h": dict(sorted(recent_24h_counts.items())),
        },
        "stats": {
            "risk_delta": _dist_stats(rd_values),
            "confidence": _dist_stats(conf_values),
            "eff": _dist_stats(eff),
            "eff_neg": _dist_stats(eff_neg),
            "eff_pos": _dist_stats(eff_pos),
            "negative_signal_ratio": float(negative_signal_ratio),
            "min_confidence_candidates_coverage": coverage_candidates,
            "recommended_min_confidence_coverage": float(reco_coverage),
            "eff_neg_positive_count_at_recommended_min_confidence": int(len(eff_neg_positive)),
        },
        "suggested": {
            "min_confidence": float(min_conf_reco),
            "alpha": float(alpha_final),
            "max_abs_delta": float(max_abs_delta_reco),
            "enable_confidence_scaling": bool(enable_confidence_scaling),
            "mode": mode,
        },
        "replay": {
            "points": int(len(replay_points)),
            "cash_delta_simulated": _dist_stats(replay_cash),
            "nonzero_ratio": float(replay_nonzero_ratio),
            "capped_ratio": float(replay_capped_ratio),
            "points_sample": replay_points[:30],
        },
        "warnings": warnings,
        "debug": {
            "p90_eff_neg": float(p90_eff_neg),
            "alpha_reco_raw": float(alpha_reco) if alpha_reco is not None else None,
            "alpha_cap_from_max_abs": float(alpha_cap) if alpha_cap is not None else None,
        },
    }

    out_abs = os.path.abspath(str(out_path or "outputs/news_overlay_calibration.json"))
    io_atomic_write_json(out_abs, output, indent=2)

    replay_p90 = output.get("replay", {}).get("cash_delta_simulated", {}).get("p90")
    print(f"[CALIBRATOR] wrote: {out_abs}")
    print(f"[CALIBRATOR] negative_signal_ratio={negative_signal_ratio:.4f}")
    print(
        f"[CALIBRATOR] suggested min_confidence={float(min_conf_reco):.2f} "
        f"alpha={float(alpha_final):.4f} max_abs_delta={float(max_abs_delta_reco):.4f} "
        f"scaling={bool(enable_confidence_scaling)} mode={mode}"
    )
    print(
        f"[CALIBRATOR] replay p90_cash_delta={float(replay_p90 or 0.0):.4f} "
        f"nonzero_ratio={float(replay_nonzero_ratio):.4f} capped_ratio={float(replay_capped_ratio):.4f}"
    )
    if warnings:
        for w in warnings:
            print(f"[CALIBRATOR][WARN] {w}")
    return 0


def main():
    """def main: docstring omitted (was garbled/non-ASCII)."""
    parser = argparse.ArgumentParser(description="Paper trading engine")
    parser.add_argument("config_path", nargs="?", default="paper_config.json", help="Path to config JSON.")
    parser.add_argument(
        "--debug-planner-once",
        action="store_true",
        help="Run deterministic planner dry-run once and exit (no market data required).",
    )
    parser.add_argument(
        "--debug-turnover-limit",
        type=float,
        default=None,
        help="Optional absolute turnover limit for planner dry-run.",
    )
    parser.add_argument(
        "--debug-system-s1-5",
        action="store_true",
        help="Run offline deterministic S1..S5 acceptance dry-run and exit.",
    )
    parser.add_argument(
        "--debug-outdir",
        type=str,
        default="outputs/gw_dryrun",
        help="Output directory for debug artifacts.",
    )
    parser.add_argument(
        "--debug-news-overlay-phase2",
        action="store_true",
        help="Run deterministic Phase 2 news overlay consumption dry-run and exit.",
    )
    parser.add_argument(
        "--debug-news-overlay-once",
        action="store_true",
        help="Run one deterministic news overlay debug check and exit (no trades).",
    )
    parser.add_argument(
        "--calibrate-news-overlay",
        action="store_true",
        help="Run news overlay calibrator + replay and exit.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=72.0,
        help="Lookback window in hours for news overlay calibrator.",
    )
    parser.add_argument(
        "--target-cash-delta",
        type=float,
        default=0.02,
        help="Target p90 cash delta for calibrator.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/news_overlay_calibration.json",
        help="Output JSON path for calibrator.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path override.",
    )
    args = parser.parse_args()

    debug_env = str(os.environ.get("GW_DEBUG_PLANNER_ONCE", "")).strip().lower() in ("1", "true", "yes", "on")
    debug_system_env = str(os.environ.get("GW_DEBUG_SYSTEM_S1_5", "")).strip().lower() in ("1", "true", "yes", "on")
    debug_news_overlay_phase2_env = str(os.environ.get("GW_DEBUG_NEWS_OVERLAY_PHASE2", "")).strip().lower() in ("1", "true", "yes", "on")
    debug_news_overlay_once_env = str(os.environ.get("GW_DEBUG_NEWS_OVERLAY_ONCE", "")).strip().lower() in ("1", "true", "yes", "on")
    if bool(args.debug_planner_once or debug_env):
        debug_run_planner_once(args.config_path, turnover_limit=args.debug_turnover_limit)
        return 0
    if bool(args.debug_system_s1_5 or debug_system_env):
        return debug_run_system_s1_s5(
            config_path=args.config_path,
            outdir=args.debug_outdir,
            turnover_limit=args.debug_turnover_limit,
        )
    config_path = args.config if isinstance(args.config, str) and args.config.strip() else args.config_path

    if bool(args.debug_news_overlay_phase2 or debug_news_overlay_phase2_env):
        return debug_run_news_overlay_phase2(
            config_path=config_path,
            outdir=args.debug_outdir,
        )
    if bool(args.debug_news_overlay_once or debug_news_overlay_once_env):
        return debug_run_news_overlay_once(
            config_path=config_path,
            outdir=args.debug_outdir,
        )
    if bool(args.calibrate_news_overlay):
        return calibrate_news_overlay(
            config_path=config_path,
            lookback_hours=args.lookback_hours,
            target_cash_delta=args.target_cash_delta,
            out_path=args.out,
        )
    print(f"Loading config: {config_path}")

    engine = PaperTradingEngine(config_path)
    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())





