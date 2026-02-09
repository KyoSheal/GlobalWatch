"""Paper trading engine (simulation only)."""

import json
import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # NOTE: comment omitted (was garbled/non-ASCII).
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        self.last_rebalance_time = None  # for cooldown checks
        self.current_regime = {}
        self.current_macro = {}
        self.current_stale_info = {}
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

        # Signal/Macro refresh decoupling state
        execution_config = self.config.get('execution', {})
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
        
        return config
    
    def validate_config(self):
        """def validate_config: docstring omitted (was garbled/non-ASCII)."""
        assert self.config['paper_mode'] == True, "paper_mode must be True"
        assert self.config['safety']['no_real_broker'] == True, "no_real_broker must be True"
        assert self.config['safety']['simulation_only'] == True, "simulation_only must be True"
        assert self.config.get('execution', {}).get('signal_refresh_minutes', 1440) > 0, "execution.signal_refresh_minutes must be > 0"
        assert self.config.get('execution', {}).get('macro_refresh_minutes', 60) > 0, "execution.macro_refresh_minutes must be > 0"
        assert self.config.get('execution', {}).get('max_stale_ratio', 0.3) >= 0, "execution.max_stale_ratio must be >= 0"
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
        macro_cfg = self.config.get('macro_integration', {})
        assert isinstance(macro_cfg.get('enable_llm_topic_signals', True), bool), "macro_integration.enable_llm_topic_signals must be bool"
        assert 0.0 <= float(macro_cfg.get('llm_topic_confidence_threshold', 0.6)) <= 1.0, "macro_integration.llm_topic_confidence_threshold must be in [0,1]"
        assert float(macro_cfg.get('llm_topic_score_threshold', 0.5)) >= 0.0, "macro_integration.llm_topic_score_threshold must be >= 0"
        assert float(macro_cfg.get('llm_topic_tilt_scale', 0.02)) >= 0.0, "macro_integration.llm_topic_tilt_scale must be >= 0"
        assert int(macro_cfg.get('topic_memory_window', 50)) >= 1, "macro_integration.topic_memory_window must be >= 1"
        assert isinstance(macro_cfg.get('topic_sector_ticker_map', {}), dict), "macro_integration.topic_sector_ticker_map must be an object"
        assert isinstance(self.config.get('industry_map', {}), dict), "industry_map must be an object"
        
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
            self.write_live_snapshot(last_snapshot)
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
            'timestamp': latest.get('timestamp', datetime.now().isoformat()),
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

        payload = {
            'timestamp': snapshot.get('timestamp', datetime.now().isoformat()),
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
            'trade_planner_normal_score_count': int((planner_meta.get('normal_score_stats', {}) or {}).get('count', 0) or 0)
        }
        return payload

    def write_live_snapshot(self, snapshot):
        """Write UI-friendly live snapshot to outputs/snapshot_live.json."""
        try:
            live_snapshot_path = self.config.get('reporting', {}).get('snapshot_live_path', 'outputs/snapshot_live.json')
            payload = self._build_live_snapshot_payload(snapshot)
            with open(live_snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
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

        snapshot = {
            'timestamp': datetime.now().isoformat(),
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
            'trade_planner_num_adv_dropped': int((self.current_planner_info or {}).get('num_adv_dropped', 0)) if isinstance(self.current_planner_info, dict) else 0
        }
        return snapshot

    def _write_post_rebalance_live_snapshot(self, trades_count, source="execute_rebalance"):
        """Refresh live snapshot immediately after trades are persisted."""
        try:
            snapshot = self._build_post_rebalance_snapshot()
            self.write_live_snapshot(snapshot)
            live_snapshot_path = self.config.get('reporting', {}).get('snapshot_live_path', 'outputs/snapshot_live.json')
            print(f"[SNAPSHOT] Post-rebalance live snapshot written (cycle={self.current_cycle}, trades={int(trades_count)}, path={live_snapshot_path}, source={source})")
        except Exception as e:
            print(f"[WARN] Post-rebalance live snapshot refresh failed: {e}")

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
            with open(trade_history_path, 'w', encoding='utf-8') as f:
                for trade in self.trades_log:
                    trade_clean = _sanitize(trade)
                    f.write(json.dumps(trade_clean, ensure_ascii=False, allow_nan=False) + '\n')
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

            market_ticker = self._normalize_market_ticker(ticker)
            t = yf.Ticker(market_ticker)
            hist = t.history(period=period, interval=interval)
            
            if hist.empty:
                print(f"[WARN] No data for {ticker} (provider symbol: {market_ticker}), skipping")
                return None
            
            return hist
        except Exception as e:
            print(f"[WARN] Error fetching data for {ticker}: {e}")
            return None

    def get_current_price(self, ticker):
        """def get_current_price: docstring omitted (was garbled/non-ASCII)."""
        if ticker == 'CASH':
            return (1.0, 0, "LIVE")
        
        try:
            if self.price_fetcher is not None:
                try:
                    fetch_result = self.price_fetcher(ticker=ticker)
                except TypeError:
                    fetch_result = self.price_fetcher(ticker)
                except Exception as e:
                    fetch_result = None
                    print(f"[WARN] Injected price_fetcher failed for {ticker}: {e}")

                if fetch_result is not None:
                    if isinstance(fetch_result, tuple) and len(fetch_result) == 3:
                        return fetch_result
                    print(f"[WARN] Injected price_fetcher returned invalid payload for {ticker}; falling back")

            import pytz
            now_et = datetime.now(pytz.timezone('US/Eastern'))
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            market_ticker = self._normalize_market_ticker(ticker)
            t = yf.Ticker(market_ticker)
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            try:
                hist = t.history(period='1d', interval='5m')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    timestamp = hist.index[-1]
                    data_age_minutes = (now_et - timestamp).total_seconds() / 60
                    
                    if data_age_minutes < 10:
                        market_status = "LIVE"
                    elif data_age_minutes < 60:
                        market_status = "RECENT"
                    else:
                        market_status = "STALE"
                    
                    print(f"[PRICE] {ticker}: ${price:.2f} (5m @ {timestamp.strftime('%H:%M ET')}, {data_age_minutes:.0f}min ago) {market_status}")
                    return (price, data_age_minutes, market_status)
            except Exception as e:
                print(f"[PRICE] {ticker}: 5m history failed - {e}")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            try:
                hist = t.history(period='1d', interval='1m')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    timestamp = hist.index[-1]
                    data_age_minutes = (now_et - timestamp).total_seconds() / 60
                    
                    if data_age_minutes < 5:
                        market_status = "LIVE"
                    elif data_age_minutes < 60:
                        market_status = "RECENT"
                    else:
                        market_status = "STALE"
                    
                    print(f"[PRICE] {ticker}: ${price:.2f} (1m @ {timestamp.strftime('%H:%M ET')}, {data_age_minutes:.0f}min ago) {market_status}")
                    return (price, data_age_minutes, market_status)
            except Exception as e:
                print(f"[PRICE] {ticker}: 1m history failed - {e}")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            try:
                info = t.info
                for price_field in ['currentPrice', 'regularMarketPrice', 'ask', 'bid']:
                    if price_field in info and info[price_field]:
                        price = float(info[price_field])
                        if price > 0:
                            print(f"[PRICE] {ticker}: ${price:.2f} (from info.{price_field}) STALE (no timestamp)")
                            return (price, 99999, "STALE")
            except Exception as e:
                print(f"[PRICE] {ticker}: info failed - {e}")
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            try:
                hist = t.history(period='5d', interval='1d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    date = hist.index[-1]
                    # NOTE: comment omitted (was garbled/non-ASCII).
                    data_age_minutes = (now_et - date).total_seconds() / 60
                    print(f"[PRICE] {ticker}: ${price:.2f} (from daily close {date.strftime('%Y-%m-%d')}, {data_age_minutes:.0f}min ago) STALE")
                    return (price, data_age_minutes, "STALE")
            except Exception as e:
                print(f"[PRICE] {ticker}: daily history failed - {e}")
                
        except Exception as e:
            print(f"[ERROR] All price methods failed for {ticker}: {e}")
        
        return (None, 99999, "STALE")

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
                    remaining_budget -= new_notional
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

        self.current_regime['dynamic_min_cash'] = cash_target
        self.current_regime['cash_target'] = cash_target
        self.current_regime['cash_target_components'] = {
            'base_cash_from_regime': base_cash_from_regime,
            'macro_cash_slope': macro_cash_slope,
            'macro_risk_score_smoothed': macro_risk_score_smoothed,
            'macro_cash_from_risk': macro_cash_from_risk,
            'macro_cash_from_topics': macro_cash_from_topics
        }

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
        cooldown_minutes = execution_config.get('rebalance_cooldown_minutes', 0)
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
        
        if cooldown_minutes > 0 and self.last_rebalance_time is not None:
            time_since_last = (datetime.now() - self.last_rebalance_time).total_seconds() / 60
            if time_since_last < cooldown_minutes:
                remaining = cooldown_minutes - time_since_last
                print(f"[COOLDOWN] Skipping rebalance - {remaining:.1f} minutes remaining")
                return []
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        stale_price_skip_minutes = execution_config.get('stale_price_skip_minutes', 60)
        max_stale_ratio = execution_config.get('max_stale_ratio', 0.3)
        stale_policy_cfg = execution_config.get('price_stale_policy', {})
        allow_buy_status = {s.upper() for s in stale_policy_cfg.get('allow_buy', ['LIVE', 'RECENT'])}
        allow_sell_status = {s.upper() for s in stale_policy_cfg.get('allow_sell', ['LIVE', 'RECENT', 'STALE'])}
        
        price_info = {}  # {ticker: (price, data_age_minutes, market_status)}
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        for ticker in self.positions.keys():
            price, age, status = self.get_current_price(ticker)
            if price is not None:
                price_info[ticker] = (price, age, status)
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        for ticker in target_weights.keys():
            if ticker == 'CASH' or ticker in price_info:
                continue
            price, age, status = self.get_current_price(ticker)
            if price is not None:
                price_info[ticker] = (price, age, status)
        
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
        stale_candidate_count = 0  # NOTE: comment omitted (was garbled/non-ASCII).
        policy_skip_count = 0  # NOTE: comment omitted (was garbled/non-ASCII).
        candidate_count = 0
        
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
            candidate_count += 1
            if status == "STALE" and age > stale_price_skip_minutes:
                stale_candidate_count += 1

            # NOTE: comment omitted (was garbled/non-ASCII).
            if side == 'BUY' and status not in allow_buy_status:
                policy_skip_count += 1
                print(f"[SKIP] {ticker} BUY status={status} not in allow_buy={sorted(allow_buy_status)}")
                continue
            if side == 'SELL' and status not in allow_sell_status:
                policy_skip_count += 1
                print(f"[SKIP] {ticker} SELL status={status} not in allow_sell={sorted(allow_sell_status)}")
                continue

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
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        stale_ratio_candidates = stale_candidate_count / candidate_count if candidate_count > 0 else 0
        
        print(f"\n[STALE CHECK] Candidate tickers: {candidate_count}, STALE: {stale_candidate_count}, Ratio: {stale_ratio_candidates:.1%}")
        
        if stale_ratio_candidates > max_stale_ratio:
            print(f"[STALE ABORT] STALE ratio {stale_ratio_candidates:.1%} > threshold {max_stale_ratio:.1%}, aborting rebalance")
            if candidate_count > 0 and stale_candidate_count == candidate_count:
                print("[INFO] All candidate trades depend on STALE prices. "
                      "This typically happens when market is closed or data is delayed.")
            abort_trace = f"stale_abort_ratio_{stale_ratio_candidates:.1%}_gt_{max_stale_ratio:.1%}"
            # NOTE: comment omitted (was garbled/non-ASCII).
            self.current_stale_info = {
                'stale_count': stale_count,
                'stale_ratio': stale_ratio,
                'price_stale_skip': policy_skip_count > 0,
                'price_stale_abort': True,  # NOTE: comment omitted (was garbled/non-ASCII).
                'stale_ratio_candidates': stale_ratio_candidates,
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
        
        # NOTE: comment omitted (was garbled/non-ASCII).
        self.current_stale_info['price_stale_skip'] = policy_skip_count > 0
        self.current_stale_info['price_stale_abort'] = False
        self.current_stale_info['stale_candidate_count'] = stale_candidate_count
        self.current_stale_info['stale_ratio_candidates'] = stale_ratio_candidates
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
        
        turnover_scale = 1.0
        turnover_capped = False
        
        print(f"\n[TURNOVER] Planned(pre): ${turnover_notional_pre:,.2f}, Limit: ${turnover_limit:,.2f} ({max_turnover_pct:.1%})")
        if bool(planner_cfg.get('enable_trade_planner', False)):
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

                planned_after = sum(abs(float(t.get('desired_trade_value', 0.0) or 0.0)) for t in planned_trades if isinstance(t, dict))
                turnover_scale = (planned_after / turnover_notional_pre) if turnover_notional_pre > 0 else 1.0
                turnover_capped = planned_after + 1e-9 < turnover_notional_pre
                print(
                    f"[PLANNER] enabled=true turnover_limit=${planner_meta.get('turnover_limit', 0.0):,.2f} "
                    f"used_total=${planner_meta.get('turnover_used_total', 0.0):,.2f} "
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
                turnover_scale = 1.0
                turnover_capped = False
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
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            trades.append({
                'timestamp': datetime.now().isoformat(),
                'cycle': self.current_cycle,
                'ticker': ticker,
                'side': 'SELL',
                'quantity': sell_qty,
                'price': price,
                'notional': proceeds,
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
            
            # NOTE: comment omitted (was garbled/non-ASCII).
            trades.append({
                'timestamp': datetime.now().isoformat(),
                'cycle': self.current_cycle,
                'ticker': ticker,
                'side': 'BUY',
                'quantity': buy_qty,
                'price': price,
                'notional': required_cash,
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
            self.last_rebalance_time = datetime.now()  # NOTE: comment omitted (was garbled/non-ASCII).
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

            trades.append({
                'timestamp': now.isoformat(),
                'ticker': h['ticker'],
                'side': 'SELL',
                'quantity': sell_qty,
                'price': h['price'],
                'notional': proceeds,
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
            self.last_rebalance_time = now
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
            'timestamp': datetime.now().isoformat(),
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
            'diagnostic_hint': self.last_diagnostic_hint
        }
        
        self.portfolio_snapshots.append(snapshot)
        self.equity_curve.append((datetime.now(), total_equity, self.cash, positions_value))

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
        self.write_live_snapshot(snapshot)
        
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

    def generate_live_summary(self):
        """def generate_live_summary: docstring omitted (was garbled/non-ASCII)."""
        if not self.portfolio_snapshots:
            return
        
        final_snapshot = self.portfolio_snapshots[-1]
        
        summary_path = self.config['reporting']['summary_report_path'].replace('.txt', '_live.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
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
        print(f"\n{'='*60}")
        print(f"Cycle {self.current_cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        now = datetime.now()

        # Macro refresh decoupling
        if self.last_macro_time is None:
            self.refresh_macro_cache(now=now)
            self.current_macro_reused = False
            print(f"[MACRO REFRESH] First refresh completed")
        else:
            macro_elapsed = (now - self.last_macro_time).total_seconds() / 60
            if macro_elapsed >= self.macro_refresh_minutes:
                self.refresh_macro_cache(now=now)
                self.current_macro_reused = False
                print(f"[MACRO REFRESH] Refreshed after {macro_elapsed:.1f} minutes")
            else:
                self.current_macro_reused = True
                self._sync_current_macro_from_cache()
                print(f"[MACRO REFRESH] Reused cached macro ({macro_elapsed:.1f}m < {self.macro_refresh_minutes}m)")

        # Signal refresh decoupling (target weights)
        if not self.cached_target_weights or self.last_signal_time is None:
            target_weights = self.calculate_target_weights()
            self.cached_target_weights = dict(target_weights)
            self.last_signal_time = now
            self.current_weights_reused = False
            print("[SIGNAL REFRESH] First target weight calculation completed")
        else:
            signal_elapsed = (now - self.last_signal_time).total_seconds() / 60
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
        print(f"ENGINE_VERSION: v2.11.3-2026-02-09")
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
        with open(snapshots_path, 'w', encoding='utf-8') as f:
            for snapshot in self.portfolio_snapshots:
                f.write(json.dumps(snapshot) + '\n')
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
        
        with open(report_path, 'w', encoding='utf-8') as f:
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


def main():
    """def main: docstring omitted (was garbled/non-ASCII)."""
    import sys
    
    config_path = 'paper_config.json'
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print(f"Loading config: {config_path}")
    
    engine = PaperTradingEngine(config_path)
    engine.run()


if __name__ == '__main__':
    main()





