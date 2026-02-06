"""
GlobalWatch Paper Trading Module
全自动无人干预的模拟交易系统

⚠️ SIMULATION ONLY - NO REAL BROKER CONNECTION
"""

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
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ChromaDB for macro signals
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not available - macro integration disabled")

# 设置无缓冲输出，解决 Windows Terminal 延迟问题
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 安全检查：确保不会连接真实 broker
REAL_BROKER_KEYWORDS = ['alpaca', 'interactive_brokers', 'ib_insync', 'robinhood', 'td_ameritrade']
for keyword in REAL_BROKER_KEYWORDS:
    try:
        __import__(keyword)
        raise RuntimeError(f"⚠️ SAFETY VIOLATION: Detected real broker library '{keyword}'. Paper trading is SIMULATION ONLY!")
    except ImportError:
        pass  # Good, no real broker library


class MacroSignalAdapter:
    """宏观信号适配器 - 连接 GlobalWatch ChromaDB"""
    
    def __init__(self, config):
        """初始化宏观信号适配器"""
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
            
            print(f"[MACRO] ✅ Connected to ChromaDB: {chroma_path}/{collection_name}")
        except Exception as e:
            print(f"[MACRO] ⚠️ Failed to connect to ChromaDB: {e}")
            self.enabled = False

    def _extract_source_key(self, metadata):
        """提取信号来源键（source/publisher/channel 等字段）。"""
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
        """将可选数值字段安全转换为 float（失败返回 None）。"""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_correct_flag(self, value):
        """解析 correct_* 字段为 [0,1] 区间。"""
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
        """向 rolling accuracy 序列追加样本并保留固定窗口。"""
        if correct_value is None:
            return

        values = history_map.setdefault(key, [])
        values.append(float(correct_value))

        if len(values) > self.quality_window:
            del values[:-self.quality_window]

    def _update_quality_calibration(self, signals):
        """读取 VERIFIED/correct_* 回填字段并更新 theme/source 的 accuracy。"""
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
        """返回 accuracy_factor 以及采用的 rolling accuracy 信息。"""
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
    
    def fetch_recent_signals(self, n=50):
        """获取最近的 N 条信号（仅 PENDING 或 VERIFIED）"""
        if not self.enabled:
            return []
        
        try:
            # 获取所有信号
            results = self.signals_collection.get(
                include=['metadatas', 'documents']
            )
            
            if not results['ids']:
                print("[MACRO] No signals found in database")
                return []
            
            # 过滤状态并按时间排序
            signals = []
            for i, metadata in enumerate(results['metadatas']):
                status = metadata.get('status', 'UNKNOWN')
                
                if status in ['PENDING', 'VERIFIED']:
                    signals.append({
                        'id': results['ids'][i],
                        'metadata': metadata,
                        'document': results['documents'][i] if i < len(results['documents']) else ''
                    })
            
            # 按时间戳排序（最新的在前）
            signals.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
            
            # 取最近 N 条
            recent_signals = signals[:n]
            
            print(f"[MACRO] Fetched {len(recent_signals)} recent signals (from {len(signals)} valid)")
            
            return recent_signals
            
        except Exception as e:
            print(f"[MACRO] Error fetching signals: {e}")
            return []
    
    def compute_signal_weight(self, signal_timestamp):
        """计算信号权重（基于时间衰减）"""
        try:
            # 解析时间戳
            signal_time = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
            now = datetime.now(signal_time.tzinfo) if signal_time.tzinfo else datetime.now()
            
            # 计算年龄（小时）
            age_hours = (now - signal_time).total_seconds() / 3600
            
            # 指数衰减：w = exp(-lambda * age_hours)
            decay_lambda = self.macro_config.get('decay_lambda_per_hour', 0.15)
            weight = np.exp(-decay_lambda * age_hours)
            
            return weight, age_hours
            
        except Exception as e:
            print(f"[MACRO] Error computing weight: {e}")
            return 0.0, 0.0
    
    def analyze_signals(self):
        """分析宏观信号并输出 macro_risk_score + tilts
        
        严格的 k-of-n 确认机制：
        A1) 只保留 signal_max_age_hours 内的信号
        A2) 每个 theme 取最近 n 条，统计 bullish/bearish 数量，>= k 才确认
        A3) 时间衰减仅用于强度计算，不用于确认
        A4) 返回增强的 confirmed_topics 结构
        A5) macro_risk_score 只由确认主题贡献，risk_off 类加分，risk_on 类可抵消
        A6) 详细调试日志
        
        Returns:
            macro_risk_score: 0-10，越大越 risk-off
            confirmed_topics: List[{theme, direction, strength, confidence_effective, top_sources, newest_timestamp}]
            macro_tilts: {ticker: tilt_delta}
            signal_summary: 信号统计摘要
        """
        if not self.enabled:
            return 0.0, [], {}, {}
        
        print(f"\n[MACRO] Analyzing macro signals from GlobalWatch...")
        
        # 获取所有信号
        all_signals = self.fetch_recent_signals(n=200)
        
        if not all_signals:
            print("[MACRO] No signals to analyze")
            return 0.0, [], {}, {}

        # 先更新基于 VERIFIED/correct_* 的质量校准
        quality_summary = self._update_quality_calibration(all_signals)
        
        # A1) 配置参数
        confirm_k, confirm_n = self.macro_config.get('confirm_k_of_n', [2, 3])
        signal_max_age_hours = self.macro_config.get('signal_max_age_hours', 48)
        decay_lambda = self.macro_config.get('decay_lambda_per_hour', 0.15)
        
        now = datetime.now()
        
        # A1) 过滤：只保留 signal_max_age_hours 内的信号
        valid_signals = []
        for signal in all_signals:
            metadata = signal['metadata']
            timestamp_str = metadata.get('timestamp', '')
            
            try:
                signal_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                # 统一时区
                if signal_time.tzinfo:
                    now_aware = datetime.now(signal_time.tzinfo)
                else:
                    now_aware = now
                    signal_time = signal_time.replace(tzinfo=None)
                
                age_hours = (now_aware - signal_time).total_seconds() / 3600
                
                if age_hours > signal_max_age_hours:
                    continue  # 超过窗口，丢弃
                
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
                # 无效时间戳，跳过
                continue
        
        print(f"[MACRO] Filtered {len(valid_signals)}/{len(all_signals)} signals within {signal_max_age_hours}h window")
        
        if not valid_signals:
            print("[MACRO] No valid signals after age filtering")
            return 0.0, [], {}, {}
        
        # A2) 按主题分组
        theme_groups = {}
        for sig in valid_signals:
            theme = sig['metadata'].get('theme', 'unknown')
            if theme not in theme_groups:
                theme_groups[theme] = []
            theme_groups[theme].append(sig)
        
        # A2) 每个主题：按时间排序，取最近 n 条
        for theme in theme_groups:
            theme_groups[theme].sort(key=lambda x: x['timestamp'], reverse=True)
            theme_groups[theme] = theme_groups[theme][:confirm_n]
        
        # A2) 确认机制：统计 bullish/bearish 数量
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
            
            # A2) 确认逻辑：bullish_count >= k 或 bearish_count >= k
            confirmed_direction = None
            confirmed_items = []
            
            if bullish_count >= confirm_k and bullish_count > bearish_count:
                confirmed_direction = 'bullish'
                confirmed_items = bullish_items
                status = f"✅ BULLISH ({bullish_count}/{len(signals_list)})"
            elif bearish_count >= confirm_k and bearish_count > bullish_count:
                confirmed_direction = 'bearish'
                confirmed_items = bearish_items
                status = f"✅ BEARISH ({bearish_count}/{len(signals_list)})"
            else:
                # 未确认
                max_count = max(bullish_count, bearish_count)
                needed = confirm_k - max_count
                if needed > 0:
                    status = f"⏳ NEED {needed} MORE"
                else:
                    status = f"⚖️ CONFLICTED ({bullish_count}v{bearish_count})"
            
            print(f"{theme:<20} {bullish_count:>5} {bearish_count:>5} {neutral_count:>5} {status:<20}")
            
            # A3) 确认后才计算强度（时间衰减）
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
                
                # 提取 top sources（最多3条）
                top_sources = []
                for item in confirmed_items[:3]:
                    doc_preview = item['document'][:100] if item['document'] else 'N/A'
                    top_sources.append(doc_preview)
                
                # 最新时间戳
                newest_timestamp = confirmed_items[0]['timestamp'] if confirmed_items else ''
                
                # A4) 增强的 confirmed_topics 结构
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
                
                # A6) 调试日志
                print(f"  [DEBUG] {theme}: direction={confirmed_direction}, "
                      f"count={bullish_count if confirmed_direction=='bullish' else bearish_count}/"
                      f"{bearish_count if confirmed_direction=='bullish' else bullish_count}, "
                      f"strength={strength:.3f}, conf_raw={confidence_raw_avg:.3f}, "
                      f"conf_eff={confidence_effective:.3f}, acc_factor={accuracy_factor_avg:.3f}, "
                      f"newest={newest_timestamp[:19]}")
        
        print("-" * 70)
        
        # A5) macro_risk_score 计算：只由确认主题贡献
        risk_off_themes = ['risk_off', 'recession', 'rates_up', 'credit_stress', 'inflation_risk', 
                           'geopolitical_risk', 'market_crash', 'volatility_spike']
        risk_on_themes = ['risk_on', 'soft_landing', 'growth_acceleration', 'dovish_fed', 
                          'earnings_beat', 'tech_rally']
        
        risk_score = 0.0
        
        for topic in confirmed_topics:
            theme_lower = topic['theme'].lower()
            strength = topic['strength']
            direction = topic['direction']
            
            # risk_off 类主题
            is_risk_off = any(keyword in theme_lower for keyword in risk_off_themes)
            # risk_on 类主题
            is_risk_on = any(keyword in theme_lower for keyword in risk_on_themes)
            
            if is_risk_off:
                if direction == 'bearish':
                    # risk_off 主题看跌 → 增加风险分数
                    risk_score += min(strength * 2.0, 3.0)  # 单主题最多贡献 3 分
                elif direction == 'bullish':
                    # risk_off 主题看涨 → 也增加风险（例如"通胀风险看涨"）
                    risk_score += min(strength * 1.5, 2.5)
            
            elif is_risk_on:
                if direction == 'bullish':
                    # risk_on 主题看涨 → 抵消风险分数
                    risk_score -= min(strength * 1.0, 2.0)
                elif direction == 'bearish':
                    # risk_on 主题看跌 → 增加风险
                    risk_score += min(strength * 1.0, 2.0)
        
        # Clip 到 [0, 10]
        risk_score = max(0.0, min(risk_score, 10.0))
        
        print(f"\n[MACRO] Risk Score: {risk_score:.1f}/10.0 (from {len(confirmed_topics)} confirmed topics)")
        print(f"[MACRO] Confirmed Topics: {len(confirmed_topics)}")
        
        # 生成 macro_tilts（基于 macro_mapping）
        macro_tilts = self._generate_tilts(confirmed_topics)
        
        if macro_tilts:
            print(f"[MACRO] Asset Tilts:")
            for ticker, tilt in macro_tilts.items():
                print(f"  {ticker}: {tilt:+.2%}")
        
        # 信号摘要
        signal_summary = {
            'total_signals_fetched': len(all_signals),
            'valid_signals_in_window': len(valid_signals),
            'themes_analyzed': len(theme_groups),
            'confirmed_topics': len(confirmed_topics),
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
        """根据确认的主题生成资产倾斜"""
        macro_mapping = self.config.get('macro_mapping', {})
        tilt_max_delta = self.macro_config.get('tilt_max_delta', 0.02)
        
        tilts = {}
        
        for topic in confirmed_topics:
            theme = topic['theme'].lower()
            direction = topic['direction']
            
            # 查找匹配的映射规则
            for rule_name, rule_config in macro_mapping.items():
                # 简单匹配：主题名包含规则名
                if rule_name.lower() in theme or theme in rule_name.lower():
                    
                    # 应用倾斜规则
                    if 'tilt' in rule_config:
                        for ticker, tilt_value in rule_config['tilt'].items():
                            # 根据方向调整倾斜
                            if direction == 'bearish':
                                tilt_value = -abs(tilt_value)  # 反向倾斜
                            
                            # 累加倾斜（但不超过上限）
                            current_tilt = tilts.get(ticker, 0.0)
                            new_tilt = current_tilt + tilt_value
                            
                            # 限制在 [-tilt_max_delta, +tilt_max_delta]
                            tilts[ticker] = max(-tilt_max_delta, min(new_tilt, tilt_max_delta))
        
        return tilts


class PaperTradingEngine:
    """模拟交易引擎"""
    
    def __init__(self, config_path='paper_config.json'):
        """初始化"""
        self.config = self.load_config(config_path)
        self.validate_config()
        
        # 初始化状态
        self.cash = self.config['initial_cash_usd']
        self.initial_cash = self.cash
        self.positions = {}  # {ticker: quantity}
        self.cost_basis = {}  # {ticker: average_cost} 追踪成本基础
        self.equity_curve = []  # [(timestamp, equity, cash, positions_value)]
        self.trades_log = []  # 交易记录
        self.portfolio_snapshots = []  # 组合快照
        
        # 运行状态
        self.start_time = None
        self.end_time = None
        self.current_cycle = 0
        self.peak_equity = self.cash
        self.status = "READY"  # READY/RUNNING/COMPLETED
        self.last_rebalance_time = None  # 用于 cooldown 检查
        self.current_regime = {}  # 当前市场状态（Regime Filter）
        self.current_macro = {}  # 当前宏观信号（Macro Integration）
        self.current_stale_info = {}  # B3) 当前价格新鲜度信息
        self.current_turnover_info = {}  # C4) 当前换手信息
        self.forced_until_time = None  # risk_off_forced 结束时间
        self.forced_regime_reason = ""
        self.scoreboard_history = []  # 2w scoreboard records
        self.last_diagnostic_hint = ""
        self.current_weights_reused = False
        self.current_macro_reused = False
        
        # E1) 宏观信号平滑
        self.macro_risk_score_history = []  # 保存最近 N 次的 risk_score
        self.macro_smoothing_window = self.config.get('macro_integration', {}).get('smoothing_window', 3)
        self.macro_smoothing_method = self.config.get('macro_integration', {}).get('smoothing_method', 'median')  # 'median' or 'ewma'
        self.macro_ewma_alpha = self.config.get('macro_integration', {}).get('ewma_alpha', 0.4)
        
        # E2) 宏观动作冷却
        self.last_macro_cash_target = self.config['objectives']['min_cash_pct']  # 上次的现金目标
        self.macro_cooldown_cycles = self.config.get('macro_integration', {}).get('cooldown_cycles', 2)
        self.macro_cooldown_remaining = 0  # 剩余冷却周期数
        
        # 价格缓存（避免重复请求）
        self.price_cache = {}  # {ticker: (price, timestamp)}
        self.price_cache_duration = 60  # 缓存60秒

        # Signal/Macro refresh decoupling state
        execution_config = self.config.get('execution', {})
        self.signal_refresh_minutes = execution_config.get('signal_refresh_minutes', 1440)
        self.macro_refresh_minutes = execution_config.get('macro_refresh_minutes', 60)
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
        
        # 宏观信号适配器
        self.macro_adapter = MacroSignalAdapter(self.config)
        
        # 尝试恢复之前的状态
        self.resume_from_checkpoint()
        
        # 创建输出目录
        os.makedirs('outputs', exist_ok=True)

        # 加载 scoreboard 历史（用于连续窗口诊断）
        self.load_scoreboard_history()
        
        # 设置随机种子（确保可复现）
        np.random.seed(self.config['safety']['random_seed'])
        
        print("✅ Paper Trading Engine initialized")
        print(f"   Initial Cash: ${self.cash:,.2f}")
        print(f"   Duration: {self.config['duration_hours']} hours")
        print(f"   Rebalance Interval: {self.config['rebalance_minutes']} minutes")
        print(f"   Universe: {len(self.config['universe'])} assets")
    
    def load_config(self, config_path):
        """加载配置文件"""
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
        stale_policy = execution_config.setdefault('price_stale_policy', {})
        stale_policy.setdefault('allow_buy', ['LIVE', 'RECENT'])
        stale_policy.setdefault('allow_sell', ['LIVE', 'RECENT', 'STALE'])
        macro_config = config.setdefault('macro_integration', {})
        macro_config.setdefault('macro_allow_new_positions', ['TLT', 'GLD'])
        reporting_config = config.setdefault('reporting', {})
        reporting_config.setdefault('scoreboard_path', 'outputs/scoreboard.jsonl')
        
        return config
    
    def validate_config(self):
        """验证配置"""
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
        assert self.config.get('execution', {}).get('price_stale_policy', {}).get('allow_buy'), "execution.price_stale_policy.allow_buy must not be empty"
        assert self.config.get('execution', {}).get('price_stale_policy', {}).get('allow_sell'), "execution.price_stale_policy.allow_sell must not be empty"
        
        print("✅ Safety checks passed: SIMULATION ONLY mode confirmed")
    
    def resume_from_checkpoint(self):
        """从检查点恢复之前的运行状态"""
        snapshots_path = self.config['reporting']['portfolio_snapshots_path']
        trades_path = self.config['reporting']['trades_log_path']
        
        # 检查是否存在检查点文件
        if not os.path.exists(snapshots_path):
            print("ℹ️  No checkpoint found - starting fresh")
            return
        
        try:
            print("\n" + "="*60)
            print("🔄 CHECKPOINT DETECTED - Attempting to resume")
            print("="*60)
            
            # 1. 读取快照文件
            with open(snapshots_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    print("⚠️  Checkpoint file is empty - starting fresh")
                    return
                
                # 加载所有快照
                for line in lines:
                    snapshot = json.loads(line.strip())
                    self.portfolio_snapshots.append(snapshot)
            
            # 2. 恢复最后的状态
            last_snapshot = self.portfolio_snapshots[-1]
            
            self.cash = last_snapshot['cash']
            self.current_cycle = last_snapshot['cycle'] + 1  # 继续下一个周期
            self.status = "RESUMED"
            
            # 恢复持仓
            self.positions = {}
            for ticker, pos in last_snapshot['positions'].items():
                self.positions[ticker] = pos['quantity']
            
            # 恢复权益曲线
            for snapshot in self.portfolio_snapshots:
                timestamp = datetime.fromisoformat(snapshot['timestamp'])
                self.equity_curve.append((
                    timestamp,
                    snapshot['total_equity'],
                    snapshot['cash'],
                    snapshot['positions_value']
                ))
            
            # 更新峰值权益
            self.peak_equity = max(s['total_equity'] for s in self.portfolio_snapshots)
            
            # 3. 读取交易记录
            if os.path.exists(trades_path):
                trades_df = pd.read_csv(trades_path)
                self.trades_log = trades_df.to_dict('records')
                
                # 从交易记录重建成本基础
                self.rebuild_cost_basis()
            
            # 4. 显示恢复信息
            print(f"✅ Successfully resumed from checkpoint")
            print(f"   Last cycle: {last_snapshot['cycle']}")
            print(f"   Last update: {last_snapshot['timestamp']}")
            print(f"   Cash: ${self.cash:,.2f}")
            print(f"   Positions: {len(self.positions)} holdings")
            print(f"   Total equity: ${last_snapshot['total_equity']:,.2f}")
            print(f"   Return: {last_snapshot['total_return']:.2%}")
            print(f"   Historical snapshots: {len(self.portfolio_snapshots)}")
            print(f"   Historical trades: {len(self.trades_log)}")
            
            # 显示当前持仓
            if self.positions:
                print(f"\n   Current Holdings:")
                for ticker, qty in sorted(self.positions.items()):
                    cost = self.cost_basis.get(ticker, 0)
                    print(f"     {ticker}: {qty} shares (avg cost: ${cost:.2f})")
            
            print("="*60 + "\n")
            
            # 询问用户是否继续
            response = input("Continue from checkpoint? (y/n): ").strip().lower()
            if response != 'y':
                print("Starting fresh as requested...")
                self.clear_checkpoint()
                return
            
        except Exception as e:
            print(f"⚠️  Failed to resume from checkpoint: {e}")
            print("   Starting fresh...")

    def load_scoreboard_history(self):
        """加载已有 scoreboard 历史，便于连续窗口诊断。"""
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
        """在每次 snapshot 后写入一条 2w scoreboard 记录。"""
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
    
    def rebuild_cost_basis(self):
        """从交易记录重建成本基础"""
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
                
                # 加权平均成本
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
    
    def clear_checkpoint(self):
        """清除检查点，重新开始"""
        self.cash = self.config['initial_cash_usd']
        self.positions = {}
        self.cost_basis = {}
        self.equity_curve = []
        self.trades_log = []
        self.portfolio_snapshots = []
        self.current_cycle = 0
        self.peak_equity = self.cash
        self.status = "READY"
    
    def get_market_data(self, ticker, period='1mo', interval='1d'):
        """获取市场数据"""
        try:
            if ticker == 'CASH':
                return None
            
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval)
            
            if hist.empty:
                print(f"⚠️ No data for {ticker}, skipping")
                return None
            
            return hist
        except Exception as e:
            print(f"⚠️ Error fetching data for {ticker}: {e}")
            return None

    def get_current_price(self, ticker):
        """获取当前价格（实时或最新）
        
        B1) 返回三元组：(price, data_age_minutes, market_status)
        - market_status ∈ {"LIVE", "RECENT", "STALE"}
        - data_age_minutes: 数据时间戳与 now 的分钟差
        - 若无法获取时间戳，market_status = "STALE", data_age_minutes = 99999
        
        Returns:
            (price, data_age_minutes, market_status) or (None, 99999, "STALE")
        """
        if ticker == 'CASH':
            return (1.0, 0, "LIVE")
        
        try:
            import pytz
            now_et = datetime.now(pytz.timezone('US/Eastern'))
            
            # 创建新的 Ticker 对象，避免缓存
            t = yf.Ticker(ticker)
            
            # 方法1: 尝试获取最新的分钟级数据（5m 间隔）
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
            
            # 方法2: 尝试 1m 间隔
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
            
            # 方法3: 尝试 info（无时间戳，视为 STALE）
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
            
            # 方法4: 降级到日线数据（视为 STALE）
            try:
                hist = t.history(period='5d', interval='1d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    date = hist.index[-1]
                    # 计算日线数据的年龄
                    data_age_minutes = (now_et - date).total_seconds() / 60
                    print(f"[PRICE] {ticker}: ${price:.2f} (from daily close {date.strftime('%Y-%m-%d')}, {data_age_minutes:.0f}min ago) STALE")
                    return (price, data_age_minutes, "STALE")
            except Exception as e:
                print(f"[PRICE] {ticker}: daily history failed - {e}")
                
        except Exception as e:
            print(f"[ERROR] All price methods failed for {ticker}: {e}")
        
        return (None, 99999, "STALE")
    
    def calculate_momentum(self, ticker, lookback_days=20):
        """计算动量指标"""
        try:
            hist = self.get_market_data(ticker, period='3mo', interval='1d')
            if hist is None or len(hist) < lookback_days:
                return 0.0
            
            recent_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[-lookback_days]) / hist['Close'].iloc[-lookback_days]
            return float(recent_return)
        except:
            return 0.0
    
    def calculate_volatility(self, ticker, lookback_days=20):
        """计算波动率"""
        try:
            hist = self.get_market_data(ticker, period='3mo', interval='1d')
            if hist is None or len(hist) < lookback_days:
                return 0.20
            
            returns = hist['Close'].pct_change().dropna()
            vol = float(returns.tail(lookback_days).std() * np.sqrt(252))
            return vol
        except:
            return 0.20
    
    def _sync_current_macro_from_cache(self):
        """将 cached_macro 投影到 current_macro，供快照和交易日志使用"""
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
        """按 macro_refresh_minutes 刷新一次宏观信号缓存"""
        if now is None:
            now = datetime.now()

        macro_risk_score_raw, confirmed_topics, macro_tilts_raw, signal_summary = self.macro_adapter.analyze_signals()

        # E1) 平滑 macro_risk_score（仅在真正刷新 macro 时更新）
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

        # E3) 过滤 macro_tilts：只保留 universe 内资产
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

    def calculate_target_weights(self):
        """计算目标权重：风险雷达（现金）+ 趋势放大器（tilt/上限）"""

        def _apply_caps_and_normalize(weights_map, cap_map, invested_budget, fill_gap_max, fill_gap_max_iters, score_map):
            """先裁剪、仅超预算下调；小缺口时可在 headroom 内软补足。"""
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

                    # 优先按当前权重比例补足；若当前权重全为0则按 score 比例；再退化为均匀分配
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

        # ========== 步骤1: Regime 基线 ==========
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

        # ========== 步骤2: 读取 cached macro ==========
        macro_cfg = self.config.get('macro_integration', {})
        execution_cfg = self.config.get('execution', {})
        macro_mapping = self.config.get('macro_mapping', {})
        tilt_max_delta = float(macro_cfg.get('tilt_max_delta', 0.02))
        allow_buy_benchmarks = bool(execution_cfg.get('allow_buy_benchmarks', False))
        macro_risk_score_raw = float(self.cached_macro.get('macro_risk_score_raw', 0.0))
        macro_risk_score_smoothed = float(self.cached_macro.get('macro_risk_score_smoothed', macro_risk_score_raw))
        macro_tilts_filtered = dict(self.cached_macro.get('macro_tilts', {}))
        confirmed_topics = self.cached_macro.get('confirmed_topics', [])
        self._sync_current_macro_from_cache()

        # ========== 通路1: 风险雷达（现金目标） ==========
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

        cash_target_unclipped = base_cash_from_regime + macro_cash_from_risk + macro_cash_from_topics
        cash_target = float(np.clip(cash_target_unclipped, base_cash_from_regime, 0.60))
        self.last_macro_cash_target = cash_target

        print(f"\n[MACRO PATH1] cash_target = base({base_cash_from_regime:.2%}) + "
              f"slope*risk({macro_cash_from_risk:+.2%}) + topic_cash_add({macro_cash_from_topics:+.2%}) -> "
              f"clip[{base_cash_from_regime:.2%},60.00%] = {cash_target:.2%}")
        if macro_cash_topic_details:
            print(f"[MACRO PATH1] topic cash_add details: {', '.join(macro_cash_topic_details)}")

        # ========== 通路2: 趋势放大器（tilt + 单资产上限） ==========
        macro_allow_new_positions = {str(x).upper() for x in macro_cfg.get('macro_allow_new_positions', ['TLT', 'GLD'])}
        defensive_tilt_assets = set(macro_allow_new_positions) | {'CASH', 'TLT', 'GLD'}
        risk_off_mode = regime_state in ('risk_off', 'risk_off_forced')

        # 构造 trade_universe：benchmarks 仅用于 regime/benchmark 计算，不自动进入交易池
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

        # ========== 步骤3: 计算原始评分权重 ==========
        strategy = self.config['strategy']
        lookback = strategy['lookback_days']
        vol_target = strategy['vol_target']
        momentum_weight = strategy['momentum_weight']
        vol_weight = strategy['vol_weight']
        fill_gap_max = float(execution_cfg.get('fill_gap_max', 0.03))
        fill_gap_max_iters = int(execution_cfg.get('fill_gap_max_iters', 2))

        asset_scores = {}

        print(f"\n📊 Evaluating {len(trade_universe_assets)} assets...")
        print(f"{'Ticker':<8} {'Momentum':>10} {'Volatility':>12} {'Score':>10} {'Status':<10}")
        print('-' * 60)

        for asset in trade_universe_assets:
            ticker = asset['ticker']
            if ticker == 'CASH':
                continue

            momentum = self.calculate_momentum(ticker, lookback)
            volatility = self.calculate_volatility(ticker, lookback)
            score = momentum_weight * momentum - vol_weight * (volatility - vol_target)

            asset_scores[ticker] = {'momentum': momentum, 'volatility': volatility, 'score': score}
            status = '✅ BUY' if score > 0 else '❌ SKIP'
            print(f"{ticker:<8} {momentum:>9.2%} {volatility:>11.2%} {score:>9.4f} {status:<10}")

        print('-' * 60)
        positive_assets = {k: v for k, v in asset_scores.items() if v['score'] > 0}
        print(f"Selected {len(positive_assets)} assets with positive scores\n")

        total_score = sum(v['score'] for v in positive_assets.values())
        if total_score > 0:
            raw_weights = {k: v['score'] / total_score for k, v in positive_assets.items()}
        else:
            raw_weights = {}

        # 先应用 cash_target 缩放
        invested_budget = max(0.0, 1.0 - cash_target)
        scaled_weights = {k: v * invested_budget for k, v in raw_weights.items()}

        # 应用 tilts（趋势放大器，不直接强买）
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

        # 生成单资产有效上限（base_max_weight + tilt_delta，delta clip 到 ±tilt_max_delta）
        max_weight_effective = {}
        for asset in trade_universe_assets:
            ticker = asset['ticker']
            if ticker == 'CASH':
                continue
            tilt_delta = float(np.clip(applied_tilts.get(ticker, 0.0), -tilt_max_delta, tilt_max_delta))
            max_weight_effective[ticker] = float(np.clip(base_max_weight + tilt_delta, 0.0, 1.0))

        # 再应用“上限裁剪 + 下调 + 小缺口软补足”
        score_map = {k: max(0.0, float(v['score'])) for k, v in positive_assets.items()}
        adjusted_weights, alloc_diag = _apply_caps_and_normalize(
            scaled_weights,
            max_weight_effective,
            invested_budget,
            fill_gap_max,
            fill_gap_max_iters,
            score_map
        )
        capped_assets = alloc_diag.get('capped_assets', [])

        cash_weight = max(0.0, 1.0 - sum(adjusted_weights.values()))
        adjusted_weights['CASH'] = cash_weight

        # 记录本轮 macro 应用结果，供 snapshot/trade context 使用
        self.current_macro['applied_tilts'] = dict(applied_tilts)
        self.current_macro['blocked_tilts'] = dict(blocked_tilts)
        self.current_macro['blocked_tilts_not_trade_universe'] = dict(blocked_tilts_not_trade_universe)
        self.current_macro['capped_assets'] = list(capped_assets)
        self.current_macro['max_weight_per_asset_effective'] = dict(max_weight_effective)
        self.current_macro['cash_target'] = cash_target
        self.current_macro['allocation_diagnostics'] = dict(alloc_diag)

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
        """执行再平衡 - 带五大保护器：cooldown / weight_threshold / min_notional / stale_price_skip / turnover_cap
        
        C1) 计算计划交易名义金额
        C2) 检查是否超过 turnover_limit
        C3) 如果超过，按比例缩放所有交易
        C4) 记录 turnover_notional / turnover_limit / turnover_scale / turnover_capped
        C5) 不破坏现有三大保护器
        D1-D5) 修复 get_current_price() 接口不一致问题
        """
        
        # D5) 自检：测试 get_current_price() 返回三元组
        if self.positions:
            test_ticker = list(self.positions.keys())[0]
            test_price, test_age, test_status = self.get_current_price(test_ticker)
            print(f"[SELF-CHECK] get_current_price('{test_ticker}') = (price={test_price}, age={test_age}min, status={test_status})")
        
        # ========== 准备交易上下文信息 ==========
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
        
        # ========== 保护器 1: Cooldown 检查 ==========
        execution_config = self.config.get('execution', {})
        cooldown_minutes = execution_config.get('rebalance_cooldown_minutes', 0)
        
        if cooldown_minutes > 0 and self.last_rebalance_time is not None:
            time_since_last = (datetime.now() - self.last_rebalance_time).total_seconds() / 60
            if time_since_last < cooldown_minutes:
                remaining = cooldown_minutes - time_since_last
                print(f"[COOLDOWN] Skipping rebalance - {remaining:.1f} minutes remaining")
                return []
        
        # ========== B2) 获取所有价格并检查新鲜度 ==========
        stale_price_skip_minutes = execution_config.get('stale_price_skip_minutes', 60)
        max_stale_ratio = execution_config.get('max_stale_ratio', 0.3)
        stale_policy_cfg = execution_config.get('price_stale_policy', {})
        allow_buy_status = {s.upper() for s in stale_policy_cfg.get('allow_buy', ['LIVE', 'RECENT'])}
        allow_sell_status = {s.upper() for s in stale_policy_cfg.get('allow_sell', ['LIVE', 'RECENT', 'STALE'])}
        
        price_info = {}  # {ticker: (price, data_age_minutes, market_status)}
        
        # 获取当前持仓的价格
        for ticker in self.positions.keys():
            price, age, status = self.get_current_price(ticker)
            if price is not None:
                price_info[ticker] = (price, age, status)
        
        # 获取目标权重中的价格
        for ticker in target_weights.keys():
            if ticker == 'CASH' or ticker in price_info:
                continue
            price, age, status = self.get_current_price(ticker)
            if price is not None:
                price_info[ticker] = (price, age, status)
        
        # B2) 统计全量价格 STALE 概况（用于快照展示）
        stale_count = 0
        total_count = len(price_info)
        
        for ticker, (price, age, status) in price_info.items():
            if status == "STALE" and age > stale_price_skip_minutes:
                stale_count += 1
        
        stale_ratio = stale_count / total_count if total_count > 0 else 0
        
        print(f"\n[PRICE CHECK] Total tickers: {total_count}, STALE: {stale_count}, Ratio: {stale_ratio:.1%} | "
              f"Policy BUY={sorted(allow_buy_status)} SELL={sorted(allow_sell_status)}")
        
        # 记录正常情况
        self.current_stale_info = {
            'stale_count': stale_count,
            'stale_ratio': stale_ratio,
            'price_stale_skip': False,
            'price_stale_abort': False,
            'decision_trace': ''
        }
        
        # ========== 计算当前持仓价值 ==========
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
        
        # ========== 计算目标价值 ==========
        target_values = {}
        for ticker, weight in target_weights.items():
            if ticker == 'CASH':
                continue
            target_values[ticker] = total_equity * weight
        
        # ========== 保护器 2: Weight Threshold 过滤 ==========
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
        
        # ========== C1) 计算计划交易（通过所有过滤器的）==========
        min_notional = execution_config.get('min_trade_notional_usd', 0)
        
        planned_trades = []  # [{ticker, side, current_value, target_value, desired_trade_value, price, age, status}]
        stale_candidate_count = 0  # 候选交易中 STALE 数量
        policy_skip_count = 0  # 因 price_stale_policy 被跳过的候选数量
        candidate_count = 0
        
        for ticker in tickers_to_trade:
            current_value = current_values.get(ticker, 0.0)
            target_value = target_values.get(ticker, 0.0)
            desired_trade_value = abs(target_value - current_value)
            
            # Min notional 检查
            if desired_trade_value < min_notional:
                print(f"[SKIP] {ticker} trade notional ${desired_trade_value:.2f} < min ${min_notional}")
                continue
            
            # 获取价格信息
            if ticker not in price_info:
                print(f"[SKIP] {ticker} no price info")
                continue
            
            price, age, status = price_info[ticker]
            
            # 确定交易方向
            side = 'BUY' if target_value > current_value else 'SELL'
            
            status = str(status).upper()
            candidate_count += 1
            if status == "STALE" and age > stale_price_skip_minutes:
                stale_candidate_count += 1

            # 统一执行 price_stale_policy
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
            
            planned_trades.append({
                'ticker': ticker,
                'side': side,
                'current_value': current_value,
                'target_value': target_value,
                'desired_trade_value': desired_trade_value,
                'price': price,
                'age': age,
                'status': status
            })
        
        # 2) 全局数据异常保护：检查候选交易 STALE 比例
        stale_ratio_candidates = stale_candidate_count / candidate_count if candidate_count > 0 else 0
        
        print(f"\n[STALE CHECK] Candidate tickers: {candidate_count}, STALE: {stale_candidate_count}, Ratio: {stale_ratio_candidates:.1%}")
        
        if stale_ratio_candidates > max_stale_ratio:
            print(f"[STALE ABORT] STALE ratio {stale_ratio_candidates:.1%} > threshold {max_stale_ratio:.1%}, aborting rebalance")
            abort_trace = f"stale_abort_ratio_{stale_ratio_candidates:.1%}_gt_{max_stale_ratio:.1%}"
            # 3) 记录到 snapshot
            self.current_stale_info = {
                'stale_count': stale_count,
                'stale_ratio': stale_ratio,
                'price_stale_skip': policy_skip_count > 0,
                'price_stale_abort': True,  # 新增：因 STALE 比例过高而中止
                'stale_candidate_count': stale_candidate_count,
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
            print(f"[DECISION] {abort_trace}")
            return []
        
        # 更新 stale_info（正常情况）
        self.current_stale_info['price_stale_skip'] = policy_skip_count > 0
        self.current_stale_info['price_stale_abort'] = False
        self.current_stale_info['stale_candidate_count'] = stale_candidate_count
        self.current_stale_info['stale_ratio_candidates'] = stale_ratio_candidates
        self.current_stale_info['decision_trace'] = f"stale_ok_{stale_ratio_candidates:.1%}_le_{max_stale_ratio:.1%}"
        
        # C1) 计算总换手
        turnover_notional_pre = sum(abs(t['desired_trade_value']) for t in planned_trades)
        
        # C2) 检查换手上限
        max_turnover_pct = execution_config.get('max_turnover_pct_per_rebalance', 0.20)
        turnover_limit = total_equity * max_turnover_pct
        
        turnover_scale = 1.0
        turnover_capped = False
        
        print(f"\n[TURNOVER] Planned(pre): ${turnover_notional_pre:,.2f}, Limit: ${turnover_limit:,.2f} ({max_turnover_pct:.1%})")
        
        # C3) 如果超过上限，按比例缩放
        if turnover_notional_pre > turnover_limit:
            turnover_scale = turnover_limit / turnover_notional_pre
            turnover_capped = True
            print(f"[TURNOVER CAP] Scaling all trades by {turnover_scale:.2%}")
            
            # 缩放所有计划交易
            scaled_trades = []
            for trade in planned_trades:
                scaled_trade_value = trade['desired_trade_value'] * turnover_scale
                
                # 缩放后仍需满足 min_notional
                if scaled_trade_value < min_notional:
                    print(f"[SKIP] {trade['ticker']} scaled notional ${scaled_trade_value:.2f} < min ${min_notional}")
                    continue
                
                trade['desired_trade_value'] = scaled_trade_value
                scaled_trades.append(trade)
            
            planned_trades = scaled_trades
            
            # 缩放后的目标换手（仍为期望名义）
            actual_turnover_scaled = sum(abs(t['desired_trade_value']) for t in planned_trades)
            print(f"[TURNOVER CAP] Planned(after scaling): ${actual_turnover_scaled:,.2f}")
        
        # C4) 先记录 pre，post 在最终成交后更新
        self.current_turnover_info = {
            'turnover_notional': turnover_notional_pre,  # backward compatibility
            'turnover_notional_pre': turnover_notional_pre,
            'turnover_notional_post': 0.0,
            'turnover_limit': turnover_limit,
            'turnover_scale': turnover_scale,
            'turnover_capped': turnover_capped
        }
        
        # ========== 执行交易 ==========
        trades = []
        turnover_notional_post = 0.0
        
        # 先处理卖出
        for trade in [t for t in planned_trades if t['side'] == 'SELL']:
            ticker = trade['ticker']
            price = trade['price']
            desired_notional = abs(trade['desired_trade_value'])
            
            # 缩放后再按整股换算，确保最终成交不超过缩放目标
            current_qty = self.positions.get(ticker, 0)
            sell_qty = int(desired_notional / price)
            sell_qty = min(sell_qty, current_qty)
            
            if sell_qty <= 0:
                continue
            
            # 执行卖出
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
            
            # 构建决策轨迹
            decision_trace = [
                'cooldown_pass',
                'weight_threshold_pass',
                'min_notional_pass',
                f'price_{trade["status"]}_age_{trade["age"]:.0f}min'
            ]
            # 1) STALE SELL 标注
            if trade['status'] == 'STALE':
                decision_trace.append('sell_allowed_on_stale')
            if turnover_capped:
                decision_trace.append(f'turnover_cap_scale_{turnover_scale:.2%}')
            if trade_context['regime_state'] in ('risk_off', 'risk_off_forced'):
                decision_trace.append('risk_off_de-risk')
            decision_trace.extend(alloc_trace)
            
            # 构建完整的交易记录
            trades.append({
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'side': 'SELL',
                'quantity': sell_qty,
                'price': price,
                'cost': cost,
                'reason': 'rebalance',
                'regime_state': trade_context['regime_state'],
                'trend_score': trade_context['trend_score'],
                'cash_target': trade_context['cash_target'],
                'macro_risk_score': trade_context['macro_risk_score'],
                'macro_topics': trade_context['macro_topics'],
                'macro_tilts': trade_context['macro_tilts'],
                'decision_trace': ' | '.join(decision_trace),
                'price_age_minutes': trade['age'],
                'price_status': trade['status']
            })
            
            print(f"[TRADE] SELL {sell_qty} {ticker} @ ${price:.2f} (notional: ${proceeds:.2f}, {trade['status']})")
        
        # 再处理买入
        for trade in [t for t in planned_trades if t['side'] == 'BUY']:
            ticker = trade['ticker']
            price = trade['price']
            desired_notional = abs(trade['desired_trade_value'])
            
            # 缩放后再按整股换算，确保最终成交不超过缩放目标
            buy_qty = int(desired_notional / price)
            
            if buy_qty <= 0:
                continue
            
            # 检查现金是否足够
            cash_before_trade = self.cash
            required_cash = buy_qty * price
            cost = required_cash * self.config['objectives']['transaction_cost_pct']
            total_required = required_cash + cost
            
            if total_required > self.cash:
                # 调整买入数量
                buy_qty = int((self.cash * 0.99) / (price * (1 + self.config['objectives']['transaction_cost_pct'])))
                
                if buy_qty <= 0:
                    print(f"[SKIP] {ticker} insufficient cash")
                    continue
                
                required_cash = buy_qty * price
                cost = required_cash * self.config['objectives']['transaction_cost_pct']
                total_required = required_cash + cost
            
            # 执行买入
            self.cash -= total_required
            turnover_notional_post += required_cash
            old_qty = self.positions.get(ticker, 0)
            old_cost = self.cost_basis.get(ticker, 0)
            
            # 更新持仓
            self.positions[ticker] = old_qty + buy_qty
            
            # 更新成本基础（加权平均）
            if old_qty > 0:
                total_cost = (old_qty * old_cost) + (buy_qty * price)
                self.cost_basis[ticker] = total_cost / (old_qty + buy_qty)
            else:
                self.cost_basis[ticker] = price
            
            # 构建决策轨迹
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
            
            # 构建完整的交易记录
            trades.append({
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'side': 'BUY',
                'quantity': buy_qty,
                'price': price,
                'cost': cost,
                'reason': 'rebalance',
                'regime_state': trade_context['regime_state'],
                'trend_score': trade_context['trend_score'],
                'cash_target': trade_context['cash_target'],
                'macro_risk_score': trade_context['macro_risk_score'],
                'macro_topics': trade_context['macro_topics'],
                'macro_tilts': trade_context['macro_tilts'],
                'decision_trace': ' | '.join(decision_trace),
                'price_age_minutes': trade['age'],
                'price_status': trade['status']
            })
            
            print(f"[TRADE] BUY {buy_qty} {ticker} @ ${price:.2f} (notional: ${required_cash:.2f}, {trade['status']})")

        # 最终成交级别 turnover 回填（用于 snapshot / trades 验收）
        self.current_turnover_info['turnover_notional_post'] = turnover_notional_post
        if turnover_capped and turnover_notional_post > turnover_limit + 1e-6:
            print(f"[WARN] turnover_notional_post ${turnover_notional_post:,.2f} > limit ${turnover_limit:,.2f}")

        for trade in trades:
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
        
        # ========== 更新交易记录和 cooldown 时间 ==========
        if trades:
            self.trades_log.extend(trades)
            self.save_trades_immediately()
            self.last_rebalance_time = datetime.now()  # 只有实际成交才更新
            print(f"[COOLDOWN] Next rebalance allowed after {cooldown_minutes} minutes")
        else:
            print(f"[INFO] No trades executed (all filtered by protections)")
        
        return trades
    
    def _build_trade_context(self):
        """构建交易上下文信息（用于记录交易理由）"""
        # Regime 信息
        regime_state = self.current_regime.get('regime_state', 'neutral')
        trend_score = self.current_regime.get('trend_score', 0.5)
        cash_target = self.current_regime.get('cash_target', self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct']))
        
        # Macro 信息 - E1) 使用平滑后的 risk_score
        macro_risk_score_raw = self.current_macro.get('macro_risk_score', 0.0)
        macro_risk_score_smoothed = self.current_macro.get('macro_risk_score_smoothed', 0.0)
        confirmed_topics = self.current_macro.get('confirmed_topics', [])
        macro_tilts = self.current_macro.get('applied_tilts', self.current_macro.get('macro_tilts', {}))
        alloc_diag = self.current_regime.get('allocation_diagnostics', self.current_macro.get('allocation_diagnostics', {}))
        
        # 格式化 macro_topics 为字符串
        if confirmed_topics:
            topics_str = '; '.join([f"{t['theme']}:{t['direction']}" for t in confirmed_topics[:3]])
        else:
            topics_str = 'none'
        
        # 格式化 macro_tilts 为字符串
        if macro_tilts:
            tilts_str = '; '.join([f"{k}:{v:+.2%}" for k, v in macro_tilts.items()])
        else:
            tilts_str = 'none'
        
        return {
            'regime_state': regime_state,
            'trend_score': trend_score,
            'cash_target': cash_target,
            'macro_risk_score': macro_risk_score_smoothed,  # E1) 使用平滑值
            'macro_risk_score_raw': macro_risk_score_raw,  # 保留原始值
            'macro_topics': topics_str,
            'macro_tilts': tilts_str,
            'macro_tilts_dict': macro_tilts,  # 保留字典格式供内部使用
            'invested_budget': alloc_diag.get('invested_budget', 0.0),
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
        """计算持仓去风险优先级分数，越小越优先卖出。"""
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
        """触发后进入 risk_off_forced，并按最差评分优先结构化减仓。"""
        now = datetime.now()
        execution_config = self.config.get('execution', {})
        regime_config = self.config.get('regime_filter', {})

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

        # 计算当前总权益与目标现金
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
            return []

        # 受 objectives.max_rebalance_pct 与 turnover cap 双重约束
        max_rebalance_pct = float(self.config.get('objectives', {}).get('max_rebalance_pct', 1.0))
        max_turnover_pct = float(execution_config.get('max_turnover_pct_per_rebalance', 1.0))
        cap_pct = min(max_rebalance_pct, max_turnover_pct)
        turnover_limit = total_equity * cap_pct

        turnover_notional_pre = min(cash_needed_initial, positions_value)
        turnover_capped = turnover_notional_pre > turnover_limit
        turnover_scale = (turnover_limit / turnover_notional_pre) if turnover_capped and turnover_notional_pre > 0 else 1.0

        holdings.sort(key=lambda x: x['score'])  # 最差分数优先卖出

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
                'cost': cost,
                'reason': 'circuit_breaker',
                'regime_state': 'risk_off_forced',
                'trend_score': trade_context['trend_score'],
                'cash_target': forced_cash_target,
                'macro_risk_score': trade_context['macro_risk_score'],
                'macro_topics': trade_context['macro_topics'],
                'macro_tilts': trade_context['macro_tilts'],
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

        for t in trades:
            t['turnover_notional_post'] = turnover_notional_post

        if trades:
            self.trades_log.extend(trades)
            self.save_trades_immediately()
            self.last_rebalance_time = now

        print(f"[CIRCUIT] Forced risk-off active until {self.forced_until_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[CIRCUIT] Target cash: {forced_cash_target:.1%}, sold notional: ${turnover_notional_post:,.2f}, remaining cash gap: ${remaining_cash_needed:,.2f}")

        return trades

    def check_risk_controls(self):
        """检查风险控制：触发后进入 risk_off_forced 并执行结构化去风险。"""
        positions_value = 0.0
        for ticker, qty in self.positions.items():
            price, age_min, status = self.get_current_price(ticker)  # D2) 解包三元组
            if price:
                positions_value += qty * price

        total_equity = self.cash + positions_value

        if total_equity > self.peak_equity:
            self.peak_equity = total_equity

        drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        max_dd = self.config['objectives']['max_drawdown_pct']

        if drawdown > max_dd:
            print(f"⚠️ CIRCUIT BREAKER: Drawdown {drawdown:.2%} exceeds limit {max_dd:.2%}")
            trades = self._run_circuit_breaker_derisk(drawdown, max_dd)
            if trades:
                print(f"[CIRCUIT] Executed {len(trades)} structured de-risk trades")
            else:
                print("[CIRCUIT] No de-risk trades executed (already at target cash or constrained)")
            return True

        return False
    def record_snapshot(self):
        """记录组合快照"""
        print(f"[DEBUG] Recording snapshot at {datetime.now().strftime('%H:%M:%S')}")
        import sys; sys.stdout.flush()
        
        positions_value = 0.0
        positions_detail = {}
        
        for ticker, qty in self.positions.items():
            price, age_min, status = self.get_current_price(ticker)  # D2) 解包三元组
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
        
        # 计算基准收益率（如果配置了）
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
                
                # 计算超额收益（策略收益 - 基准平均收益）
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
            # 基准比较字段
            'bench_returns': bench_returns,
            'bench_avg_return': bench_avg_return,
            'bench_dispersion': bench_dispersion,
            'excess_return': excess_return,
            'win_flag': win_flag,
            # Regime Filter 字段
            'regime_state': self.current_regime.get('regime_state', 'neutral'),
            'trend_score': self.current_regime.get('trend_score', 0.5),
            'dynamic_min_cash': self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct']),
            'dynamic_max_weight': self.current_regime.get('dynamic_max_weight', self.config['objectives']['max_weight_per_asset']),
            'cash_target': self.current_regime.get('cash_target', self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct'])),
            'risk_caps_applied': self.current_regime.get('risk_caps_applied', False),
            'forced_until_time': self.current_regime.get('forced_until_time', self.forced_until_time.isoformat() if self.forced_until_time else None),
            'forced_regime_reason': self.current_regime.get('forced_reason', self.forced_regime_reason),
            # Macro Integration 字段
            'macro_risk_score_raw': self.current_macro.get('macro_risk_score', 0.0),
            'macro_risk_score': self.current_macro.get('macro_risk_score', 0.0),  # 原始值
            'macro_risk_score_smoothed': self.current_macro.get('macro_risk_score_smoothed', 0.0),  # E1) 平滑值
            'confirmed_topics_count': len(self.current_macro.get('confirmed_topics', [])),
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
            'macro_tilts_ignored': self.current_macro.get('macro_tilts_ignored', {}),  # E3) 被忽略的 tilts
            'macro_cooldown_remaining': self.macro_cooldown_remaining,  # E2) 冷却剩余周期
            # B3) Price Staleness 字段
            'stale_count': self.current_stale_info.get('stale_count', 0),
            'stale_ratio': self.current_stale_info.get('stale_ratio', 0.0),
            'price_stale_skip': self.current_stale_info.get('price_stale_skip', False),
            'price_stale_abort': self.current_stale_info.get('price_stale_abort', False),  # 新增
            'stale_candidate_count': self.current_stale_info.get('stale_candidate_count', 0),  # 新增
            'stale_ratio_candidates': self.current_stale_info.get('stale_ratio_candidates', 0.0),  # 新增
            'stale_decision_trace': self.current_stale_info.get('decision_trace', ''),
            # C4) Turnover Cap 字段
            'turnover_notional': self.current_turnover_info.get('turnover_notional', 0.0),
            'turnover_notional_pre': self.current_turnover_info.get('turnover_notional_pre', self.current_turnover_info.get('turnover_notional', 0.0)),
            'turnover_notional_post': self.current_turnover_info.get('turnover_notional_post', 0.0),
            'turnover_limit': self.current_turnover_info.get('turnover_limit', 0.0),
            'turnover_scale': self.current_turnover_info.get('turnover_scale', 1.0),
            'turnover_capped': self.current_turnover_info.get('turnover_capped', False),
            'diagnostic_hint': self.last_diagnostic_hint
        }
        
        self.portfolio_snapshots.append(snapshot)
        self.equity_curve.append((datetime.now(), total_equity, self.cash, positions_value))

        # 每个 snapshot 后写一条 scoreboard
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
        
        # 每个周期生成实时摘要
        self.generate_live_summary()
        
        return snapshot

    def save_trades_immediately(self):
        """实时保存交易记录"""
        trades_path = self.config['reporting']['trades_log_path']
        if self.trades_log:
            trades_df = pd.DataFrame(self.trades_log)
            trades_df.to_csv(trades_path, index=False)
        print(f"💾 Trades updated: {trades_path}")
        import sys; sys.stdout.flush()  # 强制刷新输出

    def generate_live_summary(self):
        """生成实时摘要（不等程序结束）"""
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
            
            # Regime Filter 面板
            if final_snapshot.get('regime_state'):
                f.write(f"Market Regime:\n")
                f.write(f"  State: {final_snapshot['regime_state'].upper()}")
                
                if final_snapshot.get('risk_caps_applied'):
                    f.write(" ⚠️ RISK CAPS ACTIVE\n")
                else:
                    f.write("\n")
                
                f.write(f"  Trend Score: {final_snapshot['trend_score']:.1%}\n")
                f.write(f"  Dynamic Min Cash: {final_snapshot['dynamic_min_cash']:.1%}\n")
                f.write(f"  Dynamic Max Weight: {final_snapshot['dynamic_max_weight']:.1%}\n\n")
            
            # Macro Integration 面板
            if final_snapshot.get('macro_risk_score', 0) > 0:
                f.write(f"Macro Signals (GlobalWatch):\n")
                f.write(f"  Risk Score: {final_snapshot['macro_risk_score']:.1f}/10.0\n")
                f.write(f"  Confirmed Topics: {final_snapshot.get('confirmed_topics_count', 0)}\n")
                
                if final_snapshot.get('macro_tilts'):
                    f.write(f"  Active Tilts:\n")
                    for ticker, tilt in final_snapshot['macro_tilts'].items():
                        f.write(f"    {ticker}: {tilt:+.2%}\n")
                f.write("\n")
            
            # 基准比较面板
            if final_snapshot.get('bench_returns'):
                f.write(f"Benchmark Comparison:\n")
                f.write(f"  Strategy Return: {final_snapshot['total_return']:.2%}\n")
                f.write(f"  Benchmark Avg Return: {final_snapshot['bench_avg_return']:.2%}\n")
                f.write(f"  Excess Return: {final_snapshot['excess_return']:.2%}")
                
                if final_snapshot['win_flag']:
                    f.write(" ✅ OUTPERFORM\n")
                else:
                    f.write(" ❌ UNDERPERFORM\n")
                
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
            f.write("⚠️  LIVE DATA - Updates every cycle\n")
            f.write("⚠️  SIMULATION ONLY - NO REAL MONEY\n")
            f.write("="*60 + "\n")
        
        print(f"📊 Live summary updated: {summary_path}")
        import sys; sys.stdout.flush()  # 强制刷新输出

    def get_cost_basis(self, ticker):
        """获取股票的成本基础（平均买入价）"""
        return self.cost_basis.get(ticker, None)
    
    def compute_benchmark_returns(self, tickers, evaluation_days=10):
        """计算基准指数收益率
        
        Args:
            tickers: 基准指数列表，如 ['QQQ', 'SPY', 'VTI', 'DIA']
            evaluation_days: 评估周期（交易日），默认10天约等于2周
        
        Returns:
            bench_returns: {ticker: return_pct}
            bench_avg_return: 平均收益率
            bench_dispersion: 收益率标准差（离散度）
        """
        bench_returns = {}
        
        for ticker in tickers:
            try:
                # 获取 evaluation_days+1 天的收盘价（需要多一天计算收益）
                hist = self.get_market_data(ticker, period='1mo', interval='1d')
                
                if hist is None or len(hist) < evaluation_days + 1:
                    print(f"[BENCHMARK] {ticker}: insufficient data (need {evaluation_days+1} days)")
                    continue
                
                # 计算收益率：(最新价 - N天前价) / N天前价
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
        
        # 计算平均收益和离散度
        returns_list = list(bench_returns.values())
        bench_avg_return = float(np.mean(returns_list))
        bench_dispersion = float(np.std(returns_list))
        
        print(f"[BENCHMARK] Average: {bench_avg_return:.2%}, Dispersion: {bench_dispersion:.2%}")
        
        return bench_returns, bench_avg_return, bench_dispersion
    
    def compute_regime_state(self):
        """计算市场状态（基于四大指数 MA50 趋势）
        
        Returns:
            regime_state: 'risk_on' / 'neutral' / 'risk_off'
            trend_score: 0.0 - 1.0 (满足 close > MA50 的指数比例)
            regime_details: {ticker: {'close': float, 'ma50': float, 'above_ma': bool}}
            dynamic_min_cash: 本轮应使用的最小现金比例
            dynamic_max_weight: 本轮应使用的最大单资产权重
        """
        regime_config = self.config.get('regime_filter', {})

        # circuit breaker 强制 risk_off 窗口优先
        if self.forced_until_time is not None:
            now = datetime.now()
            if now < self.forced_until_time:
                dynamic_min_cash = regime_config.get('cash_risk_off', self.config['objectives']['min_cash_pct'])
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
            # 如果未启用 regime filter，返回默认值
            return 'neutral', 0.5, {}, self.config['objectives']['min_cash_pct'], self.config['objectives']['max_weight_per_asset']

        regime_config = self.config['regime_filter']
        ma_window = regime_config.get('ma_window', 50)
        
        # 获取基准指数列表
        bench_tickers = self.config.get('benchmarks', {}).get('tickers', ['QQQ', 'SPY', 'VTI', 'DIA'])
        
        print(f"\n[REGIME] Computing market regime using MA{ma_window}...")
        
        regime_details = {}
        above_ma_count = 0
        valid_count = 0
        
        for ticker in bench_tickers:
            try:
                # 获取足够的历史数据计算 MA50
                hist = self.get_market_data(ticker, period='3mo', interval='1d')
                
                if hist is None or len(hist) < ma_window:
                    print(f"[REGIME] {ticker}: insufficient data for MA{ma_window}")
                    continue
                
                # 计算 MA50
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
                
                status = "✅ ABOVE" if above_ma else "❌ BELOW"
                print(f"[REGIME] {ticker}: ${latest_close:.2f} vs MA50 ${latest_ma50:.2f} {status}")
                
            except Exception as e:
                print(f"[REGIME] {ticker}: error - {e}")
                continue
        
        if valid_count == 0:
            print("[REGIME] No valid data, defaulting to neutral")
            return 'neutral', 0.5, {}, self.config['objectives']['min_cash_pct'], self.config['objectives']['max_weight_per_asset']
        
        # 计算 trend_score = 满足条件的数量 / 总数
        trend_score = above_ma_count / valid_count
        
        # 根据阈值判断市场状态
        risk_on_threshold = regime_config.get('trend_score_risk_on', 0.75)
        risk_off_threshold = regime_config.get('trend_score_risk_off', 0.50)
        
        if trend_score >= risk_on_threshold:
            regime_state = 'risk_on'
        elif trend_score <= risk_off_threshold:
            regime_state = 'risk_off'
        else:
            regime_state = 'neutral'
        
        # 动态调整现金和权重上限
        dynamic_min_cash = regime_config.get(f'cash_{regime_state}', self.config['objectives']['min_cash_pct'])
        
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
        """运行一个周期"""
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
            regime_icon = "🟢" if snapshot['regime_state'] == 'risk_on' else "🟡" if snapshot['regime_state'] == 'neutral' else "🔴"
            risk_caps = " ⚠️ RISK CAPS" if snapshot.get('risk_caps_applied') else ""
            print(f"Market Regime: {regime_icon} {snapshot['regime_state'].upper()} (trend: {snapshot['trend_score']:.1%}){risk_caps}")

        if snapshot['positions']:
            print(f"\n📊 Current Holdings:")
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
                    pnl_color = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
                else:
                    pnl_str = "N/A"
                    pnl_color = "➡️"

                print(f"{ticker:<8} {qty:>6} ${current_price:>9.2f} ${value:>11,.2f} {weight:>7.1f}% {pnl_color} {pnl_str:>8}")

            print("-" * 60)

        if self.check_risk_controls():
            print("⚠️ Risk control triggered (risk_off_forced active), skipping normal rebalance")
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
        """运行模拟交易"""
        print("\n" + "="*60)
        print("🚀 Starting Paper Trading Simulation")
        print("="*60)
        print(f"⚠️  SIMULATION ONLY - NO REAL MONEY")
        print(f"⚠️  NO BROKER CONNECTION")
        print("="*60 + "\n")
        
        # F4) 版本指纹自检
        print("="*60)
        print("📋 ENGINE VERSION FINGERPRINT")
        print("="*60)
        print(f"ENGINE_VERSION: v2.5-ABCDEF-2026-02-05")
        print(f"HAS_MACRO_SMOOTH: {hasattr(self, 'macro_risk_score_history')}")
        
        # 测试 get_current_price 返回三元组
        try:
            test_result = self.get_current_price("QQQ")
            is_tuple = isinstance(test_result, tuple) and len(test_result) == 3
            print(f"PRICE_API_RETURNS_TUPLE: {is_tuple}")
            if is_tuple:
                print(f"  └─ Sample: get_current_price('QQQ') = (price={test_result[0]}, age={test_result[1]}, status='{test_result[2]}')")
        except Exception as e:
            print(f"PRICE_API_RETURNS_TUPLE: False (Error: {e})")
        
        # 检查关键功能
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
                    print(f"\n⏰ Approaching end time, running final cycle...")
                    break
                
                print(f"\n💤 Sleeping for {self.config['rebalance_minutes']} minutes...")
                print(f"   Next cycle at: {(datetime.now() + timedelta(seconds=sleep_seconds)).strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"[DEBUG] About to sleep at {datetime.now().strftime('%H:%M:%S')}")
                print(f"[DEBUG] Sleep duration: {sleep_seconds} seconds")
                import sys; sys.stdout.flush()  # 强制刷新输出
                time.sleep(sleep_seconds)
                print(f"[DEBUG] Woke up at {datetime.now().strftime('%H:%M:%S')}")
                import sys; sys.stdout.flush()  # 强制刷新输出
            
            print(f"\n{'='*60}")
            print("📊 Final Snapshot")
            print(f"{'='*60}")
            self.run_cycle()
            
            self.status = "COMPLETED"
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            self.status = "INTERRUPTED"
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.status = "ERROR"
        finally:
            self.save_results()

    def save_results(self):
        """保存结果"""
        print(f"\n{'='*60}")
        print("💾 Saving Results")
        print(f"{'='*60}")
        
        # 保存交易日志
        trades_path = self.config['reporting']['trades_log_path']
        if self.trades_log:
            trades_df = pd.DataFrame(self.trades_log)
            trades_df.to_csv(trades_path, index=False)
            print(f"✅ Trades log saved: {trades_path}")
        else:
            # 即使没有交易也创建空文件
            pd.DataFrame(columns=['timestamp', 'ticker', 'side', 'quantity', 'price', 'cost', 'reason']).to_csv(trades_path, index=False)
            print(f"✅ Trades log saved (empty): {trades_path}")
        
        # 保存组合快照
        snapshots_path = self.config['reporting']['portfolio_snapshots_path']
        with open(snapshots_path, 'w', encoding='utf-8') as f:
            for snapshot in self.portfolio_snapshots:
                f.write(json.dumps(snapshot) + '\n')
        print(f"✅ Portfolio snapshots saved: {snapshots_path}")
        
        # 生成图表和报告
        self.generate_equity_curve()
        self.generate_summary_report()
    
    def generate_equity_curve(self):
        """生成资金曲线图"""
        if not self.equity_curve:
            print("⚠️ No equity curve data to plot")
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
        print(f"✅ Equity curve saved: {curve_path}")
        
        plt.close()
    
    def generate_summary_report(self):
        """生成摘要报告"""
        if not self.portfolio_snapshots:
            print("⚠️ No snapshots to generate report")
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
            
            # Regime Filter 面板
            if final_snapshot.get('regime_state'):
                f.write(f"Final Market Regime:\n")
                f.write(f"  State: {final_snapshot['regime_state'].upper()}")
                
                if final_snapshot.get('risk_caps_applied'):
                    f.write(" ⚠️ RISK CAPS ACTIVE\n")
                else:
                    f.write("\n")
                
                f.write(f"  Trend Score: {final_snapshot['trend_score']:.1%}\n")
                f.write(f"  Dynamic Min Cash: {final_snapshot['dynamic_min_cash']:.1%}\n")
                f.write(f"  Dynamic Max Weight: {final_snapshot['dynamic_max_weight']:.1%}\n\n")
            
            # Macro Integration 面板
            if final_snapshot.get('macro_risk_score', 0) > 0:
                f.write(f"Final Macro Signals (GlobalWatch):\n")
                f.write(f"  Risk Score: {final_snapshot['macro_risk_score']:.1f}/10.0\n")
                f.write(f"  Confirmed Topics: {final_snapshot.get('confirmed_topics_count', 0)}\n")
                
                if final_snapshot.get('macro_tilts'):
                    f.write(f"  Active Tilts:\n")
                    for ticker, tilt in final_snapshot['macro_tilts'].items():
                        f.write(f"    {ticker}: {tilt:+.2%}\n")
                f.write("\n")
            
            # 基准比较面板
            if final_snapshot.get('bench_returns'):
                f.write(f"Benchmark Comparison:\n")
                f.write(f"  Strategy Return: {total_return:.2%}\n")
                f.write(f"  Benchmark Avg Return: {final_snapshot['bench_avg_return']:.2%}\n")
                f.write(f"  Excess Return: {final_snapshot['excess_return']:.2%}")
                
                if final_snapshot['win_flag']:
                    f.write(" ✅ OUTPERFORM\n")
                else:
                    f.write(" ❌ UNDERPERFORM\n")
                
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
            f.write("⚠️  SIMULATION ONLY - NO REAL MONEY\n")
            f.write("⚠️  Past performance does not guarantee future results\n")
            f.write("="*60 + "\n")
        
        print(f"✅ Summary report saved: {report_path}")
        
        print(f"\n{'='*60}")
        print("📊 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Initial Cash: ${self.initial_cash:,.2f}")
        print(f"Final Equity: ${final_snapshot['total_equity']:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # 显示基准比较
        if final_snapshot.get('bench_returns'):
            print(f"\nBenchmark Comparison:")
            print(f"  Strategy: {total_return:.2%}")
            print(f"  Benchmark Avg: {final_snapshot['bench_avg_return']:.2%}")
            print(f"  Excess Return: {final_snapshot['excess_return']:.2%}", end="")
            if final_snapshot['win_flag']:
                print(" ✅ OUTPERFORM")
            else:
                print(" ❌ UNDERPERFORM")
        
        print(f"\nTotal Trades: {len(self.trades_log)}")
        print(f"Status: {self.status}")
        print(f"{'='*60}\n")


def main():
    """主函数"""
    import sys
    
    config_path = 'paper_config.json'
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print(f"Loading config: {config_path}")
    
    engine = PaperTradingEngine(config_path)
    engine.run()


if __name__ == '__main__':
    main()

