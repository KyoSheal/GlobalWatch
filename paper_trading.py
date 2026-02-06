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
        
        Returns:
            macro_risk_score: 0-10，越大越 risk-off
            confirmed_topics: 满足 [k/n] 规则的主题列表
            macro_tilts: {ticker: tilt_delta} 资产倾斜
            signal_summary: 信号统计摘要
        """
        if not self.enabled:
            return 0.0, [], {}, {}
        
        print(f"\n[MACRO] Analyzing macro signals from GlobalWatch...")
        
        # 获取最近信号
        signals = self.fetch_recent_signals(n=50)
        
        if not signals:
            print("[MACRO] No signals to analyze")
            return 0.0, [], {}, {}
        
        # 主题投票统计
        theme_votes = {}  # {theme: {'bullish': weight_sum, 'bearish': weight_sum, 'count': n}}
        
        # 按主题和方向加权投票
        for signal in signals:
            metadata = signal['metadata']
            
            theme = metadata.get('theme', 'unknown')
            direction = metadata.get('direction', 'neutral').lower()
            confidence = metadata.get('confidence', 50.0) / 100.0  # 归一化到 0-1
            timestamp = metadata.get('timestamp', '')
            
            # 计算时间衰减权重
            weight, age_hours = self.compute_signal_weight(timestamp)
            
            # 综合权重 = 时间衰减 * 置信度
            combined_weight = weight * confidence
            
            if theme not in theme_votes:
                theme_votes[theme] = {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0, 'count': 0}
            
            if 'bullish' in direction or 'long' in direction:
                theme_votes[theme]['bullish'] += combined_weight
            elif 'bearish' in direction or 'short' in direction:
                theme_votes[theme]['bearish'] += combined_weight
            else:
                theme_votes[theme]['neutral'] += combined_weight
            
            theme_votes[theme]['count'] += 1
        
        # 应用 [k/n] 确认规则
        confirm_k, confirm_n = self.macro_config.get('confirm_k_of_n', [2, 3])
        confirmed_topics = []
        
        print(f"\n[MACRO] Theme Analysis (require {confirm_k}/{confirm_n} confirmation):")
        print(f"{'Theme':<20} {'Bullish':>10} {'Bearish':>10} {'Count':>8} {'Status':<15}")
        print("-" * 70)
        
        for theme, votes in sorted(theme_votes.items(), key=lambda x: x[1]['count'], reverse=True):
            bullish_weight = votes['bullish']
            bearish_weight = votes['bearish']
            count = votes['count']
            
            # 判断是否确认（需要至少 k 个信号）
            if count >= confirm_k:
                # 判断方向
                if bullish_weight > bearish_weight * 1.5:  # 明显偏多
                    confirmed_topics.append({'theme': theme, 'direction': 'bullish', 'strength': bullish_weight})
                    status = "✅ BULLISH"
                elif bearish_weight > bullish_weight * 1.5:  # 明显偏空
                    confirmed_topics.append({'theme': theme, 'direction': 'bearish', 'strength': bearish_weight})
                    status = "✅ BEARISH"
                else:
                    status = "⚖️ MIXED"
            else:
                status = f"⏳ NEED {confirm_k-count} MORE"
            
            print(f"{theme:<20} {bullish_weight:>10.2f} {bearish_weight:>10.2f} {count:>8} {status:<15}")
        
        print("-" * 70)
        
        # 计算 macro_risk_score（0-10）
        # 基于 bearish 主题的强度和数量
        risk_score = 0.0
        
        for topic in confirmed_topics:
            if topic['direction'] == 'bearish':
                # 每个 bearish 主题贡献风险分数
                risk_score += min(topic['strength'] * 2, 3.0)  # 单个主题最多贡献 3 分
        
        # 限制在 0-10 范围
        risk_score = min(risk_score, 10.0)
        
        # 生成 macro_tilts（基于 macro_mapping）
        macro_tilts = self._generate_tilts(confirmed_topics)
        
        # 信号摘要
        signal_summary = {
            'total_signals': len(signals),
            'confirmed_topics': len(confirmed_topics),
            'risk_score': risk_score,
            'theme_votes': theme_votes
        }
        
        print(f"\n[MACRO] Risk Score: {risk_score:.1f}/10.0")
        print(f"[MACRO] Confirmed Topics: {len(confirmed_topics)}")
        
        if macro_tilts:
            print(f"[MACRO] Asset Tilts:")
            for ticker, tilt in macro_tilts.items():
                print(f"  {ticker}: {tilt:+.2%}")
        
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
        self.status = "READY"  # READY/RUNNING/PAUSED/COMPLETED
        self.last_rebalance_time = None  # 用于 cooldown 检查
        self.current_regime = {}  # 当前市场状态（Regime Filter）
        self.current_macro = {}  # 当前宏观信号（Macro Integration）
        
        # 价格缓存（避免重复请求）
        self.price_cache = {}  # {ticker: (price, timestamp)}
        self.price_cache_duration = 60  # 缓存60秒
        
        # 宏观信号适配器
        self.macro_adapter = MacroSignalAdapter(self.config)
        
        # 尝试恢复之前的状态
        self.resume_from_checkpoint()
        
        # 创建输出目录
        os.makedirs('outputs', exist_ok=True)
        
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
        
        return config
    
    def validate_config(self):
        """验证配置"""
        assert self.config['paper_mode'] == True, "paper_mode must be True"
        assert self.config['safety']['no_real_broker'] == True, "no_real_broker must be True"
        assert self.config['safety']['simulation_only'] == True, "simulation_only must be True"
        
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
            self.clear_checkpoint()
    
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
        """获取当前价格（实时或最新）- 强制刷新，避免缓存"""
        if ticker == 'CASH':
            return 1.0
        
        try:
            # 创建新的 Ticker 对象，避免缓存
            t = yf.Ticker(ticker)
            
            # 方法1: 尝试获取最新的分钟级数据（最可靠）
            try:
                # 使用 5m 间隔，period='1d' 获取今天的数据
                hist = t.history(period='1d', interval='5m')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    timestamp = hist.index[-1]
                    # 计算数据延迟
                    import pytz
                    now_et = datetime.now(pytz.timezone('US/Eastern'))
                    data_age_minutes = (now_et - timestamp).total_seconds() / 60
                    
                    market_status = "🟢 LIVE" if data_age_minutes < 10 else "🟡 RECENT" if data_age_minutes < 60 else "🔴 STALE"
                    print(f"[PRICE] {ticker}: ${price:.2f} (5m @ {timestamp.strftime('%H:%M ET')}, {data_age_minutes:.0f}min ago) {market_status}")
                    return price
            except Exception as e:
                print(f"[PRICE] {ticker}: 5m history failed - {e}")
            
            # 方法2: 尝试 1m 间隔
            try:
                hist = t.history(period='1d', interval='1m')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    timestamp = hist.index[-1]
                    import pytz
                    now_et = datetime.now(pytz.timezone('US/Eastern'))
                    data_age_minutes = (now_et - timestamp).total_seconds() / 60
                    market_status = "🟢 LIVE" if data_age_minutes < 5 else "🟡 RECENT" if data_age_minutes < 60 else "🔴 STALE"
                    print(f"[PRICE] {ticker}: ${price:.2f} (1m @ {timestamp.strftime('%H:%M ET')}, {data_age_minutes:.0f}min ago) {market_status}")
                    return price
            except Exception as e:
                print(f"[PRICE] {ticker}: 1m history failed - {e}")
            
            # 方法3: 尝试 info（可能有缓存）
            try:
                info = t.info
                for price_field in ['currentPrice', 'regularMarketPrice', 'ask', 'bid']:
                    if price_field in info and info[price_field]:
                        price = float(info[price_field])
                        if price > 0:
                            print(f"[PRICE] {ticker}: ${price:.2f} (from info.{price_field})")
                            return price
            except Exception as e:
                print(f"[PRICE] {ticker}: info failed - {e}")
            
            # 方法4: 降级到日线数据（最后手段）
            try:
                hist = t.history(period='5d', interval='1d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    date = hist.index[-1]
                    print(f"[PRICE] {ticker}: ${price:.2f} (from daily close {date.strftime('%Y-%m-%d')}) ⚠️ NOT REAL-TIME")
                    return price
            except Exception as e:
                print(f"[PRICE] {ticker}: daily history failed - {e}")
                
        except Exception as e:
            print(f"[ERROR] All price methods failed for {ticker}: {e}")
        
        return None
    
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
    
    def calculate_target_weights(self):
        """计算目标权重 - 动量 + 波动率调整 + Regime Filter + Macro Signals"""
        
        # ========== 步骤1: 计算市场状态（Regime Filter）==========
        regime_state, trend_score, regime_details, dynamic_min_cash, dynamic_max_weight = self.compute_regime_state()
        
        # 保存 regime 信息供 snapshot 使用
        self.current_regime = {
            'regime_state': regime_state,
            'trend_score': trend_score,
            'regime_details': regime_details,
            'dynamic_min_cash': dynamic_min_cash,
            'dynamic_max_weight': dynamic_max_weight,
            'risk_caps_applied': regime_state == 'risk_off'
        }
        
        # ========== 步骤2: 获取宏观信号（Macro Integration）==========
        macro_risk_score, confirmed_topics, macro_tilts, signal_summary = self.macro_adapter.analyze_signals()
        
        # 保存宏观信号信息供 snapshot 使用
        self.current_macro = {
            'macro_risk_score': macro_risk_score,
            'confirmed_topics': confirmed_topics,
            'macro_tilts': macro_tilts,
            'signal_summary': signal_summary
        }
        
        # 根据宏观风险分数调整现金比例
        if macro_risk_score > 0:
            macro_cash_slope = self.config.get('macro_integration', {}).get('macro_cash_slope', 0.02)
            macro_cash_add = macro_risk_score * macro_cash_slope
            dynamic_min_cash = min(dynamic_min_cash + macro_cash_add, 0.50)  # 最多 50% 现金
            
            print(f"[MACRO] Adjusting min cash: {self.current_regime['dynamic_min_cash']:.1%} → {dynamic_min_cash:.1%} (risk score: {macro_risk_score:.1f})")
            
            # 更新 regime 信息
            self.current_regime['dynamic_min_cash'] = dynamic_min_cash
        
        # ========== 步骤3: 计算资产评分（动量 + 波动率）==========
        strategy = self.config['strategy']
        lookback = strategy['lookback_days']
        vol_target = strategy['vol_target']
        momentum_weight = strategy['momentum_weight']
        vol_weight = strategy['vol_weight']
        
        asset_scores = {}
        
        print(f"\n📊 Evaluating {len(self.config['universe'])-1} assets...")
        print(f"{'Ticker':<8} {'Momentum':>10} {'Volatility':>12} {'Score':>10} {'Status':<10}")
        print("-" * 60)
        
        for asset in self.config['universe']:
            ticker = asset['ticker']
            
            if ticker == 'CASH':
                continue
            
            momentum = self.calculate_momentum(ticker, lookback)
            volatility = self.calculate_volatility(ticker, lookback)
            
            score = momentum_weight * momentum - vol_weight * (volatility - vol_target)
            
            asset_scores[ticker] = {
                'momentum': momentum,
                'volatility': volatility,
                'score': score
            }
            
            # 显示每个资产的评分
            status = "✅ BUY" if score > 0 else "❌ SKIP"
            print(f"{ticker:<8} {momentum:>9.2%} {volatility:>11.2%} {score:>9.4f} {status:<10}")
        
        print("-" * 60)
        
        positive_assets = {k: v for k, v in asset_scores.items() if v['score'] > 0}
        
        print(f"Selected {len(positive_assets)} assets with positive scores\n")
        
        if not positive_assets:
            return {'CASH': 1.0}
        
        # ========== 步骤4: 计算原始权重 ==========
        total_score = sum(v['score'] for v in positive_assets.values())
        raw_weights = {k: v['score'] / total_score for k, v in positive_assets.items()}
        
        # ========== 步骤5: 应用宏观倾斜（Macro Tilts）==========
        if macro_tilts:
            print(f"\n[MACRO] Applying tilts to weights:")
            for ticker, tilt in macro_tilts.items():
                if ticker in raw_weights:
                    old_weight = raw_weights[ticker]
                    raw_weights[ticker] = max(0.0, old_weight + tilt)  # 不能为负
                    print(f"  {ticker}: {old_weight:.2%} → {raw_weights[ticker]:.2%} (tilt: {tilt:+.2%})")
                elif tilt > 0:
                    # 如果资产不在组合中但有正倾斜，可以考虑加入
                    raw_weights[ticker] = tilt
                    print(f"  {ticker}: NEW position {tilt:.2%}")
            
            # 重新归一化
            total_weight = sum(raw_weights.values())
            if total_weight > 0:
                raw_weights = {k: v / total_weight for k, v in raw_weights.items()}
        
        # ========== 步骤6: 应用动态上限（使用 regime-adjusted max_weight）==========
        adjusted_weights = {}
        for ticker, weight in raw_weights.items():
            adjusted_weights[ticker] = min(weight, dynamic_max_weight)
        
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        # ========== 步骤7: 应用动态现金下限（使用 regime + macro adjusted min_cash）==========
        total_invested = sum(adjusted_weights.values())
        if total_invested > (1 - dynamic_min_cash):
            scale_factor = (1 - dynamic_min_cash) / total_invested
            adjusted_weights = {k: v * scale_factor for k, v in adjusted_weights.items()}
        
        cash_weight = 1.0 - sum(adjusted_weights.values())
        adjusted_weights['CASH'] = cash_weight
        
        return adjusted_weights

    def execute_rebalance(self, target_weights):
        """执行再平衡 - 带三大保护器：cooldown / weight_threshold / min_notional"""
        
        # ========== 准备交易上下文信息 ==========
        trade_context = self._build_trade_context()
        
        # ========== 保护器 1: Cooldown 检查 ==========
        execution_config = self.config.get('execution', {})
        cooldown_minutes = execution_config.get('rebalance_cooldown_minutes', 0)
        
        if cooldown_minutes > 0 and self.last_rebalance_time is not None:
            time_since_last = (datetime.now() - self.last_rebalance_time).total_seconds() / 60
            if time_since_last < cooldown_minutes:
                remaining = cooldown_minutes - time_since_last
                print(f"[COOLDOWN] Skipping rebalance - {remaining:.1f} minutes remaining")
                return []
        
        # ========== 获取当前价格和持仓价值 ==========
        current_prices = {}
        current_values = {}
        positions_value = 0.0
        
        for ticker, qty in self.positions.items():
            price = self.get_current_price(ticker)
            if price is None:
                print(f"[WARN] No price for {ticker}, skipping")
                continue
            current_prices[ticker] = price
            value = qty * price
            current_values[ticker] = value
            positions_value += value
        
        total_equity = self.cash + positions_value
        
        # ========== 计算目标价值（而非目标股数）==========
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
        
        # ========== 保护器 3: Min Notional 过滤 ==========
        min_notional = execution_config.get('min_trade_notional_usd', 0)
        
        trades = []
        
        # 先处理卖出
        for ticker in tickers_to_trade:
            current_value = current_values.get(ticker, 0.0)
            target_value = target_values.get(ticker, 0.0)
            
            if target_value >= current_value:
                continue  # 不是卖出
            
            trade_value = current_value - target_value
            
            # Min notional 检查
            if trade_value < min_notional:
                print(f"[SKIP] {ticker} sell notional ${trade_value:.2f} < min ${min_notional}")
                continue
            
            # 获取价格
            price = current_prices.get(ticker)
            if price is None:
                price = self.get_current_price(ticker)
            if price is None or price <= 0:
                print(f"[WARN] Invalid price for {ticker}")
                continue
            
            # 计算卖出股数（基于价值差异）
            current_qty = self.positions.get(ticker, 0)
            target_qty = int(target_value / price)
            sell_qty = current_qty - target_qty
            
            if sell_qty <= 0:
                continue
            
            # 执行卖出
            proceeds = sell_qty * price
            cost = proceeds * self.config['objectives']['transaction_cost_pct']
            net_proceeds = proceeds - cost
            
            self.cash += net_proceeds
            self.positions[ticker] = target_qty
            
            if target_qty == 0:
                del self.positions[ticker]
                if ticker in self.cost_basis:
                    del self.cost_basis[ticker]
            
            # 构建决策轨迹
            decision_trace = ['cooldown_pass', 'weight_threshold_pass', 'min_notional_pass']
            if trade_context['regime_state'] == 'risk_off':
                decision_trace.append('risk_off_de-risk')
            
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
                'decision_trace': ' | '.join(decision_trace)
            })
            
            print(f"[TRADE] SELL {sell_qty} {ticker} @ ${price:.2f} (notional: ${proceeds:.2f})")
        
        # 再处理买入
        for ticker in tickers_to_trade:
            current_value = current_values.get(ticker, 0.0)
            target_value = target_values.get(ticker, 0.0)
            
            if target_value <= current_value:
                continue  # 不是买入
            
            trade_value = target_value - current_value
            
            # Min notional 检查
            if trade_value < min_notional:
                print(f"[SKIP] {ticker} buy notional ${trade_value:.2f} < min ${min_notional}")
                continue
            
            # 获取价格
            price = current_prices.get(ticker)
            if price is None:
                price = self.get_current_price(ticker)
            if price is None or price <= 0:
                print(f"[WARN] Invalid price for {ticker}")
                continue
            
            # 计算买入股数（基于价值差异）
            current_qty = self.positions.get(ticker, 0)
            target_qty = int(target_value / price)
            buy_qty = target_qty - current_qty
            
            if buy_qty <= 0:
                continue
            
            # 检查现金是否足够
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
            decision_trace = ['cooldown_pass', 'weight_threshold_pass', 'min_notional_pass']
            if ticker in trade_context.get('macro_tilts_dict', {}):
                tilt = trade_context['macro_tilts_dict'][ticker]
                decision_trace.append(f'macro_tilt_{tilt:+.2%}')
            if trade_context['regime_state'] == 'risk_on':
                decision_trace.append('risk_on_add-risk')
            if total_required >= self.cash * 0.99:
                decision_trace.append('cash_limited')
            
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
                'decision_trace': ' | '.join(decision_trace)
            })
            
            print(f"[TRADE] BUY {buy_qty} {ticker} @ ${price:.2f} (notional: ${required_cash:.2f})")
        
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
        cash_target = self.current_regime.get('dynamic_min_cash', self.config['objectives']['min_cash_pct'])
        
        # Macro 信息
        macro_risk_score = self.current_macro.get('macro_risk_score', 0.0)
        confirmed_topics = self.current_macro.get('confirmed_topics', [])
        macro_tilts = self.current_macro.get('macro_tilts', {})
        
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
            'macro_risk_score': macro_risk_score,
            'macro_topics': topics_str,
            'macro_tilts': tilts_str,
            'macro_tilts_dict': macro_tilts  # 保留字典格式供内部使用
        }

    def check_risk_controls(self):
        """检查风险控制"""
        positions_value = 0.0
        for ticker, qty in self.positions.items():
            price = self.get_current_price(ticker)
            if price:
                positions_value += qty * price
        
        total_equity = self.cash + positions_value
        
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        drawdown = (self.peak_equity - total_equity) / self.peak_equity
        
        max_dd = self.config['objectives']['max_drawdown_pct']
        
        if drawdown > max_dd:
            print(f"⚠️ CIRCUIT BREAKER: Drawdown {drawdown:.2%} exceeds limit {max_dd:.2%}")
            print(f"   Pausing trading and increasing cash position")
            
            self.status = "PAUSED"
            
            for ticker in list(self.positions.keys()):
                qty = self.positions[ticker]
                sell_qty = qty // 2
                
                if sell_qty > 0:
                    price = self.get_current_price(ticker)
                    if price:
                        proceeds = sell_qty * price
                        cost = proceeds * self.config['objectives']['transaction_cost_pct']
                        self.cash += (proceeds - cost)
                        self.positions[ticker] -= sell_qty
                        
                        if self.positions[ticker] == 0:
                            del self.positions[ticker]
                        
                        self.trades_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'ticker': ticker,
                            'side': 'SELL',
                            'quantity': sell_qty,
                            'price': price,
                            'cost': cost,
                            'reason': 'circuit_breaker'
                        })
            
            return True
        
        return False
    
    def record_snapshot(self):
        """记录组合快照"""
        print(f"[DEBUG] Recording snapshot at {datetime.now().strftime('%H:%M:%S')}")
        import sys; sys.stdout.flush()
        
        positions_value = 0.0
        positions_detail = {}
        
        for ticker, qty in self.positions.items():
            price = self.get_current_price(ticker)
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
            'risk_caps_applied': self.current_regime.get('risk_caps_applied', False),
            # Macro Integration 字段
            'macro_risk_score': self.current_macro.get('macro_risk_score', 0.0),
            'confirmed_topics_count': len(self.current_macro.get('confirmed_topics', [])),
            'macro_tilts': self.current_macro.get('macro_tilts', {})
        }
        
        self.portfolio_snapshots.append(snapshot)
        self.equity_curve.append((datetime.now(), total_equity, self.cash, positions_value))
        
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
        
        snapshot = self.record_snapshot()
        
        print(f"Cash: ${snapshot['cash']:,.2f}")
        print(f"Positions Value: ${snapshot['positions_value']:,.2f}")
        print(f"Total Equity: ${snapshot['total_equity']:,.2f}")
        print(f"Return: {snapshot['total_return']:.2%}")
        print(f"Drawdown: {snapshot['drawdown']:.2%}")
        print(f"Status: {snapshot['status']}")
        
        # 显示市场状态（Regime Filter）
        if snapshot.get('regime_state'):
            regime_icon = "🟢" if snapshot['regime_state'] == 'risk_on' else "🟡" if snapshot['regime_state'] == 'neutral' else "🔴"
            risk_caps = " ⚠️ RISK CAPS" if snapshot.get('risk_caps_applied') else ""
            print(f"Market Regime: {regime_icon} {snapshot['regime_state'].upper()} (trend: {snapshot['trend_score']:.1%}){risk_caps}")
        
        # 显示持仓详情
        if snapshot['positions']:
            print(f"\n📊 Current Holdings:")
            print(f"{'Ticker':<8} {'Qty':>6} {'Price':>10} {'Value':>12} {'Weight':>8} {'P&L':>10}")
            print("-" * 60)
            
            for ticker, pos in sorted(snapshot['positions'].items(), key=lambda x: x[1]['value'], reverse=True):
                qty = pos['quantity']
                current_price = pos['price']
                value = pos['value']
                weight = value / snapshot['total_equity'] * 100
                
                # 计算盈亏（如果有历史交易记录）
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
            print("⚠️ Risk control triggered, skipping rebalance")
            return
        
        print("\nCalculating target weights...")
        target_weights = self.calculate_target_weights()
        
        print("Target Weights:")
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
