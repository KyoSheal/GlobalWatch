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
        """计算目标权重 - 动量 + 波动率调整"""
        strategy = self.config['strategy']
        lookback = strategy['lookback_days']
        vol_target = strategy['vol_target']
        momentum_weight = strategy['momentum_weight']
        vol_weight = strategy['vol_weight']
        
        asset_scores = {}
        
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
        
        positive_assets = {k: v for k, v in asset_scores.items() if v['score'] > 0}
        
        if not positive_assets:
            return {'CASH': 1.0}
        
        total_score = sum(v['score'] for v in positive_assets.values())
        raw_weights = {k: v['score'] / total_score for k, v in positive_assets.items()}
        
        max_weight = self.config['objectives']['max_weight_per_asset']
        min_cash = self.config['objectives']['min_cash_pct']
        
        adjusted_weights = {}
        for ticker, weight in raw_weights.items():
            adjusted_weights[ticker] = min(weight, max_weight)
        
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        total_invested = sum(adjusted_weights.values())
        if total_invested > (1 - min_cash):
            scale_factor = (1 - min_cash) / total_invested
            adjusted_weights = {k: v * scale_factor for k, v in adjusted_weights.items()}
        
        cash_weight = 1.0 - sum(adjusted_weights.values())
        adjusted_weights['CASH'] = cash_weight
        
        return adjusted_weights

    def execute_rebalance(self, target_weights):
        """执行再平衡"""
        current_prices = {}
        positions_value = 0.0
        
        for ticker, qty in self.positions.items():
            price = self.get_current_price(ticker)
            if price is None:
                price = 100.0
            current_prices[ticker] = price
            positions_value += qty * price
        
        total_equity = self.cash + positions_value
        
        target_positions = {}
        for ticker, weight in target_weights.items():
            if ticker == 'CASH':
                continue
            
            target_value = total_equity * weight
            price = self.get_current_price(ticker)
            
            if price is None or price <= 0:
                continue
            
            target_qty = int(target_value / price)
            target_positions[ticker] = target_qty
        
        trades = []
        
        # 卖出
        for ticker, current_qty in list(self.positions.items()):
            target_qty = target_positions.get(ticker, 0)
            
            if target_qty < current_qty:
                sell_qty = current_qty - target_qty
                price = current_prices.get(ticker, self.get_current_price(ticker))
                
                if price is None:
                    continue
                
                proceeds = sell_qty * price
                cost = proceeds * self.config['objectives']['transaction_cost_pct']
                net_proceeds = proceeds - cost
                
                self.cash += net_proceeds
                self.positions[ticker] = target_qty
                
                if target_qty == 0:
                    del self.positions[ticker]
                
                trades.append({
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'side': 'SELL',
                    'quantity': sell_qty,
                    'price': price,
                    'cost': cost,
                    'reason': 'rebalance'
                })
        
        # 买入
        for ticker, target_qty in target_positions.items():
            current_qty = self.positions.get(ticker, 0)
            
            if target_qty > current_qty:
                buy_qty = target_qty - current_qty
                price = self.get_current_price(ticker)
                
                if price is None:
                    continue
                
                required_cash = buy_qty * price
                cost = required_cash * self.config['objectives']['transaction_cost_pct']
                total_required = required_cash + cost
                
                if total_required > self.cash:
                    buy_qty = int((self.cash * 0.99) / (price * (1 + self.config['objectives']['transaction_cost_pct'])))
                    
                    if buy_qty <= 0:
                        continue
                    
                    required_cash = buy_qty * price
                    cost = required_cash * self.config['objectives']['transaction_cost_pct']
                    total_required = required_cash + cost
                
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
                
                trades.append({
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'side': 'BUY',
                    'quantity': buy_qty,
                    'price': price,
                    'cost': cost,
                    'reason': 'rebalance'
                })
        
        self.trades_log.extend(trades)
        
        # 实时保存交易记录（不等到程序结束）
        if trades:
            self.save_trades_immediately()
        
        return trades

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
        
        total_equity = self.cash + positions_value
        
        total_return = (total_equity - self.initial_cash) / self.initial_cash
        drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'cycle': self.current_cycle,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'total_return': total_return,
            'drawdown': drawdown,
            'positions': positions_detail,
            'status': self.status
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
            f.write("⚠️  LIVE DATA - Updates every 15 minutes\n")
            f.write("⚠️  SIMULATION ONLY - NO REAL MONEY\n")
            f.write("="*60 + "\n")
        
        print(f"📊 Live summary updated: {summary_path}")
        import sys; sys.stdout.flush()  # 强制刷新输出

    def get_cost_basis(self, ticker):
        """获取股票的成本基础（平均买入价）"""
        return self.cost_basis.get(ticker, None)

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
        print(f"Total Trades: {len(self.trades_log)}")
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
