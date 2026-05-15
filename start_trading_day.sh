#!/bin/zsh
# GlobalWatch daily startup — runs at 5:45 AM PST (Mon-Fri)
# Market opens 6:30 AM PST; news pipeline gets 45 min head start, engine gets 15 min

cd /Users/jzhang1/Downloads/GlobalWatch
mkdir -p outputs/logs

echo "[$(date)] ── Trading day start ──────────────────────" >> outputs/logs/startup.log

# Step 1: start news pipeline in background
python3 run_news_pipeline.py --interval 30 >> outputs/logs/news_pipeline.log 2>&1 &
NEWS_PID=$!
echo "[$(date)] News pipeline started  PID=$NEWS_PID" >> outputs/logs/startup.log

# Step 2: generate industry signals once (LLM analysis of all 12 L2 sectors, ~8 min)
echo "[$(date)] Generating industry signals via LLM..." >> outputs/logs/startup.log
python3 GlobalWatch_V2.py --run-industry-runtime-once --config paper_config.json >> outputs/logs/industry_signals.log 2>&1 &
INDUSTRY_PID=$!
echo "[$(date)] Industry signal generation started  PID=$INDUSTRY_PID" >> outputs/logs/startup.log

# Step 3: wait for industry generation to complete (max 10 min), then start engine
# Industry run takes ~8 min; trading engine needs fresh signals before first cycle
sleep 600
echo "[$(date)] Industry generation window done." >> outputs/logs/startup.log

# Step 4: start main trading engine
python3 -u paper_trading.py paper_config.json >> outputs/logs/trading.log 2>&1 &
TRADE_PID=$!
echo "[$(date)] Trading engine started  PID=$TRADE_PID" >> outputs/logs/startup.log

echo "[$(date)] All processes running. News=$NEWS_PID  Industry=$INDUSTRY_PID  Engine=$TRADE_PID" >> outputs/logs/startup.log
