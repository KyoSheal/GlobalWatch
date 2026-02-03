"""
GlobalWatch Paper Trading Module
全自动无人干预的模拟交易系统

⚠️ SIMULATION ONLY - NO REAL BROKER CONNECTION
"""

import json
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 安全检查：确保不会连接真实 broker
REAL_BROKER_KEYWORDS = ['alpaca', 'interactive_brokers', 'ib_insync', 'robinhood', 'td_ameritrade']
for keyword in REAL_BROKER_KEYWORDS:
    try:
        __import__(keyword)
        raise RuntimeError(f"⚠️ SAFETY VIOLATION: Detected real broker library '{keyword}'. Paper trading is SIMULATION ONLY!")
    except ImportError:
        pass  # Good, no real broker library
