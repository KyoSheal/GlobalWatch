"""
为 paper_trading.py 添加调试日志
"""

# 读取原文件
with open('paper_trading.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到 time.sleep 那一行并添加日志
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    
    # 在 time.sleep 前添加日志
    if 'time.sleep(sleep_seconds)' in line:
        indent = ' ' * (len(line) - len(line.lstrip()))
        new_lines.insert(-1, f'{indent}print(f"[DEBUG] About to sleep at {{datetime.now().strftime(\'%H:%M:%S\')}}")\n')
        new_lines.insert(-1, f'{indent}print(f"[DEBUG] Sleep duration: {{sleep_seconds}} seconds")\n')
        new_lines.insert(-1, f'{indent}import sys; sys.stdout.flush()\n')
        new_lines.append(f'{indent}print(f"[DEBUG] Woke up at {{datetime.now().strftime(\'%H:%M:%S\')}}")\n')
        new_lines.append(f'{indent}import sys; sys.stdout.flush()\n')

# 保存
with open('paper_trading.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Added debug logs to paper_trading.py")
print("Now run: python paper_trading.py paper_config.json")
