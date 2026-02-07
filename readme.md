# GlobalWatch Paper Trading (V2.10.1)

[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)
[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)

默认文档为中文。  
Default entry is Chinese.

## 文档入口
- 中文完整版：`README.zh.md`
- English version: `README.en.md`

## 当前版本摘要
- 当前版本：`v2.10.1`
- 重点升级：Alpha 与组合风控六步升级（Step 1-6）
- 文档策略：公开运行与验收方法，不公开核心阈值和专有算法细节

## 快速开始
```bash
python -u paper_trading.py paper_config.json
```

Windows:
```bash
Start_Paper_Trading.bat
```

## 编码与乱码说明
- 中文乱码通常是终端编码问题，不是时区问题
- Windows PowerShell 建议先执行：
```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```
- GitHub 页面显示正常时，说明文件编码本身无问题
