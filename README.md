# SSQ Quant Research System

一个面向中国福利彩票双色球的可审计统计研究与公平回测系统。

本项目不承诺预测中奖号码，也不声称历史统计能够改变彩票的理论随机性。所有策略都必须与同票数、同成本的随机基线比较。

## 功能

- 双色球开奖记录的 CSV 导入与校验
- 随机、频率等可解释策略
- 严格按时间推进的滚动回测
- 红球/蓝球命中数、成本、奖金和收益记录
- 研究结果导出为 JSON

## 使用

```powershell
python -m ssq_research.cli demo
python -m ssq_research.cli backtest --csv data/ssq.csv --window 30 --tickets 1
```

CSV 字段：`issue,draw_date,red_1,red_2,red_3,red_4,red_5,red_6,blue`。

## 研究边界

- 不使用目标期数据生成该期预测。
- 不把命中个数当成中奖概率优势证明。
- 不把历史奖金估算当成实际可领取金额。
- 真实数据必须保留来源 URL、抓取时间和原始文件哈希。

## License

MIT

