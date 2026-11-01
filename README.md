# SSQ Quant Research System

一个面向中国福利彩票双色球的可审计统计研究与公平回测系统。

本项目不承诺预测中奖号码，也不声称历史统计能够改变彩票的理论随机性。所有策略都必须与同票数、同成本的随机基线比较。

## 功能

- 双色球开奖记录的 CSV 导入与校验
- 随机、频率等可解释策略
- 严格按时间推进的滚动回测
- 红球/蓝球命中数、成本、奖金和收益记录
- 研究结果导出为 JSON
- 固定奖金基线与可变奖金的明确标记（一/二等奖金额未知时记为 `None`，不当作 0 元）
- 蓝球退化检测指标（`distinct_blue` / `blue_entropy` / `top1_share` / `longest_run`）
- 可选的双头（红 sigmoid + 蓝 softmax）多层 LSTM 训练、预测和 walk-forward 回测流水线

## 使用

```powershell
python -m ssq_research.cli demo
python -m ssq_research.cli backtest --csv data/ssq.csv --window 30 --seed 0 --output backtest.json
```

CSV 字段：`issue,draw_date,red_1,red_2,red_3,red_4,red_5,red_6,blue`。

启用 LSTM：

```powershell
python -m pip install -e ".[lstm]"
```

LSTM 只作为独立实验策略；训练时 `shuffle=False`，回测按历史前缀逐期推进，不得与随机基线分离报告。

网络输出为双头：`red_head`（33 路 sigmoid，6/33 多标签）与 `blue_head`（16 路 softmax，1/16 单选）。
回测每期同时产出两条记录，**`strategy` 字段可以区分它们**：

- `lstm_argmax`：确定性点估计，取概率最高的 6 红 + 1 蓝，不含任何随机性。
- `lstm_sample`：使用 `numpy.random.Generator`（由 `seed` 派生）按概率无放回抽 6 红 + 1 蓝。

判断「模型到底学到了没有」请先看退化指标，而不是看命中了几个号：

```python
from ssq_research.core import blue_degeneracy_report

report = blue_degeneracy_report(lstm_rows)   # lstm_rows 来自 walk_forward_lstm
# {'distinct_blue': 1, 'blue_entropy': 0.0, 'blue_entropy_max': 4.0,
#  'top1_share': 1.0, 'longest_run': 300, 'degenerate': True}
```

`distinct_blue` 长期只有 1~3、`blue_entropy` 远低于 4 bit，说明模型输出已经退化，
此时的「预测」只是常数，不构成任何优势证据。`degenerate` 的判定需要至少 16 次抽样
（样本太少不可靠），并且 `distinct_blue <= 7` 或熵低于 2 bit 即视为退化。

注意 `lstm_argmax` 与 `lstm_sample` 的读法不同：**`lstm_sample` 的蓝球会随 `SEED` 变化**
（说明抽样确实在按概率走），而 **`lstm_argmax` 的蓝球恒定是正常的** —— 它不含随机性，
恒定恰恰是「蓝球概率分布接近均匀、模型没有可用信号」的证据，不是 bug。

## 随机种子

Notebook 默认 `SEED = None`（运行时随机生成并打印实际值），所以两次运行的抽样结果不再逐位相同。
需要复现某次结果时，把该次日志里打印的 `SEED=...` 填回配置格即可 —— 随机但不失可追溯。
CLI 同理：`--seed` 缺省为 0，同 seed 的回测完全可复现。

## 研究边界

- 不使用目标期数据生成该期预测。
- 不把命中个数当成中奖概率优势证明。
- 不把历史奖金估算当成实际可领取金额。
- 真实数据必须保留来源 URL、抓取时间和原始文件哈希。
- 一/二等奖金额依赖奖池，一律记为 `payout=None`、`net=None`，不得用 0 参与求和。

## 变更记录

当前版本 **0.2.2**：在 0.2.0「修复 LSTM 蓝球确定性退化（红/蓝双头、蓝球 softmax、概率抽样与
退化检测）」与 0.2.1「notebook 默认 `SEED` 改为运行时随机并打印、放宽退化判据、修掉候选去重静默重复」
的基础上，补齐测试覆盖（101 用例，双头结构 / softmax 归一 / 退化阈值边界），运行时代码零改动。
完整记录见 [CHANGELOG.md](CHANGELOG.md)。

## License

MIT

