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
- 概率与显著性分析：Wilson 置信区间、卡方拟合优度检验、各奖级理论概率（由组合数算出，非硬编码）
- 多算法对照：频率（可加时间衰减）、马尔可夫链、ARIMA、LSTM、集成，全部与随机基线同窗口同成本逐期对比
- 可视化与单文件 HTML 仪表盘（图表内嵌 SVG，Kaggle 里直接下载打开即可）
- 可选 Gradio 实时查询界面

## 使用

```powershell
python -m ssq_research.cli demo
python -m ssq_research.cli backtest --csv data/ssq.csv --window 30 --seed 0 --output backtest.json
```

生成单文件 HTML 研究报告（概率分析 + 多算法对照 + 图表 + 单号查询）：

```powershell
python -m pip install -e ".[algorithms,viz]"
python -m ssq_research.cli report --csv data/ssq.csv --window 100 --max-periods 600 --output report.html
```

- `--no-algorithms`：只跑随机与频率基线，不跑评分算法
- `--with-arima`：额外跑 ARIMA 评分器（需要 statsmodels）
- `--max-periods`：限制回测期数，0 表示全部（全量较慢）

Kaggle 上一键运行见 [`notebooks/ssq_kaggle.ipynb`](notebooks/ssq_kaggle.ipynb)：
一份跑完取数、概率分析、多算法对照、可选 LSTM、HTML 报告与可选 Gradio 界面，默认 3 分钟内出结果。

Notebook 只保留这一份；原来按专项拆分的三份已删除，功能全部收进这一份。

### 数据来源

取数按顺序尝试，任一成功即停（`fetch_ssq_draws_with_fallback`）：

1. `https://data.17500.cn/ssq_asc.txt` —— 单文件全量文本，一次请求
2. `http://data.17500.cn/ssq_asc.txt` —— 同站明文，只为受限网络保留
3. 中国福彩官网 JSON 接口 —— 分页抓取，覆盖 2013 年至今
4. `ssq_research/data/ssq_snapshot.csv` —— 随包发布的快照，截至 2026-11-01

前三者失败会逐条打印原因；回退到快照时会显式警告，并给出快照的最新期号——
**静默使用旧数据比取数失败更糟**。要固定输入时把 `DATA_CSV` 指向自己的 CSV。

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

## 概率分析的口径

报告里的「高频号码 Top 5」是**历史频次排名**，不是预测。每个号都附 95% Wilson 区间与理论值，
「判定」列会明确写出该号的冷热是否超出随机波动。

- 红球理论出现率 `6/33 = 18.18%`，蓝球 `1/16 = 6.25%`。
- 卡方拟合优度检验（红 df=32、蓝 df=15）判断历史分布是否偏离均匀。p 值大表示一致，
  此时任何「热号/冷号」结论都只是噪声。
- 同时检验 33 个号时，**即使开奖完全公平**，也必然有一两个号的单球 95% 区间不覆盖理论值。
  这是多重比较的必然结果，不是发现了规律。
- 各奖级概率由组合数计算（`math.comb`），一等奖恒为 `1 / 17,721,088`，与选号方式无关。

## 算法说明

五个算法都实现为同一个契约 `score(history) -> (49,)`：对 49 个槽位（前 33 个红球、后 16 个蓝球）
各给一个分数，归一化后按分数取号。统一契约的意义是让它们接进同一套滚动回测，
与随机基线的对比才成立。

### 频率统计（`FrequencyScorer`）

统计每个号在窗口内的出现次数，按频次排序取号。`half_life` 开关（`frequency_recent`）把历史
按指数衰减加权，让近期开奖权重更高。

- 参数：`half_life=None`（等权）；`frequency_recent` 取 30 期半衰期。
- 局限：这是全项目最朴素的基线，也是必须被超越的那条线。公平开奖下它不可能稳定优于随机。

### 马尔可夫链（`MarkovScorer`）

红球：统计每个号「上期出现→本期出现」与「上期未出现→本期出现」的条件频率；
蓝球：构建 16×16 的一阶转移矩阵，用当前蓝球所在行的分布作为下一期分数。

- 参数：`smoothing=1.0`（拉普拉斯平滑，避免未出现的转移概率为 0）。
- 局限：一阶假设（只依赖上一期）在真实开奖中并不成立，且 16×16 矩阵分摊到 3500 期历史后
  每格平均不到 14 个样本，估计噪声很大。

### ARIMA（`ArimaScorer`）

把每个号的「出现/不出现」当作一条 0/1 时间序列，逐序列拟合 ARIMA(1,0,0) 并预测下一期。

- 参数：`order=(1,0,0)`、`max_history=240`、`fallback_window=10`。
- `estimator="mle"` 用 statsmodels 的极大似然，49 个序列逐一拟合，600 期回测要几分钟到几十分钟。
- `estimator="css"` 用闭式条件最小二乘，同样的模型一次性向量化算完 49 个序列（单次约 0.6 毫秒），
  回测默认走这条。两者是同一模型的两种估计，差异在小数点后第三位量级。
- 局限：ARIMA(1,0,0) 没有差分也没有移动平均项，实质就是带常数项的 AR(1)。0/1 序列的
  自相关本就很弱，估计出的系数接近 0，预测值向均值收缩——所以它的分数接近均匀分布。

### LSTM（`LstmScorer` / `lstm.py`）

双头结构：`red_head = Dense(33, sigmoid)` 配二元交叉熵，`blue_head = Dense(16, softmax)` 配
稀疏分类交叉熵。输入是最近 `window` 期的 49 维 0/1 编码。

- 参数：`window=10`、`hidden_units=32`、`layers=2`、`dropout=0.1`、`epochs`、`retrain_every`。
- 为什么是双头：早期版本用单个 `Dense(49, sigmoid)`，49 个独立二分类的交叉熵最优解就是
  「历史边际出现率」这个常数向量，与初始化无关。于是蓝球 argmax 永远落在同一个号上——
  两次运行结果完全相同，看起来像「预测稳定」，其实是退化成常数。双头 + softmax 强制
  蓝球输出成竞争关系，让分布能够表达「不确信」。
- 取号两种模式：`argmax`（确定性，无可复现问题但也无随机性）与 `sample`（按概率无放回抽样，
  带 `seed` 可复现）。
- 局限：训练完的蓝球概率通常落在 0.056~0.068 之间（均匀为 0.0625），经验熵接近 4 bit 上限。
  这说明模型没有学到可用信号，`argmax` 给出的仍是常数解。

### 集成学习（`EnsembleScorer`）

把上面若干评分器归一化后加权平均，默认等权。

- 局限：它降低的是模型之间的**方差**，不是开奖的随机性。几个都打不过随机基线的模型，
  加权平均后依然打不过——这一点在报告里有逐期数据支撑，不是一句断言。

### 怎么读退化指标

`distinct_blue` 长期只有 1~3、`blue_entropy` 远低于 4 bit 时，输出已经塌缩成常数，此时
命中数高只是「押同一个号押中了」的偶然，不是预测能力。**先看退化指标，再看命中率。**

## 多算法为什么打不过随机基线

每个算法都实现为 `score(history) -> (49,)` 的评分器，再统一接入同一套滚动回测，因此
`frequency`、`markov`、`arima`、`lstm`、`ensemble` 与 `random` 是在同窗口、同成本下逐期比较的。

「集成学习提高预测准确性」这句话对公平开奖不成立：它降低的是多个模型输出之间的方差，
不是开奖本身的随机性——**几个都无法超越随机基线的模型，加权平均后依然无法超越**。
报告会给出每个策略相对随机基线的双比例检验 p 值，以及 `distinct_blue` / `blue_entropy`
退化指标，用来区分「真的赢了」和「输出退化成常数」。

## 研究边界

- 不使用目标期数据生成该期预测。
- 不把命中个数当成中奖概率优势证明。
- 不把历史奖金估算当成实际可领取金额。
- 真实数据必须保留来源 URL、抓取时间和原始文件哈希。
- 一/二等奖金额依赖奖池，一律记为 `payout=None`、`net=None`，不得用 0 参与求和。

## 变更记录

已发布版本 **0.2.2**：修复 LSTM 蓝球确定性退化，并补齐测试覆盖。

`main` 分支上另有已合入、**尚未发版**的功能：概率与显著性分析（Wilson 区间、卡方检验、组合数计算的奖级
概率）、多算法评分对照（频率/马尔可夫/ARIMA/LSTM/集成）、可视化与单文件 HTML 仪表盘、可选 Gradio 界面。
所有算法仍强制与随机基线对比，退化指标随报告一起输出。这部分功能的版本号与发布时间由维护者决定，
暂按「未发布」记录，完整历史见 [CHANGELOG.md](CHANGELOG.md)。

## License

MIT

