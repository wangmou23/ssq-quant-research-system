# 变更记录

本项目的所有重要变更都记录在此文件中，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布] — 计划中的 0.3.0

功能性新增：概率与显著性分析、多算法评分对照、可视化与单文件 HTML 仪表盘。
**运行时代码新增，既有 API 未破坏**：`demo` / `backtest` 行为、`core` 与 `lstm` 的对外签名均未改动。

状态说明：功能已合入 `main`，但**版本号仍为 `0.2.2`、未打 tag**。发布时机与版本号由维护者决定。
本节标题待发版时改为具体版本号与日期。

先说结论，免得被下面的功能列表误导：这些算法**不能提高中奖概率**。它们被实现成同一套评分契约，
是为了让「没有优势」这件事可被检验和复现，而不是为了给出更准的号码。一等奖概率恒为
`1 / 17,721,088`，与选号方式无关。

### 新增

- `ssq_research.stats`（纯标准库，零依赖）：
  - `wilson_interval`：二项比例的 Wilson 置信区间。蓝球 p = 1/16 时正态近似会明显失真，故不用教科书公式。
  - `red_frequency` / `blue_frequency`：每号出现次数、历史率、理论率、区间、是否超出区间。
  - `chi_square_survival`：卡方尾概率。为免除 scipy 依赖，自行实现正则化不完全伽马函数
    （级数 + 连分式），并用 `chi2_sf(32,32)≈0.4667` 等参考点校验。
  - `red_uniformity` / `blue_uniformity`：卡方拟合优度检验（红 df=32、蓝 df=15）。
  - `prize_odds`：六个奖级的组合数与概率，**由 `math.comb` 计算而非硬编码**；
    校验值：一等奖 1/17,721,088、二等奖 1/1,181,406、三等奖 1/109,389、四等奖 1/2,303、
    五等奖 1/128.9、六等奖 5.889%。
  - `compare_proportions`：两个命中率的双比例检验，用于判断「比随机基线好」是否超出二项噪声。
  - `build_probability_report`：把上面这些汇总成一个冻结容器。
- `ssq_research.scores`：统一评分契约 `score(history) -> (49,)`，与取号步骤解耦。
  - `FrequencyScorer`：历史出现率，`half_life` 可开启指数时间衰减。
  - `MarkovScorer`：红球逐号两状态出现链（Laplace 平滑）；蓝球 16×16 转移矩阵，
    可表达「上一个蓝球之后常跟哪个」，这是纯频率统计做不到的。
  - `ArimaScorer`：对每个号的 0/1 出现序列拟合 ARIMA（默认 (1,0,0)），拟合失败回退为近期均值，
    避免单条序列不收敛中断整轮回测；`max_history` 限制成本。
  - `LstmScorer`：包装已训练的双头模型，使 LSTM 与统计基线进入同一张对照表。
  - `EnsembleScorer`：先归一化再加权，避免原始量纲大的评分器压过其他成员。
  - `pick_from_scores` / `ScorerStrategy`：`argmax` 与 `sample` 两种出票模式，策略名带模式后缀
    （`frequency_argmax` / `frequency_sample`），回测行不会混淆。
- `ssq_research.viz`：`frequency_chart_svg`（含 95% 误差棒与理论参考线）、`rolling_rate_svg`、
  `backtest_net_svg`、`score_chart_svg`。图内标签一律英文，因为 matplotlib 默认字体在 Kaggle 上
  没有中文字形，中文会渲染成方框。
- `ssq_research.report`：单文件 HTML 仪表盘。图表以内嵌 SVG 输出、数据以 JSON 内嵌，
  无外部请求，Kaggle 里下载即可打开。含结论横幅、卡方结果卡、Top5 显著性表、奖级概率表、
  多算法对照表（带相对随机基线的 p 值）、退化检测表、单号查询（纯 JS）。
- `ssq_research.webapp`：可选 Gradio 界面，`launch(share=True)` 可在 Kaggle 通过隧道访问。
  每次出票都会同时打印一等奖概率，避免把「生成得快」误读成「更可能中」。
- `notebooks/ssq_multi_algo_kaggle.ipynb`：一键跑通取数 → 概率分析 → 多算法对照 → HTML 报告。
- `tests/test_stats.py`（纯标准库）与 `tests/test_scores.py`（`importorskip("numpy")`）。

### 变更

- `cli` 新增 `report` 子命令：`--window`（默认 100）、`--seed`、`--max-periods`、
  `--no-algorithms`、`--with-arima`。
- `pyproject.toml` 新增可选依赖组：`algorithms`（numpy + statsmodels）、`viz`（matplotlib）、
  `web`（gradio）、`all`。
- 版本号**未变动**，仍为 `0.2.2`；发版时再决定新版本号以及是否升到 0.3.0。

### 设计取舍

- `stats` 坚持零依赖，`scores` / `viz` / `webapp` 的第三方依赖全部惰性导入并给出可执行的安装提示，
  这样 `demo` / `backtest` 在裸 Python 环境下仍可用。
- 报告的「高频 Top 5」按频次排序，但每行强制附带 95% 区间与判定列。只给排名不给区间，
  等于把噪声包装成结论。
- 一/二等奖 `payout=None` 的行在累计净收益里跳过而不是记 0，否则中头奖会显示成亏损 2 元。

## [0.2.2] - 2026-11-01

测试补强版本。**运行时代码零改动**：`ssq_research/` 与 `notebooks/` 一行未改，行为与 API 与 0.2.1 完全一致，
版本号提升只为让这份测试覆盖可被引用。

### 测试

- 新增 `tests/test_lstm_model.py`（14 用例，需要 tensorflow，用 `importorskip` 守护）：双头结构
  （`blue_head` 必须是 `softmax`、`red_head` 必须是 `sigmoid`）、loss 配置、`make_sequences` 的双头
  标签布局、`predict_probabilities` 单条/batch 契约、`set_global_seed` 的可复现性。直接断言
  `model.get_layer("blue_head").activation.__name`，比断言 loss 字符串更贴近根因。
- 新增 `tests/test_lstm_heads.py`（25 个用例 / 45 个参数化实例，纯 numpy，<1s）：`encode_draw`
  红蓝边界、`split_probabilities` / `combine_heads` / `pack_targets` 往返、`normalize_weights`
  的归一与退化回退、`split_head_outputs` 对旧单头 checkpoint 的兼容。
- `tests/test_core.py` 由 14 增至 20 用例：用策略 spy 取代原先「`red_hits >= 0`」这类恒真断言
  （基数不可能为负，原断言什么都没验证），真正证明滚动窗口只含历史、且不含目标期及之后的数据。
- `tests/test_diagnostics.py` 保持 10 用例不变（0.2.1 的两个阈值边界在此被独立变异复核）。
- 合计 **101 用例**：`python -m pytest -q` → `101 passed`；无 TensorFlow 环境
  `--ignore=tests/test_lstm_model.py` → `87 passed`。
- 0.2.1 条目里「由 QA 后续落地」的测试已在 0.2.2 全部落地。
- 独立变异复核（由本次提交方执行，确认测试非空转）：把 `blue_head` 改回 `sigmoid` → 3 个用例失败；
  `MAX_HEALTHY_DISTINCT_BLUE` 由 7 改 3 → 1 个用例失败。

## [0.2.1] - 2026-11-01

补丁版本，闭合 0.2.0 遗留的验收缺口：notebook 默认 `SEED` 写死导致两次运行输出仍然逐位相同。

### 修复

- `notebooks/ssq_lstm_kaggle.ipynb`：`SEED` 由常量 `42` 改为 `None`，运行时用
  `int(time.time_ns() % 2**31)` 生成并**打印实际 SEED**；`walk_forward_lstm` 与
  `np.random.default_rng` 共用同一个 `SEED` 变量。0.2.0 修掉了 argmax 恒定的机制，
  但写死的种子仍让「两次运行不该完全一样」这条验收标准在默认配置下不成立。
  默认随机、日志留痕的折中同时满足 README 的可审计立场：要复现某次结果，把日志里的 SEED 填回即可。
- `notebooks/ssq_lstm_kaggle.ipynb`：候选组合去重失败时不再沿用上一次的红球（那会静默输出
  重复组合、让「5 组候选」名不副实），改为跳过该条并打印警告；不足 5 组时明确说明实际得到
  N 组互异候选，不补齐、不假装凑满。
- `core.blue_degeneracy_report`：`degenerate` 判据由 `distinct_blue <= 3` 放宽为
  `distinct_blue <= 7`，堵住阈值盲区 —— 四个蓝球各占 25% 时熵恰好 `log2(4) == 2.0`，
  原判据 `entropy < 2.0` 为假，会把「只探索了 1/4 号池」的退化分布漏报为健康。
  docstring 补写「`total <= 15` 时不作判定（样本太少不可靠）」这一下限行为。
- `notebooks/ssq_lstm_kaggle.ipynb`：最终候选格新增 `lstm_argmax` 与 `lstm_sample` 的蓝球
  退化对照与读法说明，避免读者把「argmax 组恒定」误判为 bug 复发 —— 它不含随机性，
  恒定恰恰是「蓝球分布接近均匀、模型无信号」的证据。

### 变更

- （补记 0.2.0 的破坏性变更）`lstm.walk_forward_lstm` 返回行的 `strategy` 字段由 `"lstm"`
  改为 `"lstm_argmax"` / `"lstm_sample"`，且**返回行数翻倍**（每期两条）。任何下游
  `row["strategy"] == "lstm"` 的过滤会静默失效、任何按行数统计成本的地方会翻倍。
  0.2.0 的条目里只写了功能没标注破坏性，此处补记。
- 新增常量 `core.MAX_HEALTHY_DISTINCT_BLUE = 7` 与 `core.MIN_TRIALS_FOR_DEGENERACY = 16`，
  退化判据的两个阈值不再散落在判据表达式里。
- `pyproject.toml` 与 `ssq_research.__version__` 同步 `0.2.1`。

### 测试

- `tests/test_diagnostics.py` 补两个边界用例：四球均匀分布（熵恰为 2.0）必须判为退化、
  15 次以内样本不作判定。现有 36 个用例全绿。
- 双头模型 / softmax 归一 / `predict_probabilities` 拼接 / 退化阈值的独立测试由 QA
  在 `tests/test_lstm_model.py`、`tests/test_lstm_heads.py` 落地。

## [0.2.0] - 2026-11-01

本次迭代修复「LSTM 蓝球预测完全确定性 / 退化」缺陷：两次独立运行会输出完全相同的蓝球，
根因是输出头把 49 个号码当成 49 个互相独立的二分类，其最优解就是历史边际出现率构成的常数
向量，而蓝球边际率在长样本上接近均匀 1/16，`argmax` 因此每期都落在同一个号上。

### 修复

- `lstm.build_model`：输出层由单个 `Dense(49, sigmoid)` + BCE 改为双头 ——
  `red_head`（`Dense(33, sigmoid)` + BCE，6/33 多标签）与
  `blue_head`（`Dense(16, softmax)` + sparse categorical cross-entropy，1/16 单选）。
  这样模型才能表达「蓝球是 16 选 1 的互斥分布」，而不是把它当成 16 个独立伯努利。
- `lstm.pick_from_probabilities`：`seed` 参数在 0.1.0 中收进来后从未使用，属于死参数。
  现拆分为 `pick_argmax`（确定性点估计，docstring 明确标注不含随机性）与
  `pick_sampled`（使用 `numpy.random.Generator` 按概率无放回抽 6 红 + 1 蓝），
  `pick_from_probabilities` 保留为 `pick_sampled` 的向后兼容包装，`seed` 真正生效。
- `lstm.walk_forward_lstm`：同一前缀同时输出 `lstm_argmax` 与 `lstm_sample` 两条记录，
  退化与否在结果里直接可见；每行新增 `seed`、`deterministic`、`blue_max_probability`。
- `core.FrequencyStrategy`：蓝球平局由「永远取小号」改为按传入 `rng` 随机选，
  修掉逐期推进时蓝球长期锁死同一个号的问题；红球保持确定性排序，原因写入 docstring。
- `prize.evaluate_prize` / `core.build_result_row`：一、二等奖 `payout` 由 `0.0` 改为 `None`，
  `net` 同步为 `None`，不再把中头奖记成亏 2 元；`core.summarize` 求和时跳过 `None`
  并新增 `variable_prize_count` 计数。
- `core.summarize`：新增除零与全 `None` 保护，空行集或全未结算时不再抛异常。
- `data_fetch.parse_ssq_text`：逐行校验红球 6 个互异且 1..33、蓝球 1..16，
  失败行跳过并计入 `skipped_line_count`，不再静默丢行。
- `notebooks/ssq_lstm_kaggle.ipynb`：最终候选组合的蓝球改为 seeded 无放回抽样，
  并输出蓝球概率 Top5、经验熵与退化判定，不再是 `argsort` 取前 5 的确定性派生。

### 新增

- `lstm.set_global_seed(seed, deterministic=False)`，同时设定 Python / NumPy / TensorFlow 随机源；
  `LstmConfig` 新增 `seed: int = 0` 与 `deterministic: bool = False`（默认不启用 op determinism，避免显著变慢）。
- `lstm.predict_probabilities(model, recent_draws)`：唯一负责把双头输出拼接回 49 维
  （红 33 + 蓝 16）的地方，`walk_forward_lstm` 与 notebook 均调用它，避免切片逻辑重复。
- `lstm.make_sequences(..., return_heads=True)`：同时产出 `y_red (N,33)` 与 `y_blue (N,)`；
  默认 `return_heads=False` 时行为与 0.1.0 完全一致。
- `lstm.split_probabilities` / `lstm.combine_heads` / `lstm.pack_targets`：红蓝边界的单一来源常量
  `RED_SLICE` / `BLUE_SLICE`，防止编码器与目标构造错位。
- `core.blue_degeneracy_report(rows)`：输出 `distinct_blue`、`blue_entropy`（附 `blue_entropy_max = 4.0`）、
  `top1_share`、`longest_run`、`degenerate`，让「模型没学到东西」在结果里可见而不是被当成一次成功预测。
- `core.summarize` 每条输出新增 `distinct_blue`、`blue_entropy`、`settled_trials`、`variable_prize_count`。
- `data_fetch.ParseStats` / `parse_ssq_text_with_stats` / `fetch_ssq_draws_with_stats`：
  回传 `total_lines`、`parsed_lines`、`skipped_line_count` 与 `skipped_reasons`。
- `tests/test_picks.py`、`tests/test_diagnostics.py`：抽样策略与退化检测的回归测试，无需 TensorFlow。

### 变更

- 回测结果行新增 `pick_reds`、`pick_blue` 字段，用于计算退化指标；已有字段一个未删。
- `cli backtest` 新增 `--seed`，并在输出中打印 `summarize` 与退化报告。
- 版本号 `0.1.0` → `0.2.0`（`pyproject.toml` 与 `ssq_research.__version__` 同步）。

## [0.1.0] - 初始版本

- 双色球 CSV 导入与校验、随机/频率策略、滚动回测、奖金与收益记录、JSON 导出。
- 可选多层 LSTM 训练与 walk-forward 回测流水线（单一 49 维 sigmoid 输出头）。
