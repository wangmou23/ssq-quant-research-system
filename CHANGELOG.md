# 变更记录

本项目的所有重要变更都记录在此文件中，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
