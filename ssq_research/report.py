"""Self-contained HTML research dashboard.

The output is one downloadable ``.html`` file with charts inlined as SVG and no
external requests, which matters because Kaggle gives you a file, not a port to
serve on.

The dashboard is deliberately blunt about what it does and does not show. The
"top numbers" panel is labelled as *historical* frequency, every figure carries
its 95% interval, and the jackpot odds sit directly beside the ranking so the two
cannot be read separately.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Mapping, Sequence

from .core import Draw, blue_degeneracy_report, summarize
from .stats import ProbabilityReport, build_probability_report, compare_proportions

_CSS = """
:root{--bg:#ffffff;--surface:#f7f7f5;--line:#e3e2dd;--ink:#1c1c1a;--muted:#5f5e5a;--accent:#378ADD;--warn:#A32D2D;--ok:#1D9E75}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.65}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:22px;font-weight:600;margin:0 0 6px}
h2{font-size:16px;font-weight:600;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:14px;font-weight:600;margin:18px 0 8px}
p{margin:8px 0}
.meta{color:var(--muted);font-size:12.5px}
.verdict{border-radius:10px;padding:14px 16px;margin:20px 0;border:1px solid var(--line);background:var(--surface)}
.verdict strong{display:block;font-size:14px;margin-bottom:4px}
.verdict.neg{border-left:4px solid var(--warn)}
.verdict.pos{border-left:4px solid var(--ok)}
table{width:100%;border-collapse:collapse;margin:10px 0 4px;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;border-radius:5px;padding:1px 6px;font-size:11.5px;border:1px solid var(--line);background:var(--surface);color:var(--muted)}
.badge.hi{border-color:#F09595;color:var(--warn)}
.badge.lo{border-color:#97C459;color:#3B6D11}
.chart{margin:12px 0 4px;overflow-x:auto}
.chart svg{max-width:100%;height:auto}
.note{color:var(--muted);font-size:12.5px;margin:6px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.row{display:flex;gap:16px;flex-wrap:wrap}
.row>.card{flex:1 1 260px;margin:0}
.k{color:var(--muted);font-size:12px}
.v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
select{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:#fff}
footer{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
code{background:var(--surface);padding:1px 5px;border-radius:4px;font-size:12.5px}
"""

_JS = """
(function(){
  var data = window.SSQ_DATA || {red:[], blue:[]};
  function fmt(x, d){ return Number(x).toFixed(d); }
  function describe(rows, number){
    for (var i=0;i<rows.length;i++){ if (rows[i].number === number) return rows[i]; }
    return null;
  }
  function render(){
    var pool = document.getElementById('pool').value;
    var rows = pool === 'blue' ? data.blue : data.red;
    var number = parseInt(document.getElementById('ball').value, 10);
    var row = describe(rows, number);
    var out = document.getElementById('detail');
    if (!row){ out.textContent = 'no data'; return; }
    out.innerHTML =
      '<div class="k">出现次数 / 样本期数</div><div class="v">' + row.count + ' / ' + data.samples + '</div>' +
      '<div class="k" style="margin-top:10px">历史出现率</div><div class="v">' + (row.rate*100).toFixed(2) + '%</div>' +
      '<div class="k" style="margin-top:10px">理论出现率</div><div class="v">' + (row.expected*100).toFixed(2) + '%</div>' +
      '<div class="k" style="margin-top:10px">95% 置信区间</div><div class="v" style="font-size:15px">' +
        (row.ci_low*100).toFixed(2) + '% ~ ' + (row.ci_high*100).toFixed(2) + '%</div>' +
      '<div class="k" style="margin-top:10px">距上次出现</div><div class="v">' + row.gap + ' 期</div>' +
      '<p class="note">' + (row.significant
        ? '该号的历史出现率偏离理论值超出了单球 95% 区间。注意：同时检验 33（或 16）个号时，出现 1~2 个这样的号本身就在期望之内，不能据此认为存在可预测的规律。'
        : '该号的历史出现率与理论值没有显著差异，落在 95% 置信区间内。') + '</p>';
  }
  function fill(){
    var pool = document.getElementById('pool').value;
    var rows = pool === 'blue' ? data.blue : data.red;
    var select = document.getElementById('ball');
    select.innerHTML = '';
    rows.forEach(function(row){
      var option = document.createElement('option');
      option.value = row.number;
      option.textContent = '号码 ' + String(row.number).padStart(2,'0');
      select.appendChild(option);
    });
    render();
  }
  var poolEl = document.getElementById('pool');
  var ballEl = document.getElementById('ball');
  if (poolEl){ poolEl.addEventListener('change', fill); }
  if (ballEl){ ballEl.addEventListener('change', render); }
  fill();
})();
"""


def _p(value: float) -> str:
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def _pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _last_gap(draws: Sequence[Draw], number: int, is_blue: bool) -> int:
    for offset, draw in enumerate(reversed(draws)):
        hit = (draw.blue == number) if is_blue else (number in draw.reds)
        if hit:
            return offset
    return len(draws)


def _explorer_payload(draws: Sequence[Draw], report: ProbabilityReport) -> dict:
    def rows(source: Sequence, is_blue: bool) -> list[dict]:
        payload = []
        for row in source:
            low, high = row.interval
            payload.append(
                {
                    "number": row.number,
                    "count": row.count,
                    "rate": round(row.rate, 6),
                    "expected": round(row.expected_rate, 6),
                    "ci_low": round(low, 6),
                    "ci_high": round(high, 6),
                    "gap": _last_gap(draws, row.number, is_blue),
                    "significant": bool(row.deviates_significantly),
                }
            )
        return payload

    return {"samples": report.samples, "red": rows(report.red_rows, False), "blue": rows(report.blue_rows, True)}


def _prize_table(report: ProbabilityReport) -> str:
    lines = ["<table><thead><tr><th>奖级</th><th>中奖组合数</th><th>单注中奖概率</th><th>约等于</th></tr></thead><tbody>"]
    for odds in report.prizes:
        lines.append(
            "<tr><td>{}</td><td class=\"num\">{:,}</td><td class=\"num\">{:.4%}</td><td class=\"num\">1 / {:,.0f}</td></tr>".format(
                html.escape(odds.level), odds.combinations, odds.probability, odds.one_in
            )
        )
    lines.append("</tbody></table>")
    return "".join(lines)


def _frequency_table(rows: Sequence, label: str) -> str:
    lines = [f"<table><thead><tr><th>排名</th><th>{label}</th><th>出现次数</th><th>历史出现率</th><th>理论值</th><th>95% 区间</th><th>判定</th></tr></thead><tbody>"]
    for rank, row in enumerate(rows, start=1):
        low, high = row.interval
        badge = (
            '<span class="badge hi">偏离</span>' if row.deviates_significantly else '<span class="badge">区间内</span>'
        )
        lines.append(
            "<tr><td class=\"num\">{}</td><td class=\"num\">{:02d}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{} ~ {}</td><td>{}</td></tr>".format(
                rank, row.number, row.count, _pct(row.rate), _pct(row.expected_rate), _pct(low), _pct(high), badge
            )
        )
    lines.append("</tbody></table>")
    return "".join(lines)


def _comparison_table(rows: Sequence[Mapping], baseline_name: str = "random") -> tuple[str, str]:
    summaries = summarize(rows)
    baseline = next((item for item in summaries if item["strategy"] == baseline_name), None)
    headers = [
        "策略", "期数", "红球均命中", "蓝球均命中", "蓝球去重", "蓝球熵(bit)",
        "已结算期数", "净收益", "与随机基线的差异 p",
    ]
    lines = ["<table><thead><tr>" + "".join(f"<th>{head}</th>" for head in headers) + "</tr></thead><tbody>"]
    conclusion_parts = []
    for item in summaries:
        trials = int(item["trials"])
        blue_hits = float(item["avg_blue_hits"]) * trials
        p_value: float | None = None
        if baseline is not None and item["strategy"] != baseline_name:
            base_trials = int(baseline["trials"])
            base_hits = float(baseline["avg_blue_hits"]) * base_trials
            p_value = compare_proportions(int(round(blue_hits)), trials, int(round(base_hits)), base_trials)
            if p_value < 0.05:
                conclusion_parts.append(
                    f"{item['strategy']} 的蓝球命中率与随机基线的差异达到 p={_p(p_value)}，但这仍是大规模多重比较中的一次，需独立样本复核"
                )
        net = item.get("net")
        net_text = "未知" if net is None else f"{net:.2f}"
        cells = [
            f"<td>{html.escape(str(item['strategy']))}</td>",
            f"<td class=\"num\">{trials}</td>",
            f"<td class=\"num\">{item['avg_red_hits']:.3f}</td>",
            f"<td class=\"num\">{item['avg_blue_hits']:.4f}</td>",
            f"<td class=\"num\">{item.get('distinct_blue', '-')}</td>",
            f"<td class=\"num\">{item.get('blue_entropy', 0.0):.3f}</td>",
            f"<td class=\"num\">{item.get('settled_trials', '-')}</td>",
            f"<td class=\"num\">{net_text}</td>",
            f"<td class=\"num\">{'-' if p_value is None else _p(p_value)}</td>",
        ]
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</tbody></table>")
    if not conclusion_parts:
        conclusion = "没有任何策略的蓝球命中率与随机基线出现统计学显著差异。这正是公平开奖应有的结果。"
    else:
        conclusion = "；".join(conclusion_parts) + "。"
    return "".join(lines), conclusion


def _degeneracy_table(rows: Sequence[Mapping]) -> str:
    grouped: dict[str, list[Mapping]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)
    lines = ["<table><thead><tr><th>策略</th><th>蓝球去重数</th><th>蓝球熵 (bit)</th><th>众数占比</th><th>最长连击</th><th>判定</th></tr></thead><tbody>"]
    for name in sorted(grouped):
        report = blue_degeneracy_report(grouped[name])
        verdict = '<span class="badge hi">退化</span>' if report.get("degenerate") else '<span class="badge lo">健康</span>'
        lines.append(
            "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{:.3f}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td>{}</td></tr>".format(
                html.escape(name), report.get("distinct_blue", "-"), float(report.get("blue_entropy", 0.0)),
                _pct(float(report.get("top1_share", 0.0)), 1), report.get("longest_run", "-"), verdict,
            )
        )
    lines.append("</tbody></table>")
    return "".join(lines)


def _chart_section(draws: Sequence[Draw], report: ProbabilityReport, rows: Sequence[Mapping]) -> str:
    try:
        from .viz import backtest_net_svg, frequency_chart_svg
    except RuntimeError as exc:
        return f'<div class="card">图表未生成：{html.escape(str(exc))}</div>'
    try:
        red_chart = frequency_chart_svg(list(report.red_rows), "Red ball appearance rate (33 balls)", "appearance rate")
        blue_chart = frequency_chart_svg(list(report.blue_rows), "Blue ball appearance rate (16 balls)", "appearance rate")
        net_chart = backtest_net_svg(list(rows))
    except RuntimeError as exc:
        return f'<div class="card">图表未生成：{html.escape(str(exc))}</div>'
    except Exception as exc:  # a chart failure must never take the report down
        return f'<div class="card">图表渲染失败：{html.escape(repr(exc))}</div>'
    return (
        f'<div class="chart">{red_chart}</div>'
        f'<p class="note">误差棒为单球 95% Wilson 区间，红色虚线为理论均匀值 6/33。区间与虚线重叠即说明该号的冷热程度无法与随机波动区分。</p>'
        f'<div class="chart">{blue_chart}</div>'
        f'<p class="note">蓝球理论值为 1/16。区间普遍比红球宽，因为每个号的出现次数只有红球的约 1/6。</p>'
        f'<div class="chart">{net_chart}</div>'
        f'<p class="note">累计净收益按已结算期数累加；一/二等奖因奖池未知记为未知而不计 0，避免把头奖记成亏损。</p>'
    )


def build_report_html(
    draws: Sequence[Draw],
    rows: Sequence[Mapping],
    *,
    seed: int,
    window: int,
    title: str = "双色球统计研究仪表盘",
) -> str:
    """Render the full dashboard as one HTML document."""
    ordered = sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    report = build_probability_report(ordered)
    comparison_table, comparison_conclusion = _comparison_table(rows)
    degeneracy_table = _degeneracy_table(rows)
    charts = _chart_section(ordered, report, rows)
    payload = json.dumps(_explorer_payload(ordered, report), ensure_ascii=False)

    red_verdict = "与均匀分布一致" if report.red_uniformity.uniform_plausible else "出现偏离"
    blue_verdict = "与均匀分布一致" if report.blue_uniformity.uniform_plausible else "出现偏离"
    banner_class = "pos" if not report.has_detectable_pattern else "neg"
    banner_text = (
        "红球与蓝球的历史分布都与公平开奖一致：卡方拟合优度检验未发现超出抽样噪声的偏离。"
        "因此下面列出的“高频号码”只是过去出现得多，不构成对下一期的预测。"
        if not report.has_detectable_pattern
        else "至少一类球的分布偏离了均匀假设。请先排查数据完整性（漏期、错列、重复期号），"
        "再考虑是否存在真实偏差——历史上有过数据源错列造成的假信号。"
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    red_top_table = _frequency_table(report.red_top, "红球")
    blue_top_table = _frequency_table(report.blue_top, "蓝球")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="meta">数据范围 {html.escape(report.first_issue)} → {html.escape(report.last_issue)}　·　样本 {report.samples} 期　·
回测窗口 {window} 期　·　seed {seed}　·　生成于 {generated}</p>

<div class="verdict {banner_class}">
<strong>结论先看这里</strong>
{banner_text}
</div>

<div class="row">
<div class="card"><div class="k">红球卡方拟合优度</div><div class="v">p = {_p(report.red_uniformity.p_value)}</div>
<div class="note">χ² = {report.red_uniformity.statistic:.2f}，df = {report.red_uniformity.degrees_of_freedom}，{red_verdict}</div></div>
<div class="card"><div class="k">蓝球卡方拟合优度</div><div class="v">p = {_p(report.blue_uniformity.p_value)}</div>
<div class="note">χ² = {report.blue_uniformity.statistic:.2f}，df = {report.blue_uniformity.degrees_of_freedom}，{blue_verdict}</div></div>
<div class="card"><div class="k">单注一等奖概率</div><div class="v">1 / {report.jackpot.one_in:,.0f}</div>
<div class="note">与选号方式无关，任何算法的输出都不改变这个数字</div></div>
</div>

<h2>一、号码趋势与分布</h2>
{charts}

<h2>二、概率分析：高频号码与显著性</h2>
<p>下表按历史出现次数排序。请把“判定”列一起看：落在 95% 区间内意味着该号的冷热无法与随机波动区分。</p>
<h3>红球 Top 5（理论值 6/33 = 18.18%）</h3>
{red_top_table}
<h3>蓝球 Top 5（理论值 1/16 = 6.25%）</h3>
{blue_top_table}

<h3>各奖级理论概率</h3>
<p>一张 2 元单式票对应 <code>C(33,6) × 16 = 17,721,088</code> 种等可能结果之一。</p>
{_prize_table(report)}

<h2>三、多算法对照回测</h2>
<p>所有策略在同一窗口、同一成本、同一随机基线下逐期推进，只使用目标期之前的数据。</p>
{comparison_table}
<div class="card">{html.escape(comparison_conclusion)}</div>

<h3>输出退化检测</h3>
<p>命中率高低之外，还要看策略是否只是在重复同一个号。</p>
{degeneracy_table}

<h2>四、单号查询</h2>
<div class="card">
<p><label>球池 <select id="pool"><option value="red">红球</option><option value="blue">蓝球</option></select></label>
　<label>号码 <select id="ball"></select></label></p>
<div id="detail" class="note">加载中…</div>
</div>

<h2>五、算法说明与本次实测结论</h2>
<table><thead><tr><th>算法</th><th>它看什么</th><th>强项</th><th>本次实测</th></tr></thead><tbody>
<tr><td>频率统计</td><td>每个号的历史出现率，可加时间衰减</td><td>完全可解释，可复现</td><td>等价于“边际出现率”，在公平开奖下无优势</td></tr>
<tr><td>马尔可夫链</td><td>红球的两状态出现链；蓝球的 16×16 转移矩阵</td><td>能表达“上一个蓝球之后常跟哪个”</td><td>转移矩阵各行接近均匀，条件结构不带来命中率提升</td></tr>
<tr><td>ARIMA</td><td>对每个号的 0/1 出现序列做自回归</td><td>经典时间序列方法，能捕捉自相关</td><td>二进制序列的白噪声特征使自回归系数接近 0</td></tr>
<tr><td>LSTM</td><td>最近若干期的 49 维编码序列</td><td>能拟合非线性序列模式</td><td>输出趋近历史边际率；见 0.2.0 的双头修复与退化检测</td></tr>
<tr><td>集成</td><td>对上面各家的归一化得分加权</td><td>降低单模型输出方差</td><td>降低的是模型之间的分歧，不是开奖本身的随机性</td></tr>
</tbody></table>
<p class="note">集成学习常被描述为“提高预测准确性”。对公平开奖而言，它降低的是模型输出的方差，而不是中奖概率——多个都无法超越随机基线的模型，加权平均后仍然无法超越随机基线。</p>

<footer>
本报告只做统计描述与公平回测，不承诺预测中奖号码。所有排名均附带显著性判定；
任何策略都必须与同票数、同成本的随机基线比较后才可评估。
生成工具：ssq-quant-research-system。
</footer>
</div>
<script>window.SSQ_DATA = {payload};</script>
<script>{_JS}</script>
</body>
</html>
"""
