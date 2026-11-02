"""Optional Gradio interface for live querying.

Kaggle does not expose inbound ports, so a plain web server is useless there;
``launch(share=True)`` publishes through Gradio's tunnel instead. That is why this
module is an optional extra and why the static HTML report from
:mod:`ssq_research.report` stays the recommended artifact — it needs no tunnel and
no dependency beyond matplotlib.

The interface deliberately prints the theoretical odds next to every result. A
ticket generator that hides the base rate invites the reader to mistake variety
for skill.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .core import Draw, Pick, load_csv
from .stats import build_probability_report


def _gr():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('the web app requires gradio; install with: pip install -e ".[web]"') from exc
    return gr


def load_draws(draws: Sequence[Draw] | None = None, csv_path: str | Path | None = None) -> list[Draw]:
    """Resolve the sample from memory or from a CSV file."""
    if draws is not None:
        return sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    if csv_path:
        return load_csv(Path(csv_path))
    raise ValueError("provide draws or csv_path")


def _scorer_names() -> list[str]:
    from .scores import available_algorithms

    status = available_algorithms()
    names = [name for name, ok in status.items() if ok and name != "lstm"]
    return sorted(names)


def _resolve_scorer(name: str):
    from .scores import ArimaScorer, build_ensemble, build_scorers

    scorers = build_scorers()
    if name == "ensemble":
        return build_ensemble(list(scorers.values()))
    if name == "arima":
        return ArimaScorer()
    if name in scorers:
        return scorers[name]
    raise ValueError(f"unknown algorithm: {name}")


def score_table(draws: Sequence[Draw], algorithm: str = "frequency", top: int = 10) -> tuple[list[list], list[list]]:
    """Top red and blue scores for the latest available history."""
    from .scores import normalize, top_indices

    scorer = _resolve_scorer(algorithm)
    scores = normalize(scorer.score(list(draws)))
    red = scores[:33]
    blue = scores[33:]
    red_rows = [[int(index) + 1, round(float(red[index]) * 100, 3)] for index in top_indices(red, min(top, 33))]
    blue_rows = [[int(index) + 1, round(float(blue[index]) * 100, 3)] for index in top_indices(blue, min(5, 16))]
    return red_rows, blue_rows


def suggest_ticket(draws: Sequence[Draw], algorithm: str = "frequency", mode: str = "sample", seed: int = 0) -> tuple[str, str]:
    """Produce one ticket plus the sentence that must accompany it."""
    from .scores import ScorerStrategy

    scorer = _resolve_scorer(algorithm)
    import numpy as np

    rng = np.random.default_rng(seed)
    pick: Pick = ScorerStrategy(scorer, mode).pick(tuple(draws), rng)
    odds = build_probability_report(list(draws)).jackpot
    ticket = "红球 " + " ".join(f"{number:02d}" for number in pick.reds) + f"　蓝球 {pick.blue:02d}"
    caveat = (
        f"算法：{algorithm} / {mode}　seed={seed}\n\n"
        f"这张票的中一等奖概率是 1 / {odds.one_in:,.0f}，与随手乱选完全相同。"
        f"上面的排名描述的是历史，不是下一期。"
    )
    return ticket, caveat


def build_app(draws: Sequence[Draw] | None = None, csv_path: str | Path | None = None):
    """Construct the Gradio blocks without launching them."""
    gr = _gr()
    sample = load_draws(draws, csv_path)
    report = build_probability_report(sample)
    names = _scorer_names()

    summary = (
        f"样本 {report.samples} 期（{report.first_issue} → {report.last_issue}）　·　"
        f"红球卡方 p = {report.red_uniformity.p_value:.3f}　·　蓝球卡方 p = {report.blue_uniformity.p_value:.3f}\n\n"
        + (
            "两类球的历史分布都与公平开奖一致，下面的排名只是历史频次，不含预测信息。"
            if not report.has_detectable_pattern
            else "检测到分布偏离，请先排查数据完整性再解读排名。"
        )
    )

    with gr.Blocks(title="SSQ 统计研究") as app:
        gr.Markdown("# 双色球统计研究\n" + summary)
        with gr.Row():
            algorithm = gr.Dropdown(choices=names, value=names[0] if names else None, label="算法")
            mode = gr.Dropdown(choices=["sample", "argmax"], value="sample", label="出票模式")
            seed = gr.Slider(minimum=0, maximum=9999, step=1, value=0, label="seed")
        with gr.Row():
            generate = gr.Button("生成候选", variant="primary")
            refresh = gr.Button("刷新排名")
        ticket = gr.Textbox(label="候选号码", lines=1)
        caveat = gr.Textbox(label="必须一起看的话", lines=3)
        with gr.Row():
            red_table = gr.Dataframe(headers=["红球", "得分 %"], label="红球排名", interactive=False)
            blue_table = gr.Dataframe(headers=["蓝球", "得分 %"], label="蓝球排名", interactive=False)

        def _ticket(current_algorithm, current_mode, current_seed):
            return suggest_ticket(sample, current_algorithm, current_mode, int(current_seed))

        def _tables(current_algorithm):
            red_rows, blue_rows = score_table(sample, current_algorithm)
            return red_rows, blue_rows

        generate.click(_ticket, inputs=[algorithm, mode, seed], outputs=[ticket, caveat])
        refresh.click(_tables, inputs=[algorithm], outputs=[red_table, blue_table])
        app.load(_tables, inputs=[algorithm], outputs=[red_table, blue_table])
    return app


def launch(draws: Sequence[Draw] | None = None, csv_path: str | Path | None = None, share: bool = True, **kwargs):
    """Build and launch the app. On Kaggle pass ``share=True``."""
    app = build_app(draws, csv_path)
    return app.launch(share=share, **kwargs)
