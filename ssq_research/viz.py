"""SVG chart rendering for the HTML report.

Chart text is bilingual: Chinese is used only when a CJK font can actually be
found on the machine, otherwise every label falls back to English. Assuming the
font exists is how a report ends up full of tofu boxes, so the check is done once
and cached.

Every function returns an SVG string rather than a file, so a generated report is
a single self-contained document that can be downloaded from Kaggle as one
artifact.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Mapping, Sequence

from .core import Draw

SERIES_COLOURS = ("#378ADD", "#1D9E75", "#BA7517", "#D4537E", "#7F77DD", "#5F5E5A")
REFERENCE_COLOUR = "#E24B4A"

#: Font family names to look for, in preference order.
CJK_FONT_NAMES = (
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Droid Sans Fallback",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
)
#: Font files to register when the family is not already known to matplotlib.
CJK_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)

_cjk_family: str | None = None
_cjk_probed = False


def ensure_cjk_font() -> str | None:
    """Return a usable CJK font family name, or ``None`` when there is none.

    The result is cached: probing matplotlib's font list on every chart would be
    slow, and the answer cannot change within a process.
    """
    global _cjk_family, _cjk_probed
    if _cjk_probed:
        return _cjk_family
    _cjk_probed = True
    try:
        from matplotlib import font_manager
    except ImportError:  # pragma: no cover - matplotlib is required anyway
        return None

    known = {entry.name for entry in font_manager.fontManager.ttflist}
    for family in CJK_FONT_NAMES:
        if family in known:
            _cjk_family = family
            return family
    for candidate in CJK_FONT_PATHS:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            _cjk_family = font_manager.FontProperties(fname=str(path)).get_name()
            return _cjk_family
        except Exception:  # noqa: BLE001 - a broken font file must not kill the report
            continue
    return None


def has_cjk_font() -> bool:
    """Whether chart text can safely be written in Chinese."""
    return ensure_cjk_font() is not None


def reset_font_cache() -> None:
    """Forget the cached probe result.

    Call this after installing a font at runtime (a notebook doing
    ``apt-get install fonts-wqy-zenhei`` mid-session is the usual case),
    otherwise the first "no font" answer is cached for the whole process.
    """
    global _cjk_family, _cjk_probed
    _cjk_family = None
    _cjk_probed = False


def _label(english: str, chinese: str) -> str:
    """Pick the label matching the font situation."""
    return chinese if has_cjk_font() else english


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only on bare installs
        raise RuntimeError('charts require matplotlib; install with: pip install -e ".[viz]"') from exc
    family = ensure_cjk_font()
    if family:
        stack = [family] + [name for name in matplotlib.rcParams["font.sans-serif"] if name != family]
        matplotlib.rcParams["font.sans-serif"] = stack
        # A CJK font without a proper minus glyph renders negative numbers as boxes.
        matplotlib.rcParams["axes.unicode_minus"] = False
    return plt



def _to_svg(figure) -> str:
    plt = _plt()
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    markup = buffer.getvalue()
    start = markup.find("<svg")
    return markup[start:] if start >= 0 else markup


def frequency_chart_svg(rows: Sequence, title: str, ylabel: str | None = None) -> str:
    """Bar chart of observed rate per ball with 95% Wilson intervals.

    The dashed line is the uniform expectation. When the error bars straddle it,
    the apparent hot/cold gap is indistinguishable from sampling noise.
    """
    plt = _plt()
    figure, axes = plt.subplots(figsize=(9.2, 3.4))
    numbers = [row.number for row in rows]
    rates = [row.rate for row in rows]
    lower = [max(0.0, row.rate - row.interval[0]) for row in rows]
    upper = [max(0.0, row.interval[1] - row.rate) for row in rows]
    expected = rows[0].expected_rate
    axes.bar(numbers, rates, width=0.72, color=SERIES_COLOURS[0], label=_label("observed", "观测值"))
    axes.errorbar(numbers, rates, yerr=[lower, upper], fmt="none", ecolor="#888780", elinewidth=0.7, capsize=1.5)
    axes.axhline(expected, color=REFERENCE_COLOUR, linewidth=1.0, linestyle="--",
                 label=_label(f"uniform {expected:.4f}", f"均匀 {expected:.4f}"))
    axes.set_title(title, fontsize=9)
    axes.set_xlabel(_label("ball", "球号"), fontsize=8)
    axes.set_ylabel(ylabel or _label("Appearance rate", "出现率"), fontsize=8)
    axes.tick_params(labelsize=7)
    axes.set_xticks(numbers if len(numbers) <= 16 else numbers[::2])
    axes.legend(fontsize=7, frameon=False, loc="upper right")
    figure.tight_layout()
    return _to_svg(figure)


def rolling_rate_svg(draws: Sequence[Draw], number: int, window: int = 30, is_blue: bool = False) -> str:
    """Rolling appearance rate of one ball over the sample."""
    plt = _plt()
    ordered = sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    indicators = [
        1.0 if ((draw.blue == number) if is_blue else (number in draw.reds)) else 0.0 for draw in ordered
    ]
    rolling: list[float] = []
    running = 0.0
    for index, value in enumerate(indicators):
        running += value
        if index >= window:
            running -= indicators[index - window]
        rolling.append(running / min(index + 1, window))
    figure, axes = plt.subplots(figsize=(9.2, 2.8))
    axes.plot(range(len(rolling)), rolling, color=SERIES_COLOURS[1], linewidth=1.1)
    expected = (1.0 / 16.0) if is_blue else (6.0 / 33.0)
    axes.axhline(expected, color=REFERENCE_COLOUR, linewidth=1.0, linestyle="--",
                 label=_label(f"uniform {expected:.4f}", f"均匀 {expected:.4f}"))
    axes.set_title(_label(f"Rolling {window}-period rate of ball {number}",
                          f"球 {number:02d} 的 {window} 期滚动出现率"), fontsize=9)
    axes.set_xlabel(_label("period index", "期序"), fontsize=8)
    axes.set_ylabel(_label("rate", "出现率"), fontsize=8)
    axes.tick_params(labelsize=7)
    axes.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    return _to_svg(figure)


def backtest_net_svg(rows: Sequence[Mapping], title: str = "Cumulative net by strategy") -> str:
    """Cumulative net return per strategy.

    Rows whose prize amount is unknown (first/second prize, ``net is None``) are
    skipped rather than counted as zero, so a strategy cannot look better or worse
    just because it hit a jackpot the payout data cannot price.
    """
    plt = _plt()
    grouped: dict[str, list[Mapping]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)
    figure, axes = plt.subplots(figsize=(9.2, 3.2))
    for index, (name, strategy_rows) in enumerate(sorted(grouped.items())):
        running = 0.0
        curve: list[float] = []
        for row in strategy_rows:
            net = row.get("net")
            if net is not None:
                running += float(net)
            curve.append(running)
        axes.plot(range(len(curve)), curve, linewidth=1.1, label=name, color=SERIES_COLOURS[index % len(SERIES_COLOURS)])
    axes.axhline(0.0, color="#888780", linewidth=0.8)
    axes.set_title(title, fontsize=9)
    axes.set_xlabel(_label("backtest period", "回测期序"), fontsize=8)
    axes.set_ylabel(_label("cumulative net (CNY)", "累计净收益（元）"), fontsize=8)
    axes.tick_params(labelsize=7)
    axes.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    return _to_svg(figure)


def score_chart_svg(numbers: Sequence[int], values: Sequence[float], title: str, reference: float | None = None) -> str:
    """Horizontal bar chart of algorithm scores for a short list of balls."""
    plt = _plt()
    figure, axes = plt.subplots(figsize=(6.4, max(2.2, 0.34 * len(numbers))))
    positions = list(range(len(numbers)))
    axes.barh(positions, values, color=SERIES_COLOURS[2], height=0.62)
    axes.set_yticks(positions)
    axes.set_yticklabels([str(number) for number in numbers], fontsize=7)
    axes.invert_yaxis()
    axes.set_title(title, fontsize=9)
    axes.set_xlabel(_label("score", "评分"), fontsize=8)
    axes.tick_params(labelsize=7)
    if reference is not None:
        axes.axvline(reference, color=REFERENCE_COLOUR, linewidth=1.0, linestyle="--")
    figure.tight_layout()
    return _to_svg(figure)
