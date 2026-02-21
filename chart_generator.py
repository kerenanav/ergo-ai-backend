"""
chart_generator.py — Matplotlib chart functions returning base64 PNG strings.

IMPORTANT: matplotlib.use("Agg") is called at module level before any other
matplotlib import. This enables non-interactive (Agg) rendering, which is
required for async/multi-threaded FastAPI use.

All public functions return a base64-encoded PNG string that can be:
  - Embedded directly in JSON API responses
  - Decoded and passed to ReportLab via _b64_to_rl_image() in report_generator.py

Thread safety: each function creates and closes its own Figure. plt.close(fig)
inside to_base64() prevents memory leaks under concurrent load.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — must be before pyplot import

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Brand palette ────────────────────────────────────────────────────────────
_NAVY   = "#1a237e"
_INDIGO = "#283593"
_GREEN  = "#2e7d32"
_RED    = "#c62828"
_AMBER  = "#f57f17"
_LIGHT  = "#e8eaf6"


# ────────────────────────────────────────────────────────────────────────────
# Internal helper
# ────────────────────────────────────────────────────────────────────────────

def to_base64(fig: plt.Figure) -> str:
    """Serialise *fig* to base64 PNG and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)  # rewind before reading
    return base64.b64encode(buf.read()).decode("ascii")


# ────────────────────────────────────────────────────────────────────────────
# Chart functions
# ────────────────────────────────────────────────────────────────────────────

def profit_composition_chart(
    n_accepted: int,
    n_rejected: int,
    expected_rev: float,
    realized: float | None = None,
) -> str:
    """Pie (decision split) + bar (expected vs realized revenue)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor("white")

    # Left pie: accepted vs rejected
    ax_pie = axes[0]
    labels = ["Accepted", "Rejected"]
    vals   = [max(n_accepted, 0), max(n_rejected, 0)]
    colors = [_GREEN, _RED]
    if sum(vals) == 0:
        vals = [1, 0]
    ax_pie.pie(vals, labels=labels, colors=colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 10})
    ax_pie.set_title("Decision Split", fontsize=11, color=_NAVY, pad=8)

    # Right bar: expected (+ realized if available)
    ax_bar = axes[1]
    bar_labels = ["Expected"]
    bar_vals   = [expected_rev]
    bar_colors = [_INDIGO]
    if realized is not None:
        bar_labels.append("Realized")
        bar_vals.append(realized)
        bar_colors.append(_GREEN if realized >= 0 else _RED)

    bars = ax_bar.bar(bar_labels, bar_vals, color=bar_colors, width=0.5)
    ax_bar.set_ylabel("Revenue (€)", fontsize=9)
    ax_bar.set_title("Revenue", fontsize=11, color=_NAVY)
    ax_bar.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}")
    )
    for bar, val in zip(bars, bar_vals):
        y_pos = bar.get_height() + abs(bar.get_height()) * 0.01
        ax_bar.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"€{val:,.0f}", ha="center", va="bottom", fontsize=8)
    ax_bar.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return to_base64(fig)


def capacity_utilization_chart(n_accepted: int, capacity: float) -> str:
    """Donut-style pie showing capacity utilization."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("white")

    utilization = min(n_accepted / max(float(capacity), 1), 1.0)
    remaining   = max(1.0 - utilization, 0.0)

    ax.pie(
        [utilization, remaining],
        colors=[_INDIGO, _LIGHT],
        startangle=90,
        autopct="%1.0f%%",
        textprops={"fontsize": 11},
        wedgeprops={"width": 0.6},
    )
    ax.set_title(
        f"Capacity Utilization\n{n_accepted} / {int(capacity)} rooms",
        fontsize=11, color=_NAVY,
    )
    fig.tight_layout()
    return to_base64(fig)


def decision_distribution_chart(
    p_cancel_values: list[float],
    decisions: list[int],
) -> str:
    """Histogram of P(cancel) split by accepted vs rejected."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")

    p = np.array(p_cancel_values, dtype=float)
    d = np.array(decisions, dtype=int)
    accepted = p[d == 1]
    rejected = p[d == 0]
    bins = np.linspace(0, 1, 21)

    ax.hist(accepted, bins=bins, color=_GREEN, alpha=0.7, label="Accepted", edgecolor="white")
    ax.hist(rejected, bins=bins, color=_RED,   alpha=0.7, label="Rejected",  edgecolor="white")
    ax.set_xlabel("P(cancel)", fontsize=9)
    ax.set_ylabel("Count",     fontsize=9)
    ax.set_title("Decision Distribution by Cancellation Risk", fontsize=11, color=_NAVY)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return to_base64(fig)


def expected_vs_realized_chart(
    expected: list[float],
    realized: list[float] | None,
    outcome_available: bool = True,
) -> str:
    """Side-by-side bars for expected vs realized revenue per booking."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")

    n = len(expected)
    x = np.arange(1, n + 1)

    ax.bar(x - 0.2, expected, width=0.4, color=_INDIGO, alpha=0.8, label="Expected")
    if outcome_available and realized and len(realized) == n:
        ax.bar(x + 0.2, realized, width=0.4, color=_GREEN, alpha=0.8, label="Realized")

    ax.set_xlabel("Booking", fontsize=9)
    ax.set_ylabel("Revenue (€)", fontsize=9)
    ax.set_title("Expected vs Realized Revenue", fontsize=11, color=_NAVY)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))

    if not outcome_available:
        ax.text(0.5, 0.5, "Realized data not available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888", style="italic")

    fig.tight_layout()
    return to_base64(fig)


def cumulative_profit_chart(
    ai_profits: list[float],
    hist_profits: list[float],
    periods: list[str] | None = None,
) -> str:
    """Cumulative revenue line chart: AI strategy vs accept-all baseline."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")

    x        = list(range(1, len(ai_profits) + 1))
    ai_cum   = np.cumsum(ai_profits)
    hist_cum = np.cumsum(hist_profits)

    ax.plot(x, ai_cum,   color=_INDIGO, marker="o", linewidth=2, label="AI Strategy")
    ax.plot(x, hist_cum, color=_AMBER,  marker="s", linewidth=2, label="Baseline (accept-all)", linestyle="--")
    ax.fill_between(x, ai_cum, hist_cum,
                    where=(ai_cum >= hist_cum), alpha=0.12, color=_GREEN, label="_nolegend_")
    ax.fill_between(x, ai_cum, hist_cum,
                    where=(ai_cum < hist_cum),  alpha=0.12, color=_RED,   label="_nolegend_")

    if periods and len(periods) == len(x):
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontsize=7, rotation=30, ha="right")

    ax.set_xlabel("Period", fontsize=9)
    ax.set_ylabel("Cumulative Revenue (€)", fontsize=9)
    ax.set_title("Cumulative Profit: AI vs Baseline", fontsize=11, color=_NAVY)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))

    fig.tight_layout()
    return to_base64(fig)


def incremental_value_chart(
    deltas: list[float],
    periods: list[str] | None = None,
) -> str:
    """Bar chart of per-period delta (AI revenue − baseline revenue)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")

    x      = list(range(1, len(deltas) + 1))
    colors = [_GREEN if d >= 0 else _RED for d in deltas]

    ax.bar(x, deltas, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)

    if periods and len(periods) == len(x):
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontsize=7, rotation=30, ha="right")

    ax.set_xlabel("Period", fontsize=9)
    ax.set_ylabel("Delta Revenue (€)", fontsize=9)
    ax.set_title("Incremental Value: AI − Baseline", fontsize=11, color=_NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))

    fig.tight_layout()
    return to_base64(fig)


def per_period_delta_chart(fold_results: list[dict]) -> str:
    """Bar chart of per-fold delta derived from backtest fold results."""
    periods = [f'F{r["fold"]}' for r in fold_results]
    deltas  = [
        r.get("ai_realized_revenue", 0.0) - r.get("baseline_realized_revenue", 0.0)
        for r in fold_results
    ]
    return incremental_value_chart(deltas, periods)


def penalty_sensitivity_chart(penalty_sens: list[dict]) -> str:
    """Line chart: expected revenue vs cancellation penalty."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")

    x = [r["penalty_value"]           for r in penalty_sens]
    y = [r["total_expected_revenue"]   for r in penalty_sens]

    ax.plot(x, y, color=_INDIGO, marker="o", linewidth=2)
    ax.set_xlabel("Cancellation Penalty (€)", fontsize=9)
    ax.set_ylabel("Expected Revenue (€)",      fontsize=9)
    ax.set_title("Sensitivity: Cancellation Penalty", fontsize=11, color=_NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))

    fig.tight_layout()
    return to_base64(fig)


def capacity_sensitivity_chart(cap_sens: list[dict]) -> str:
    """Line chart: expected revenue vs capacity."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")

    x = [r["capacity_value"]           for r in cap_sens]
    y = [r["total_expected_revenue"]    for r in cap_sens]

    ax.plot(x, y, color=_INDIGO, marker="o", linewidth=2)
    ax.set_xlabel("Capacity (rooms)",      fontsize=9)
    ax.set_ylabel("Expected Revenue (€)",  fontsize=9)
    ax.set_title("Sensitivity: Capacity",  fontsize=11, color=_NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))

    fig.tight_layout()
    return to_base64(fig)
