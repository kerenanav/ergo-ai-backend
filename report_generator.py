"""
report_generator.py — PDF report generation with ReportLab (Optide V2).

Produces a professional A4 report in English or Italian containing:
  - Scope metadata (ID, domain, status, period, params snapshot)
  - Executive summary
  - Layer-1 model CV metrics table
  - Layer-3 backtesting fold results (with embedded charts)
  - Layer-2 optimization snapshot (with embedded charts)
  - Sensitivity analysis tables and charts
  - Outcome availability notice when realized data is absent

All text is sourced from i18n.py; charts are embedded as base64 PNG via
_b64_to_rl_image(). The PDF is returned as raw bytes.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from i18n import t

# ── Brand palette ─────────────────────────────────────────────────────────────
_NAVY   = colors.HexColor("#1a237e")
_INDIGO = colors.HexColor("#283593")
_LIGHT  = colors.HexColor("#e8eaf6")
_GRID   = colors.HexColor("#9fa8da")
_WHITE  = colors.white
_GREEN  = colors.HexColor("#1b5e20")
_RED    = colors.HexColor("#b71c1c")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _col_widths(*fractions: float, total_cm: float = 17.0) -> list[float]:
    return [f * total_cm * cm for f in fractions]


def _table_style(header_bg: Any = _INDIGO) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), header_bg),
        ("TEXTCOLOR",     (0, 0), (-1,  0), _WHITE),
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1,  0), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, _GRID),
        ("ALIGN",         (1, 1), (-1, -1), "RIGHT"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ])


def _pct_color(value: float) -> str:
    hex_color = "#1b5e20" if value >= 0 else "#b71c1c"
    sign = "+" if value >= 0 else ""
    return f'<font color="{hex_color}"><b>{sign}{value:.1f}%</b></font>'


def _b64_to_rl_image(b64_str: str, width_cm: float = 15.0) -> RLImage:
    """Decode a base64 PNG string and return a ReportLab Image flowable.

    The BytesIO buffer must remain open until doc.build() completes.
    We keep a reference by attaching it to the Image object.
    """
    buf = io.BytesIO(base64.b64decode(b64_str))
    img = RLImage(buf, width=width_cm * cm)
    img._buf = buf  # prevent garbage collection before doc.build()
    return img


# ────────────────────────────────────────────────────────────────────────────
# Section builders
# ────────────────────────────────────────────────────────────────────────────

def _scope_section(
    scope_dict: dict,
    lang: str,
    sty_h2: ParagraphStyle,
    sty_note: ParagraphStyle,
) -> list[Any]:
    """Build a scope metadata table flowable list."""
    story: list[Any] = []
    story.append(Paragraph(t("scope_metadata", lang), sty_h2))

    rows = [
        [t("scope_id_label", lang),        scope_dict.get("scope_id", "—")],
        [t("scope_status_label", lang),     scope_dict.get("status", "—")],
        [t("domain_label", lang),           scope_dict.get("domain_config_id", "—")],
        [t("scope_n_uids", lang),           str(scope_dict.get("n_uids", "—"))],
        [t("scope_created_at", lang),       scope_dict.get("created_at", "—")[:19]],
    ]
    if scope_dict.get("locked_at"):
        rows.append(["Locked at", scope_dict["locked_at"][:19]])

    snap = scope_dict.get("params_snapshot", {})
    if snap:
        snap_str = "  |  ".join(f"{k}={v}" for k, v in snap.items())
        rows.append([t("params_snapshot_label", lang), snap_str])

    tbl = Table(
        [[Paragraph(str(r[0]), sty_note), Paragraph(str(r[1]), sty_note)] for r in rows],
        colWidths=_col_widths(0.35, 0.65),
    )
    tbl.setStyle(_table_style(_NAVY))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))
    return story


# ────────────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────────────

def generate_report(
    lang: str,
    model_metrics: dict[str, Any],
    backtest_summary: dict[str, Any] | None = None,
    sensitivity_results: dict[str, Any] | None = None,
    optimization_snapshot: dict[str, Any] | None = None,
    scope_dict: dict[str, Any] | None = None,
    outcome_available: bool = True,
) -> bytes:
    """Generate a PDF report and return the raw bytes.

    Parameters
    ----------
    lang:
        ``"en"`` or ``"it"``.
    model_metrics:
        Output of ``CancellationPredictor.fit()``
        (keys: cv_metrics, mean_auc, mean_brier).
    backtest_summary:
        Serialised ``BacktestSummary.to_dict()`` or equivalent dict.
    sensitivity_results:
        Output of the ``/sensitivity`` endpoint.
    optimization_snapshot:
        Single-run optimisation summary for the executive snapshot.
    scope_dict:
        Scope metadata dict (scope_id, status, domain_config_id, …).
    outcome_available:
        If False, realized-revenue columns and sections are suppressed.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2.0 * cm,
        leftMargin=2.0 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.0 * cm,
        title=t("report_title", lang),
    )

    styles = getSampleStyleSheet()

    sty_title = ParagraphStyle("OpTitle",  parent=styles["Title"],   fontSize=20,
                                textColor=_NAVY, spaceAfter=6, alignment=TA_CENTER)
    sty_h1    = ParagraphStyle("OpH1",     parent=styles["Heading1"], fontSize=13,
                                textColor=_INDIGO, spaceBefore=18, spaceAfter=6)
    sty_h2    = ParagraphStyle("OpH2",     parent=styles["Heading2"], fontSize=11,
                                textColor=_INDIGO, spaceBefore=12, spaceAfter=4)
    sty_body  = ParagraphStyle("OpBody",   parent=styles["Normal"],   fontSize=9,
                                leading=13, alignment=TA_JUSTIFY)
    sty_note  = ParagraphStyle("OpNote",   parent=styles["Normal"],   fontSize=8,
                                textColor=colors.HexColor("#555555"), leading=11,
                                alignment=TA_LEFT)

    story: list[Any] = []

    # ── Title block ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(t("report_title", lang), sty_title))
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")
    story.append(Paragraph(
        f'<font color="#555555"><i>{t("generated_at", lang)}: {now_str}</i></font>',
        sty_body,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=_NAVY, spaceAfter=8))

    # ── Scope metadata (if provided) ─────────────────────────────────────────
    if scope_dict:
        story.extend(_scope_section(scope_dict, lang, sty_h2, sty_note))

    # ── Outcome availability notice ───────────────────────────────────────────
    if not outcome_available:
        story.append(Paragraph(
            f'<font color="#b71c1c"><b>{t("outcome_not_available_short", lang)}</b></font>',
            sty_body,
        ))
        story.append(Paragraph(t("outcome_not_available_long", lang), sty_note))
        story.append(Spacer(1, 0.3 * cm))

    # ── Executive summary ─────────────────────────────────────────────────────
    story.append(Paragraph(t("executive_summary", lang), sty_h1))
    story.append(Paragraph(t("exec_intro", lang), sty_body))
    story.append(Spacer(1, 0.3 * cm))

    mean_auc   = model_metrics.get("mean_auc",   0.0)
    mean_brier = model_metrics.get("mean_brier", 0.0)
    story.append(Paragraph(
        t("exec_model", lang).format(auc=mean_auc, brier=mean_brier),
        sty_body,
    ))

    if backtest_summary and outcome_available:
        n    = backtest_summary.get("n_splits", 0)
        ai   = backtest_summary.get("total_ai_revenue", 0.0)
        base = backtest_summary.get("total_baseline_revenue", 0.0)
        imp  = backtest_summary.get("total_improvement_pct", 0.0)
        story.append(Paragraph(
            t("exec_backtest", lang).format(n=n, ai=ai, base=base, imp=imp),
            sty_body,
        ))

    # Lambda status notice
    lam_status = (optimization_snapshot or {}).get("lambda_status", "inactive")
    lam_key    = "lambda_active" if lam_status == "active" else "lambda_inactive"
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f'<i>{t("lambda_status_label", lang)}: {t(lam_key, lang)}</i>',
        sty_note,
    ))
    story.append(Spacer(1, 0.5 * cm))

    # ── Layer 1: Model performance ────────────────────────────────────────────
    story.append(Paragraph(t("model_performance", lang), sty_h1))
    story.append(Paragraph(t("calibration_note", lang), sty_note))
    story.append(Spacer(1, 0.3 * cm))

    cv = model_metrics.get("cv_metrics", [])
    if cv:
        headers = [t("fold", lang), t("auc_score", lang), t("brier_score", lang)]
        rows = [[str(m["fold"]), f'{m["auc"]:.4f}', f'{m["brier"]:.4f}'] for m in cv]
        rows.append([
            f'<b>{t("mean", lang)}</b>',
            f"<b>{mean_auc:.4f}</b>",
            f"<b>{mean_brier:.4f}</b>",
        ])
        tbl = Table(
            [[Paragraph(h, sty_note) for h in headers]]
            + [[Paragraph(str(c), sty_note) for c in row] for row in rows],
            colWidths=_col_widths(0.15, 0.425, 0.425),
        )
        tbl.setStyle(_table_style())
        story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ── Layer 2: Optimization snapshot ───────────────────────────────────────
    if optimization_snapshot:
        story.append(Paragraph(t("optimization_results", lang), sty_h1))
        snap = optimization_snapshot
        kv = [
            [t("accepted_bookings", lang), str(snap.get("n_accepted", "—"))],
            [t("rejected_bookings", lang), str(snap.get("n_rejected", "—"))],
            [t("expected_revenue",  lang), f'${snap.get("total_expected_revenue", 0):,.2f}'],
            [t("expected_occupancy",lang), f'{snap.get("expected_occupancy", 0):.1f}'],
            [t("solver_status",     lang), snap.get("solver_status", "—")],
            [t("lambda_status_label",lang),snap.get("lambda_status", "inactive")],
        ]
        if not outcome_available:
            kv.append([t("outcome_unavailable", lang), "—"])
        tbl = Table(
            [[Paragraph(r[0], sty_note), Paragraph(r[1], sty_note)] for r in kv],
            colWidths=_col_widths(0.55, 0.45),
        )
        tbl.setStyle(_table_style(_NAVY))
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

        # Embed optimization charts
        charts = snap.get("charts", {})
        for chart_key in ("profit_composition", "capacity_utilization", "decision_distribution"):
            if chart_key in charts:
                try:
                    story.append(_b64_to_rl_image(charts[chart_key], width_cm=14.0))
                    story.append(Spacer(1, 0.3 * cm))
                except Exception:
                    pass

    # ── Layer 3: Backtesting ──────────────────────────────────────────────────
    if backtest_summary:
        story.append(PageBreak())
        story.append(Paragraph(t("backtesting_results", lang), sty_h1))

        if not outcome_available:
            story.append(Paragraph(t("backtest_outcome_required", lang), sty_note))
        else:
            folds = backtest_summary.get("folds", [])
            if folds:
                headers = [
                    t("fold", lang),
                    t("period", lang),
                    t("n_bookings", lang),
                    t("ai_accepted", lang),
                    t("ai_realized_rev", lang),
                    t("baseline_realized_rev", lang),
                    t("improvement_pct", lang),
                ]
                rows = []
                for f in folds:
                    period = f'{f.get("period_start", "–")} → {f.get("period_end", "–")}'
                    rows.append([
                        str(f["fold"]),
                        period,
                        f'{f["n_bookings"]:,}',
                        str(f.get("ai_accepted", "–")),
                        f'${f["ai_realized_revenue"]:,.0f}',
                        f'${f["baseline_realized_revenue"]:,.0f}',
                        _pct_color(f["improvement_pct"]),
                    ])

                total_ai   = backtest_summary.get("total_ai_revenue", 0.0)
                total_base = backtest_summary.get("total_baseline_revenue", 0.0)
                total_imp  = backtest_summary.get("total_improvement_pct", 0.0)
                rows.append([
                    f'<b>{t("total", lang)}</b>', "", "", "",
                    f"<b>${total_ai:,.0f}</b>",
                    f"<b>${total_base:,.0f}</b>",
                    _pct_color(total_imp),
                ])

                tbl = Table(
                    [[Paragraph(h, sty_note) for h in headers]]
                    + [[Paragraph(str(c), sty_note) for c in row] for row in rows],
                    colWidths=_col_widths(0.05, 0.22, 0.09, 0.10, 0.18, 0.18, 0.10),
                )
                tbl.setStyle(_table_style())
                story.append(tbl)

            # Backtest params footnote
            bparams = backtest_summary.get("params", {})
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(
                f'<i>{t("backtest_params", lang).format(
                    penalty=bparams.get("cancellation_penalty", "–"),
                    capacity=bparams.get("capacity", "–"),
                    lambda_risk=bparams.get("lambda_risk", "–"),
                    n_samples=bparams.get("n_samples_per_fold", "–"),
                )}</i>',
                sty_note,
            ))

            # Embed backtest charts
            bt_charts = backtest_summary.get("charts", {})
            for chart_key in ("cumulative_profit", "incremental_value", "per_period_delta"):
                if chart_key in bt_charts:
                    try:
                        story.append(Spacer(1, 0.3 * cm))
                        story.append(_b64_to_rl_image(bt_charts[chart_key], width_cm=14.0))
                    except Exception:
                        pass

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    if sensitivity_results:
        story.append(PageBreak())
        story.append(Paragraph(t("sensitivity_analysis", lang), sty_h1))

        # Penalty sensitivity
        pen_sens = sensitivity_results.get("penalty_sensitivity", [])
        if pen_sens:
            story.append(Paragraph(t("penalty_sensitivity", lang), sty_h2))
            headers = [
                t("penalty_multiplier", lang),
                t("penalty_value", lang),
                t("expected_revenue", lang),
                t("accepted_bookings", lang),
            ]
            rows = [[
                f'{r["penalty_multiplier"]:.0%}',
                f'€{r["penalty_value"]:,.0f}',
                f'${r["total_expected_revenue"]:,.0f}',
                str(r["n_accepted"]),
            ] for r in pen_sens]
            tbl = Table(
                [[Paragraph(h, sty_note) for h in headers]]
                + [[Paragraph(c, sty_note) for c in row] for row in rows],
                colWidths=_col_widths(0.22, 0.22, 0.33, 0.23),
            )
            tbl.setStyle(_table_style())
            story.append(tbl)
            if "penalty_sensitivity_chart" in sensitivity_results:
                try:
                    story.append(Spacer(1, 0.3 * cm))
                    story.append(_b64_to_rl_image(sensitivity_results["penalty_sensitivity_chart"]))
                except Exception:
                    pass
            story.append(Spacer(1, 0.4 * cm))

        # Capacity sensitivity
        cap_sens = sensitivity_results.get("capacity_sensitivity", [])
        if cap_sens:
            story.append(Paragraph(t("capacity_sensitivity", lang), sty_h2))
            headers = [
                t("capacity_multiplier", lang),
                t("capacity_value", lang),
                t("expected_revenue", lang),
                t("expected_occupancy", lang),
            ]
            rows = [[
                f'{r["capacity_multiplier"]:.0%}',
                f'{r["capacity_value"]:.0f}',
                f'${r["total_expected_revenue"]:,.0f}',
                f'{r["expected_occupancy"]:.1f}',
            ] for r in cap_sens]
            tbl = Table(
                [[Paragraph(h, sty_note) for h in headers]]
                + [[Paragraph(c, sty_note) for c in row] for row in rows],
                colWidths=_col_widths(0.22, 0.22, 0.33, 0.23),
            )
            tbl.setStyle(_table_style())
            story.append(tbl)
            if "capacity_sensitivity_chart" in sensitivity_results:
                try:
                    story.append(Spacer(1, 0.3 * cm))
                    story.append(_b64_to_rl_image(sensitivity_results["capacity_sensitivity_chart"]))
                except Exception:
                    pass
            story.append(Spacer(1, 0.4 * cm))

        # Lambda sensitivity
        lam_sens = sensitivity_results.get("lambda_sensitivity", [])
        if lam_sens:
            story.append(Paragraph(t("lambda_sensitivity", lang), sty_h2))
            headers = [
                t("lambda", lang),
                t("expected_revenue", lang),
                t("accepted_bookings", lang),
                t("expected_occupancy", lang),
            ]
            rows = [[
                f'{r["lambda"]:.2f}×',
                f'${r["total_expected_revenue"]:,.0f}',
                str(r["n_accepted"]),
                f'{r["expected_occupancy"]:.1f}',
            ] for r in lam_sens]
            tbl = Table(
                [[Paragraph(h, sty_note) for h in headers]]
                + [[Paragraph(c, sty_note) for c in row] for row in rows],
                colWidths=_col_widths(0.18, 0.33, 0.27, 0.22),
            )
            tbl.setStyle(_table_style())
            story.append(tbl)

        # Sensitivity params footnote
        sp = sensitivity_results.get("params", {})
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            f'<i>Base: {t("capacity", lang)}={sp.get("base_capacity")} | '
            f'{t("penalty", lang)}=€{sp.get("base_penalty")} | '
            f'λ={sp.get("base_lambda")} | n_sample={sp.get("n_sample")}</i>',
            sty_note,
        ))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(story)
    return buffer.getvalue()
