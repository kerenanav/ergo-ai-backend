"""
i18n.py — Bilingual string catalogue (English / Italian) for Ergo.ai reports.

Usage:
    from i18n import t
    title = t("report_title", "it")  # → "Ergo.ai Report di Decision Intelligence"
"""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # ── Report structure ──────────────────────────────────────────────
        "report_title": "Ergo.ai Decision Intelligence Report",
        "executive_summary": "Executive Summary",
        "model_performance": "Model Performance (Layer 1 — LightGBM + Platt Scaling)",
        "optimization_results": "Optimization Results (Layer 2 — MILP / HiGHS)",
        "backtesting_results": "Backtesting Results (Layer 3 — Rolling CV)",
        "sensitivity_analysis": "Sensitivity Analysis",
        "generated_at": "Generated at",
        "page": "Page",
        # ── ML metrics ───────────────────────────────────────────────────
        "fold": "Fold",
        "auc_score": "ROC-AUC",
        "brier_score": "Brier Score",
        "mean": "Mean",
        "cancellation_probability": "Cancellation Probability",
        "calibration_note": (
            "Model trained with TimeSeriesSplit (no look-ahead bias). "
            "Probabilities calibrated via Platt scaling."
        ),
        # ── Financial / optimization ──────────────────────────────────────
        "expected_revenue": "Expected Revenue",
        "ai_revenue": "AI Strategy Revenue (realized)",
        "baseline_revenue": "Baseline Revenue (accept-all)",
        "improvement": "Revenue Improvement",
        "accepted_bookings": "Accepted Bookings",
        "rejected_bookings": "Rejected Bookings",
        "expected_occupancy": "Expected Occupancy (rooms)",
        "capacity": "Capacity",
        "penalty": "Cancellation Penalty",
        "lambda": "Overbooking Lambda",
        # ── Backtesting ───────────────────────────────────────────────────
        "n_bookings": "# Bookings",
        "ai_expected_rev": "AI Expected Rev.",
        "ai_realized_rev": "AI Realized Rev.",
        "baseline_realized_rev": "Baseline Realized Rev.",
        "improvement_pct": "Improvement %",
        "total": "Total",
        # ── Sensitivity ───────────────────────────────────────────────────
        "penalty_sensitivity": "Sensitivity to Cancellation Penalty (50 % – 200 %)",
        "capacity_sensitivity": "Sensitivity to Capacity (±20 %)",
        "lambda_sensitivity": "Sensitivity to Overbooking Lambda (0× – 3×)",
        "penalty_multiplier": "Penalty Multiplier",
        "capacity_multiplier": "Capacity Multiplier",
        "penalty_value": "Penalty (€)",
        "capacity_value": "Capacity (rooms)",
        # ── Executive summary ─────────────────────────────────────────────
        "exec_intro": (
            "Ergo.ai is a Decision Intelligence Engine for hotel booking demand. "
            "It combines a calibrated machine-learning model (LightGBM + Platt scaling) "
            "with a Mixed-Integer Linear Program (HiGHS solver) to maximise expected "
            "revenue under capacity constraints, and validates every decision policy "
            "through rolling backtesting against historical data."
        ),
        "exec_model": "The predictive model achieves a mean cross-validated ROC-AUC of {auc:.4f} "
                      "and a Brier score of {brier:.4f}.",
        "exec_backtest": (
            "Over {n} backtest folds the AI strategy realised ${ai:,.0f} in total revenue, "
            "versus ${base:,.0f} for the accept-all baseline — "
            "a {imp:+.1f} % improvement."
        ),
    },

    "it": {
        # ── Struttura report ─────────────────────────────────────────────
        "report_title": "Ergo.ai — Report di Decision Intelligence",
        "executive_summary": "Sommario Esecutivo",
        "model_performance": "Performance del Modello (Layer 1 — LightGBM + Platt Scaling)",
        "optimization_results": "Risultati dell'Ottimizzazione (Layer 2 — MILP / HiGHS)",
        "backtesting_results": "Risultati del Backtesting (Layer 3 — Rolling CV)",
        "sensitivity_analysis": "Analisi di Sensibilità",
        "generated_at": "Generato il",
        "page": "Pagina",
        # ── Metriche ML ──────────────────────────────────────────────────
        "fold": "Piega",
        "auc_score": "ROC-AUC",
        "brier_score": "Punteggio Brier",
        "mean": "Media",
        "cancellation_probability": "Probabilità di Cancellazione",
        "calibration_note": (
            "Modello addestrato con TimeSeriesSplit (nessun look-ahead bias). "
            "Le probabilità sono calibrate tramite Platt scaling."
        ),
        # ── Finanziario / ottimizzazione ─────────────────────────────────
        "expected_revenue": "Ricavo Atteso",
        "ai_revenue": "Ricavo Strategia AI (realizzato)",
        "baseline_revenue": "Ricavo Baseline (accetta tutto)",
        "improvement": "Miglioramento del Ricavo",
        "accepted_bookings": "Prenotazioni Accettate",
        "rejected_bookings": "Prenotazioni Rifiutate",
        "expected_occupancy": "Occupazione Attesa (camere)",
        "capacity": "Capacità",
        "penalty": "Penale Cancellazione",
        "lambda": "Lambda Overbooking",
        # ── Backtesting ───────────────────────────────────────────────────
        "n_bookings": "# Prenotazioni",
        "ai_expected_rev": "Ricavo Atteso AI",
        "ai_realized_rev": "Ricavo Realizzato AI",
        "baseline_realized_rev": "Ricavo Realizzato Baseline",
        "improvement_pct": "Miglioramento %",
        "total": "Totale",
        # ── Sensibilità ──────────────────────────────────────────────────
        "penalty_sensitivity": "Sensibilità alla Penale di Cancellazione (50 % – 200 %)",
        "capacity_sensitivity": "Sensibilità alla Capacità (±20 %)",
        "lambda_sensitivity": "Sensibilità al Lambda di Overbooking (0× – 3×)",
        "penalty_multiplier": "Moltiplicatore Penale",
        "capacity_multiplier": "Moltiplicatore Capacità",
        "penalty_value": "Penale (€)",
        "capacity_value": "Capacità (camere)",
        # ── Sommario esecutivo ────────────────────────────────────────────
        "exec_intro": (
            "Ergo.ai è un Decision Intelligence Engine per la gestione della domanda "
            "di prenotazioni alberghiere. Combina un modello di machine learning calibrato "
            "(LightGBM + Platt scaling) con un programma misto-intero lineare (solver HiGHS) "
            "per massimizzare il ricavo atteso sotto vincoli di capacità, validando ogni "
            "politica decisionale tramite backtesting rolling sui dati storici."
        ),
        "exec_model": "Il modello predittivo ottiene un ROC-AUC medio cross-validato di {auc:.4f} "
                      "e un Brier score di {brier:.4f}.",
        "exec_backtest": (
            "Nei {n} fold di backtesting la strategia AI ha realizzato ${ai:,.0f} di ricavo totale, "
            "contro ${base:,.0f} della baseline accetta-tutto — "
            "un miglioramento del {imp:+.1f} %."
        ),
    },
}


def t(key: str, lang: str = "en") -> str:
    """Return the translated string for *key* in *lang* (falls back to English)."""
    catalogue = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return catalogue.get(key, TRANSLATIONS["en"].get(key, key))
