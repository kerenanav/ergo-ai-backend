"""
i18n.py — Bilingual string catalogue (English / Italian) for Optide V2.

Usage:
    from i18n import t
    msg = t("scope_not_locked", "it")
"""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # ── Report structure ──────────────────────────────────────────────
        "report_title":          "Optide Decision Intelligence Report",
        "executive_summary":     "Executive Summary",
        "model_performance":     "Model Performance (Layer 1 — LightGBM + Platt Scaling)",
        "optimization_results":  "Optimization Results (Layer 2 — MILP / HiGHS)",
        "backtesting_results":   "Backtesting Results (Layer 3 — Rolling CV)",
        "sensitivity_analysis":  "Sensitivity Analysis",
        "generated_at":          "Generated at",
        "page":                  "Page",

        # ── Scope metadata ────────────────────────────────────────────────
        "scope_metadata":         "Scope Metadata",
        "scope_id_label":         "Scope ID",
        "scope_status_label":     "Status",
        "domain_label":           "Domain",
        "selection_rule_label":   "Selection Rule",
        "params_snapshot_label":  "Parameters",
        "scope_created_at":       "Created at",
        "scope_period":           "Period",
        "scope_n_uids":           "Records selected",

        # ── Scope lifecycle ───────────────────────────────────────────────
        "scope_not_locked":       "Scope must be locked. Run POST /optimize first.",
        "scope_mismatch":         "Scope params_snapshot does not match request params.",
        "scope_not_found":        "Scope not found.",
        "clone_warning":          "This is a new experiment. Results are NOT directly comparable with the original scope.",

        # ── ML metrics ───────────────────────────────────────────────────
        "fold":              "Fold",
        "auc_score":         "ROC-AUC",
        "brier_score":       "Brier Score",
        "mean":              "Mean",
        "cancellation_probability": "Cancellation Probability",
        "calibration_note": (
            "Model trained with TimeSeriesSplit (no look-ahead bias). "
            "Probabilities calibrated via Platt scaling."
        ),

        # ── Financial / optimization ──────────────────────────────────────
        "expected_revenue":    "Expected Revenue",
        "ai_revenue":          "AI Strategy Revenue (realized)",
        "baseline_revenue":    "Baseline Revenue (accept-all)",
        "improvement":         "Revenue Improvement",
        "accepted_bookings":   "Accepted Bookings",
        "rejected_bookings":   "Rejected Bookings",
        "expected_occupancy":  "Expected Occupancy (rooms)",
        "capacity":            "Capacity",
        "penalty":             "Cancellation Penalty",
        "lambda":              "Overbooking Lambda",
        "solver_status":       "Solver Status",

        # ── Lambda / risk ─────────────────────────────────────────────────
        "lambda_inactive":     "Risk aversion inactive in V1 (risk_enabled=false).",
        "lambda_active":       "Risk aversion active.",
        "lambda_status_label": "Lambda Status",

        # ── Outcome availability ──────────────────────────────────────────
        "outcome_available_yes":    "Outcome available (historical data)",
        "outcome_available_no":     "Outcome not available (real-time data)",
        "outcome_unavailable":      "N/A (outcome_available=False)",
        "outcome_not_available_short": "Outcome not available (real-time).",
        "outcome_not_available_long": (
            "Realized profit cannot be computed because real outcomes are not yet "
            "available for these records. Showing expected value estimated by the ML model only."
        ),
        "backtest_outcome_required": (
            "Backtesting requires real historical outcomes. "
            "Cannot run on data without ml_target_column."
        ),

        # ── Backtesting ───────────────────────────────────────────────────
        "n_bookings":               "# Bookings",
        "period":                   "Period",
        "ai_accepted":              "AI Accepted",
        "ai_expected_rev":          "AI Expected Rev.",
        "ai_realized_rev":          "AI Realized Rev.",
        "baseline_realized_rev":    "Baseline Realized Rev.",
        "improvement_pct":          "Improvement %",
        "total":                    "Total",
        "risk_aversion":            "Risk Aversion",
        "bookings_per_period":      "Bookings per Period",
        "historical_policy_used":   "Historical Policy",
        "backtest_params": (
            "Parameters used: Cancellation Penalty=€{penalty}  |  "
            "Capacity={capacity}  |  Risk Aversion={lambda_risk}  |  "
            "Bookings per Period={n_samples}"
        ),
        "confidence_interval":      "95% Confidence Interval",

        # ── Sensitivity ───────────────────────────────────────────────────
        "penalty_sensitivity":   "Sensitivity to Cancellation Penalty (50% – 200%)",
        "capacity_sensitivity":  "Sensitivity to Capacity (±20%)",
        "lambda_sensitivity":    "Sensitivity to Overbooking Lambda (0× – 3×)",
        "penalty_multiplier":    "Penalty Multiplier",
        "capacity_multiplier":   "Capacity Multiplier",
        "penalty_value":         "Penalty (€)",
        "capacity_value":        "Capacity (rooms)",

        # ── Charts ────────────────────────────────────────────────────────
        "chart_profit_composition":    "Profit Composition",
        "chart_capacity_utilization":  "Capacity Utilization",
        "chart_decision_distribution": "Decision Distribution",
        "chart_expected_vs_realized":  "Expected vs Realized",
        "chart_cumulative_profit":     "Cumulative Profit",
        "chart_incremental_value":     "Incremental Value",
        "chart_per_period_delta":      "Per-Period Delta",
        "chart_not_available":         "Chart not available.",

        # ── Executive summary ─────────────────────────────────────────────
        "exec_intro": (
            "Optide is a Decision Intelligence Engine for tabular demand data. "
            "It combines a calibrated machine-learning model (LightGBM + Platt scaling) "
            "with a Mixed-Integer Linear Program (HiGHS solver) to maximise expected "
            "revenue under capacity constraints, and validates every decision policy "
            "through rolling backtesting against historical data."
        ),
        "exec_model": (
            "The predictive model achieves a mean cross-validated ROC-AUC of {auc:.4f} "
            "and a Brier score of {brier:.4f}."
        ),
        "exec_backtest": (
            "Over {n} backtest folds the AI strategy realised ${ai:,.0f} in total revenue, "
            "versus ${base:,.0f} for the accept-all baseline — "
            "a {imp:+.1f}% improvement."
        ),
        "conclusion_header": "Conclusion",
    },

    "it": {
        # ── Struttura report ─────────────────────────────────────────────
        "report_title":          "Optide — Report di Decision Intelligence",
        "executive_summary":     "Sommario Esecutivo",
        "model_performance":     "Performance del Modello (Layer 1 — LightGBM + Platt Scaling)",
        "optimization_results":  "Risultati dell'Ottimizzazione (Layer 2 — MILP / HiGHS)",
        "backtesting_results":   "Risultati del Backtesting (Layer 3 — Rolling CV)",
        "sensitivity_analysis":  "Analisi di Sensibilità",
        "generated_at":          "Generato il",
        "page":                  "Pagina",

        # ── Metadati scope ────────────────────────────────────────────────
        "scope_metadata":         "Metadati Scope",
        "scope_id_label":         "ID Scope",
        "scope_status_label":     "Stato",
        "domain_label":           "Dominio",
        "selection_rule_label":   "Regola di Selezione",
        "params_snapshot_label":  "Parametri",
        "scope_created_at":       "Creato il",
        "scope_period":           "Periodo",
        "scope_n_uids":           "Record selezionati",

        # ── Ciclo vita scope ──────────────────────────────────────────────
        "scope_not_locked":     "Lo scope deve essere bloccato. Eseguire prima POST /optimize.",
        "scope_mismatch":       "params_snapshot dello scope non corrisponde ai parametri della richiesta.",
        "scope_not_found":      "Scope non trovato.",
        "clone_warning":        "Questo è un nuovo esperimento. I risultati NON sono direttamente comparabili con lo scope originale.",

        # ── Metriche ML ──────────────────────────────────────────────────
        "fold":              "Piega",
        "auc_score":         "ROC-AUC",
        "brier_score":       "Punteggio Brier",
        "mean":              "Media",
        "cancellation_probability": "Probabilità di Cancellazione",
        "calibration_note": (
            "Modello addestrato con TimeSeriesSplit (nessun look-ahead bias). "
            "Le probabilità sono calibrate tramite Platt scaling."
        ),

        # ── Finanziario / ottimizzazione ─────────────────────────────────
        "expected_revenue":    "Ricavo Atteso",
        "ai_revenue":          "Ricavo Strategia AI (realizzato)",
        "baseline_revenue":    "Ricavo Baseline (accetta tutto)",
        "improvement":         "Miglioramento del Ricavo",
        "accepted_bookings":   "Prenotazioni Accettate",
        "rejected_bookings":   "Prenotazioni Rifiutate",
        "expected_occupancy":  "Occupazione Attesa (camere)",
        "capacity":            "Capacità",
        "penalty":             "Penale Cancellazione",
        "lambda":              "Lambda Overbooking",
        "solver_status":       "Stato Solver",

        # ── Lambda / rischio ─────────────────────────────────────────────
        "lambda_inactive":     "Avversione al rischio non attiva in V1 (risk_enabled=false).",
        "lambda_active":       "Avversione al rischio attiva.",
        "lambda_status_label": "Stato Lambda",

        # ── Disponibilità outcome ─────────────────────────────────────────
        "outcome_available_yes":    "Outcome disponibile (dati storici)",
        "outcome_available_no":     "Outcome non disponibile (dati real-time)",
        "outcome_unavailable":      "N/D (outcome_available=False)",
        "outcome_not_available_short": "Outcome non disponibile (real-time).",
        "outcome_not_available_long": (
            "Il profitto realizzato non è calcolabile perché gli outcome reali non sono ancora "
            "disponibili per questi record. Viene mostrato solo il valore atteso "
            "stimato dal modello ML."
        ),
        "backtest_outcome_required": (
            "Il backtesting richiede outcome storici reali. "
            "Impossibile eseguire su dati senza ml_target_column."
        ),

        # ── Backtesting ───────────────────────────────────────────────────
        "n_bookings":               "# Prenotazioni",
        "period":                   "Periodo",
        "ai_accepted":              "Accettate AI",
        "ai_expected_rev":          "Ricavo Atteso AI",
        "ai_realized_rev":          "Ricavo Realizzato AI",
        "baseline_realized_rev":    "Ricavo Realizzato Baseline",
        "improvement_pct":          "Miglioramento %",
        "total":                    "Totale",
        "risk_aversion":            "Avversione al Rischio",
        "bookings_per_period":      "Prenotazioni per Periodo",
        "historical_policy_used":   "Politica Storica",
        "backtest_params": (
            "Parametri utilizzati: Penalità Cancellazione=€{penalty}  |  "
            "Capacità={capacity}  |  Avversione al Rischio={lambda_risk}  |  "
            "Prenotazioni per Periodo={n_samples}"
        ),
        "confidence_interval":      "Intervallo di Confidenza 95%",

        # ── Sensibilità ──────────────────────────────────────────────────
        "penalty_sensitivity":   "Sensibilità alla Penale di Cancellazione (50% – 200%)",
        "capacity_sensitivity":  "Sensibilità alla Capacità (±20%)",
        "lambda_sensitivity":    "Sensibilità al Lambda di Overbooking (0× – 3×)",
        "penalty_multiplier":    "Moltiplicatore Penale",
        "capacity_multiplier":   "Moltiplicatore Capacità",
        "penalty_value":         "Penale (€)",
        "capacity_value":        "Capacità (camere)",

        # ── Grafici ───────────────────────────────────────────────────────
        "chart_profit_composition":    "Composizione del Profitto",
        "chart_capacity_utilization":  "Utilizzo della Capacità",
        "chart_decision_distribution": "Distribuzione Decisioni",
        "chart_expected_vs_realized":  "Atteso vs Realizzato",
        "chart_cumulative_profit":     "Profitto Cumulativo",
        "chart_incremental_value":     "Valore Incrementale",
        "chart_per_period_delta":      "Delta per Periodo",
        "chart_not_available":         "Grafico non disponibile.",

        # ── Sommario esecutivo ────────────────────────────────────────────
        "exec_intro": (
            "Optide è un Decision Intelligence Engine per dati di domanda tabulari. "
            "Combina un modello di machine learning calibrato (LightGBM + Platt scaling) "
            "con un programma misto-intero lineare (solver HiGHS) per massimizzare il "
            "ricavo atteso sotto vincoli di capacità, validando ogni politica decisionale "
            "tramite backtesting rolling sui dati storici."
        ),
        "exec_model": (
            "Il modello predittivo ottiene un ROC-AUC medio cross-validato di {auc:.4f} "
            "e un Brier score di {brier:.4f}."
        ),
        "exec_backtest": (
            "Nei {n} fold di backtesting la strategia AI ha realizzato ${ai:,.0f} di ricavo totale, "
            "contro ${base:,.0f} della baseline accetta-tutto — "
            "un miglioramento del {imp:+.1f}%."
        ),
        "conclusion_header": "Conclusione",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Return the translated string for *key* in *lang* (falls back to English)."""
    catalogue = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return catalogue.get(key, TRANSLATIONS["en"].get(key, key))
