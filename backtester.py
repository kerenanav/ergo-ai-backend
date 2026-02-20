"""
backtester.py — Layer 3: Rolling walk-forward backtesting.

Methodology
───────────
For each TimeSeriesSplit fold:
  1. Train a fresh LightGBM + Platt model on the chronologically earlier data.
  2. Predict P(cancel) on the following (held-out) window.
  3. Run the MILP optimiser to obtain the AI accept/reject decisions.
  4. Apply the actual booking outcomes (is_canceled ground truth) to compute
     *realised* revenue for both the AI strategy and the accept-all baseline.

Comparison
──────────
Baseline strategy : accept every booking.
AI strategy       : accept only MILP-selected bookings.

Revenue accounting (per booking i, ground truth outcome o_i ∈ {0=show, 1=cancel}):

    if decision_i = 1:
        revenue_i = (1 − o_i) · gross_rev_i  −  o_i · penalty
    if decision_i = 0:
        revenue_i = 0   (revenue foregone, no penalty)

All seeds are fixed (RANDOM_SEED = 42).  No Excel output.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from constraint_compiler import OptimizationParams
from decision_engine import DecisionEngine, OptimizationResult
from predictive_model import CancellationPredictor

logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    n_bookings: int

    # Predictive metrics
    auc: float
    brier: float

    # AI strategy
    ai_accepted: int
    ai_rejected: int
    ai_expected_revenue: float
    """Expected revenue from the optimiser (using P(cancel) estimates)."""
    ai_realized_revenue: float
    """Actual revenue realised given ground-truth cancellation outcomes."""

    # Baseline: accept every booking
    baseline_realized_revenue: float

    # Relative improvement
    improvement_pct: float


@dataclass
class BacktestSummary:
    n_splits: int
    folds: list[FoldResult]
    mean_auc: float
    mean_brier: float
    total_ai_revenue: float
    total_baseline_revenue: float
    total_improvement_pct: float
    params: dict

    def to_dict(self) -> dict:
        """Serialise to a plain dict (nested dataclasses converted too)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Rolling walk-forward backtest: AI strategy vs. accept-all baseline."""

    def __init__(
        self,
        predictor: CancellationPredictor,
        engine: DecisionEngine,
    ) -> None:
        self.predictor = predictor
        self.engine    = engine

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        params: OptimizationParams,
        n_splits: int = 5,
    ) -> BacktestSummary:
        """Execute the rolling backtest.

        Parameters
        ----------
        df:
            Full preprocessed dataset (output of ``predictor.load_raw()``).
        params:
            Optimisation parameters (capacity, penalty, lambda).
        n_splits:
            Number of TimeSeriesSplit folds.

        Returns
        -------
        BacktestSummary with per-fold and aggregate metrics.
        """
        logger.info(
            "Starting backtest — n_splits=%d  capacity=%.0f  penalty=%.2f  λ=%.2f",
            n_splits, params.capacity, params.cancellation_penalty, params.lambda_risk,
        )

        splits        = self.predictor.fit_and_predict_splits(df, n_splits=n_splits)
        fold_results: list[FoldResult] = []

        for split in splits:
            fold_idx  = split["fold"]
            val_idx   = split["val_idx"]
            p_cancel  = split["p_cancel"]
            y_true    = split["y_true"]     # ground-truth: 1 = cancelled, 0 = stayed

            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            # ── Optimise ───────────────────────────────────────────────
            opt_result: OptimizationResult = self.engine.optimize(df_val, p_cancel, params)

            # ── Financial accounting ───────────────────────────────────
            gross_rev = self._gross_revenue(df_val)

            ai_realized  = self._realized_revenue(
                decisions=opt_result.decisions,
                y_true=y_true,
                gross_rev=gross_rev,
                penalty=params.cancellation_penalty,
            )
            # Baseline: accept all (decisions = all-ones)
            base_decisions = np.ones(len(y_true), dtype=int)
            base_realized  = self._realized_revenue(
                decisions=base_decisions,
                y_true=y_true,
                gross_rev=gross_rev,
                penalty=params.cancellation_penalty,
            )

            improvement_pct = (
                (ai_realized - base_realized) / max(abs(base_realized), 1.0)
            ) * 100.0

            fold_result = FoldResult(
                fold=fold_idx,
                n_bookings=len(val_idx),
                auc=split["auc"],
                brier=split["brier"],
                ai_accepted=opt_result.n_accepted,
                ai_rejected=opt_result.n_rejected,
                ai_expected_revenue=round(opt_result.total_expected_revenue, 2),
                ai_realized_revenue=round(ai_realized, 2),
                baseline_realized_revenue=round(base_realized, 2),
                improvement_pct=round(improvement_pct, 2),
            )
            fold_results.append(fold_result)

            logger.info(
                "Fold %d — AI realised $%.0f | Baseline $%.0f | Δ %+.1f%%",
                fold_idx, ai_realized, base_realized, improvement_pct,
            )

        # ── Aggregate ─────────────────────────────────────────────────
        mean_auc   = round(float(np.mean([f.auc   for f in fold_results])), 4)
        mean_brier = round(float(np.mean([f.brier for f in fold_results])), 4)
        total_ai   = sum(f.ai_realized_revenue   for f in fold_results)
        total_base = sum(f.baseline_realized_revenue for f in fold_results)
        total_imp  = ((total_ai - total_base) / max(abs(total_base), 1.0)) * 100.0

        logger.info(
            "Backtest complete — total AI $%.0f | baseline $%.0f | Δ %+.1f%%",
            total_ai, total_base, total_imp,
        )

        return BacktestSummary(
            n_splits=n_splits,
            folds=fold_results,
            mean_auc=mean_auc,
            mean_brier=mean_brier,
            total_ai_revenue=round(total_ai, 2),
            total_baseline_revenue=round(total_base, 2),
            total_improvement_pct=round(total_imp, 2),
            params={
                "capacity":             params.capacity,
                "cancellation_penalty": params.cancellation_penalty,
                "lambda_risk":          params.lambda_risk,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gross_revenue(df_val: pd.DataFrame) -> np.ndarray:
        """adr × total_nights for each booking in the validation fold."""
        adr = df_val["adr"].fillna(0).values.astype(np.float64)

        if "total_nights" in df_val.columns:
            nights = df_val["total_nights"].fillna(1).values.astype(np.float64)
        elif "stays_in_week_nights" in df_val.columns:
            nights = (
                df_val["stays_in_weekend_nights"].fillna(0)
                + df_val["stays_in_week_nights"].fillna(0)
            ).values.astype(np.float64)
        else:
            nights = np.ones(len(df_val), dtype=np.float64)

        return adr * np.maximum(nights, 1.0)

    @staticmethod
    def _realized_revenue(
        decisions: np.ndarray,
        y_true: np.ndarray,
        gross_rev: np.ndarray,
        penalty: float,
    ) -> float:
        """Compute realised revenue given ground-truth outcomes.

        For each accepted booking:
            • Guest shows up (y_true = 0) → +gross_rev_i
            • Guest cancels  (y_true = 1) → −penalty
        Rejected bookings contribute 0.
        """
        total = 0.0
        for i, dec in enumerate(decisions):
            if dec == 1:
                if y_true[i] == 0:       # stayed
                    total += gross_rev[i]
                else:                     # cancelled
                    total -= penalty
        return total
