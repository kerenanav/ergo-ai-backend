"""
backtester.py — Layer 3: Temporal backtest AI vs accept-all baseline.

Methodology
───────────
The dataset is sorted chronologically by arrival_date and divided into
n_splits equal time windows.  For each window:

  1. Sample n_samples_per_fold bookings from that window.
  2. Predict P(cancel) using the pre-trained model (no retraining).
  3. Run the MILP optimiser → AI accept/reject decisions.
  4. Apply ground-truth is_canceled outcomes to both strategies:

     AI strategy   : only accepted bookings generate revenue / loss.
     Baseline       : every booking in the sample is accepted.

Revenue accounting (per booking i, ground truth o_i ∈ {0=show, 1=cancel}):
    if decision_i = 1 (accepted):
        revenue_i = (1 − o_i) · adr_i · nights_i  −  o_i · penalty
    if decision_i = 0 (rejected):
        revenue_i = 0

Comparison is causally valid: same sample, same outcomes, different decisions.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

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
    period_start: str
    period_end: str

    # AI strategy
    ai_accepted: int
    ai_rejected: int
    ai_expected_revenue: float
    ai_realized_revenue: float

    # Baseline: accept every booking
    baseline_realized_revenue: float

    # Delta
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
    """Temporal backtest: AI strategy vs. accept-all baseline."""

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
        n_samples_per_fold: int = 100,
        model_metrics: dict | None = None,
    ) -> BacktestSummary:
        """Execute the temporal backtest.

        Parameters
        ----------
        df:
            Full preprocessed dataset (output of predictor.load_raw()).
            Must contain is_canceled and arrival_date columns.
        params:
            Optimisation parameters (capacity, penalty, lambda_risk).
        n_splits:
            Number of equal time windows to evaluate.
        n_samples_per_fold:
            Bookings sampled from each time window.
        model_metrics:
            Pre-trained model metrics dict with mean_auc / mean_brier.
        """
        if model_metrics is None:
            model_metrics = {}

        print(
            f"BACKTEST PARAMETRI: n_splits={n_splits}  n_samples_per_fold={n_samples_per_fold}"
            f"  capacity={params.capacity}  cancellation_penalty={params.cancellation_penalty}"
            f"  lambda_risk={params.lambda_risk}",
            flush=True,
        )
        logger.info(
            "Starting backtest — n_splits=%d  n_samples=%d  "
            "capacity=%.0f  penalty=%.2f  lambda=%.2f",
            n_splits, n_samples_per_fold,
            params.capacity, params.cancellation_penalty, params.lambda_risk,
        )

        # Sort chronologically
        if "arrival_date" in df.columns:
            df_sorted = df.sort_values("arrival_date").reset_index(drop=True)
        else:
            df_sorted = df.reset_index(drop=True)

        fold_size     = len(df_sorted) // n_splits
        fold_results: list[FoldResult] = []

        for fold in range(n_splits):
            start_idx  = fold * fold_size
            end_idx    = start_idx + fold_size if fold < n_splits - 1 else len(df_sorted)
            df_period  = df_sorted.iloc[start_idx:end_idx].copy().reset_index(drop=True)

            # Sample bookings from this time window
            n_sample = min(n_samples_per_fold, len(df_period))
            rng      = np.random.default_rng(RANDOM_SEED + fold)
            idx      = rng.choice(len(df_period), size=n_sample, replace=False)
            df_sample = df_period.iloc[idx].copy().reset_index(drop=True)

            # Date labels for the frontend chart
            if "arrival_date" in df_sample.columns:
                period_start = str(df_sample["arrival_date"].min().date())
                period_end   = str(df_sample["arrival_date"].max().date())
            else:
                period_start = str(fold + 1)
                period_end   = str(fold + 1)

            print(
                f"  Fold {fold+1}: period={period_start}→{period_end}"
                f"  n={n_sample}  capacity={params.capacity}"
                f"  penalty={params.cancellation_penalty}  lambda={params.lambda_risk}",
                flush=True,
            )

            # Predict with pre-trained model (no retraining)
            p_cancel = self.predictor.predict_proba(df_sample)

            # AI decision via MILP
            opt_result: OptimizationResult = self.engine.optimize(df_sample, p_cancel, params)

            # Ground-truth outcomes
            y_true = (
                df_sample["is_canceled"].values.astype(int)
                if "is_canceled" in df_sample.columns
                else np.zeros(n_sample, dtype=int)
            )
            gross_rev = self._gross_revenue(df_sample)

            # Realized revenues
            ai_realized = self._realized_revenue(
                decisions=opt_result.decisions,
                y_true=y_true,
                gross_rev=gross_rev,
                penalty=params.cancellation_penalty,
            )
            base_realized = self._realized_revenue(
                decisions=np.ones(n_sample, dtype=int),
                y_true=y_true,
                gross_rev=gross_rev,
                penalty=params.cancellation_penalty,
            )

            improvement_pct = (
                (ai_realized - base_realized) / max(abs(base_realized), 1.0)
            ) * 100.0

            fold_results.append(FoldResult(
                fold=fold + 1,
                n_bookings=n_sample,
                period_start=period_start,
                period_end=period_end,
                ai_accepted=opt_result.n_accepted,
                ai_rejected=opt_result.n_rejected,
                ai_expected_revenue=round(opt_result.total_expected_revenue, 2),
                ai_realized_revenue=round(ai_realized, 2),
                baseline_realized_revenue=round(base_realized, 2),
                improvement_pct=round(improvement_pct, 2),
            ))

            logger.info(
                "Fold %d [%s → %s] — AI $%.0f | Baseline $%.0f | delta %+.1f%%",
                fold + 1, period_start, period_end,
                ai_realized, base_realized, improvement_pct,
            )

        # Aggregate
        total_ai   = sum(f.ai_realized_revenue       for f in fold_results)
        total_base = sum(f.baseline_realized_revenue  for f in fold_results)
        total_imp  = ((total_ai - total_base) / max(abs(total_base), 1.0)) * 100.0

        logger.info(
            "Backtest complete — total AI $%.0f | baseline $%.0f | delta %+.1f%%",
            total_ai, total_base, total_imp,
        )

        return BacktestSummary(
            n_splits=n_splits,
            folds=fold_results,
            mean_auc=model_metrics.get("mean_auc", 0.0),
            mean_brier=model_metrics.get("mean_brier", 0.0),
            total_ai_revenue=round(total_ai, 2),
            total_baseline_revenue=round(total_base, 2),
            total_improvement_pct=round(total_imp, 2),
            params={
                "capacity":             params.capacity,
                "cancellation_penalty": params.cancellation_penalty,
                "lambda_risk":          params.lambda_risk,
                "n_samples_per_fold":   n_samples_per_fold,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gross_revenue(df_val: pd.DataFrame) -> np.ndarray:
        """adr × nights for each booking."""
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
        """Realised revenue given ground-truth cancellation outcomes.

        Accepted booking (decision = 1):
            guest stayed  (y_true = 0) → +gross_rev_i
            guest cancelled (y_true = 1) → −penalty
        Rejected booking (decision = 0) → 0
        """
        total = 0.0
        for i, dec in enumerate(decisions):
            if dec == 1:
                if y_true[i] == 0:
                    total += gross_rev[i]
                else:
                    total -= penalty
        return total
