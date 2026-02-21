"""
backtester.py — Layer 3: Temporal backtest AI vs accept-all baseline.

Methodology
───────────
The dataset (filtered to scope.selected_uids if a scope is provided) is sorted
chronologically and divided into n_splits equal time windows. For each window:

  1. Sample n_samples_per_fold bookings from that window.
  2. Predict P(cancel) using the pre-trained model (no retraining).
  3. Run the MILP optimiser → AI accept/reject decisions.
  4. Apply ground-truth ml_target outcomes to both strategies.

Revenue accounting (per booking i, ground truth o_i in {0=show, 1=cancel}):
    if decision_i = 1 (accepted):
        revenue_i = (1 - o_i) * gross_rev_i  -  o_i * penalty
    if decision_i = 0 (rejected):
        revenue_i = 0

Baseline: accept every booking in the sample.

Guard rails:
  - 422 if outcome_available=False (no ground-truth outcomes to compare against)
  - 409 if scope.status != "locked" (enforced by ScopeManager.validate_for_backtest)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from chart_generator import cumulative_profit_chart, incremental_value_chart, per_period_delta_chart
from constraint_compiler import OptimizationParams
from decision_engine import DecisionEngine, OptimizationResult
from domain_config import DomainConfig, evaluate_revenue
from predictive_model import CancellationPredictor

logger     = logging.getLogger(__name__)
RANDOM_SEED: int = 42


# ────────────────────────────────────────────────────────────────────────────
# Result containers
# ────────────────────────────────────────────────────────────────────────────

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

    # Baseline
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
    outcome_available: bool = True
    lambda_status: str = "inactive"
    charts: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.charts is None:
            self.charts = {}

    def to_dict(self) -> dict:
        return asdict(self)


# ────────────────────────────────────────────────────────────────────────────
# Backtester
# ────────────────────────────────────────────────────────────────────────────

class Backtester:
    """Temporal backtest: AI strategy vs. accept-all baseline."""

    def __init__(
        self,
        predictor: CancellationPredictor,
        engine: DecisionEngine,
    ) -> None:
        self.predictor = predictor
        self.engine    = engine

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        params: OptimizationParams,
        cfg: DomainConfig | None = None,
        scope=None,
        outcome_available: bool = True,
        n_splits: int = 5,
        n_samples_per_fold: int = 100,
        model_metrics: dict | None = None,
    ) -> BacktestSummary:
        """Execute the temporal backtest.

        Parameters
        ----------
        df:
            Full preprocessed dataset (output of predictor.load_raw()).
            Must contain the ml_target_column and _uid column.
        params:
            Optimisation parameters (capacity, penalty, lambda_risk).
        cfg:
            DomainConfig for this domain (optional, enables formula-based revenue).
        scope:
            If provided, only rows whose _uid is in scope.selected_uids are used.
        outcome_available:
            If False, raises ValueError immediately (caller maps to HTTP 422).
        n_splits:
            Number of equal chronological time windows.
        n_samples_per_fold:
            Bookings sampled from each time window.
        model_metrics:
            Pre-trained model metrics dict with mean_auc / mean_brier.
        """
        if not outcome_available:
            raise ValueError("outcome_not_available")

        if model_metrics is None:
            model_metrics = {}

        # Filter to scope UIDs if provided
        if scope is not None and "_uid" in df.columns:
            df = df[df["_uid"].isin(scope.selected_uids)].copy()
            logger.info("Backtest filtered to %d rows (scope %s)", len(df), scope.scope_id)

        target_col = cfg.ml_target_column if cfg else "is_canceled"

        logger.info(
            "Starting backtest — n_splits=%d  n_samples=%d  "
            "capacity=%.0f  penalty=%.2f  lambda=%.2f",
            n_splits, n_samples_per_fold,
            params.capacity, params.cancellation_penalty, params.lambda_risk,
        )

        # Sort chronologically
        sort_col = "_time" if "_time" in df.columns else (
            "arrival_date" if "arrival_date" in df.columns else None
        )
        if sort_col:
            df_sorted = df.sort_values(sort_col).reset_index(drop=True)
        else:
            df_sorted = df.reset_index(drop=True)

        fold_size    = max(len(df_sorted) // n_splits, 1)
        fold_results: list[FoldResult] = []

        for fold in range(n_splits):
            start_idx = fold * fold_size
            end_idx   = start_idx + fold_size if fold < n_splits - 1 else len(df_sorted)
            df_period = df_sorted.iloc[start_idx:end_idx].copy().reset_index(drop=True)

            n_sample  = min(n_samples_per_fold, len(df_period))
            if n_sample == 0:
                continue

            rng      = np.random.default_rng(RANDOM_SEED + fold)
            idx      = rng.choice(len(df_period), size=n_sample, replace=False)
            df_sample = df_period.iloc[idx].copy().reset_index(drop=True)

            # Period labels
            period_start, period_end = self._period_labels(df_sample, sort_col, fold)

            # Predict with pre-trained model (no retraining)
            p_cancel = self.predictor.predict_proba(df_sample)

            # AI decision via MILP
            opt_result: OptimizationResult = self.engine.optimize(
                df_sample, p_cancel, params, cfg=cfg
            )

            # Ground-truth outcomes
            y_true = (
                df_sample[target_col].values.astype(int)
                if target_col in df_sample.columns
                else np.zeros(n_sample, dtype=int)
            )

            # Revenue
            gross_rev = self._gross_revenue(df_sample, cfg)

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

        # Generate charts
        charts = self._build_charts(fold_results)

        lambda_status = "inactive" if not (cfg and cfg.risk_enabled) else "active"

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
            outcome_available=outcome_available,
            lambda_status=lambda_status,
            charts=charts,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _period_labels(
        df_sample: pd.DataFrame,
        sort_col: str | None,
        fold: int,
    ) -> tuple[str, str]:
        if sort_col and sort_col in df_sample.columns:
            try:
                col = pd.to_datetime(df_sample[sort_col], errors="coerce")
                return str(col.min().date()), str(col.max().date())
            except Exception:
                pass
        return str(fold + 1), str(fold + 1)

    @staticmethod
    def _gross_revenue(
        df_sample: pd.DataFrame,
        cfg: DomainConfig | None,
    ) -> np.ndarray:
        if cfg and cfg.revenue_formula:
            try:
                return evaluate_revenue(df_sample, cfg.revenue_formula)
            except Exception as exc:
                logger.debug("evaluate_revenue failed: %s — falling back to adr*nights", exc)

        # Fallback: adr * nights
        adr = df_sample["adr"].fillna(0).values.astype(np.float64) if "adr" in df_sample.columns else np.zeros(len(df_sample))
        if "total_nights" in df_sample.columns:
            nights = df_sample["total_nights"].fillna(1).values.astype(np.float64)
        elif "stays_in_week_nights" in df_sample.columns:
            nights = (
                df_sample["stays_in_weekend_nights"].fillna(0)
                + df_sample["stays_in_week_nights"].fillna(0)
            ).values.astype(np.float64)
        else:
            nights = np.ones(len(df_sample), dtype=np.float64)
        return adr * np.maximum(nights, 1.0)

    @staticmethod
    def _realized_revenue(
        decisions: np.ndarray,
        y_true: np.ndarray,
        gross_rev: np.ndarray,
        penalty: float,
    ) -> float:
        total = 0.0
        for i, dec in enumerate(decisions):
            if dec == 1:
                total += float(gross_rev[i]) if y_true[i] == 0 else -penalty
        return total

    @staticmethod
    def _build_charts(fold_results: list[FoldResult]) -> dict[str, str]:
        if not fold_results:
            return {}
        try:
            fd = [asdict(f) for f in fold_results]
            ai_profits   = [f.ai_realized_revenue       for f in fold_results]
            hist_profits = [f.baseline_realized_revenue  for f in fold_results]
            periods      = [f.period_start               for f in fold_results]
            return {
                "cumulative_profit":   cumulative_profit_chart(ai_profits, hist_profits, periods),
                "incremental_value":   incremental_value_chart(
                    [a - b for a, b in zip(ai_profits, hist_profits)], periods
                ),
                "per_period_delta":    per_period_delta_chart(fd),
            }
        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc)
            return {}
