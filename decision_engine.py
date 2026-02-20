"""
decision_engine.py — Layer 2: MILP-based booking optimiser (HiGHS via scipy).

Algorithm
─────────
1. Build financial quantities (BookingFinancials) from booking DataFrame +
   P(cancel) predictions from Layer 1.
2. Construct the objective vector c (negate EV for minimisation).
3. Compile capacity constraints (C1 expected-occupancy, C2 hard ceiling).
4. Solve the 0-1 MILP with scipy.optimize.milp, which internally calls HiGHS.
5. If HiGHS fails or times-out, fall back to a deterministic greedy heuristic
   (sort by descending EV, accept until capacity is filled).

All random seeds are fixed; results are fully reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, milp

from constraint_compiler import OptimizationParams, compile_constraints
from objective_builder import BookingFinancials, build_objective

logger = logging.getLogger(__name__)

# HiGHS solver time limit (seconds)
_SOLVER_TIME_LIMIT: float = 120.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """All outputs from a single optimisation run."""

    decisions: np.ndarray
    """Binary array: 1 = accept, 0 = reject."""

    p_cancel: np.ndarray
    """Cancellation probabilities passed to the solver."""

    expected_values: np.ndarray
    """Per-booking expected profit EV_i (before the accept/reject decision)."""

    gross_revenues: np.ndarray
    """Per-booking gross revenue if the guest shows up (adr × nights)."""

    total_expected_revenue: float
    """Sum of EV_i for accepted bookings: Σ decisions_i · EV_i."""

    expected_occupancy: float
    """Expected number of rooms occupied: Σ decisions_i · p_nocancel_i."""

    n_accepted: int
    n_rejected: int
    solver_status: str


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Wraps scipy.optimize.milp (HiGHS) for the booking-acceptance problem."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        df_bookings: pd.DataFrame,
        p_cancel: np.ndarray,
        params: OptimizationParams,
    ) -> OptimizationResult:
        """Optimise the accept/reject decisions for a batch of bookings.

        Parameters
        ----------
        df_bookings:
            DataFrame with at least ``adr`` and ``total_nights`` columns.
        p_cancel:
            Shape-(n,) array of cancellation probabilities from Layer 1.
        params:
            Capacity, penalty, and lambda hyper-parameters.

        Returns
        -------
        OptimizationResult
        """
        n = len(df_bookings)
        if n == 0:
            return self._empty_result()

        print(f"PARAMETRI OTTIMIZZAZIONE: capacity={params.capacity}, cancellation_penalty={params.cancellation_penalty}, lambda_risk={params.lambda_risk}, n_bookings={n}", flush=True)

        # ── Extract financial inputs ────────────────────────────────────
        adr = df_bookings["adr"].fillna(0).values.astype(np.float64)

        # total_nights may or may not be pre-computed in the DataFrame
        if "total_nights" in df_bookings.columns:
            nights = df_bookings["total_nights"].fillna(1).values.astype(np.float64)
        elif "stays_in_week_nights" in df_bookings.columns:
            nights = (
                df_bookings["stays_in_weekend_nights"].fillna(0)
                + df_bookings["stays_in_week_nights"].fillna(0)
            ).values.astype(np.float64)
        else:
            nights = np.ones(n, dtype=np.float64)

        nights = np.maximum(nights, 1.0)

        financials = BookingFinancials(
            adr=adr,
            total_nights=nights,
            p_cancel=p_cancel,
            cancellation_penalty=params.cancellation_penalty,
            lambda_risk=params.lambda_risk,
        )

        c           = build_objective(financials)          # shape (n,) to minimise
        p_no_cancel = financials.p_no_cancel
        constraints = compile_constraints(p_no_cancel, params)

        # Variable bounds: 0 ≤ x_i ≤ 1
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        # All variables are binary (integer in [0, 1])
        integrality = np.ones(n, dtype=int)

        # ── Solve with HiGHS ───────────────────────────────────────────
        result = milp(
            c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"disp": False, "time_limit": _SOLVER_TIME_LIMIT},
        )

        if result.success and result.x is not None:
            decisions     = np.round(result.x).astype(int)
            solver_status = "optimal"
            logger.info(
                "HiGHS optimal — accepted %d/%d  obj=%.2f",
                decisions.sum(), n, -result.fun,
            )
        else:
            logger.warning(
                "HiGHS solver returned status '%s'. Falling back to greedy.",
                result.message,
            )
            decisions     = self._greedy_fallback(financials, params)
            solver_status = "greedy_fallback"

        # ── Build result ───────────────────────────────────────────────
        ev              = financials.expected_value
        total_rev       = float(np.dot(decisions, ev))
        exp_occ         = float(np.dot(decisions, p_no_cancel))

        return OptimizationResult(
            decisions=decisions,
            p_cancel=p_cancel,
            expected_values=ev,
            gross_revenues=financials.gross_revenue,
            total_expected_revenue=total_rev,
            expected_occupancy=exp_occ,
            n_accepted=int(decisions.sum()),
            n_rejected=int(n - decisions.sum()),
            solver_status=solver_status,
        )

    # ------------------------------------------------------------------
    # Greedy fallback
    # ------------------------------------------------------------------

    def _greedy_fallback(
        self,
        financials: BookingFinancials,
        params: OptimizationParams,
    ) -> np.ndarray:
        """Accept bookings in descending EV order until capacity count is reached."""
        ev        = financials.expected_value
        cap_limit = int(round(params.capacity))

        order     = np.argsort(-ev)   # descending expected value
        decisions = np.zeros(len(ev), dtype=int)
        count     = 0

        for i in order:
            if ev[i] <= 0.0:          # no-value or negative-value booking
                break
            if count < cap_limit:
                decisions[i] = 1
                count        += 1

        logger.info(
            "Greedy fallback — accepted %d/%d  capacity=%d",
            decisions.sum(), len(ev), cap_limit,
        )
        return decisions

    # ------------------------------------------------------------------
    # Empty result for zero-booking input
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> OptimizationResult:
        empty = np.array([], dtype=np.float64)
        return OptimizationResult(
            decisions=np.array([], dtype=int),
            p_cancel=empty,
            expected_values=empty,
            gross_revenues=empty,
            total_expected_revenue=0.0,
            expected_occupancy=0.0,
            n_accepted=0,
            n_rejected=0,
            solver_status="empty",
        )
