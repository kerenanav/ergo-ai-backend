"""
decision_engine.py — Layer 2: MILP-based booking optimiser (HiGHS via scipy).

V2 changes:
  - OptimizationResult gains uids: list[str] and lambda_status: str
  - optimize() accepts cfg: DomainConfig for revenue formula
  - Revenue computed via evaluate_revenue(df, cfg.revenue_formula)
  - lambda_status = "inactive" when cfg.risk_enabled = False (always in V1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, milp

from constraint_compiler import OptimizationParams, compile_constraints
from objective_builder import BookingFinancials, build_objective

logger = logging.getLogger(__name__)

_SOLVER_TIME_LIMIT: float = 120.0


@dataclass
class OptimizationResult:
    """All outputs from a single optimisation run."""

    decisions: np.ndarray
    uids: list[str]
    p_cancel: np.ndarray
    expected_values: np.ndarray
    gross_revenues: np.ndarray
    total_expected_revenue: float
    expected_occupancy: float
    n_accepted: int
    n_rejected: int
    solver_status: str
    lambda_status: str     # "inactive" (V1) or "active" (future V2)


class DecisionEngine:
    """Wraps scipy.optimize.milp (HiGHS) for the booking-acceptance problem."""

    def optimize(
        self,
        df_bookings: pd.DataFrame,
        p_cancel: np.ndarray,
        params: OptimizationParams,
        cfg: Any | None = None,          # DomainConfig
        uids: list[str] | None = None,
    ) -> OptimizationResult:
        n = len(df_bookings)
        if n == 0:
            return self._empty_result()

        # ── Revenue from domain formula ─────────────────────────────────
        if cfg is not None:
            from domain_config import evaluate_revenue
            gross_rev = evaluate_revenue(df_bookings, cfg.revenue_formula)
        else:
            adr = df_bookings["adr"].fillna(0).values.astype(np.float64)
            if "total_nights" in df_bookings.columns:
                nights = df_bookings["total_nights"].fillna(1).values.astype(np.float64)
            elif "stays_in_week_nights" in df_bookings.columns:
                nights = (
                    df_bookings["stays_in_weekend_nights"].fillna(0)
                    + df_bookings["stays_in_week_nights"].fillna(0)
                ).values.astype(np.float64)
            else:
                nights = np.ones(n, dtype=np.float64)
            gross_rev = adr * np.maximum(nights, 1.0)

        risk_enabled = (cfg.risk_enabled if cfg is not None else False)
        lambda_status = "inactive" if not risk_enabled else "active"

        financials = BookingFinancials(
            gross_revenue=gross_rev,
            p_cancel=p_cancel,
            cancellation_penalty=params.cancellation_penalty,
            lambda_risk=params.lambda_risk,
            risk_enabled=risk_enabled,
        )

        domain_constraints = (cfg.constraints if cfg is not None else None)
        c           = build_objective(financials)
        p_no_cancel = financials.p_no_cancel
        constraints = compile_constraints(p_no_cancel, params, domain_constraints)

        bounds      = Bounds(lb=np.zeros(n), ub=np.ones(n))
        integrality = np.ones(n, dtype=int)

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
            logger.info("HiGHS optimal — accepted %d/%d  obj=%.2f",
                        decisions.sum(), n, -result.fun)
        else:
            logger.warning("HiGHS '%s'. Falling back to greedy.", result.message)
            decisions     = self._greedy_fallback(financials, params)
            solver_status = "greedy_fallback"

        ev       = financials.expected_value
        total_ev = float(np.dot(decisions, ev))
        exp_occ  = float(np.dot(decisions, p_no_cancel))

        uid_list = uids if uids is not None else [str(i) for i in range(n)]

        return OptimizationResult(
            decisions=decisions,
            uids=uid_list,
            p_cancel=p_cancel,
            expected_values=ev,
            gross_revenues=gross_rev,
            total_expected_revenue=total_ev,
            expected_occupancy=exp_occ,
            n_accepted=int(decisions.sum()),
            n_rejected=int(n - decisions.sum()),
            solver_status=solver_status,
            lambda_status=lambda_status,
        )

    def _greedy_fallback(
        self,
        financials: BookingFinancials,
        params: OptimizationParams,
    ) -> np.ndarray:
        ev        = financials.expected_value
        cap_limit = int(round(params.capacity))
        order     = np.argsort(-ev)
        decisions = np.zeros(len(ev), dtype=int)
        count     = 0
        for i in order:
            if ev[i] <= 0.0:
                break
            if count < cap_limit:
                decisions[i] = 1
                count += 1
        logger.info("Greedy fallback — accepted %d/%d", decisions.sum(), len(ev))
        return decisions

    @staticmethod
    def _empty_result() -> OptimizationResult:
        empty = np.array([], dtype=np.float64)
        return OptimizationResult(
            decisions=np.array([], dtype=int),
            uids=[],
            p_cancel=empty,
            expected_values=empty,
            gross_revenues=empty,
            total_expected_revenue=0.0,
            expected_occupancy=0.0,
            n_accepted=0,
            n_rejected=0,
            solver_status="empty",
            lambda_status="inactive",
        )
