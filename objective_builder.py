"""
objective_builder.py — Constructs the MILP objective-function coefficients.

Expected value of accepting booking i (V2 form):
────────────────────────────────────────────────
    EV(i) = gross_revenue_i · P(show_i) − penalty · P(cancel_i) − Risk(i)

where
    gross_revenue_i = eval(revenue_formula) for row i   [from DomainConfig]
    P(show_i)       = 1 − P(cancel_i)

Risk term (V1: always zero):
    if risk_enabled=True:
        Risk(i) = λ · P(cancel_i) · P(show_i) · gross_revenue_i
        [variance-based: bounded by 0.25 · λ · gross_revenue_i]
    if risk_enabled=False (V1 default):
        Risk(i) = 0   → lambda_status = "inactive"

The MILP *minimises*, so we negate EV to convert maximisation → minimisation:
    c_i = −EV(i)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BookingFinancials:
    """Vectorised financial quantities for a batch of bookings."""

    gross_revenue: np.ndarray
    """Revenue per booking from evaluate_revenue(df, revenue_formula). Shape (n,)."""

    p_cancel: np.ndarray
    """Cancellation probability from Layer-1 model. Shape (n,)."""

    cancellation_penalty: float
    """Flat penalty charged per cancellation (€)."""

    lambda_risk: float = 0.0
    """Risk multiplier. Only active if risk_enabled=True."""

    risk_enabled: bool = False
    """False in V1 — risk term is always zero regardless of lambda_risk."""

    def __post_init__(self) -> None:
        self.gross_revenue = np.asarray(self.gross_revenue, dtype=np.float64)
        self.p_cancel = np.clip(np.asarray(self.p_cancel, dtype=np.float64), 0.0, 1.0)
        self.gross_revenue = np.maximum(self.gross_revenue, 0.0)

    @property
    def p_no_cancel(self) -> np.ndarray:
        """Show-up probability: 1 − P(cancel)."""
        return 1.0 - self.p_cancel

    @property
    def expected_value(self) -> np.ndarray:
        """
        Expected profit for each booking:

            EV_i = gross_revenue_i · P(show_i)
                   − cancellation_penalty · P(cancel_i)
                   [− λ · P(cancel_i) · P(show_i) · gross_revenue_i  if risk_enabled]

        With risk_enabled=False (V1): the risk term is always 0.
        lambda_status = "inactive" is set in decision_engine.py.
        """
        base = (
            self.p_no_cancel * self.gross_revenue
            - self.cancellation_penalty * self.p_cancel
        )
        if self.risk_enabled and self.lambda_risk != 0.0:
            risk_term = (
                self.lambda_risk
                * self.p_cancel
                * self.p_no_cancel
                * self.gross_revenue
            )
            return base - risk_term
        return base


def build_objective(financials: BookingFinancials) -> np.ndarray:
    """Return coefficient vector *c* for scipy.optimize.milp (minimisation).

    The solver minimises ``c @ x``, so we negate EV to transform
    maximisation → minimisation.

    Returns
    -------
    np.ndarray of shape (n,) with c_i = −EV_i.
    """
    return -financials.expected_value
