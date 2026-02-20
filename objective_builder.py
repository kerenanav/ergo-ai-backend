"""
objective_builder.py — Constructs the MILP objective-function coefficients.

Expected value of accepting booking i
──────────────────────────────────────
    EV(i) = P(no_cancel_i) · revenue_i  −  P(cancel_i) · penalty

where
    revenue_i  = adr_i × total_nights_i

The MILP *minimises*, so we negate EV to convert maximisation → minimisation:

    c_i = −EV(i)

Bookings with EV(i) < 0 are never worth accepting at the given penalty level;
the solver will set x_i = 0 automatically once capacity is a binding constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BookingFinancials:
    """Vectorised financial quantities for a batch of bookings."""

    adr: np.ndarray
    """Average Daily Rate per booking (€/night)."""

    total_nights: np.ndarray
    """Length of stay in nights (must be ≥ 1)."""

    p_cancel: np.ndarray
    """Cancellation probability from Layer-1 model, shape (n,)."""

    cancellation_penalty: float
    """Flat penalty charged per cancellation (€)."""

    lambda_risk: float = 0.0
    """
    Risk multiplier for the risk term in the objective.
    Higher values penalise bookings with high p_cancel more aggressively.
    """

    def __post_init__(self) -> None:
        self.adr           = np.asarray(self.adr,          dtype=np.float64)
        self.total_nights  = np.asarray(self.total_nights,  dtype=np.float64)
        self.p_cancel      = np.asarray(self.p_cancel,      dtype=np.float64)
        # Clip to valid probability range
        self.p_cancel = np.clip(self.p_cancel, 0.0, 1.0)
        # Clip nights to avoid nonsense (0-night stays)
        self.total_nights = np.maximum(self.total_nights, 1.0)

    @property
    def p_no_cancel(self) -> np.ndarray:
        """Show-up probability: 1 − P(cancel)."""
        return 1.0 - self.p_cancel

    @property
    def gross_revenue(self) -> np.ndarray:
        """Revenue assuming the guest does not cancel: adr × nights."""
        return self.adr * self.total_nights

    @property
    def expected_value(self) -> np.ndarray:
        """
        Expected profit for each booking:

            EV_i = gross_revenue_i · P(show_i)
                   − cancellation_penalty · P(cancel_i)
                   − lambda_risk · P(cancel_i) · gross_revenue_i

        The risk term  lambda_risk · P(cancel_i) · gross_revenue_i
        penalises bookings where both cancellation probability and
        revenue at stake are high. lambda_risk = 0 collapses to the
        standard EV formula.
        """
        risk_term = self.lambda_risk * self.p_cancel * self.gross_revenue
        return (
            self.p_no_cancel * self.gross_revenue
            - self.cancellation_penalty * self.p_cancel
            - risk_term
        )


def build_objective(financials: BookingFinancials) -> np.ndarray:
    """Return coefficient vector *c* for scipy.optimize.milp (minimisation).

    The solver minimises ``c @ x``, so we negate the expected-value vector
    to transform our *maximisation* objective into a minimisation one.

    Returns
    -------
    np.ndarray of shape (n,) with c_i = −EV_i.
    """
    return -financials.expected_value
