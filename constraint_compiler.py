"""
constraint_compiler.py — Builds scipy LinearConstraint objects for the MILP.

The booking-acceptance problem is a 0-1 knapsack variant:

    Decision variables : x_i ∈ {0, 1}   (1 = accept booking i, 0 = reject)

    Constraints
    ───────────
    C1 (expected-occupancy)  : Σ x_i · p_nocancel_i  ≤  capacity · λ
         Keeps expected show-ups within λ × nominal capacity.
         λ > 1 enables deliberate overbooking; λ < 1 adds a safety buffer.

    C2 (hard-booking ceiling) : Σ x_i  ≤  capacity · λ · δ
         Even if every guest cancels, we don't accept an absurdly large
         number of reservations.  δ = 2.0 is a conservative ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import LinearConstraint

# Safety multiplier for the hard-booking ceiling constraint (C2)
_HARD_CEILING_FACTOR: float = 2.0


@dataclass
class OptimizationParams:
    """All tunable parameters for a single optimization run."""

    capacity: float
    """Nominal hotel capacity in rooms."""

    cancellation_penalty: float
    """Financial penalty charged per cancelled booking (€)."""

    lambda_overbooking: float = 1.0
    """
    Overbooking multiplier applied to the expected-occupancy constraint.
    1.0 → no overbooking; 1.2 → 20 % overbooking allowed.
    Set to a very small positive number (e.g., 0.01) to practically forbid
    all bookings (extreme risk-aversion scenario in sensitivity analysis).
    """

    extra_constraints: list = field(default_factory=list)
    """Reserved for future custom constraints."""


def compile_constraints(
    p_no_cancel: np.ndarray,
    params: OptimizationParams,
) -> list[LinearConstraint]:
    """Return the list of ``scipy.optimize.LinearConstraint`` objects.

    Parameters
    ----------
    p_no_cancel:
        Shape (n,) array of show-up probabilities, i.e., 1 − P(cancel).
    params:
        Optimization hyper-parameters.

    Returns
    -------
    A list of ``LinearConstraint`` instances ready to pass to
    ``scipy.optimize.milp``.
    """
    n = len(p_no_cancel)
    if n == 0:
        return []

    constraints: list[LinearConstraint] = []

    # ── C1: Expected-occupancy constraint ───────────────────────────────────
    # Σ x_i · p_nocancel_i  ≤  capacity · λ
    ub_occ = params.capacity * max(params.lambda_overbooking, 1e-6)
    A_occ  = p_no_cancel.reshape(1, n)
    constraints.append(LinearConstraint(A_occ, lb=-np.inf, ub=ub_occ))

    # ── C2: Hard-booking ceiling ────────────────────────────────────────────
    # Σ x_i  ≤  capacity · λ · δ
    ub_hard = params.capacity * max(params.lambda_overbooking, 1e-6) * _HARD_CEILING_FACTOR
    A_hard  = np.ones((1, n))
    constraints.append(LinearConstraint(A_hard, lb=-np.inf, ub=ub_hard))

    return constraints
