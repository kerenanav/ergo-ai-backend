"""
constraint_compiler.py — Builds scipy LinearConstraint objects for the MILP.

V2 Constraint Pipeline (5 Steps):
──────────────────────────────────
Step A — Normalise: rule dict → (LHS, op, RHS) triple
Step B — Classify:  triple → C1 | C2 | C3 | C5 | NULL
Step C — Extract:   per-type parameter extraction
Step D — Assemble:  build LinearConstraint objects
Step E — Select:    MILP (any binary var) vs LP (all continuous)

Supported constraint types in V1:
    C1 — Capacity:  Σ x_i ≤ capacity  (global total, resource_key logged)
    C2 — Bounds:    0 ≤ x_i ≤ 1       (handled via integrality + Bounds, not LC)

V2 reserved (not yet active):
    C3 — Demand / coverage (soft)
    C5 — Activation  x ≤ M·y

All constraints use scipy LinearConstraint (lb=-inf, ub=RHS).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import LinearConstraint

logger = logging.getLogger(__name__)


@dataclass
class OptimizationParams:
    """All tunable parameters for a single optimization run."""

    capacity: float
    """Hard count limit — total accepted bookings cannot exceed this."""

    cancellation_penalty: float
    """Financial penalty per cancelled booking (€)."""

    lambda_risk: float = 0.0
    """Risk multiplier (V1: always inactive when risk_enabled=False)."""

    extra_constraints: list = field(default_factory=list)
    """Reserved for future custom constraints."""


# ────────────────────────────────────────────────────────────────────────────
# Step B — Type classifier
# ────────────────────────────────────────────────────────────────────────────

_TYPE_MAP = {
    "C1": "C1",
    "capacity": "C1",
    "count": "C1",
    "C2": "C2",
    "bounds": "C2",
    "expected_occupancy": "C2_occ",
    "C3": "C3",
    "demand": "C3",
    "C5": "C5",
    "activation": "C5",
}


def _classify(constraint_dict: dict) -> str:
    """Step B: classify a raw constraint dict to a canonical type string."""
    raw = str(constraint_dict.get("type", "")).strip()
    return _TYPE_MAP.get(raw, "NULL")


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def compile_constraints(
    p_no_cancel: np.ndarray,
    params: OptimizationParams,
    domain_constraints: list[dict] | None = None,
) -> list[LinearConstraint]:
    """Build and return scipy LinearConstraint objects.

    Parameters
    ----------
    p_no_cancel:
        Shape-(n,) array of show-up probabilities (1 − P(cancel)).
    params:
        Capacity, penalty, lambda from the current request.
    domain_constraints:
        Optional list of constraint dicts from DomainConfig.constraints.
        If None, falls back to the two standard V1 constraints.

    Steps A–E are applied internally.
    """
    n = len(p_no_cancel)
    if n == 0:
        return []

    constraints: list[LinearConstraint] = []

    if domain_constraints:
        # V2 mode: iterate config constraint list (Steps A-E)
        for rule in domain_constraints:
            c_type = _classify(rule)

            if c_type == "C1":
                # Step C: runtime params.capacity is the binding limit.
                # config capacity_value is a reference default only — the
                # request parameter always overrides it so that callers can
                # send capacity=50 and get at most 50 accepted bookings.
                cap = params.capacity
                resource_key = rule.get("resource_key", [])
                if resource_key:
                    logger.debug(
                        "C1 constraint resource_key=%s capacity=%.0f (applied globally in V1)",
                        resource_key, cap,
                    )
                # Step D: assemble — global count constraint
                A = np.ones((1, n))
                constraints.append(LinearConstraint(A, lb=-np.inf, ub=cap))

            elif c_type == "C2":
                # C2 bounds (0 ≤ x ≤ 1) are handled by Bounds + integrality.
                # No LinearConstraint needed; log and skip.
                logger.debug("C2 bounds handled by solver Bounds/integrality — skipped as LC")

            elif c_type == "C2_occ":
                # Expected-occupancy variant: Σ x_i · p_no_cancel_i ≤ capacity
                cap = float(rule.get("capacity_value", params.capacity))
                A_occ = p_no_cancel.reshape(1, n)
                constraints.append(LinearConstraint(A_occ, lb=-np.inf, ub=cap))

            elif c_type in ("C3", "C5"):
                logger.info("Constraint type %s noted but not active in V1 — skipped", c_type)

            else:
                logger.debug("Unknown constraint type '%s' — skipped (Step B: NULL)", rule.get("type"))

    else:
        # V1 fallback: two standard constraints
        # C1: hard count
        A_count = np.ones((1, n))
        constraints.append(LinearConstraint(A_count, lb=-np.inf, ub=params.capacity))

        # C2-occ: expected occupancy
        A_occ = p_no_cancel.reshape(1, n)
        constraints.append(LinearConstraint(A_occ, lb=-np.inf, ub=params.capacity))

    # Step E: MILP vs LP selection is handled in decision_engine.py
    # (integrality=ones → MILP; integrality=zeros → LP)

    return constraints
