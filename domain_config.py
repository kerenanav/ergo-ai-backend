"""
domain_config.py — DomainConfig dataclass and helpers for Optide V2.

Supports:
  - Loading domain configuration from JSON files
  - Validating config completeness
  - Computing stable row UIDs via sha256[:16]
  - Building a time column from raw columns or time_builder spec
  - Adding _uid column to DataFrames
  - Evaluating revenue formulas via df.eval()
  - Listing available configs
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ────────────────────────────────────────────────────────────────────────────
# DomainConfig dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainConfig:
    """All domain-specific configuration loaded from a JSON config file."""

    domain_id: str
    domain_name: str
    config_version: str
    decision_mode: str

    # Time handling
    time_column: str | None
    time_builder: dict | None

    # Column definitions
    action_columns: list[str]
    resource_columns: list[str]
    id_column: str | None
    uid_strategy: str
    uid_key_columns: list[str]

    # Financial formulas
    revenue_formula: str
    cost_formula: str | None

    # ML configuration
    ml_target_column: str | None
    ml_features: list[str]

    # Baseline policy
    historical_policy: str
    historical_policy_column: str | None

    # Constraints
    constraints: list[dict]

    # Default optimization parameters
    cancellation_penalty_default: float
    risk_enabled: bool
    lambda_risk_default: float

    # Data
    dataset_path: str


# ────────────────────────────────────────────────────────────────────────────
# Load / validate
# ────────────────────────────────────────────────────────────────────────────

def load_domain_config(path: str | Path) -> DomainConfig:
    """Load a DomainConfig from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return DomainConfig(
        domain_id=raw["domain_id"],
        domain_name=raw.get("domain_name", raw["domain_id"]),
        config_version=raw.get("config_version", "1.0"),
        decision_mode=raw.get("decision_mode", "row_level"),
        time_column=raw.get("time_column"),
        time_builder=raw.get("time_builder"),
        action_columns=raw.get("action_columns", []),
        resource_columns=raw.get("resource_columns", []),
        id_column=raw.get("id_column"),
        uid_strategy=raw.get("uid_strategy", "hash"),
        uid_key_columns=raw.get("uid_key_columns", []),
        revenue_formula=raw.get("revenue_formula", "0"),
        cost_formula=raw.get("cost_formula"),
        ml_target_column=raw.get("ml_target_column"),
        ml_features=raw.get("ml_features", []),
        historical_policy=raw.get("historical_policy", "accept_all"),
        historical_policy_column=raw.get("historical_policy_column"),
        constraints=raw.get("constraints", []),
        cancellation_penalty_default=float(raw.get("cancellation_penalty_default", 50.0)),
        risk_enabled=bool(raw.get("risk_enabled", False)),
        lambda_risk_default=float(raw.get("lambda_risk_default", 0.0)),
        dataset_path=raw.get("dataset_path", ""),
    )


def validate_domain_config(cfg: DomainConfig) -> list[str]:
    """Return a list of warning strings for any config issues found."""
    warnings: list[str] = []

    if not cfg.uid_key_columns:
        warnings.append("uid_key_columns is empty — UIDs will not be stable")
    if not cfg.ml_features:
        warnings.append("ml_features is empty — model training will fail")
    if not cfg.ml_target_column:
        warnings.append("ml_target_column not set — backtest will be unavailable")
    if cfg.historical_policy == "column" and not cfg.historical_policy_column:
        warnings.append("historical_policy=column but historical_policy_column is null")
    if (cfg.historical_policy_column
            and cfg.ml_target_column
            and cfg.historical_policy_column == cfg.ml_target_column):
        warnings.append("historical_policy_column must not equal ml_target_column (leakage risk)")

    return warnings


# ────────────────────────────────────────────────────────────────────────────
# UID helpers
# ────────────────────────────────────────────────────────────────────────────

def compute_row_uid(row: Any, uid_key_columns: list[str]) -> str:
    """Compute a stable 16-char UID from selected columns using sha256."""
    key = "|".join(str(row[c]) for c in uid_key_columns)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def add_uid_column(df: pd.DataFrame, cfg: DomainConfig) -> pd.DataFrame:
    """Return a copy of *df* with a '_uid' column added."""
    df = df.copy()
    uid_cols = [c for c in cfg.uid_key_columns if c in df.columns]
    if not uid_cols:
        logger.warning("No uid_key_columns found in DataFrame — using row index as UID")
        df["_uid"] = [f"row_{i}" for i in range(len(df))]
    else:
        df["_uid"] = df.apply(lambda row: compute_row_uid(row, uid_cols), axis=1)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Time column builder
# ────────────────────────────────────────────────────────────────────────────

def build_time_column(df: pd.DataFrame, cfg: DomainConfig) -> pd.Series:
    """Return a datetime Series for sorting/filtering.

    Priority:
    1. time_column (direct datetime parse)
    2. time_builder (composite year/month/day columns)
    3. NaT series (no time info available)
    """
    if cfg.time_column and cfg.time_column in df.columns:
        return pd.to_datetime(df[cfg.time_column], errors="coerce")

    if cfg.time_builder:
        tb = cfg.time_builder
        year_col  = tb.get("year_col")
        month_col = tb.get("month_col")
        day_col   = tb.get("day_col")

        if year_col and year_col in df.columns:
            year = df[year_col].astype(int)
        else:
            year = pd.Series(2000, index=df.index)

        if month_col and month_col in df.columns:
            month_raw = df[month_col]
            # In pandas 2.x string columns may be StringDtype, not object.
            # Try numeric cast first; fall back to month-name mapping on failure.
            try:
                month = month_raw.astype(int)
            except (ValueError, TypeError):
                month = month_raw.astype(str).str.lower().map(MONTH_MAP).fillna(1).astype(int)
        else:
            month = pd.Series(1, index=df.index)

        if day_col and day_col in df.columns:
            day = df[day_col].astype(int)
        else:
            day = pd.Series(1, index=df.index)

        return pd.to_datetime(
            {"year": year, "month": month, "day": day}, errors="coerce"
        )

    return pd.Series(pd.NaT, index=df.index)


# ────────────────────────────────────────────────────────────────────────────
# Revenue formula evaluator
# ────────────────────────────────────────────────────────────────────────────

def evaluate_revenue(df: pd.DataFrame, formula: str) -> np.ndarray:
    """Evaluate *formula* against *df* row-by-row, returning non-negative floats.

    Tries df.eval() first (fast, vectorised).
    Falls back to row-level eval() if that fails.
    """
    try:
        result = df.eval(formula, engine="python")
        return np.maximum(np.asarray(result, dtype=np.float64), 0.0)
    except Exception as exc:
        logger.debug("df.eval() failed for formula %r: %s — falling back to row eval", formula, exc)

    n = len(df)
    out = np.zeros(n, dtype=np.float64)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            val = eval(formula, {"__builtins__": {}}, dict(row))  # noqa: S307
            out[i] = max(float(val), 0.0)
        except Exception:
            out[i] = 0.0
    return out


# ────────────────────────────────────────────────────────────────────────────
# Config discovery
# ────────────────────────────────────────────────────────────────────────────

def list_domain_configs(configs_dir: str | Path = "configs") -> list[str]:
    """Return domain_id strings for all *_config.json files found."""
    configs_dir = Path(configs_dir)
    if not configs_dir.exists():
        return []
    return [
        p.stem.replace("_config", "")
        for p in sorted(configs_dir.glob("*_config.json"))
    ]
