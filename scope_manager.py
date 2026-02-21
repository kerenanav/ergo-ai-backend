"""
scope_manager.py — Scope lifecycle management for Optide V2.

A Scope is the unit of work: it anchors every operation (optimize, backtest,
report) to a fixed set of rows (identified by stable UIDs), with a locked state
and a snapshotted parameter set.

Lifecycle:
  draft  → (POST /optimize completes) → locked
  locked → (POST /scope/clone) → new draft

Persistence: each Scope is stored as scopes/{scope_id}.json on disk.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from domain_config import DomainConfig

logger = logging.getLogger(__name__)

SCOPES_DIR  = Path("scopes")
RANDOM_SEED = 42


# ────────────────────────────────────────────────────────────────────────────
# Scope dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Scope:
    """A single unit of work with stable UIDs and snapshotted parameters."""

    scope_id: str
    domain_config_id: str
    status: str                    # "draft" | "locked"
    selected_uids: list[str]
    params_snapshot: dict
    selection_rule: dict
    created_at: str
    locked_at: str | None = None
    optimize_completed: bool = False
    backtest_completed: bool = False
    n_uids: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Scope":
        return Scope(**d)


# ────────────────────────────────────────────────────────────────────────────
# ScopeManager
# ────────────────────────────────────────────────────────────────────────────

class ScopeManager:
    """Manages Scope objects: creation, persistence, lifecycle transitions."""

    def __init__(self, scopes_dir: str | Path = SCOPES_DIR) -> None:
        self._dir = Path(scopes_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._scopes: dict[str, Scope] = {}

    # ── Startup ──────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Load all scopes from disk into memory (called at startup)."""
        loaded = 0
        for path in sorted(self._dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                scope = Scope.from_dict(data)
                self._scopes[scope.scope_id] = scope
                loaded += 1
                logger.debug("Loaded scope %s (status=%s)", scope.scope_id, scope.status)
            except Exception as exc:
                logger.warning("Could not load scope from %s: %s", path, exc)
        logger.info("Loaded %d scopes from %s", loaded, self._dir)

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_scope(self, scope_id: str) -> Scope:
        """Return the Scope with *scope_id*, raising KeyError if not found."""
        scope = self._scopes.get(scope_id)
        if scope is None:
            raise KeyError(scope_id)
        return scope

    def list_scopes(self) -> list[dict]:
        """Return a summary list of all scopes."""
        return [
            {
                "scope_id":        s.scope_id,
                "domain_config_id": s.domain_config_id,
                "status":          s.status,
                "n_uids":          s.n_uids,
                "created_at":      s.created_at,
                "locked_at":       s.locked_at,
            }
            for s in self._scopes.values()
        ]

    # ── Create ───────────────────────────────────────────────────────────────

    def create_scope(
        self,
        domain_config_id: str,
        params: dict,
        selection_rule: dict,
        df: pd.DataFrame,
        cfg: DomainConfig,
        seed: int = RANDOM_SEED,
    ) -> Scope:
        """Create a new draft Scope and persist it."""
        scope_id      = uuid.uuid4().hex[:12]
        selected_uids = self.compute_selected_uids(df, selection_rule, cfg, seed)

        scope = Scope(
            scope_id=scope_id,
            domain_config_id=domain_config_id,
            status="draft",
            selected_uids=selected_uids,
            params_snapshot=dict(params),
            selection_rule=selection_rule,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            n_uids=len(selected_uids),
        )
        self._scopes[scope_id] = scope
        self.save(scope)
        logger.info("Created scope %s — domain=%s  n_uids=%d", scope_id, domain_config_id, len(selected_uids))
        return scope

    # ── Lifecycle transitions ────────────────────────────────────────────────

    def lock_scope(self, scope_id: str) -> Scope:
        """Mark a scope as locked (called after /optimize completes)."""
        scope = self.get_scope(scope_id)
        scope.status             = "locked"
        scope.locked_at          = datetime.now(tz=timezone.utc).isoformat()
        scope.optimize_completed = True
        self.save(scope)
        logger.info("Locked scope %s", scope_id)
        return scope

    def mark_backtest_completed(self, scope_id: str) -> Scope:
        """Record that /backtest has run on this scope."""
        scope = self.get_scope(scope_id)
        scope.backtest_completed = True
        self.save(scope)
        return scope

    def clone_scope(self, scope_id: str, new_params: dict) -> tuple[Scope, dict]:
        """Clone a scope with different parameters, returning (new_scope, warning)."""
        original      = self.get_scope(scope_id)
        new_scope_id  = uuid.uuid4().hex[:12]
        merged_params = {**original.params_snapshot, **new_params}

        new_scope = Scope(
            scope_id=new_scope_id,
            domain_config_id=original.domain_config_id,
            status="draft",
            selected_uids=list(original.selected_uids),
            params_snapshot=merged_params,
            selection_rule=original.selection_rule,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            n_uids=original.n_uids,
        )
        self._scopes[new_scope_id] = new_scope
        self.save(new_scope)

        warning = {
            "en": "This is a new experiment. Results are NOT directly comparable with the original scope.",
            "it": "Questo è un nuovo esperimento. I risultati NON sono direttamente comparabili con lo scope originale.",
        }
        logger.info("Cloned scope %s → %s", scope_id, new_scope_id)
        return new_scope, warning

    # ── Validation ───────────────────────────────────────────────────────────

    def validate_for_backtest(self, scope: Scope) -> None:
        """Raise ValueError('scope_not_locked') if the scope is not locked."""
        if scope.status != "locked":
            raise ValueError("scope_not_locked")

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, scope: Scope) -> None:
        """Persist a scope to scopes/{scope_id}.json."""
        path = self._dir / f"{scope.scope_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scope.to_dict(), f, indent=2)

    # ── UID selection ────────────────────────────────────────────────────────

    @staticmethod
    def compute_selected_uids(
        df: pd.DataFrame,
        selection_rule: dict,
        cfg: DomainConfig,
        seed: int = RANDOM_SEED,
    ) -> list[str]:
        """Select rows from *df* according to *selection_rule* and return their UIDs.

        Supported rule types:
          - random  : sample *n* rows randomly (default)
          - all     : include all rows
        Optional: start_date / end_date filter on df["_time"] column.
        """
        df_filtered = df.copy()

        # Date range filter (requires _time column added upstream)
        start_date = selection_rule.get("start_date")
        end_date   = selection_rule.get("end_date")
        if "_time" in df_filtered.columns:
            if start_date:
                df_filtered = df_filtered[
                    df_filtered["_time"] >= pd.Timestamp(start_date)
                ]
            if end_date:
                df_filtered = df_filtered[
                    df_filtered["_time"] <= pd.Timestamp(end_date)
                ]

        rule_type = selection_rule.get("type", "random")

        if rule_type == "all":
            subset = df_filtered
        else:
            # random (default)
            n   = min(int(selection_rule.get("n", 100)), len(df_filtered))
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df_filtered), size=n, replace=False)
            subset = df_filtered.iloc[idx]

        if "_uid" in subset.columns:
            return subset["_uid"].tolist()
        else:
            return [str(i) for i in subset.index.tolist()]
