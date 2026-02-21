"""
main.py — Optide V2 Decision Intelligence Engine  (FastAPI backend)

Endpoints
─────────
GET  /health
GET  /domain/list
GET  /domain/templates
POST /domain/upload_csv         → multipart CSV → auto-suggest mapping + preview
POST /domain/confirm_mapping    → save config + train model + hot-load domain
POST /scope/create
POST /scope/clone
GET  /scope/list
GET  /scope/{scope_id}
POST /optimize          → scope_id optional; Format A locks scope, Format B is scopeless
POST /backtest          → 409 if not locked | 422 if no outcomes
POST /sensitivity
POST /report            → 409 if not locked; returns PDF bytes
POST /historical_baseline

Start:
    python main.py
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from backtester import Backtester
from chart_generator import (
    capacity_utilization_chart,
    capacity_sensitivity_chart,
    decision_distribution_chart,
    expected_vs_realized_chart,
    penalty_sensitivity_chart,
    profit_composition_chart,
)
from constraint_compiler import OptimizationParams
from decision_engine import DecisionEngine, OptimizationResult
from domain_config import (
    DomainConfig,
    add_uid_column,
    build_time_column,
    evaluate_revenue,
    list_domain_configs,
    load_domain_config,
    validate_domain_config,
)
from predictive_model import CancellationPredictor
from report_generator import generate_report
from scope_manager import Scope, ScopeManager

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("optide")

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIGS_DIR  = Path("configs")
SCOPES_DIR   = Path("scopes")
DATA_DIR     = Path("data")
RANDOM_SEED  = 42

# ── Application state ─────────────────────────────────────────────────────────
_state: dict[str, Any] = {
    "ready":          False,
    "domain_configs": {},   # domain_id → DomainConfig
    "datasets":       {},   # domain_id → pd.DataFrame (with _uid, _time)
    "predictors":     {},   # domain_id → CancellationPredictor
    "engines":        {},   # domain_id → DecisionEngine
    "model_metrics":  {},   # domain_id → {cv_metrics, mean_auc, mean_brier}
    "scope_manager":  None,
    "results":        {},   # scope_id → optimize result snapshot
    "backtests":      {},   # scope_id → BacktestSummary.to_dict()
    "sensitivities":  {},   # scope_id or "global" → sensitivity dict
}


# ────────────────────────────────────────────────────────────────────────────
# Lifespan: load configs, datasets, models
# ────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Optide V2 starting up …")

    CONFIGS_DIR.mkdir(exist_ok=True)
    SCOPES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # Scope manager
    sm = ScopeManager(SCOPES_DIR)
    sm.load_all()
    _state["scope_manager"] = sm

    # Discover and load all domain configs
    domain_ids = list_domain_configs(CONFIGS_DIR)
    if not domain_ids:
        logger.warning("No domain configs found in %s/ — server will start but no domains are ready.", CONFIGS_DIR)

    for domain_id in domain_ids:
        _load_domain(domain_id)

    # Server is ready if at least one domain loaded successfully
    _state["ready"] = bool(_state["domain_configs"])
    logger.info(
        "Startup complete — domains ready: %s",
        list(_state["domain_configs"].keys()) or "none",
    )

    yield

    logger.info("Optide V2 shutting down.")


def _load_domain(domain_id: str) -> bool:
    """Load config, dataset, and model for one domain. Returns True on success."""
    config_path = CONFIGS_DIR / f"{domain_id}_config.json"
    if not config_path.exists():
        logger.warning("Config file missing: %s", config_path)
        return False

    try:
        cfg = load_domain_config(config_path)
    except Exception as exc:
        logger.error("Failed to load config %s: %s", config_path, exc)
        return False

    warnings = validate_domain_config(cfg)
    for w in warnings:
        logger.warning("[%s] config warning: %s", domain_id, w)

    # Load dataset
    data_path = Path(cfg.dataset_path)
    if not data_path.exists():
        logger.warning("[%s] Dataset not found: %s — domain will be config-only.", domain_id, data_path)
        _state["domain_configs"][domain_id] = cfg
        return True

    try:
        predictor_for_load = CancellationPredictor(domain_config=cfg)
        df = predictor_for_load.load_raw()
        time_s = build_time_column(df, cfg)
        if time_s.notna().any():
            df["_time"] = time_s
        df = add_uid_column(df, cfg)
        _state["datasets"][domain_id] = df
        logger.info("[%s] Dataset loaded — %d rows", domain_id, len(df))
    except Exception as exc:
        logger.error("[%s] Dataset load failed: %s", domain_id, exc)
        _state["domain_configs"][domain_id] = cfg
        return True

    # Load pre-trained model
    model_path = f"model_{domain_id}.pkl"
    if os.path.exists(model_path):
        try:
            predictor = joblib.load(model_path)
            logger.info("[%s] Loaded model from %s", domain_id, model_path)
        except Exception as exc:
            logger.error("[%s] Could not load %s: %s", domain_id, model_path, exc)
            predictor = None
    else:
        logger.warning(
            "[%s] model_%s.pkl not found — run `python train_model.py %s` first.",
            domain_id, domain_id, domain_id,
        )
        predictor = None

    _state["domain_configs"][domain_id] = cfg
    if predictor is not None:
        _state["predictors"][domain_id] = predictor
        _state["engines"][domain_id]    = DecisionEngine()
        _state["model_metrics"][domain_id] = {
            "cv_metrics": predictor.cv_metrics,
            "mean_auc":   round(float(np.mean([m["auc"]   for m in predictor.cv_metrics])), 4) if predictor.cv_metrics else 0.0,
            "mean_brier": round(float(np.mean([m["brier"] for m in predictor.cv_metrics])), 4) if predictor.cv_metrics else 0.0,
        }

    return True


# ────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Optide — Decision Intelligence Engine",
    description="Scope-driven booking optimisation: LightGBM + MILP (HiGHS). V2.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class _NgrokMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response


app.add_middleware(_NgrokMiddleware)


# ────────────────────────────────────────────────────────────────────────────
# Guards
# ────────────────────────────────────────────────────────────────────────────

def _require_ready() -> None:
    if not _state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="No domains ready. Place CSV + config in working directory and restart.",
        )


def _require_domain(domain_id: str) -> DomainConfig:
    cfg = _state["domain_configs"].get(domain_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found.")
    return cfg


def _require_predictor(domain_id: str) -> tuple[CancellationPredictor, DecisionEngine]:
    predictor = _state["predictors"].get(domain_id)
    engine    = _state["engines"].get(domain_id)
    if predictor is None or engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model for domain '{domain_id}' not loaded. Run `python train_model.py {domain_id}` first.",
        )
    return predictor, engine


def _require_scope(scope_id: str) -> Scope:
    sm: ScopeManager = _state["scope_manager"]
    try:
        return sm.get_scope(scope_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Scope '{scope_id}' not found.")


def _require_locked(scope: Scope) -> None:
    if scope.status != "locked":
        raise HTTPException(
            status_code=409,
            detail={
                "error":      "scope_not_locked",
                "message_en": "Scope must be locked. Run POST /optimize first.",
                "message_it": "Lo scope deve essere bloccato. Eseguire prima POST /optimize.",
            },
        )


def _scope_df(scope: Scope, domain_id: str) -> pd.DataFrame:
    """Return the subset of the domain dataset matching the scope UIDs."""
    df_full = _state["datasets"].get(domain_id)
    if df_full is None:
        raise HTTPException(status_code=503, detail=f"Dataset for domain '{domain_id}' not loaded.")
    if "_uid" in df_full.columns:
        df = df_full[df_full["_uid"].isin(scope.selected_uids)].copy().reset_index(drop=True)
    else:
        df = df_full.copy()
    if len(df) == 0:
        raise HTTPException(status_code=422, detail="Scope UIDs not found in dataset.")
    return df


def _outcome_available(df: pd.DataFrame, cfg: DomainConfig) -> bool:
    target = cfg.ml_target_column
    if not target:
        return False
    if target not in df.columns:
        return False
    return not df[target].isna().any()


def _realized_profit(
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


# ────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ────────────────────────────────────────────────────────────────────────────

class SelectionRule(BaseModel):
    type: str       = "random"
    n: int          = Field(default=100, ge=1, le=10_000)
    start_date: Optional[str] = None
    end_date:   Optional[str] = None


class ScopeParams(BaseModel):
    cancellation_penalty: float = Field(default=50.0,  ge=0)
    capacity:             float = Field(default=100.0, ge=1)
    lambda_risk:          float = Field(default=0.0,   ge=0)


class CreateScopeRequest(BaseModel):
    domain_id:      str
    selection_rule: SelectionRule = SelectionRule()
    params:         ScopeParams   = ScopeParams()
    seed:           int           = 42


class CloneScopeRequest(BaseModel):
    scope_id:   str
    new_params: dict = {}


class OptimizeRequest(BaseModel):
    # Format A (scope-based, recommended):
    scope_id:             Optional[str]   = None
    # Format B/C (scopeless): domain_id optional; defaults to "hotel_v1" if omitted
    domain_id:            Optional[str]   = None
    n_samples:            int             = Field(default=100, ge=1, le=10_000)
    # Shared params — have defaults so the frontend can omit them
    cancellation_penalty: float           = 50.0
    capacity:             float           = 100.0
    lambda_risk:          float           = 0.0
    # Legacy envelope: {"params": {"cancellation_penalty": ..., "capacity": ...}}
    params:               Optional[dict]  = None


class BacktestRequest(BaseModel):
    scope_id:          str
    n_splits:          int = Field(default=5,   ge=2, le=10)
    n_samples_per_fold:int = Field(default=100, ge=10, le=1000)


class SensitivityRequest(BaseModel):
    scope_id:    Optional[str]   = None
    domain_id:   Optional[str]   = None
    capacity:    float           = Field(default=100.0, ge=1)
    base_penalty:float           = Field(default=50.0,  ge=0)
    base_lambda: float           = Field(default=0.0,   ge=0)
    n_sample:    int             = Field(default=500,   ge=50, le=5000)


class ReportRequest(BaseModel):
    scope_id: str
    lang:     str = "en"


class HistoricalBaselineRequest(BaseModel):
    scope_id:             Optional[str]   = None
    cancellation_penalty: float           = 50.0
    capacity:             float           = 100.0
    lambda_risk:          float           = 0.0
    n_samples:            int             = Field(default=100, ge=1, le=10_000)


class ConfirmMappingRequest(BaseModel):
    filename:                    str
    domain_name:                 str
    revenue_formula:             str
    ml_target_column:            str
    ml_features:                 list[str]
    time_column:                 Optional[str]       = None
    uid_key_columns:             Optional[list[str]] = None
    cancellation_penalty_default:float               = 50.0
    capacity_default:            float               = 100.0
    lambda_risk_default:         float               = 0.0


# ────────────────────────────────────────────────────────────────────────────
# GET /health
# ────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict[str, Any]:
    domains_ready = list(_state["predictors"].keys())
    domains_config_only = [
        d for d in _state["domain_configs"] if d not in _state["predictors"]
    ]
    return {
        "status":             "ready" if _state.get("ready") else "not_ready",
        "domains_ready":      domains_ready,
        "domains_config_only":domains_config_only,
        "model_metrics":      _state.get("model_metrics", {}),
        "n_scopes":           len(_state["scope_manager"]._scopes) if _state["scope_manager"] else 0,
    }


# ────────────────────────────────────────────────────────────────────────────
# GET /domain/list
# ────────────────────────────────────────────────────────────────────────────

@app.get("/domain/list", tags=["domain"])
async def domain_list() -> dict[str, Any]:
    """List all domain configs discovered at startup."""
    result = []
    for domain_id, cfg in _state["domain_configs"].items():
        result.append({
            "domain_id":   domain_id,
            "domain_name": cfg.domain_name,
            "model_ready": domain_id in _state["predictors"],
            "dataset_rows": len(_state["datasets"].get(domain_id, [])) or None,
            "revenue_formula": cfg.revenue_formula,
            "ml_target_column": cfg.ml_target_column,
        })
    return {"domains": result}


# ────────────────────────────────────────────────────────────────────────────
# GET /domain/templates
# ────────────────────────────────────────────────────────────────────────────

@app.get("/domain/templates", tags=["domain"])
async def domain_templates() -> dict[str, Any]:
    """Return example config templates for hotel and logistics domains."""
    return {
        "templates": [
            {
                "domain_id":   "hotel_v1",
                "domain_name": "Hotel Booking Demand",
                "revenue_formula": "adr*(stays_in_weekend_nights+stays_in_week_nights)",
                "ml_target_column": "is_canceled",
                "dataset_path": "hotel_bookings.csv",
            },
            {
                "domain_id":   "logistics_v1",
                "domain_name": "Logistics Route Acceptance",
                "revenue_formula": "base_rate * weight_kg * distance_km",
                "ml_target_column": "cancelled",
                "dataset_path": "logistics_data.csv",
            },
        ]
    }


# ────────────────────────────────────────────────────────────────────────────
# POST /domain/upload_csv
# ────────────────────────────────────────────────────────────────────────────

def _suggest_column_mapping(df: pd.DataFrame) -> dict[str, Any]:
    """
    Auto-suggest column roles from column names + data statistics.
    Returns a suggestions dict with keys: time_column, revenue_formula,
    ml_target_column, ml_features, uid_key_columns.
    """
    cols = list(df.columns)

    # --- time_column: first col matching date/time keywords, not purely binary
    time_col: str | None = None
    for c in cols:
        if re.search(r"date|time|year|month|day|period|week", c, re.I):
            uniq = df[c].nunique()
            if uniq > 2:
                time_col = c
                break

    # --- revenue_formula: first col matching financial keywords (numeric)
    rev_col: str | None = None
    for c in cols:
        if re.search(r"adr|price|revenue|rate|fare|amount|value|cost|fee", c, re.I):
            if pd.api.types.is_numeric_dtype(df[c]):
                rev_col = c
                break
    rev_formula = rev_col or ""

    # --- ml_target_column: binary column (only 0/1) matching churn/cancel keywords
    target_col: str | None = None
    binary_cols = [
        c for c in cols
        if pd.api.types.is_numeric_dtype(df[c]) and set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})
    ]
    # prefer a named match
    for c in binary_cols:
        if re.search(r"cancel|churn|reject|default|fraud|abandon|drop", c, re.I):
            target_col = c
            break
    if target_col is None and binary_cols:
        target_col = binary_cols[0]

    # --- ml_features: numeric + low-cardinality string cols, minus target/time
    exclude = {target_col, time_col}
    features = []
    for c in cols:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(c)
        elif df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            if df[c].nunique() <= 50:
                features.append(c)

    # --- uid_key_columns: suggest first few identifying columns
    uid_candidates = [c for c in cols if re.search(r"id|key|uid|booking|reservation|order", c, re.I)]
    uid_key = uid_candidates[:3] if uid_candidates else cols[:2]

    return {
        "time_column":      time_col,
        "revenue_formula":  rev_formula,
        "ml_target_column": target_col,
        "ml_features":      features,
        "uid_key_columns":  uid_key,
    }


@app.post("/domain/upload_csv", tags=["domain"])
async def domain_upload_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a CSV file for a new domain.
    Saves to data/{filename} and returns auto-suggested column mapping + preview.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=5000)  # read up to 5000 rows for suggestion
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {exc}") from exc

    # Save full file to data/
    dest = DATA_DIR / file.filename
    dest.write_bytes(raw)
    logger.info("Uploaded CSV saved → %s  (%d rows preview)", dest, len(df))

    suggestions = _suggest_column_mapping(df)

    preview_records = df.head(5).where(pd.notnull(df), None).to_dict(orient="records")

    return {
        "filename":  file.filename,
        "saved_to":  str(dest),
        "n_rows":    "unknown (full file saved; preview on 5000 rows)",
        "n_cols":    len(df.columns),
        "columns":   list(df.columns),
        "dtypes":    {c: str(df[c].dtype) for c in df.columns},
        "suggestions": suggestions,
        "preview":   preview_records,
    }


# ────────────────────────────────────────────────────────────────────────────
# POST /domain/confirm_mapping
# ────────────────────────────────────────────────────────────────────────────

@app.post("/domain/confirm_mapping", tags=["domain"])
async def domain_confirm_mapping(request: ConfirmMappingRequest) -> dict[str, Any]:
    """
    Confirm column mapping, generate domain config JSON, train ML model in-process,
    and hot-load the new domain into runtime state.
    """
    csv_path = DATA_DIR / request.filename
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found in data/. Upload it first via /domain/upload_csv.",
        )

    # Derive domain_id from domain_name (slugify)
    domain_id = re.sub(r"[^a-z0-9]+", "_", request.domain_name.lower()).strip("_")
    if not domain_id:
        raise HTTPException(status_code=422, detail="domain_name produces an empty domain_id.")

    # uid_key_columns: fallback to first 2 ml_features if not provided
    uid_key_columns = request.uid_key_columns or request.ml_features[:2]

    # Build config dict
    constraints: list[dict] = [{"type": "C1", "capacity_value": request.capacity_default, "alpha": 1.0}]
    config_dict = {
        "domain_id":                   domain_id,
        "domain_name":                 request.domain_name,
        "config_version":              "1.0",
        "decision_mode":               "row_level",
        "time_column":                 request.time_column,
        "time_builder":                None,
        "action_columns":              [],
        "resource_columns":            [],
        "id_column":                   None,
        "uid_strategy":                "hash",
        "uid_key_columns":             uid_key_columns,
        "revenue_formula":             request.revenue_formula,
        "cost_formula":                None,
        "ml_target_column":            request.ml_target_column,
        "ml_features":                 request.ml_features,
        "historical_policy":           "accept_all",
        "historical_policy_column":    None,
        "constraints":                 constraints,
        "cancellation_penalty_default":request.cancellation_penalty_default,
        "risk_enabled":                False,
        "lambda_risk_default":         request.lambda_risk_default,
        "dataset_path":                str(csv_path),
    }

    config_path = CONFIGS_DIR / f"{domain_id}_config.json"
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    logger.info("Saved domain config → %s", config_path)

    # Train model in-process
    model_trained   = False
    model_metrics_d: dict[str, Any] = {}
    train_error     = None

    try:
        from domain_config import load_domain_config
        from predictive_model import CancellationPredictor

        cfg_new   = load_domain_config(config_path)
        predictor = CancellationPredictor(domain_config=cfg_new)
        df_train  = predictor.load_raw()

        if cfg_new.ml_target_column not in df_train.columns:
            train_error = f"Target column '{cfg_new.ml_target_column}' not found in CSV."
        else:
            metrics = predictor.fit(df_train)
            model_path = f"model_{domain_id}.pkl"
            joblib.dump(predictor, model_path)
            model_trained   = True
            model_metrics_d = metrics
            logger.info(
                "[%s] Model trained → %s  AUC=%.4f  Brier=%.4f",
                domain_id, model_path, metrics["mean_auc"], metrics["mean_brier"],
            )
    except Exception as exc:
        train_error = str(exc)
        logger.error("[%s] Training failed: %s", domain_id, exc)

    # Hot-load domain into runtime state
    loaded = _load_domain(domain_id)
    if loaded:
        # Mark server ready if it wasn't already
        _state["ready"] = True
        logger.info("[%s] Hot-loaded into runtime state.", domain_id)

    return {
        "domain_id":    domain_id,
        "config_saved": True,
        "config_path":  str(config_path),
        "model_trained":model_trained,
        "model_metrics":model_metrics_d,
        "domain_loaded":loaded,
        "train_error":  train_error,
    }


# ────────────────────────────────────────────────────────────────────────────
# POST /scope/create
# ────────────────────────────────────────────────────────────────────────────

@app.post("/scope/create", tags=["scope"])
async def scope_create(request: CreateScopeRequest) -> dict[str, Any]:
    """Create a new draft scope for a domain."""
    _require_ready()
    cfg = _require_domain(request.domain_id)

    df_full = _state["datasets"].get(request.domain_id)
    if df_full is None:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset for domain '{request.domain_id}' not loaded.",
        )

    sm: ScopeManager = _state["scope_manager"]
    params_dict = request.params.model_dump()

    scope = sm.create_scope(
        domain_config_id=request.domain_id,
        params=params_dict,
        selection_rule=request.selection_rule.model_dump(),
        df=df_full,
        cfg=cfg,
        seed=request.seed,
    )

    return {
        "scope_id":  scope.scope_id,
        "status":    scope.status,
        "n_uids":    scope.n_uids,
        "domain_id": scope.domain_config_id,
        "created_at":scope.created_at,
        "params_snapshot": scope.params_snapshot,
    }


# ────────────────────────────────────────────────────────────────────────────
# POST /scope/clone
# ────────────────────────────────────────────────────────────────────────────

@app.post("/scope/clone", tags=["scope"])
async def scope_clone(request: CloneScopeRequest) -> dict[str, Any]:
    """Clone an existing scope with new parameters (creates a fresh draft)."""
    _require_ready()
    scope = _require_scope(request.scope_id)
    sm: ScopeManager = _state["scope_manager"]
    new_scope, warning = sm.clone_scope(request.scope_id, request.new_params)

    return {
        "new_scope_id":    new_scope.scope_id,
        "original_scope_id": request.scope_id,
        "status":          new_scope.status,
        "n_uids":          new_scope.n_uids,
        "params_snapshot": new_scope.params_snapshot,
        "warning":         warning,
    }


# ────────────────────────────────────────────────────────────────────────────
# GET /scope/list
# ────────────────────────────────────────────────────────────────────────────

@app.get("/scope/list", tags=["scope"])
async def scope_list() -> dict[str, Any]:
    """List all scopes."""
    sm: ScopeManager = _state["scope_manager"]
    return {"scopes": sm.list_scopes()}


# ────────────────────────────────────────────────────────────────────────────
# GET /scope/{scope_id}
# ────────────────────────────────────────────────────────────────────────────

@app.get("/scope/{scope_id}", tags=["scope"])
async def scope_get(scope_id: str) -> dict[str, Any]:
    """Return scope metadata."""
    scope = _require_scope(scope_id)
    return scope.to_dict()


# ────────────────────────────────────────────────────────────────────────────
# POST /optimize
# ────────────────────────────────────────────────────────────────────────────

@app.post("/optimize", tags=["optimization"])
async def optimize(request: OptimizeRequest) -> dict[str, Any]:
    """
    Run MILP optimisation.

    Format A (scope-based):
        {"scope_id": "...", "cancellation_penalty": 50, "capacity": 100}
        → Uses scope's selected UIDs; locks scope on success.

    Format B (explicit domain, scopeless):
        {"domain_id": "hotel_v1", "n_samples": 100, "cancellation_penalty": 50, "capacity": 100}
        → Samples n_samples rows; no scope created/locked.

    Format C (direct params, no scope_id, no domain_id):
        {"cancellation_penalty": 100, "capacity": 100, "lambda_risk": 0, "n_samples": 100}
        → Defaults to domain_id="hotel_v1"; same as Format B.
    """
    _require_ready()

    # ── Unwrap legacy params envelope if present ──────────────────────────────
    # Allows {"params": {"cancellation_penalty": X, "capacity": Y, ...}}
    if request.params:
        p = request.params
        if "cancellation_penalty" in p:
            request.cancellation_penalty = float(p["cancellation_penalty"])
        if "capacity" in p:
            request.capacity = float(p["capacity"])
        if "lambda_risk" in p:
            request.lambda_risk = float(p["lambda_risk"])
        if "n_samples" in p:
            request.n_samples = int(p["n_samples"])

    # ── Determine mode ────────────────────────────────────────────────────────
    use_scope = request.scope_id is not None

    if use_scope:
        # ── Format A: scope-based ─────────────────────────────────────────────
        scope     = _require_scope(request.scope_id)
        domain_id = scope.domain_config_id
        cfg       = _require_domain(domain_id)
        predictor, engine = _require_predictor(domain_id)

        snap        = scope.params_snapshot
        penalty     = request.cancellation_penalty if request.cancellation_penalty != 50.0 \
                      else snap.get("cancellation_penalty", cfg.cancellation_penalty_default)
        capacity    = request.capacity if request.capacity != 100.0 \
                      else snap.get("capacity", 100.0)
        lambda_risk = request.lambda_risk if request.lambda_risk != 0.0 \
                      else snap.get("lambda_risk", cfg.lambda_risk_default)

        df = _scope_df(scope, domain_id)

    else:
        # ── Format B / C: scopeless ───────────────────────────────────────────
        # Format C: no domain_id → default to "hotel_v1"
        domain_id = request.domain_id or "hotel_v1"
        cfg       = _require_domain(domain_id)
        predictor, engine = _require_predictor(domain_id)
        scope     = None

        penalty     = request.cancellation_penalty
        capacity    = request.capacity
        lambda_risk = request.lambda_risk

        df_full = _state["datasets"].get(domain_id)
        if df_full is None:
            raise HTTPException(status_code=503, detail=f"Dataset for '{domain_id}' not loaded.")
        n = min(request.n_samples, len(df_full))
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(df_full), size=n, replace=False)
        df  = df_full.iloc[idx].copy().reset_index(drop=True)

    # ── Shared optimisation logic ─────────────────────────────────────────────
    params = OptimizationParams(
        capacity=capacity,
        cancellation_penalty=penalty,
        lambda_risk=lambda_risk,
    )

    outcome_avail = _outcome_available(df, cfg)

    logger.info(
        "/optimize %s domain=%s n=%d capacity=%.0f penalty=%.2f lambda=%.2f",
        f"scope={request.scope_id}" if use_scope else f"(no scope) n_samples={request.n_samples}",
        domain_id, len(df), capacity, penalty, lambda_risk,
    )

    try:
        p_cancel = predictor.predict_proba(df)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    uids = df["_uid"].tolist() if "_uid" in df.columns else [str(i) for i in range(len(df))]

    try:
        result: OptimizationResult = engine.optimize(df, p_cancel, params, cfg=cfg, uids=uids)
    except Exception as exc:
        logger.exception("Optimisation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Realized profit (only if outcomes available)
    gross_rev = evaluate_revenue(df, cfg.revenue_formula)
    total_realized: float | None = None
    baseline_realized: float | None = None

    if outcome_avail:
        y_true = df[cfg.ml_target_column].values.astype(int)
        total_realized    = round(_realized_profit(result.decisions, y_true, gross_rev, penalty), 2)
        baseline_realized = round(_realized_profit(
            np.ones(len(df), dtype=int), y_true, gross_rev, penalty
        ), 2)
        diff = total_realized - baseline_realized
        logger.info(
            "[%s] AI realized=%.0f  Baseline=%.0f  delta=%+.0f (%s)",
            f"scope={request.scope_id}" if use_scope else "no-scope",
            total_realized, baseline_realized, diff,
            "AI wins" if diff >= 0 else "Baseline wins",
        )

    # Charts
    p_cancel_list  = p_cancel.tolist()
    decisions_list = result.decisions.tolist()
    charts: dict[str, str] = {}
    try:
        charts["profit_composition"]  = profit_composition_chart(
            result.n_accepted, result.n_rejected,
            result.total_expected_revenue, total_realized,
        )
        charts["capacity_utilization"]  = capacity_utilization_chart(result.n_accepted, capacity)
        charts["decision_distribution"] = decision_distribution_chart(p_cancel_list, decisions_list)
        if outcome_avail:
            expected_list = result.expected_values.tolist()
            realized_list = [
                float(gross_rev[i]) if (result.decisions[i] == 1 and df[cfg.ml_target_column].iloc[i] == 0)
                else (-penalty if result.decisions[i] == 1 else 0.0)
                for i in range(len(df))
            ]
            charts["expected_vs_realized"] = expected_vs_realized_chart(
                expected_list[:50], realized_list[:50], outcome_avail
            )
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

    # Per-booking decisions list
    decisions_out = []
    for i in range(len(df)):
        accepted = bool(result.decisions[i])
        real_val: float | None = None
        if outcome_avail:
            y_i = int(df[cfg.ml_target_column].iloc[i])
            real_val = round(float(gross_rev[i]) if (accepted and y_i == 0)
                             else (-penalty if accepted else 0.0), 2)
        decisions_out.append({
            "uid":             uids[i],
            "accept":          accepted,
            "p_cancel":        round(float(p_cancel[i]), 4),
            "expected_value":  round(float(result.expected_values[i]), 2),
            "gross_revenue":   round(float(gross_rev[i]), 2),
            "realized_profit": real_val,
            "risk_label": (
                "high"   if p_cancel[i] > 0.60 else
                "medium" if p_cancel[i] > 0.35 else "low"
            ),
        })

    params_used = {
        "capacity":             capacity,
        "cancellation_penalty": penalty,
        "lambda_risk":          lambda_risk,
    }

    # ── Format A only: lock scope and cache snapshot ──────────────────────────
    scope_id_out   = request.scope_id
    scope_status   = None
    if use_scope:
        sm: ScopeManager = _state["scope_manager"]
        sm.lock_scope(scope.scope_id)
        scope_status = "locked"

        snapshot = {
            "scope_id":               scope.scope_id,
            "domain_id":              domain_id,
            "n_accepted":             result.n_accepted,
            "n_rejected":             result.n_rejected,
            "total_expected_revenue": round(result.total_expected_revenue, 2),
            "total_realized_profit":  total_realized,
            "baseline_realized_profit": baseline_realized,
            "expected_occupancy":     round(result.expected_occupancy, 2),
            "solver_status":          result.solver_status,
            "lambda_status":          result.lambda_status,
            "outcome_available":      outcome_avail,
            "charts":                 charts,
            "params_used":            params_used,
        }
        _state["results"][scope.scope_id] = snapshot

    return {
        "scope_id":               scope_id_out,
        "scope_status":           scope_status,
        "domain_id":              domain_id,
        "n_bookings":             len(df),
        "n_accepted":             result.n_accepted,
        "n_rejected":             result.n_rejected,
        "total_expected_revenue": round(result.total_expected_revenue, 2),
        "total_realized_profit":  total_realized,
        "baseline_realized_profit": baseline_realized,
        "expected_occupancy":     round(result.expected_occupancy, 2),
        "solver_status":          result.solver_status,
        "lambda_status":          result.lambda_status,
        "outcome_available":      outcome_avail,
        "params_used":            params_used,
        "charts":                 charts,
        "decisions":              decisions_out,
    }


# ────────────────────────────────────────────────────────────────────────────
# POST /backtest
# ────────────────────────────────────────────────────────────────────────────

@app.post("/backtest", tags=["backtesting"])
async def backtest(request: BacktestRequest) -> dict[str, Any]:
    """
    Rolling walk-forward backtest: AI strategy vs accept-all baseline.
    Returns 409 if the scope is not locked.
    Returns 422 if no ground-truth outcomes are available.
    """
    _require_ready()
    scope = _require_scope(request.scope_id)
    _require_locked(scope)

    domain_id = scope.domain_config_id
    cfg       = _require_domain(domain_id)
    predictor, engine = _require_predictor(domain_id)

    df_full = _state["datasets"].get(domain_id)
    if df_full is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    outcome_avail = _outcome_available(
        df_full[df_full["_uid"].isin(scope.selected_uids)] if "_uid" in df_full.columns else df_full,
        cfg,
    )
    if not outcome_avail:
        raise HTTPException(
            status_code=422,
            detail={
                "error":      "outcome_not_available",
                "message_en": "Backtesting requires real historical outcomes. "
                              "Cannot run on data without ml_target_column.",
                "message_it": "Il backtesting richiede outcome storici reali. "
                              "Impossibile eseguire su dati senza ml_target_column.",
            },
        )

    snap    = scope.params_snapshot
    params  = OptimizationParams(
        capacity=snap.get("capacity", 100.0),
        cancellation_penalty=snap.get("cancellation_penalty", cfg.cancellation_penalty_default),
        lambda_risk=snap.get("lambda_risk", cfg.lambda_risk_default),
    )
    model_metrics = _state["model_metrics"].get(domain_id, {})

    bt = Backtester(predictor, engine)
    try:
        summary = bt.run(
            df_full,
            params,
            cfg=cfg,
            scope=scope,
            outcome_available=True,
            n_splits=request.n_splits,
            n_samples_per_fold=request.n_samples_per_fold,
            model_metrics=model_metrics,
        )
    except ValueError as exc:
        if "outcome_not_available" in str(exc):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sm: ScopeManager = _state["scope_manager"]
    sm.mark_backtest_completed(scope.scope_id)

    result_dict = summary.to_dict()
    _state["backtests"][scope.scope_id] = result_dict

    return result_dict


# ────────────────────────────────────────────────────────────────────────────
# POST /sensitivity
# ────────────────────────────────────────────────────────────────────────────

@app.post("/sensitivity", tags=["analysis"])
async def sensitivity(request: SensitivityRequest) -> dict[str, Any]:
    """
    Sensitivity sweep: penalty (50–200%), capacity (±20%), lambda (0–3×).
    Can be called with scope_id (uses scope domain) or domain_id directly.
    """
    _require_ready()

    # Resolve domain
    if request.scope_id:
        scope     = _require_scope(request.scope_id)
        domain_id = scope.domain_config_id
    elif request.domain_id:
        domain_id = request.domain_id
        scope     = None
    else:
        raise HTTPException(status_code=422, detail="Provide scope_id or domain_id.")

    cfg               = _require_domain(domain_id)
    predictor, engine = _require_predictor(domain_id)
    df_full           = _state["datasets"].get(domain_id)
    if df_full is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    # Sample from scope or full dataset
    if scope is not None and "_uid" in df_full.columns:
        df_base = df_full[df_full["_uid"].isin(scope.selected_uids)].copy()
    else:
        df_base = df_full.copy()

    rng      = np.random.default_rng(RANDOM_SEED)
    n_sample = min(request.n_sample, len(df_base))
    idx      = rng.choice(len(df_base), size=n_sample, replace=False)
    df_sample = df_base.iloc[idx].copy().reset_index(drop=True)

    try:
        p_cancel = predictor.predict_proba(df_sample)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _run(params: OptimizationParams) -> dict[str, Any]:
        r = engine.optimize(df_sample, p_cancel, params, cfg=cfg)
        return {
            "total_expected_revenue": round(r.total_expected_revenue, 2),
            "n_accepted":             r.n_accepted,
            "expected_occupancy":     round(r.expected_occupancy, 2),
        }

    # 1. Penalty sensitivity: ×0.5 → ×2.0 (7 steps)
    penalty_sens = []
    for m in np.linspace(0.5, 2.0, 7):
        p    = request.base_penalty * float(m)
        entry = _run(OptimizationParams(
            capacity=request.capacity,
            cancellation_penalty=p,
            lambda_risk=request.base_lambda,
        ))
        entry["penalty_multiplier"] = round(float(m), 4)
        entry["penalty_value"]      = round(p, 2)
        penalty_sens.append(entry)

    # 2. Capacity sensitivity: ×0.8 → ×1.2 (9 steps)
    cap_sens = []
    for m in np.linspace(0.8, 1.2, 9):
        c     = request.capacity * float(m)
        entry = _run(OptimizationParams(
            capacity=c,
            cancellation_penalty=request.base_penalty,
            lambda_risk=request.base_lambda,
        ))
        entry["capacity_multiplier"] = round(float(m), 4)
        entry["capacity_value"]      = round(c, 2)
        cap_sens.append(entry)

    # 3. Lambda sensitivity: 0 → 3 (7 steps)
    lam_sens = []
    for lam in np.linspace(0.0, 3.0, 7):
        entry = _run(OptimizationParams(
            capacity=request.capacity,
            cancellation_penalty=request.base_penalty,
            lambda_risk=float(lam),
        ))
        entry["lambda"] = round(float(lam), 4)
        lam_sens.append(entry)

    # Charts
    charts: dict[str, str] = {}
    try:
        charts["penalty_sensitivity"]  = penalty_sensitivity_chart(penalty_sens)
        charts["capacity_sensitivity"] = capacity_sensitivity_chart(cap_sens)
    except Exception as exc:
        logger.warning("Sensitivity chart failed: %s", exc)

    result = {
        "penalty_sensitivity":  penalty_sens,
        "capacity_sensitivity": cap_sens,
        "lambda_sensitivity":   lam_sens,
        "charts":               charts,
        "params": {
            "base_capacity": request.capacity,
            "base_penalty":  request.base_penalty,
            "base_lambda":   request.base_lambda,
            "n_sample":      n_sample,
        },
    }

    cache_key = request.scope_id or request.domain_id or "global"
    _state["sensitivities"][cache_key] = result
    return result


# ────────────────────────────────────────────────────────────────────────────
# POST /report
# ────────────────────────────────────────────────────────────────────────────

@app.post(
    "/report",
    tags=["reporting"],
    responses={200: {"content": {"application/pdf": {}}}},
)
async def report(request: ReportRequest) -> Response:
    """
    Generate and download a PDF report for a locked scope.
    Returns 409 if the scope is not locked.
    """
    _require_ready()
    scope = _require_scope(request.scope_id)
    _require_locked(scope)

    domain_id = scope.domain_config_id
    cfg       = _require_domain(domain_id)

    lang            = request.lang if request.lang in ("en", "it") else "en"
    model_metrics   = _state["model_metrics"].get(domain_id, {})
    opt_snapshot    = _state["results"].get(scope.scope_id)
    bt_dict         = _state["backtests"].get(scope.scope_id)
    sensitivity     = _state["sensitivities"].get(scope.scope_id) or \
                      _state["sensitivities"].get(domain_id)

    if opt_snapshot is None:
        raise HTTPException(
            status_code=422,
            detail="No optimization result cached for this scope. Run /optimize first.",
        )

    outcome_avail = opt_snapshot.get("outcome_available", True)

    try:
        pdf_bytes = generate_report(
            lang=lang,
            model_metrics=model_metrics,
            backtest_summary=bt_dict,
            sensitivity_results=sensitivity,
            optimization_snapshot=opt_snapshot,
            scope_dict=scope.to_dict(),
            outcome_available=outcome_avail,
        )
    except Exception as exc:
        logger.exception("PDF generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = f"optide_report_{scope.scope_id}_{lang}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ────────────────────────────────────────────────────────────────────────────
# POST /historical_baseline
# ────────────────────────────────────────────────────────────────────────────

@app.post("/historical_baseline", tags=["analysis"])
async def historical_baseline(request: HistoricalBaselineRequest) -> dict[str, Any]:
    """
    Accept-all baseline using actual outcome data.

    With scope_id: uses the scope's selected UIDs (scope must exist).
    Without scope_id: samples n_samples rows from hotel_v1 (default domain).
    Returns 422 if outcome data is unavailable.
    """
    _require_ready()

    if request.scope_id is not None:
        # ── Scope-based path ──────────────────────────────────────────────────
        scope     = _require_scope(request.scope_id)
        domain_id = scope.domain_config_id
        cfg       = _require_domain(domain_id)
        df        = _scope_df(scope, domain_id)
        penalty   = request.cancellation_penalty
        scope_id_out = scope.scope_id
    else:
        # ── Scopeless path (default domain = hotel_v1) ────────────────────────
        domain_id = "hotel_v1"
        cfg       = _require_domain(domain_id)
        df_full   = _state["datasets"].get(domain_id)
        if df_full is None:
            raise HTTPException(status_code=503, detail=f"Dataset for '{domain_id}' not loaded.")
        n = min(request.n_samples, len(df_full))
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(df_full), size=n, replace=False)
        df  = df_full.iloc[idx].copy().reset_index(drop=True)
        penalty      = request.cancellation_penalty
        scope_id_out = None

    if not _outcome_available(df, cfg):
        raise HTTPException(
            status_code=422,
            detail="Outcome data not available for this dataset (real-time data).",
        )

    gross_rev     = evaluate_revenue(df, cfg.revenue_formula)
    y_true        = df[cfg.ml_target_column].values.astype(int)
    n             = len(df)
    realized      = _realized_profit(np.ones(n, dtype=int), y_true, gross_rev, penalty)
    cancellations = int(y_true.sum())

    logger.info(
        "[baseline %s] n=%d  cancellations=%d  realized=%.0f",
        f"scope={scope_id_out}" if scope_id_out else "no-scope",
        n, cancellations, realized,
    )

    return {
        "scope_id":                  scope_id_out,
        "domain_id":                 domain_id,
        "bookings_accepted":         n,
        "total_realized_profit":     round(realized, 2),
        "total_gross_revenue":       round(float(gross_rev.sum()), 2),
        "realized_cancellations":    cancellations,
        "cancellation_penalty_used": penalty,
    }


# ────────────────────────────────────────────────────────────────────────────
# GET /report  (backward-compat alias — GET /report?scope_id=&lang=)
# ────────────────────────────────────────────────────────────────────────────

@app.get(
    "/report",
    tags=["reporting"],
    responses={200: {"content": {"application/pdf": {}}}},
)
async def report_get(
    scope_id: str = Query(...),
    lang: str     = Query(default="en", pattern="^(en|it)$"),
) -> Response:
    """GET alias for /report (query-param version)."""
    return await report(ReportRequest(scope_id=scope_id, lang=lang))


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
