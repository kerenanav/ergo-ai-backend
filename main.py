"""
main.py — Ergo.ai Decision Intelligence Engine  (FastAPI backend)

Endpoints
─────────
GET  /health        — liveness + model-readiness check
POST /predict       — P(cancel) per booking  [Layer 1]
POST /optimize      — MILP accept/reject decisions  [Layer 2]
POST /backtest      — rolling walk-forward backtest  [Layer 3]
POST /sensitivity   — sensitivity sweep (penalty / capacity / lambda)
GET  /report        — PDF report (lang=en|it)

Start the server (either way works):
    python main.py
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

The hotel_bookings.csv file must be in the working directory.
Download from: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
import joblib
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backtester import Backtester, BacktestSummary
from constraint_compiler import OptimizationParams
from decision_engine import DecisionEngine, OptimizationResult
from predictive_model import CancellationPredictor
from report_generator import generate_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ergo_ai")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = "hotel_bookings.csv"
MODEL_PATH  = "model.pkl"
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Application state (populated during startup lifespan)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan: load data and train model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pre-trained model and prepare all singletons at startup."""
    logger.info("Ergo.ai starting up …")

    if not os.path.exists(DATA_PATH):
        logger.warning(
            "Dataset not found at '%s'. "
            "Download hotel_bookings.csv from Kaggle (jessemostipak/hotel-booking-demand) "
            "and place it in the working directory.",
            DATA_PATH,
        )
        _state["ready"] = False
    else:
        if os.path.exists(MODEL_PATH):
            logger.info("Loading pre-trained model from %s …", MODEL_PATH)
            predictor = joblib.load(MODEL_PATH)
        else:
            logger.warning("model.pkl not found — training from scratch (slow) …")
            predictor = CancellationPredictor(data_path=DATA_PATH)
            df_train  = predictor.load_raw()
            predictor.fit(df_train)

        engine     = DecisionEngine()
        backtester = Backtester(predictor, engine)

        df = predictor.load_raw()
        model_metrics = {
            "cv_metrics": predictor.cv_metrics,
            "mean_auc":   round(float(np.mean([m["auc"]   for m in predictor.cv_metrics])), 4),
            "mean_brier": round(float(np.mean([m["brier"] for m in predictor.cv_metrics])), 4),
        }

        _state.update(
            predictor=predictor,
            engine=engine,
            backtester=backtester,
            df=df,
            model_metrics=model_metrics,
            ready=True,
            last_backtest=None,
            last_sensitivity=None,
            last_optimize=None,
        )
        logger.info(
            "Model ready — Mean AUC=%.4f  Brier=%.4f  rows=%d",
            model_metrics["mean_auc"],
            model_metrics["mean_brier"],
            len(df),
        )

    yield

    logger.info("Ergo.ai shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ergo.ai — Decision Intelligence Engine",
    description=(
        "Hotel booking demand optimisation via calibrated LightGBM + MILP (HiGHS). "
        "Endpoints: /health, /predict, /optimize, /backtest, /sensitivity, /report"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NgrokHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response

app.add_middleware(NgrokHeaderMiddleware)



def _require_ready() -> None:
    if not _state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not ready. "
                "Place hotel_bookings.csv in the working directory and restart the server."
            ),
        )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class BookingFeatures(BaseModel):
    """Features available at booking time (no post-facto information)."""

    hotel: str                        = "City Hotel"
    lead_time: float                  = 30
    arrival_date_year: int            = 2016
    arrival_date_month: str           = "July"
    arrival_date_week_number: int     = 27
    arrival_date_day_of_month: int    = 1
    stays_in_weekend_nights: float    = 1
    stays_in_week_nights: float       = 2
    adults: float                     = 2
    children: float                   = 0
    babies: float                     = 0
    meal: str                         = "BB"
    country: str                      = "PRT"
    market_segment: str               = "Online TA"
    distribution_channel: str         = "TA/TO"
    is_repeated_guest: int            = 0
    previous_cancellations: float     = 0
    previous_bookings_not_canceled: float = 0
    reserved_room_type: str           = "A"
    booking_changes: float            = 0
    deposit_type: str                 = "No Deposit"
    agent: float                      = 0
    company: float                    = 0
    days_in_waiting_list: float       = 0
    customer_type: str                = "Transient"
    adr: float                        = 80.0
    required_car_parking_spaces: float = 0
    total_of_special_requests: float  = 0
    total_nights: Optional[float]     = None  # computed if absent

    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    bookings: list[BookingFeatures] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    n_bookings: int
    predictions: list[dict[str, Any]]


class OptimizeRequest(BaseModel):
    bookings: list[BookingFeatures] = Field(default=[], description="Ignored: dataset sample is used instead")
    cancellation_penalty: float     = 50.0
    capacity: float                 = 100.0
    lambda_risk: float              = 1.0
    n_samples: int                  = Field(default=100, ge=10, le=500, description="Number of bookings to sample from the dataset")


class OptimizeResponse(BaseModel):
    n_bookings: int
    n_accepted: int
    n_rejected: int
    total_expected_revenue: float
    total_realized_profit: float
    expected_occupancy: float
    solver_status: str
    params_used: dict[str, float]
    decisions: list[dict[str, Any]]


class BacktestRequest(BaseModel):
    n_splits: int               = Field(default=5,    ge=2,  le=10)
    n_samples: int              = Field(default=100,  ge=10, le=500, description="Bookings sampled per time window")
    capacity: float             = Field(default=100.0, ge=1)
    cancellation_penalty: float = Field(default=50.0, ge=0)
    lambda_risk: float          = Field(default=1.0,  ge=0)


class SensitivityRequest(BaseModel):
    capacity: float            = Field(default=100.0, ge=1)
    base_penalty: float        = Field(default=50.0,  ge=0)
    base_lambda: float         = Field(default=1.0,   ge=0)
    n_sample: int              = Field(default=500,   ge=50, le=5000,
                                       description="Bookings sampled from dataset for speed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bookings_to_df(bookings: list[BookingFeatures]) -> pd.DataFrame:
    """Convert a list of Pydantic booking models to a pandas DataFrame."""
    rows = []
    for b in bookings:
        d = b.model_dump()
        if d.get("total_nights") is None:
            d["total_nights"] = max(
                1.0,
                float(d.get("stays_in_weekend_nights", 0) or 0)
                + float(d.get("stays_in_week_nights",  0) or 0),
            )
        rows.append(d)
    return pd.DataFrame(rows)


def _opt_result_to_dict(
    result: OptimizationResult,
    y_true: np.ndarray | None = None,
    gross_rev: np.ndarray | None = None,
    penalty: float = 0.0,
) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(result.decisions)):
        accepted = bool(result.decisions[i])
        if y_true is not None and gross_rev is not None and accepted:
            realized = float(gross_rev[i]) if y_true[i] == 0 else -penalty
        else:
            realized = 0.0
        rows.append({
            "booking_id":       i,
            "accept":           accepted,
            "p_cancel":         round(float(result.p_cancel[i]), 4),
            "p_no_cancel":      round(float(1.0 - result.p_cancel[i]), 4),
            "expected_value":   round(float(result.expected_values[i]), 2),
            "gross_revenue":    round(float(result.gross_revenues[i]), 2),
            "realized_profit":  round(realized, 2),
            "risk_label":       (
                "high"   if result.p_cancel[i] > 0.60
                else "medium" if result.p_cancel[i] > 0.35
                else "low"
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"], summary="Liveness and readiness check")
async def health() -> dict[str, Any]:
    """Return the service status and current model metrics."""
    metrics = _state.get("model_metrics") or {}
    return {
        "status":         "ready" if _state.get("ready") else "not_ready",
        "dataset_loaded": _state.get("ready", False),
        "model_metrics":  metrics,
        "dataset_rows":   len(_state["df"]) if _state.get("ready") else None,
    }


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["ml"],
    summary="Predict cancellation probability for each booking [Layer 1]",
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Return P(cancel) and risk label for each supplied booking."""
    _require_ready()

    predictor: CancellationPredictor = _state["predictor"]
    df = _bookings_to_df(request.bookings)

    try:
        p_cancel = predictor.predict_from_raw(df)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    predictions = [
        {
            "booking_id":  i,
            "p_cancel":    round(float(p), 4),
            "p_no_cancel": round(float(1.0 - p), 4),
            "risk_label":  "high" if p > 0.60 else "medium" if p > 0.35 else "low",
        }
        for i, p in enumerate(p_cancel)
    ]

    return PredictResponse(n_bookings=len(predictions), predictions=predictions)


# ---------------------------------------------------------------------------
# POST /optimize
# ---------------------------------------------------------------------------

@app.post(
    "/optimize",
    response_model=OptimizeResponse,
    tags=["optimization"],
    summary="MILP booking-acceptance optimisation [Layer 2]",
)
async def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """
    Run the HiGHS MILP optimiser to maximise expected revenue subject
    to capacity constraints.  Returns an accept/reject decision per booking.
    """
    _require_ready()

    print(f"PARAMETRI RICEVUTI: capacity={request.capacity}, cancellation_penalty={request.cancellation_penalty}, lambda_risk={request.lambda_risk}, n_samples={request.n_samples}", flush=True)
    logger.info(
        "/optimize called — capacity=%.0f  penalty=%.2f  lambda_risk=%.2f  n_samples=%d",
        request.capacity, request.cancellation_penalty, request.lambda_risk, request.n_samples,
    )

    predictor: CancellationPredictor = _state["predictor"]
    engine:    DecisionEngine        = _state["engine"]
    df_full:   pd.DataFrame          = _state["df"]

    # Sample n_samples real bookings from the loaded dataset (bookings from frontend are ignored)
    n_sample = min(request.n_samples, len(df_full))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(df_full), size=n_sample, replace=False)
    df = df_full.iloc[idx].copy().reset_index(drop=True)

    try:
        p_cancel = predictor.predict_proba(df)
    except Exception as exc:
        logger.exception("Prediction step failed inside /optimize")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    params = OptimizationParams(
        capacity=request.capacity,
        cancellation_penalty=request.cancellation_penalty,
        lambda_risk=request.lambda_risk,
    )

    try:
        result: OptimizationResult = engine.optimize(df, p_cancel, params)
    except Exception as exc:
        logger.exception("Optimisation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Realized profit using actual is_canceled outcomes ──────────────────
    adr_arr = df["adr"].fillna(0).values.astype(np.float64)
    if "total_nights" in df.columns:
        nights_arr = df["total_nights"].fillna(0).values.astype(np.float64)
    elif "stays_in_week_nights" in df.columns:
        nights_arr = (
            df["stays_in_weekend_nights"].fillna(0)
            + df["stays_in_week_nights"].fillna(0)
        ).values.astype(np.float64)
    else:
        nights_arr = np.ones(len(df), dtype=np.float64)
    nights_arr = np.maximum(nights_arr, 1.0)
    gross_rev_arr = adr_arr * nights_arr

    y_true_arr = (
        df["is_canceled"].values.astype(int)
        if "is_canceled" in df.columns
        else np.zeros(len(df), dtype=int)
    )

    # AI realized profit: only accepted bookings count
    total_realized_profit = 0.0
    for i, dec in enumerate(result.decisions):
        if dec == 1:
            if y_true_arr[i] == 0:
                total_realized_profit += float(gross_rev_arr[i])
            else:
                total_realized_profit -= request.cancellation_penalty

    # Baseline realized profit: accept-all on same sample
    baseline_realized_profit = 0.0
    for i in range(len(df)):
        if y_true_arr[i] == 0:
            baseline_realized_profit += float(gross_rev_arr[i])
        else:
            baseline_realized_profit -= request.cancellation_penalty

    diff = total_realized_profit - baseline_realized_profit
    print(
        f"\n[AI vs Baseline — same {len(df)} bookings]\n"
        f"  AI       : accepted={result.n_accepted:>4}/{len(df)}  "
        f"realized_profit={total_realized_profit:>10,.2f}\n"
        f"  Baseline : accepted={len(df):>4}/{len(df)}  "
        f"realized_profit={baseline_realized_profit:>10,.2f}\n"
        f"  Diff     : {diff:>+10,.2f}  "
        f"({'AI wins' if diff >= 0 else 'Baseline wins'})\n"
        f"  [actual cancellations in sample: {int(y_true_arr.sum())}/{len(df)}]",
        flush=True,
    )

    # Cache for report
    _state["last_optimize"] = {
        "n_accepted":             result.n_accepted,
        "n_rejected":             result.n_rejected,
        "total_expected_revenue": round(result.total_expected_revenue, 2),
        "expected_occupancy":     round(result.expected_occupancy, 2),
        "solver_status":          result.solver_status,
    }

    return OptimizeResponse(
        n_bookings=len(df),
        n_accepted=result.n_accepted,
        n_rejected=result.n_rejected,
        total_expected_revenue=round(result.total_expected_revenue, 2),
        total_realized_profit=round(total_realized_profit, 2),
        expected_occupancy=round(result.expected_occupancy, 2),
        solver_status=result.solver_status,
        params_used={
            "capacity":             request.capacity,
            "cancellation_penalty": request.cancellation_penalty,
            "lambda_risk":          request.lambda_risk,
        },
        decisions=_opt_result_to_dict(
            result,
            y_true=y_true_arr,
            gross_rev=gross_rev_arr,
            penalty=request.cancellation_penalty,
        ),
    )


# ---------------------------------------------------------------------------
# GET /dataset_info
# ---------------------------------------------------------------------------

@app.get("/dataset_info", tags=["system"], summary="Info sul dataset caricato")
async def dataset_info() -> dict[str, Any]:
    """Restituisce statistiche sul dataset hotel_bookings caricato in memoria."""
    _require_ready()

    df: pd.DataFrame = _state["df"]

    # Date range
    date_min = str(df["arrival_date"].min().date()) if "arrival_date" in df.columns else None
    date_max = str(df["arrival_date"].max().date()) if "arrival_date" in df.columns else None

    # Distribuzione per hotel
    hotel_dist = df["hotel"].value_counts().to_dict() if "hotel" in df.columns else {}

    # Distribuzione per market_segment
    segment_dist = (
        df["market_segment"].value_counts().to_dict()
        if "market_segment" in df.columns else {}
    )

    return {
        "total_rows":        len(df),
        "date_range":        {"min": date_min, "max": date_max},
        "hotel_distribution":   hotel_dist,
        "segment_distribution": segment_dist,
        "n_samples_range":   {"min": 10, "max": 500, "default": 100},
    }


# ---------------------------------------------------------------------------
# POST /historical_baseline
# ---------------------------------------------------------------------------

class HistoricalBaselineRequest(BaseModel):
    n_samples: int            = Field(default=100, ge=10, le=500)
    cancellation_penalty: float = Field(default=50.0, ge=0)


@app.post(
    "/historical_baseline",
    tags=["analysis"],
    summary="Simulate accept-all historical baseline on a random sample",
)
async def historical_baseline(request: HistoricalBaselineRequest) -> dict[str, Any]:
    """
    Accept all n_samples bookings without optimisation (historical baseline).
    Uses actual is_canceled outcomes to compute realised profit/loss.
    """
    _require_ready()

    predictor: CancellationPredictor = _state["predictor"]
    df_full:   pd.DataFrame          = _state["df"]

    # Same fixed seed as /optimize for consistency
    n = min(request.n_samples, len(df_full))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(df_full), size=n, replace=False)
    df = df_full.iloc[idx].copy().reset_index(drop=True)

    # Gross revenue — same logic as /optimize for consistency
    adr = df["adr"].fillna(0).values.astype(np.float64)
    if "total_nights" in df.columns:
        nights = df["total_nights"].fillna(0).values.astype(np.float64)
    elif "stays_in_week_nights" in df.columns:
        nights = (
            df["stays_in_weekend_nights"].fillna(0)
            + df["stays_in_week_nights"].fillna(0)
        ).values.astype(np.float64)
    else:
        nights = np.ones(n, dtype=np.float64)
    nights = np.maximum(nights, 1.0)
    gross_rev = adr * nights

    # Realised outcome using actual is_canceled
    y_true = (
        df["is_canceled"].values.astype(int)
        if "is_canceled" in df.columns
        else np.zeros(n, dtype=int)
    )
    realized_cancellations = int(y_true.sum())

    # Baseline: accept ALL n bookings
    realized_profit = 0.0
    for i in range(n):
        if y_true[i] == 0:
            realized_profit += float(gross_rev[i])
        else:
            realized_profit -= request.cancellation_penalty

    print(
        f"\n[historical_baseline — {n} bookings, all accepted]\n"
        f"  cancellations : {realized_cancellations}/{n}\n"
        f"  gross_revenue : {float(gross_rev.sum()):>10,.2f}\n"
        f"  realized_profit: {realized_profit:>10,.2f}  "
        f"(losses from cancellations: {realized_cancellations} × {request.cancellation_penalty} "
        f"= {realized_cancellations * request.cancellation_penalty:,.2f})",
        flush=True,
    )

    return {
        "bookings_accepted":         n,
        "total_realized_profit":     round(realized_profit, 2),
        "total_gross_revenue":       round(float(gross_rev.sum()), 2),
        "realized_cancellations":    realized_cancellations,
        "cancellation_penalty_used": request.cancellation_penalty,
    }


# ---------------------------------------------------------------------------
# POST /backtest
# ---------------------------------------------------------------------------

@app.post(
    "/backtest",
    tags=["backtesting"],
    summary="Rolling walk-forward backtest: AI vs accept-all baseline [Layer 3]",
)
async def backtest(request: BacktestRequest) -> dict[str, Any]:
    """
    Temporal backtest: AI strategy vs accept-all baseline.

    The dataset is split into n_splits chronological windows.
    For each window, n_samples bookings are sampled, the pre-trained model
    predicts P(cancel), the MILP optimiser makes accept/reject decisions,
    and realised revenue is computed from actual is_canceled outcomes.
    """
    _require_ready()

    print(
        f"BACKTEST RICEVUTO: n_splits={request.n_splits}  n_samples={request.n_samples}"
        f"  capacity={request.capacity}  cancellation_penalty={request.cancellation_penalty}"
        f"  lambda_risk={request.lambda_risk}",
        flush=True,
    )
    logger.info(
        "/backtest called — n_splits=%d  n_samples=%d  capacity=%.0f  penalty=%.2f  lambda=%.2f",
        request.n_splits, request.n_samples,
        request.capacity, request.cancellation_penalty, request.lambda_risk,
    )

    backtester: Backtester  = _state["backtester"]
    df: pd.DataFrame        = _state["df"]
    model_metrics: dict     = _state.get("model_metrics", {})

    params = OptimizationParams(
        capacity=request.capacity,
        cancellation_penalty=request.cancellation_penalty,
        lambda_risk=request.lambda_risk,
    )

    try:
        summary: BacktestSummary = backtester.run(
            df,
            params,
            n_splits=request.n_splits,
            n_samples_per_fold=request.n_samples,
            model_metrics=model_metrics,
        )
    except Exception as exc:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Cache for /report
    _state["last_backtest"] = summary

    result_dict = summary.to_dict()

    return {
        "n_splits":              result_dict["n_splits"],
        "mean_auc":              result_dict["mean_auc"],
        "mean_brier":            result_dict["mean_brier"],
        "total_ai_revenue":      result_dict["total_ai_revenue"],
        "total_baseline_revenue":result_dict["total_baseline_revenue"],
        "total_improvement_pct": result_dict["total_improvement_pct"],
        "params":                result_dict["params"],
        "folds":                 result_dict["folds"],
    }


# ---------------------------------------------------------------------------
# POST /sensitivity
# ---------------------------------------------------------------------------

@app.post(
    "/sensitivity",
    tags=["analysis"],
    summary="Sensitivity sweep over penalty (50–200%), capacity (±20%), lambda (0–3×)",
)
async def sensitivity(request: SensitivityRequest) -> dict[str, Any]:
    """
    Resample *n_sample* bookings from the loaded dataset, compute P(cancel)
    once with the fully-trained model, then re-run the MILP optimiser across
    a grid of parameter values.

    Returns three sensitivity curves:
    - ``penalty_sensitivity``   : cancellation penalty ×0.5 → ×2.0
    - ``capacity_sensitivity``  : capacity ×0.8 → ×1.2
    - ``lambda_sensitivity``    : lambda 0.0 → 3.0
    """
    _require_ready()

    print(f"PARAMETRI RICEVUTI: {request}", flush=True)
    logger.info(
        "/sensitivity called — capacity=%.0f  base_penalty=%.2f  base_lambda=%.2f  n_sample=%d",
        request.capacity, request.base_penalty, request.base_lambda, request.n_sample,
    )

    predictor: CancellationPredictor = _state["predictor"]
    engine:    DecisionEngine        = _state["engine"]
    df:        pd.DataFrame          = _state["df"]

    # Random sample from training data (fixed seed → reproducible)
    rng       = np.random.default_rng(RANDOM_SEED)
    n_sample  = min(request.n_sample, len(df))
    idx       = rng.choice(len(df), size=n_sample, replace=False)
    df_sample = df.iloc[idx].copy().reset_index(drop=True)

    try:
        p_cancel = predictor.predict_proba(df_sample)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    base_params = OptimizationParams(
        capacity=request.capacity,
        cancellation_penalty=request.base_penalty,
        lambda_risk=request.base_lambda,
    )

    def _run_opt(params: OptimizationParams) -> dict[str, Any]:
        r = engine.optimize(df_sample, p_cancel, params)
        return {
            "total_expected_revenue": round(r.total_expected_revenue, 2),
            "n_accepted":             r.n_accepted,
            "expected_occupancy":     round(r.expected_occupancy, 2),
        }

    # ── 1. Penalty sensitivity: 50% → 200% in 7 steps ──────────────────
    penalty_mults = np.linspace(0.5, 2.0, 7)
    penalty_sens  = []
    for m in penalty_mults:
        p     = OptimizationParams(
            capacity=base_params.capacity,
            cancellation_penalty=base_params.cancellation_penalty * float(m),
            lambda_risk=base_params.lambda_risk,
        )
        entry = _run_opt(p)
        entry["penalty_multiplier"] = round(float(m), 4)
        entry["penalty_value"]      = round(float(base_params.cancellation_penalty * m), 2)
        penalty_sens.append(entry)

    # ── 2. Capacity sensitivity: ±20% in 9 steps ───────────────────────
    cap_mults    = np.linspace(0.8, 1.2, 9)
    capacity_sens = []
    for m in cap_mults:
        p     = OptimizationParams(
            capacity=base_params.capacity * float(m),
            cancellation_penalty=base_params.cancellation_penalty,
            lambda_risk=base_params.lambda_risk,
        )
        entry = _run_opt(p)
        entry["capacity_multiplier"] = round(float(m), 4)
        entry["capacity_value"]      = round(float(base_params.capacity * m), 2)
        capacity_sens.append(entry)

    # ── 3. Lambda sensitivity: 0× → 3× in 7 steps ──────────────────────
    lambdas      = np.linspace(0.0, 3.0, 7)
    lambda_sens  = []
    for lam in lambdas:
        eff_lam = max(float(lam), 1e-4)  # avoid degenerate zero-capacity constraint
        p       = OptimizationParams(
            capacity=base_params.capacity,
            cancellation_penalty=base_params.cancellation_penalty,
            lambda_risk=eff_lam,
        )
        entry         = _run_opt(p)
        entry["lambda"] = round(float(lam), 4)
        lambda_sens.append(entry)

    result = {
        "penalty_sensitivity":  penalty_sens,
        "capacity_sensitivity": capacity_sens,
        "lambda_sensitivity":   lambda_sens,
        "params": {
            "base_capacity": request.capacity,
            "base_penalty":  request.base_penalty,
            "base_lambda":   request.base_lambda,
            "n_sample":      n_sample,
        },
    }

    # Cache for /report
    _state["last_sensitivity"] = result
    return result


# ---------------------------------------------------------------------------
# GET /report
# ---------------------------------------------------------------------------

@app.get(
    "/report",
    tags=["reporting"],
    summary="Download a PDF report in English (en) or Italian (it)",
    responses={200: {"content": {"application/pdf": {}}}},
)
async def report(
    lang: str = Query(
        default="en",
        pattern="^(en|it)$",
        description="Report language: 'en' (English) or 'it' (Italian)",
    ),
) -> Response:
    """
    Generate and download a PDF report.

    The report includes:
    - Executive summary
    - Layer-1 model CV metrics
    - Layer-3 backtesting results (if ``/backtest`` was called)
    - Layer-2 optimization snapshot (if ``/optimize`` was called)
    - Sensitivity analysis (if ``/sensitivity`` was called)

    Call ``/backtest`` and ``/sensitivity`` first to populate all report sections.
    """
    _require_ready()

    model_metrics  = _state.get("model_metrics", {})
    last_backtest  = _state.get("last_backtest")
    last_sensitivity = _state.get("last_sensitivity")
    last_optimize  = _state.get("last_optimize")

    # Convert BacktestSummary dataclass → dict for the report generator
    bt_dict: dict[str, Any] | None = None
    if last_backtest is not None:
        bt_dict = (
            last_backtest.to_dict()
            if isinstance(last_backtest, BacktestSummary)
            else last_backtest
        )

    try:
        pdf_bytes = generate_report(
            lang=lang,
            model_metrics=model_metrics,
            backtest_summary=bt_dict,
            sensitivity_results=last_sensitivity,
            optimization_snapshot=last_optimize,
        )
    except Exception as exc:
        logger.exception("PDF generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = f"ergo_ai_report_{lang}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Entry point — python main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
