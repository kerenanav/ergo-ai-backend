"""
predictive_model.py — Layer 1: Calibrated ML model for P(cancel | features).

Design principles
-----------------
* No look-ahead bias: all features are known at booking time.
  Post-facto columns (reservation_status, reservation_status_date) are dropped.
* Temporal ordering: rows are sorted by arrival_date before any split.
* TimeSeriesSplit: folds respect chronological order — training always precedes
  the validation period.
* Platt scaling (CalibratedClassifierCV, method='sigmoid') converts raw LGBM
  scores to well-calibrated probabilities.
* All random seeds are fixed (RANDOM_SEED = 42).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED: int = 42
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------

# All features known at booking time (no reservation_status / _date)
FEATURE_COLS: list[str] = [
    "hotel",
    "lead_time",
    "arrival_date_month",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",           # derived: weekend + week nights
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_COLS: list[str] = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

MONTH_MAP: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# Columns to drop before any feature engineering (post-facto / target leakage)
LEAKY_COLS: list[str] = ["reservation_status", "reservation_status_date"]


# ---------------------------------------------------------------------------
# CancellationPredictor
# ---------------------------------------------------------------------------

class CancellationPredictor:
    """LightGBM + Platt scaling classifier for hotel cancellation probability.

    Typical usage
    -------------
    predictor = CancellationPredictor("hotel_bookings.csv")
    df        = predictor.load_raw()
    metrics   = predictor.fit(df)             # trains & calibrates
    p_cancel  = predictor.predict_proba(df)   # returns P(cancel) for each row
    """

    def __init__(self, data_path: str = "hotel_bookings.csv") -> None:
        self.data_path = Path(data_path)
        self.model: CalibratedClassifierCV | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.cv_metrics: list[dict[str, Any]] = []
        self.is_fitted: bool = False

    # -----------------------------------------------------------------------
    # Loading & preprocessing
    # -----------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """Load CSV from disk and return a preprocessed (training-ready) DataFrame."""
        df = pd.read_csv(self.data_path)
        logger.info("Loaded %d rows from %s", len(df), self.data_path)
        return self._preprocess_training(df)

    def _base_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shared preprocessing: null-filling, type coercions, derived features."""
        df = df.copy()

        # Drop leaky columns
        df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])

        # Fill nulls
        df["children"] = df["children"].fillna(0.0)
        df["country"]  = df["country"].fillna("Unknown")
        df["agent"]    = df["agent"].fillna(0.0)
        df["company"]  = df["company"].fillna(0.0)

        # Derived feature: total nights booked
        df["total_nights"] = (
            df["stays_in_weekend_nights"].fillna(0)
            + df["stays_in_week_nights"].fillna(0)
        ).clip(lower=0)

        # Build arrival_date for ordering (not used as a feature directly)
        if "arrival_date_year" in df.columns:
            df["_month_num"] = (
                df["arrival_date_month"]
                .map(MONTH_MAP)
                .fillna(1)
                .astype(int)
            )
            df["arrival_date"] = pd.to_datetime(
                {
                    "year":  df["arrival_date_year"],
                    "month": df["_month_num"],
                    "day":   df["arrival_date_day_of_month"],
                },
                errors="coerce",
            )
            df = df.drop(columns=["_month_num"])

        return df

    def _preprocess_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing for training: base transforms + temporal sort."""
        df = self._base_preprocess(df)
        if "arrival_date" in df.columns:
            df = df.sort_values("arrival_date").reset_index(drop=True)
        return df

    def _preprocess_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing for inference: base transforms only (no sort needed)."""
        return self._base_preprocess(df)

    # -----------------------------------------------------------------------
    # Encoding (label encoding for categoricals)
    # -----------------------------------------------------------------------

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        """Fit one LabelEncoder per categorical column on the full dataset."""
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored label encoders; unknown categories → -1."""
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                known = set(le.classes_)
                df[col] = (
                    df[col]
                    .astype(str)
                    .map(lambda x: int(le.transform([x])[0]) if x in known else -1)
                )
        return df

    # -----------------------------------------------------------------------
    # Feature matrix extraction
    # -----------------------------------------------------------------------

    def _get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the feature DataFrame (keeps column names → no sklearn warning)."""
        cols = [c for c in FEATURE_COLS if c in df.columns]
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            logger.debug("Features missing from input (using 0): %s", missing)
        # Return a DataFrame so LightGBM always sees consistent feature names
        return df[cols].fillna(0).astype(np.float32)

    def _get_X_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = self._get_X(df)
        y = df["is_canceled"].values.astype(int)
        return X, y

    # -----------------------------------------------------------------------
    # Model factory
    # -----------------------------------------------------------------------

    def _make_lgbm(self) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Train with TimeSeriesSplit CV + Platt scaling; returns CV metrics dict.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-processed training DataFrame.  If None, loads from ``data_path``.

        Returns
        -------
        dict with keys ``cv_metrics``, ``mean_auc``, ``mean_brier``.
        """
        if df is None:
            df = self.load_raw()

        # Fit encoders on full dataset (no target info leaks from encoding)
        self._fit_encoders(df)
        df_enc = self._encode(df)
        X, y = self._get_X_y(df_enc)

        tscv = TimeSeriesSplit(n_splits=5)
        fold_metrics: list[dict[str, Any]] = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            base = self._make_lgbm()
            # cv=3 inner Platt calibration uses a held-out portion of the training fold
            cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            cal.fit(X_tr, y_tr)

            p_va  = cal.predict_proba(X_va)[:, 1]
            auc   = float(roc_auc_score(y_va, p_va))
            brier = float(brier_score_loss(y_va, p_va))

            fold_metrics.append(
                {"fold": fold + 1, "auc": round(auc, 4), "brier": round(brier, 4)}
            )
            logger.info("CV fold %d/%d — AUC=%.4f  Brier=%.4f", fold + 1, 5, auc, brier)

        # Final model trained on the full dataset
        base_final = self._make_lgbm()
        self.model = CalibratedClassifierCV(base_final, method="sigmoid", cv=5)
        self.model.fit(X, y)
        self.is_fitted   = True
        self.cv_metrics  = fold_metrics

        mean_auc   = round(float(np.mean([m["auc"]   for m in fold_metrics])), 4)
        mean_brier = round(float(np.mean([m["brier"] for m in fold_metrics])), 4)
        logger.info("Final model ready — Mean AUC=%.4f  Mean Brier=%.4f", mean_auc, mean_brier)

        return {"cv_metrics": fold_metrics, "mean_auc": mean_auc, "mean_brier": mean_brier}

    # -----------------------------------------------------------------------
    # Inference — already-preprocessed DataFrame
    # -----------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(cancel) for every row of an already-preprocessed DataFrame.

        Assumes *df* has been through ``_base_preprocess`` already.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        df_enc = self._encode(df)
        X = self._get_X(df_enc)
        return self.model.predict_proba(X)[:, 1]

    # -----------------------------------------------------------------------
    # Inference — raw API input DataFrame
    # -----------------------------------------------------------------------

    def predict_from_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess raw input then predict P(cancel).

        Use this for data arriving through the /predict and /optimize API
        endpoints where no prior preprocessing has been applied.
        """
        df_proc = self._preprocess_inference(df)
        return self.predict_proba(df_proc)

    # -----------------------------------------------------------------------
    # Backtesting helper: per-fold train + predict (strict no-look-ahead)
    # -----------------------------------------------------------------------

    def fit_and_predict_splits(
        self,
        df: pd.DataFrame | None = None,
        n_splits: int = 5,
    ) -> list[dict[str, Any]]:
        """Train a fresh model per fold and return fold-level predictions.

        Each fold trains only on chronologically *earlier* data and predicts on
        the immediately following window — the strictest form of walk-forward
        validation.

        Returns
        -------
        List of dicts with keys:
            fold, val_idx, y_true, p_cancel, auc, brier
        """
        if df is None:
            df = self.load_raw()

        # Fit encoders once on the full dataset (encoding carries no target info)
        self._fit_encoders(df)
        df_enc = self._encode(df)
        X, y = self._get_X_y(df_enc)

        tscv  = TimeSeriesSplit(n_splits=n_splits)
        splits: list[dict[str, Any]] = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            base = self._make_lgbm()
            cal  = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            cal.fit(X_tr, y_tr)

            p_cancel = cal.predict_proba(X_va)[:, 1]
            auc   = float(roc_auc_score(y_va, p_cancel))
            brier = float(brier_score_loss(y_va, p_cancel))

            splits.append(
                {
                    "fold":     fold + 1,
                    "val_idx":  va_idx,
                    "y_true":   y_va,
                    "p_cancel": p_cancel,
                    "auc":      round(auc, 4),
                    "brier":    round(brier, 4),
                }
            )
            logger.info(
                "Backtest fold %d/%d — AUC=%.4f  Brier=%.4f  n_val=%d",
                fold + 1, n_splits, auc, brier, len(va_idx),
            )

        return splits
