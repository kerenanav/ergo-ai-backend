"""
predictive_model.py — Layer 1: LightGBM + Platt scaling cancellation predictor.

DomainConfig-driven: feature set, target column, and time column all come
from the loaded DomainConfig rather than being hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

_DEFAULT_FEATURES = [
    "lead_time", "arrival_date_week_number",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "meal", "country",
    "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "deposit_type", "days_in_waiting_list",
    "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
]
_DEFAULT_TARGET = "is_canceled"
_DEFAULT_DATA   = "hotel_bookings.csv"

_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


class CancellationPredictor:
    """LightGBM + Platt scaling predictor, DomainConfig-driven."""

    def __init__(
        self,
        data_path: str | None = None,
        domain_config: Any | None = None,
    ) -> None:
        if domain_config is not None:
            self.data_path    = str(domain_config.dataset_path)
            self.feature_cols = list(domain_config.ml_features)
            self.target_col   = str(domain_config.ml_target_column)
            self.cfg          = domain_config
        else:
            self.data_path    = str(data_path or _DEFAULT_DATA)
            self.feature_cols = list(_DEFAULT_FEATURES)
            self.target_col   = _DEFAULT_TARGET
            self.cfg          = None

        self.model: CalibratedClassifierCV | None = None
        self.encoders: dict[str, LabelEncoder]    = {}
        self.cv_metrics: list[dict]               = []

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "arrival_date_month" in df.columns and df["arrival_date_month"].dtype == object:
            df["arrival_date_month"] = df["arrival_date_month"].map(_MONTH_MAP).fillna(0).astype(int)
        if "total_nights" not in df.columns:
            wknd = df.get("stays_in_weekend_nights", pd.Series(0, index=df.index)).fillna(0)
            wkdy = df.get("stays_in_week_nights",    pd.Series(0, index=df.index)).fillna(0)
            df["total_nights"] = (wknd + wkdy).clip(lower=1)
        for col in ["agent", "company"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        logger.info("Loaded %d rows from %s", len(df), self.data_path)
        df = self._preprocess(df)
        if self.cfg is not None:
            from domain_config import build_time_column
            time_s = build_time_column(df, self.cfg)
            if time_s.notna().any():
                df = df.assign(_time=time_s).sort_values("_time").drop(columns=["_time"]).reset_index(drop=True)
        elif "arrival_date_year" in df.columns:
            sort_cols = [c for c in ["arrival_date_year", "arrival_date_month",
                                     "arrival_date_day_of_month"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols).reset_index(drop=True)
        return df

    def _encode_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
            if (df[col].dtype == object
                    or pd.api.types.is_string_dtype(df[col])
                    or str(df[col].dtype) == "category"):
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                else:
                    if col in self.encoders:
                        le = self.encoders[col]
                        df[col] = df[col].astype(str).apply(
                            lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
                        )
                    else:
                        df[col] = 0
        return df

    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        df_enc = self._encode_features(df, fit=fit)
        return df_enc[self.feature_cols].fillna(0).values.astype(np.float32)

    def _get_y(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.target_col].fillna(0).values.astype(int)

    def _make_lgbm(self) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )

    def fit(self, df: pd.DataFrame | None = None, n_splits: int = 5) -> dict[str, Any]:
        if df is None:
            df = self.load_raw()
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")

        X = self._prepare_X(df, fit=True)
        y = self._get_y(df)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_metrics: list[dict] = []

        for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
            fold_model = CalibratedClassifierCV(self._make_lgbm(), method="sigmoid")
            fold_model.fit(X[train_idx], y[train_idx])
            p_val = fold_model.predict_proba(X[val_idx])[:, 1]
            auc   = roc_auc_score(y[val_idx], p_val)
            brier = brier_score_loss(y[val_idx], p_val)
            cv_metrics.append({"fold": fold_i + 1, "auc": round(auc, 4), "brier": round(brier, 4)})
            logger.info("Fold %d — AUC=%.4f  Brier=%.4f", fold_i + 1, auc, brier)

        self.model = CalibratedClassifierCV(self._make_lgbm(), method="sigmoid")
        self.model.fit(X, y)
        self.cv_metrics = cv_metrics

        mean_auc   = round(float(np.mean([m["auc"]   for m in cv_metrics])), 4)
        mean_brier = round(float(np.mean([m["brier"] for m in cv_metrics])), 4)
        logger.info("Training complete — Mean AUC=%.4f  Brier=%.4f", mean_auc, mean_brier)
        return {"cv_metrics": cv_metrics, "mean_auc": mean_auc, "mean_brier": mean_brier}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        df = self._preprocess(df)
        X = self._prepare_X(df, fit=False)
        return self.model.predict_proba(X)[:, 1]

    def predict_from_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df)
