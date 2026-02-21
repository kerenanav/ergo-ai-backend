"""
train_model.py — Train and save the cancellation predictor for one or all domains.

Usage:
    python train_model.py                      # train hotel_v1 (default)
    python train_model.py hotel_v1             # explicit domain_id
    python train_model.py all                  # train every config in configs/

Output:
    model_{domain_id}.pkl   (e.g. model_hotel_v1.pkl)

The generated pickle must be committed to the repository so that the server
can load it at startup without retraining.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib

from domain_config import load_domain_config, list_domain_configs, validate_domain_config
from predictive_model import CancellationPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path("configs")


def train_domain(domain_id: str) -> bool:
    """Train and save the model for *domain_id*. Returns True on success."""
    config_path = CONFIGS_DIR / f"{domain_id}_config.json"
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return False

    cfg      = load_domain_config(config_path)
    warnings = validate_domain_config(cfg)
    for w in warnings:
        logger.warning("[%s] %s", domain_id, w)

    data_path = Path(cfg.dataset_path)
    if not data_path.exists():
        logger.error(
            "[%s] Dataset not found: %s — place the CSV in the working directory.",
            domain_id, data_path,
        )
        return False

    model_path = f"model_{domain_id}.pkl"
    logger.info("[%s] Loading data from %s …", domain_id, data_path)

    predictor = CancellationPredictor(domain_config=cfg)
    df        = predictor.load_raw()
    logger.info("[%s] %d rows loaded.", domain_id, len(df))

    if cfg.ml_target_column not in df.columns:
        logger.error(
            "[%s] Target column '%s' not found in dataset — cannot train.",
            domain_id, cfg.ml_target_column,
        )
        return False

    logger.info("[%s] Training model (LightGBM + Platt scaling, TimeSeriesSplit-5) …", domain_id)
    metrics = predictor.fit(df)

    joblib.dump(predictor, model_path)
    logger.info(
        "[%s] Saved → %s  |  Mean AUC=%.4f  Brier=%.4f",
        domain_id, model_path,
        metrics["mean_auc"], metrics["mean_brier"],
    )
    return True


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "hotel_v1"

    if arg == "all":
        domain_ids = list_domain_configs(CONFIGS_DIR)
        if not domain_ids:
            logger.error("No config files found in %s/", CONFIGS_DIR)
            sys.exit(1)
        logger.info("Training all domains: %s", domain_ids)
        results = {d: train_domain(d) for d in domain_ids}
        for d, ok in results.items():
            logger.info("  %-20s %s", d, "OK" if ok else "FAILED")
        if not all(results.values()):
            sys.exit(1)
    else:
        ok = train_domain(arg)
        sys.exit(0 if ok else 1)
