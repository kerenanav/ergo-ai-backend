"""
train_model.py — Addestra il modello di cancellazione e lo salva su disco.

Esegui questo script una volta in locale prima del deploy:
    python train_model.py

Il file model.pkl generato deve essere committato nel repository
in modo che il server lo carichi all'avvio senza riaddestrare.
"""

import logging

import joblib

from predictive_model import CancellationPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH  = "hotel_bookings.csv"
MODEL_PATH = "model.pkl"

if __name__ == "__main__":
    logger.info("Caricamento dati da %s ...", DATA_PATH)
    predictor = CancellationPredictor(data_path=DATA_PATH)
    df = predictor.load_raw()

    logger.info("Addestramento modello (potrebbe richiedere qualche minuto) ...")
    metrics = predictor.fit(df)

    logger.info("Salvataggio modello in %s ...", MODEL_PATH)
    joblib.dump(predictor, MODEL_PATH)

    logger.info(
        "Completato — Mean AUC=%.4f  Mean Brier=%.4f  salvato in %s",
        metrics["mean_auc"],
        metrics["mean_brier"],
        MODEL_PATH,
    )
