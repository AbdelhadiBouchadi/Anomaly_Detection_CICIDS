"""
Real-time / Batch Inference
Load saved models and predict on new network traffic flows.
"""

import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from typing import Dict, Optional


class AnomalyPredictor:
    """
    Loads trained models and provides prediction on new traffic samples.
    Supports ensemble voting across all four models.
    """

    MODEL_MAP = {
        'Random Forest':    'rf_model.pkl',
        'XGBoost':          'xgb_model.pkl',
        'Isolation Forest': 'iso_forest.pkl',
        'Autoencoder':      'autoencoder.h5',
    }

    def __init__(self, model_dir: str = 'outputs'):
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.ae_threshold: Optional[float] = None

    def load(self):
        """Load all available models and scaler."""
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded")

        for name, fname in self.MODEL_MAP.items():
            fpath = os.path.join(self.model_dir, fname)
            if not os.path.exists(fpath):
                continue
            if fname.endswith('.pkl'):
                self.models[name] = joblib.load(fpath)
            else:
                self.models[name] = tf.keras.models.load_model(fpath)
            print(f"✓ {name} loaded")

        return self

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            X = self.scaler.transform(X)
        return X

    def predict_single(self, X: np.ndarray, model_name: str) -> Dict:
        """Predict using a single model. Returns score + label."""
        model = self.models[model_name]
        X_p = self.preprocess(X)

        if model_name in ('Random Forest', 'XGBoost'):
            prob = model.predict_proba(X_p)[:, 1]
            label = (prob > 0.5).astype(int)
            return {'score': prob, 'label': label}

        elif model_name == 'Isolation Forest':
            score = -model.score_samples(X_p)
            label = (model.predict(X_p) == -1).astype(int)
            return {'score': score, 'label': label}

        elif model_name == 'Autoencoder':
            recon = model.predict(X_p, verbose=0)
            mse = np.mean(np.square(X_p - recon), axis=1)
            thr = self.ae_threshold or np.percentile(mse, 90)
            return {'score': mse, 'label': (mse > thr).astype(int)}

    def predict_ensemble(self, X: np.ndarray, strategy: str = 'majority') -> np.ndarray:
        """
        Ensemble prediction across all loaded models.
        strategy: 'majority' | 'any' | 'all'
        """
        votes = []
        for name in self.models:
            result = self.predict_single(X, name)
            votes.append(result['label'])

        votes = np.stack(votes, axis=1)  # (n_samples, n_models)

        if strategy == 'majority':
            return (votes.mean(axis=1) >= 0.5).astype(int)
        elif strategy == 'any':
            return (votes.sum(axis=1) >= 1).astype(int)
        elif strategy == 'all':
            return (votes.sum(axis=1) == len(self.models)).astype(int)

    def report(self, X: np.ndarray, sample_ids=None) -> pd.DataFrame:
        """Full report per sample across all models."""
        rows = []
        for i, x in enumerate(X):
            row = {'sample': sample_ids[i] if sample_ids else i}
            for name in self.models:
                res = self.predict_single(x.reshape(1, -1), name)
                row[f'{name}_score'] = res['score'][0]
                row[f'{name}_label'] = 'ATTACK' if res['label'][0] else 'BENIGN'
            rows.append(row)
        return pd.DataFrame(rows)


# ─── CLI usage ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    predictor = AnomalyPredictor('outputs').load()
    print("\nAll models ready for inference.")
    print("Usage: predictor.predict_ensemble(X_new, strategy='majority')")
