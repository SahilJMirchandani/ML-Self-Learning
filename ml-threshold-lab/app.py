"""VESIT ML Virtual Lab: Classification Threshold Tuning.

Trains Logistic Regression on scikit-learn's Breast Cancer Wisconsin dataset.
The browser explores thresholds on a validation set, then reveals a separate
held-out test result for the selected threshold.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
RANDOM_STATE = 42

# In scikit-learn's source dataset, target 0 = malignant and target 1 = benign.
# Here malignant is explicitly re-mapped to positive class 1.
dataset = load_breast_cancer()
X = dataset.data
y_malignant = (dataset.target == 0).astype(int)

# 60% train, 20% validation, 20% final test. Stratification preserves
# the malignant/benign proportions in each split.
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y_malignant,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y_malignant,
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_remaining,
)

# The scaler learns parameters from training data only; validation and test
# data are transformed with those learned values through the pipeline.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
)
model.fit(X_train, y_train)
malignant_class_index = list(model.classes_).index(1)
validation_scores = model.predict_proba(X_validation)[:, malignant_class_index]
test_scores = model.predict_proba(X_test)[:, malignant_class_index]
validation_auc = float(roc_auc_score(y_validation, validation_scores))
test_auc = float(roc_auc_score(y_test, test_scores))


def cases_for(y_true: np.ndarray, scores: np.ndarray, prefix: str) -> list[dict]:
    """Return probabilities and ground truth for the teaching interface."""
    return [
        {
            "id": f"{prefix}-{i + 1:03d}",
            "actual": int(y_true[i]),
            "malignantProbability": float(scores[i]),
        }
        for i in range(len(y_true))
    ]


# All probability values come from the trained model. The browser computes
# threshold-dependent confusion counts and metrics from validation cases.
# Test predictions are provided separately and revealed only after the learner
# freezes a chosen validation threshold in the interface.
LAB_DATA = {
    "dataset": {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "trainSamples": int(X_train.shape[0]),
        "validationSamples": int(X_validation.shape[0]),
        "testSamples": int(X_test.shape[0]),
        "validationMalignant": int(y_validation.sum()),
        "validationBenign": int(len(y_validation) - y_validation.sum()),
        "testMalignant": int(y_test.sum()),
        "testBenign": int(len(y_test) - y_test.sum()),
        "split": "60% train / 20% validation / 20% test",
        "randomState": RANDOM_STATE,
        "source": "scikit-learn built-in benchmark dataset",
    },
    "model": {
        "name": "Logistic Regression",
        "preprocessing": "StandardScaler fitted on training data only",
        "positiveClass": "Malignant",
        "thresholdDefault": 0.50,
        "validationRocAuc": validation_auc,
        "testRocAuc": test_auc,
    },
    "cases": cases_for(y_validation, validation_scores, "VAL"),
    "testCases": cases_for(y_test, test_scores, "TEST"),
}


class LabHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            payload = json.dumps({
                "status": "ok",
                "model": "trained",
                "trainSamples": len(X_train),
                "validationSamples": len(X_validation),
                "testSamples": len(X_test),
            }).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return
        if path == "/api/data":
            body = json.dumps(LAB_DATA, separators=(",", ":")).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path in ("/", "/index.html"):
            try:
                page = (ROOT / "index.html").read_bytes()
            except OSError:
                self._send(500, b"Lab interface is missing.", "text/plain; charset=utf-8")
                return
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        self._send(404, b'{"error":"Not found"}', "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        # Avoid logging individual prediction values or personal data.
        print(f"[ml-lab] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), LabHandler)
    print(
        f"ML Threshold Tuning Lab listening on port {PORT} | "
        f"{len(X_train)} train / {len(X_validation)} validation / {len(X_test)} test | "
        f"validation ROC-AUC={validation_auc:.4f} | test ROC-AUC={test_auc:.4f}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
