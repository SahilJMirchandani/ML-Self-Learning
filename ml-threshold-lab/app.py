"""VESIT ML Virtual Lab: Classification Threshold Tuning.

Trains a Logistic Regression classifier on scikit-learn's Breast Cancer Wisconsin
dataset and serves test-set probabilities to a small browser-based lab.
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

# In scikit-learn's source dataset, target 0 = malignant and target 1 = benign.
# Re-map malignant to positive class 1 so the lab's recall measures malignant-case detection.
dataset = load_breast_cancer()
X = dataset.data
y_malignant = (dataset.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_malignant,
    test_size=0.25,
    random_state=42,
    stratify=y_malignant,
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=3000, random_state=42),
)
model.fit(X_train, y_train)
malignant_class_index = list(model.classes_).index(1)
test_scores = model.predict_proba(X_test)[:, malignant_class_index]
auc = float(roc_auc_score(y_test, test_scores))

# Preserve full precision for threshold calculations in the browser.
LAB_DATA = {
    "dataset": {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "trainSamples": int(X_train.shape[0]),
        "testSamples": int(X_test.shape[0]),
        "testMalignant": int(y_test.sum()),
        "testBenign": int(len(y_test) - y_test.sum()),
        "split": "75% train / 25% test",
        "randomState": 42,
        "source": "scikit-learn built-in benchmark dataset",
    },
    "model": {
        "name": "Logistic Regression",
        "preprocessing": "StandardScaler",
        "positiveClass": "Malignant",
        "thresholdDefault": 0.50,
        "rocAuc": auc,
    },
    "cases": [
        {
            "id": f"TEST-{i + 1:03d}",
            "actual": int(y_test[i]),
            "malignantProbability": float(test_scores[i]),
        }
        for i in range(len(y_test))
    ],
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
            self._send(200, b'{"status":"ok","model":"trained"}', "application/json; charset=utf-8")
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
        # Keep deploy/runtime logs useful without exposing case-level data.
        print(f"[ml-lab] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), LabHandler)
    print(
        f"ML Threshold Tuning Lab listening on port {PORT} | "
        f"{len(X_train)} train / {len(X_test)} test | test ROC-AUC={auc:.4f}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
