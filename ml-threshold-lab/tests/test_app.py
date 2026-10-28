"""Regression tests for the ML classification threshold-tuning lab."""
from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

import app  # noqa: E402


def evaluate(cases: list[dict], threshold: float) -> dict[str, float | int]:
    """Calculate binary classification counts/metrics as the browser does."""
    tp = tn = fp = fn = 0
    for case in cases:
        predicted = int(case["malignantProbability"] >= threshold)
        actual = case["actual"]
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
    }


class ModelInvariantTests(unittest.TestCase):
    def test_expected_dataset_and_split_sizes(self) -> None:
        meta = app.LAB_DATA["dataset"]
        self.assertEqual(meta["samples"], 569)
        self.assertEqual(meta["features"], 30)
        self.assertEqual(
            (meta["trainSamples"], meta["validationSamples"], meta["testSamples"]),
            (341, 114, 114),
        )
        self.assertEqual(
            meta["trainSamples"] + meta["validationSamples"] + meta["testSamples"],
            meta["samples"],
        )

    def test_stratified_splits_keep_all_examples(self) -> None:
        meta = app.LAB_DATA["dataset"]
        self.assertEqual(meta["validationMalignant"] + meta["validationBenign"], meta["validationSamples"])
        self.assertEqual(meta["testMalignant"] + meta["testBenign"], meta["testSamples"])
        self.assertGreater(meta["validationMalignant"], 0)
        self.assertGreater(meta["validationBenign"], 0)
        self.assertGreater(meta["testMalignant"], 0)
        self.assertGreater(meta["testBenign"], 0)

    def test_probabilities_and_auc_are_valid(self) -> None:
        for split in ("cases", "testCases"):
            cases = app.LAB_DATA[split]
            self.assertTrue(cases)
            self.assertEqual(len(cases), 114)
            for case in cases:
                self.assertIn(case["actual"], (0, 1))
                self.assertGreaterEqual(case["malignantProbability"], 0.0)
                self.assertLessEqual(case["malignantProbability"], 1.0)
        model = app.LAB_DATA["model"]
        self.assertGreaterEqual(model["validationRocAuc"], 0.0)
        self.assertLessEqual(model["validationRocAuc"], 1.0)
        self.assertGreaterEqual(model["testRocAuc"], 0.0)
        self.assertLessEqual(model["testRocAuc"], 1.0)

    def test_threshold_counts_reconcile_and_positive_predictions_are_monotonic(self) -> None:
        cases = app.LAB_DATA["cases"]
        low, middle, high = (evaluate(cases, t) for t in (0.25, 0.50, 0.75))
        for result in (low, middle, high):
            self.assertEqual(result["tp"] + result["fn"], app.LAB_DATA["dataset"]["validationMalignant"])
            self.assertEqual(result["tn"] + result["fp"], app.LAB_DATA["dataset"]["validationBenign"])
            self.assertEqual(result["tp"] + result["tn"] + result["fp"] + result["fn"], 114)
            for key in ("accuracy", "precision", "recall", "specificity", "f1"):
                self.assertGreaterEqual(result[key], 0.0)
                self.assertLessEqual(result[key], 1.0)
        self.assertGreaterEqual(low["tp"] + low["fp"], middle["tp"] + middle["fp"])
        self.assertGreaterEqual(middle["tp"] + middle["fp"], high["tp"] + high["fp"])

    def test_validation_and_test_examples_are_separate(self) -> None:
        val_ids = {row["id"] for row in app.LAB_DATA["cases"]}
        test_ids = {row["id"] for row in app.LAB_DATA["testCases"]}
        self.assertFalse(val_ids & test_ids)
        self.assertTrue(all(key.startswith("VAL-") for key in val_ids))
        self.assertTrue(all(key.startswith("TEST-") for key in test_ids))


class HTTPEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.LabHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_health_endpoint_reports_split(self) -> None:
        with urlopen(self.base_url + "/api/health", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model"], "trained")
        self.assertEqual(payload["trainSamples"], 341)
        self.assertEqual(payload["validationSamples"], 114)
        self.assertEqual(payload["testSamples"], 114)

    def test_data_endpoint_separates_validation_from_test(self) -> None:
        with urlopen(self.base_url + "/api/data", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
        self.assertEqual(len(payload["cases"]), 114)
        self.assertEqual(len(payload["testCases"]), 114)
        self.assertIn("validationRocAuc", payload["model"])
        self.assertIn("testRocAuc", payload["model"])

    def test_root_serves_the_interactive_lab(self) -> None:
        with urlopen(self.base_url + "/", timeout=3) as response:
            html = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
        self.assertIn("Classification", html)
        self.assertIn("threshold-slider", html)
        self.assertIn("test-eval-btn", html)
        self.assertIn("export-btn", html)

    def test_unknown_routes_return_404(self) -> None:
        with self.assertRaises(Exception) as context:
            urlopen(self.base_url + "/not-a-real-route", timeout=3)
        self.assertEqual(getattr(context.exception, "code", None), 404)


if __name__ == "__main__":
    unittest.main()
