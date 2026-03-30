"""Tests for utility metrics."""

import pytest
import numpy as np

from power_graph_risk.utils.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    precision_recall_f1,
    risk_auc,
    normalise_risk_scores,
    compute_vulnerability_index,
)


class TestRegressionMetrics:
    def test_mae_perfect(self):
        y = np.array([0.1, 0.5, 0.9])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_mae_known_value(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 0.5])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.5)

    def test_mse_perfect(self):
        y = np.array([0.2, 0.4, 0.6])
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_mse_known_value(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 0.0])
        assert mean_squared_error(y_true, y_pred) == pytest.approx(1.0)

    def test_rmse_equals_sqrt_mse(self):
        y_true = np.random.default_rng(0).uniform(0, 1, 20)
        y_pred = np.random.default_rng(1).uniform(0, 1, 20)
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
            np.sqrt(mean_squared_error(y_true, y_pred))
        )

    def test_rmse_nonneg(self):
        y_true = np.array([0.3, 0.7])
        y_pred = np.array([0.5, 0.2])
        assert root_mean_squared_error(y_true, y_pred) >= 0.0


class TestPrecisionRecallF1:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        p, r, f1 = precision_recall_f1(y_true, y_true)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_all_negative_predictions(self):
        y_true = np.array([1, 1, 0])
        y_pred = np.array([0, 0, 0])
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_all_positive_predictions(self):
        y_true = np.array([1, 0, 0])
        y_pred = np.array([1, 1, 1])
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        assert p == pytest.approx(1.0 / 3.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(2.0 * (1.0 / 3.0) / (1.0 + 1.0 / 3.0))

    def test_known_example(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        # TP=1, FP=1, FN=1
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)


class TestRiskAUC:
    def test_perfect_classifier(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        auc = risk_auc(y_true, y_score)
        assert auc == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier_near_half(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_score = rng.uniform(0, 1, 100)
        auc = risk_auc(y_true, y_score)
        # Should be roughly 0.5 for a random classifier
        assert 0.3 <= auc <= 0.7

    def test_auc_in_unit_interval(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.6, 0.4, 0.9, 0.1, 0.7])
        auc = risk_auc(y_true, y_score)
        assert 0.0 <= auc <= 1.0

    def test_all_same_class(self):
        y_true = np.zeros(5, dtype=int)
        y_score = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        auc = risk_auc(y_true, y_score)
        assert auc == pytest.approx(0.5, abs=1e-6)


class TestNormaliseRiskScores:
    def test_constant_scores(self):
        scores = np.array([0.5, 0.5, 0.5])
        normed = normalise_risk_scores(scores)
        np.testing.assert_array_equal(normed, np.zeros(3))

    def test_min_max(self):
        scores = np.array([1.0, 3.0, 5.0])
        normed = normalise_risk_scores(scores)
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)

    def test_preserves_order(self):
        scores = np.array([0.2, 0.8, 0.5])
        normed = normalise_risk_scores(scores)
        assert normed[0] < normed[2] < normed[1]


class TestComputeVulnerabilityIndex:
    def test_no_centrality(self):
        risk = {1: 0.8, 2: 0.3, 3: 0.5}
        vuln = compute_vulnerability_index(risk)
        assert vuln == pytest.approx(risk)

    def test_with_centrality(self):
        risk = {1: 1.0, 2: 0.0}
        centrality = {1: 0.0, 2: 1.0}
        vuln = compute_vulnerability_index(risk, centrality, centrality_weight=0.5)
        assert vuln[1] == pytest.approx(0.5)
        assert vuln[2] == pytest.approx(0.5)

    def test_pure_centrality(self):
        risk = {1: 0.0}
        cent = {1: 0.7}
        vuln = compute_vulnerability_index(risk, cent, centrality_weight=1.0)
        assert vuln[1] == pytest.approx(0.7)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            compute_vulnerability_index({1: 0.5}, centrality_weight=1.5)

    def test_output_clipped(self):
        risk = {1: 1.1}
        vuln = compute_vulnerability_index(risk)
        assert vuln[1] == pytest.approx(1.0)
