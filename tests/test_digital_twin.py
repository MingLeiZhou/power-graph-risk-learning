"""Tests for the DigitalTwin and supporting classes."""

import time
import pytest
import numpy as np

from power_graph_risk.data.power_grid import PowerGridGraph, BusNode, LineEdge, NodeType
from power_graph_risk.digital_twin.twin import (
    DigitalTwin,
    TwinSnapshot,
    AnomalyDetector,
    StateEstimator,
)
from power_graph_risk.risk.assessor import RiskAssessor


def make_small_grid() -> PowerGridGraph:
    grid = PowerGridGraph()
    for i in range(1, 5):
        nt = NodeType.SLACK if i == 1 else NodeType.PQ
        grid.add_bus(BusNode(node_id=i, node_type=nt, p_inject=20.0 if i == 1 else 0.0, p_load=5.0 * i))
    grid.add_line(LineEdge(from_bus=1, to_bus=2))
    grid.add_line(LineEdge(from_bus=2, to_bus=3))
    grid.add_line(LineEdge(from_bus=3, to_bus=4))
    grid.add_line(LineEdge(from_bus=1, to_bus=4))
    return grid


class TestStateEstimator:
    def test_first_update_equals_input(self):
        se = StateEstimator(alpha=0.5)
        X = np.ones((3, 4))
        out = se.update(X)
        np.testing.assert_array_equal(out, X)

    def test_ema_smoothing(self):
        se = StateEstimator(alpha=0.5)
        X1 = np.zeros((2, 3))
        X2 = np.ones((2, 3)) * 2.0
        se.update(X1)
        out = se.update(X2)
        # EMA: 0.5 * 2.0 + 0.5 * 0.0 = 1.0
        np.testing.assert_allclose(out, np.ones((2, 3)))

    def test_reset(self):
        se = StateEstimator(alpha=0.3)
        se.update(np.ones((2, 2)))
        se.reset()
        assert se._state is None

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            StateEstimator(alpha=0.0)
        with pytest.raises(ValueError):
            StateEstimator(alpha=1.5)


class TestAnomalyDetector:
    def test_no_anomaly_on_startup(self):
        det = AnomalyDetector(window=10, threshold=3.0)
        fv = np.ones(6)
        # First few observations — not enough history
        for _ in range(4):
            result = det.update(1, fv)
        assert not result  # last result should be False (not enough history)

    def test_anomaly_detected(self):
        det = AnomalyDetector(window=20, threshold=2.0)
        # Build up stable history
        for _ in range(15):
            det.update(1, np.zeros(6))
        # Now inject a large spike
        spike = np.ones(6) * 100.0
        is_anomaly = det.update(1, spike)
        assert is_anomaly

    def test_reset_clears_history(self):
        det = AnomalyDetector(window=20, threshold=2.0)
        for _ in range(10):
            det.update(1, np.ones(6))
        det.reset(1)
        assert 1 not in det._history or len(det._history.get(1, [])) == 0

    def test_reset_all(self):
        det = AnomalyDetector(window=20, threshold=2.0)
        for _ in range(5):
            det.update(1, np.ones(6))
            det.update(2, np.ones(6))
        det.reset()
        assert len(det._history) == 0


class TestDigitalTwin:
    def test_construction(self):
        grid = make_small_grid()
        twin = DigitalTwin(grid)
        assert isinstance(twin, DigitalTwin)

    def test_grid_property_is_copy(self):
        grid = make_small_grid()
        twin = DigitalTwin(grid)
        # Modifying original should not affect twin
        grid.update_bus(1, voltage_mag=0.5)
        assert twin.grid.buses[1].voltage_mag != 0.5

    def test_no_snapshots_initially(self):
        twin = DigitalTwin(make_small_grid())
        assert twin.latest_snapshot is None
        assert twin.history == []

    def test_update_creates_snapshot(self):
        twin = DigitalTwin(make_small_grid())
        snap = twin.update({1: {"voltage_mag": 1.05}})
        assert isinstance(snap, TwinSnapshot)
        assert len(twin.history) == 1

    def test_update_timestamp(self):
        twin = DigitalTwin(make_small_grid())
        t = 1_700_000_000.0
        snap = twin.update({1: {"voltage_mag": 1.02}}, timestamp=t)
        assert snap.timestamp == pytest.approx(t)

    def test_update_unknown_bus_warns(self):
        twin = DigitalTwin(make_small_grid())
        with pytest.warns(UserWarning):
            twin.update({999: {"voltage_mag": 1.0}})

    def test_train_marks_fitted(self):
        twin = DigitalTwin(make_small_grid())
        twin.train(epochs=5)
        assert twin.assessor._fitted

    def test_update_after_train_produces_risk_report(self):
        twin = DigitalTwin(make_small_grid())
        twin.train(epochs=5)
        snap = twin.update({1: {"voltage_mag": 1.05}})
        assert snap.risk_report is not None
        assert 0.0 <= snap.risk_report.system_risk <= 1.0

    def test_risk_trend_empty_before_train(self):
        twin = DigitalTwin(make_small_grid())
        twin.update({1: {"voltage_mag": 1.0}})
        # No risk report yet (unfitted assessor)
        trend = twin.risk_trend()
        assert trend == []

    def test_risk_trend_populated_after_train(self):
        twin = DigitalTwin(make_small_grid())
        twin.train(epochs=5)
        twin.update({1: {"voltage_mag": 1.0}})
        twin.update({2: {"p_load": 12.0}})
        trend = twin.risk_trend()
        assert len(trend) == 2

    def test_simulate_contingency_returns_report(self):
        twin = DigitalTwin(make_small_grid())
        twin.train(epochs=5)
        report = twin.simulate_contingency([(1, 2)])
        from power_graph_risk.risk.assessor import RiskReport
        assert isinstance(report, RiskReport)

    def test_simulate_contingency_does_not_modify_twin_grid(self):
        twin = DigitalTwin(make_small_grid())
        twin.train(epochs=5)
        twin.simulate_contingency([(1, 2)])
        # Internal grid should still have the line in service
        assert twin.grid.lines[(1, 2)].in_service

    def test_simulate_contingency_unknown_line_warns(self):
        twin = DigitalTwin(make_small_grid())
        twin.train(epochs=5)
        with pytest.warns(UserWarning):
            twin.simulate_contingency([(1, 99)])

    def test_history_respects_maxlen(self):
        twin = DigitalTwin(make_small_grid(), history_len=3)
        for i in range(10):
            twin.update({1: {"voltage_mag": 1.0 + i * 0.01}})
        assert len(twin.history) == 3

    def test_repr(self):
        twin = DigitalTwin(make_small_grid())
        r = repr(twin)
        assert "DigitalTwin" in r
