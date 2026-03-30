"""Tests for RiskAssessor and RiskScoreHead."""

import pytest
import numpy as np

from power_graph_risk.data.power_grid import PowerGridGraph, BusNode, LineEdge, NodeType
from power_graph_risk.models.gnn import GNNModel
from power_graph_risk.models.risk_model import RiskScoreHead
from power_graph_risk.risk.assessor import RiskAssessor, RiskReport, CascadeSimulator


def make_grid() -> PowerGridGraph:
    return PowerGridGraph.ieee_14_bus()


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


class TestRiskScoreHead:
    def test_forward_output_shape(self):
        head = RiskScoreHead(in_dim=8, hidden_dims=[16, 8])
        emb = np.random.default_rng(0).standard_normal((10, 8))
        scores = head(emb)
        assert scores.shape == (10,)

    def test_scores_in_unit_interval(self):
        head = RiskScoreHead(in_dim=8, hidden_dims=[16])
        emb = np.random.default_rng(1).standard_normal((20, 8))
        scores = head(emb)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_fit_reduces_loss(self):
        head = RiskScoreHead(in_dim=8, hidden_dims=[16], seed=0)
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((20, 8))
        targets = rng.uniform(0, 1, size=20)
        losses = head.fit(emb, targets, epochs=200, lr=1e-2)
        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_repr(self):
        head = RiskScoreHead(in_dim=8)
        assert "RiskScoreHead" in repr(head)


class TestCascadeSimulator:
    def test_n1_risk_keys(self):
        grid = make_small_grid()
        sim = CascadeSimulator(grid)
        risks = sim.n_minus_1_risk()
        # Should have one entry per in-service line
        assert len(risks) == grid.num_lines

    def test_n1_risk_values_bounded(self):
        grid = make_small_grid()
        sim = CascadeSimulator(grid)
        risks = sim.n_minus_1_risk()
        for score in risks.values():
            assert 0.0 <= score <= 1.0

    def test_bus_risk_keys(self):
        grid = make_small_grid()
        sim = CascadeSimulator(grid)
        bus_risk = sim.bus_risk_from_n1()
        assert set(bus_risk.keys()) == set(grid.buses.keys())

    def test_bus_risk_normalised(self):
        grid = make_small_grid()
        sim = CascadeSimulator(grid)
        bus_risk = sim.bus_risk_from_n1()
        for v in bus_risk.values():
            assert 0.0 <= v <= 1.0

    def test_grid_restored_after_simulation(self):
        """N-1 simulation must not leave lines tripped."""
        grid = make_small_grid()
        sim = CascadeSimulator(grid)
        sim.n_minus_1_risk()
        for line in grid.lines.values():
            assert line.in_service


class TestRiskAssessor:
    def test_assess_without_fit_warns(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=True)
        with pytest.warns(UserWarning):
            report = assessor.assess(grid)
        assert isinstance(report, RiskReport)

    def test_fit_returns_losses(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False)
        losses = assessor.fit(grid, epochs=20, lr=1e-3)
        assert len(losses) == 20
        assert all(isinstance(l, float) for l in losses)

    def test_fit_marks_fitted(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False)
        assessor.fit(grid, epochs=5)
        assert assessor._fitted

    def test_assess_after_fit_returns_report(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False)
        assessor.fit(grid, epochs=10)
        report = assessor.assess(grid)
        assert isinstance(report, RiskReport)

    def test_report_keys_match_buses(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False)
        assessor.fit(grid, epochs=10)
        report = assessor.assess(grid)
        assert set(report.bus_risk_scores.keys()) == set(grid.buses.keys())

    def test_report_scores_in_unit_interval(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False)
        assessor.fit(grid, epochs=10)
        report = assessor.assess(grid)
        for score in report.bus_risk_scores.values():
            assert 0.0 <= score <= 1.0

    def test_system_risk_in_unit_interval(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False)
        assessor.fit(grid, epochs=10)
        report = assessor.assess(grid)
        assert 0.0 <= report.system_risk <= 1.0

    def test_high_risk_buses_subset(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False, risk_threshold=0.5)
        assessor.fit(grid, epochs=10)
        report = assessor.assess(grid)
        for b in report.high_risk_buses:
            assert report.bus_risk_scores[b] >= 0.5

    def test_custom_targets(self):
        grid = make_small_grid()
        n = grid.num_buses
        targets = np.ones(n) * 0.8
        assessor = RiskAssessor(normalise_features=False)
        losses = assessor.fit(grid, targets=targets, epochs=50, lr=1e-2)
        assert losses[-1] < losses[0]

    def test_topk_vulnerable_buses(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False)
        assessor.fit(grid, epochs=10)
        top2 = assessor.topk_vulnerable_buses(grid, k=2)
        assert len(top2) == 2
        # Sorted descending
        assert top2[0][1] >= top2[1][1]

    def test_topk_more_than_n(self):
        grid = make_small_grid()
        assessor = RiskAssessor(normalise_features=False)
        assessor.fit(grid, epochs=10)
        top = assessor.topk_vulnerable_buses(grid, k=100)
        assert len(top) == grid.num_buses

    def test_repr(self):
        assessor = RiskAssessor()
        assert "RiskAssessor" in repr(assessor)

    def test_fit_with_14_bus(self):
        grid = make_grid()
        assessor = RiskAssessor(normalise_features=True)
        losses = assessor.fit(grid, epochs=10)
        assert len(losses) == 10
