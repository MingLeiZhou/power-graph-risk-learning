"""Tests for the PowerGridGraph data model."""

import pytest
import numpy as np

from power_graph_risk.data.power_grid import (
    PowerGridGraph,
    BusNode,
    LineEdge,
    NodeType,
)


def make_simple_grid() -> PowerGridGraph:
    """3-bus, 3-line triangle grid for testing."""
    grid = PowerGridGraph()
    grid.add_bus(BusNode(node_id=1, node_type=NodeType.SLACK, voltage_mag=1.05, p_inject=50.0))
    grid.add_bus(BusNode(node_id=2, node_type=NodeType.PV, voltage_mag=1.01, p_inject=20.0, p_load=10.0))
    grid.add_bus(BusNode(node_id=3, node_type=NodeType.PQ, p_load=60.0, q_load=15.0))
    grid.add_line(LineEdge(from_bus=1, to_bus=2, resistance=0.01, reactance=0.05, thermal_limit=80.0))
    grid.add_line(LineEdge(from_bus=2, to_bus=3, resistance=0.02, reactance=0.06, thermal_limit=80.0))
    grid.add_line(LineEdge(from_bus=1, to_bus=3, resistance=0.03, reactance=0.09, thermal_limit=80.0))
    return grid


class TestBusNode:
    def test_feature_vector_length(self):
        bus = BusNode(node_id=1, voltage_mag=1.0, p_inject=10.0)
        fv = bus.feature_vector()
        assert fv.shape == (PowerGridGraph.NODE_FEATURE_DIM,)

    def test_feature_vector_values(self):
        bus = BusNode(node_id=5, voltage_mag=1.02, voltage_angle=0.1,
                      p_inject=30.0, q_inject=5.0, p_load=20.0, q_load=8.0)
        fv = bus.feature_vector()
        assert fv[0] == pytest.approx(1.02)
        assert fv[1] == pytest.approx(0.1)
        assert fv[2] == pytest.approx(30.0)
        assert fv[3] == pytest.approx(5.0)
        assert fv[4] == pytest.approx(20.0)
        assert fv[5] == pytest.approx(8.0)


class TestLineEdge:
    def test_impedance(self):
        line = LineEdge(from_bus=1, to_bus=2, resistance=0.01, reactance=0.05)
        assert line.impedance == pytest.approx(complex(0.01, 0.05))

    def test_admittance(self):
        line = LineEdge(from_bus=1, to_bus=2, resistance=0.01, reactance=0.05)
        z = complex(0.01, 0.05)
        expected = 1.0 / z
        assert abs(line.admittance - expected) < 1e-9

    def test_admittance_zero_impedance(self):
        line = LineEdge(from_bus=1, to_bus=2, resistance=0.0, reactance=0.0)
        assert line.admittance == complex(0.0, 0.0)

    def test_feature_vector_length(self):
        line = LineEdge(from_bus=1, to_bus=2)
        assert line.feature_vector().shape == (PowerGridGraph.EDGE_FEATURE_DIM,)

    def test_feature_vector_in_service(self):
        line = LineEdge(from_bus=1, to_bus=2, in_service=True)
        assert line.feature_vector()[4] == pytest.approx(1.0)
        line.in_service = False
        assert line.feature_vector()[4] == pytest.approx(0.0)


class TestPowerGridGraph:
    def test_construction(self):
        grid = make_simple_grid()
        assert grid.num_buses == 3
        assert grid.num_lines == 3

    def test_duplicate_bus_raises(self):
        grid = PowerGridGraph()
        grid.add_bus(BusNode(node_id=1))
        with pytest.raises(ValueError, match="already exists"):
            grid.add_bus(BusNode(node_id=1))

    def test_line_missing_bus_raises(self):
        grid = PowerGridGraph()
        grid.add_bus(BusNode(node_id=1))
        with pytest.raises(ValueError, match="not found"):
            grid.add_line(LineEdge(from_bus=1, to_bus=99))

    def test_duplicate_line_raises(self):
        grid = PowerGridGraph()
        grid.add_bus(BusNode(node_id=1))
        grid.add_bus(BusNode(node_id=2))
        grid.add_line(LineEdge(from_bus=1, to_bus=2))
        with pytest.raises(ValueError, match="already exists"):
            grid.add_line(LineEdge(from_bus=2, to_bus=1))

    def test_node_feature_matrix_shape(self):
        grid = make_simple_grid()
        X = grid.node_feature_matrix()
        assert X.shape == (3, PowerGridGraph.NODE_FEATURE_DIM)

    def test_adjacency_matrix_shape_and_symmetry(self):
        grid = make_simple_grid()
        A = grid.adjacency_matrix()
        assert A.shape == (3, 3)
        np.testing.assert_array_almost_equal(A, A.T)

    def test_adjacency_diagonal_zero(self):
        grid = make_simple_grid()
        A = grid.adjacency_matrix()
        np.testing.assert_array_equal(np.diag(A), np.zeros(3))

    def test_weighted_adjacency(self):
        grid = make_simple_grid()
        A_w = grid.adjacency_matrix(weighted=True)
        A_b = grid.adjacency_matrix(weighted=False)
        # Weighted entries should differ from binary
        assert not np.allclose(A_w, A_b)
        # All nonzero entries should be positive
        assert (A_w[A_b > 0] > 0).all()

    def test_edge_index_shape(self):
        grid = make_simple_grid()
        ei = grid.edge_index()
        # 3 lines × 2 directions = 6 directed edges
        assert ei.shape == (2, 6)

    def test_edge_feature_matrix_shape(self):
        grid = make_simple_grid()
        ef = grid.edge_feature_matrix()
        assert ef.shape == (6, PowerGridGraph.EDGE_FEATURE_DIM)

    def test_laplacian_symmetric(self):
        grid = make_simple_grid()
        L = grid.laplacian(normalised=True)
        np.testing.assert_array_almost_equal(L, L.T)

    def test_laplacian_unnormalised(self):
        grid = make_simple_grid()
        L = grid.laplacian(normalised=False)
        A = grid.adjacency_matrix()
        D = np.diag(A.sum(axis=1))
        np.testing.assert_array_almost_equal(L, D - A)

    def test_is_connected(self):
        grid = make_simple_grid()
        assert grid.is_connected()

    def test_disconnected_grid(self):
        grid = PowerGridGraph()
        grid.add_bus(BusNode(node_id=1))
        grid.add_bus(BusNode(node_id=2))
        assert not grid.is_connected()

    def test_connected_components(self):
        grid = PowerGridGraph()
        grid.add_bus(BusNode(node_id=1))
        grid.add_bus(BusNode(node_id=2))
        grid.add_bus(BusNode(node_id=3))
        grid.add_line(LineEdge(from_bus=1, to_bus=2))
        comps = grid.connected_components()
        assert len(comps) == 2

    def test_criticality_scores_keys(self):
        grid = make_simple_grid()
        scores = grid.criticality_scores()
        assert set(scores.keys()) == {1, 2, 3}

    def test_criticality_scores_range(self):
        grid = make_simple_grid()
        scores = grid.criticality_scores()
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_update_bus(self):
        grid = make_simple_grid()
        grid.update_bus(1, voltage_mag=1.1)
        assert grid.buses[1].voltage_mag == pytest.approx(1.1)

    def test_update_bus_invalid_attr_raises(self):
        grid = make_simple_grid()
        with pytest.raises(AttributeError):
            grid.update_bus(1, nonexistent=99)

    def test_set_line_service(self):
        grid = make_simple_grid()
        grid.set_line_service(1, 2, False)
        assert not grid.lines[(1, 2)].in_service

    def test_set_line_service_missing_raises(self):
        grid = make_simple_grid()
        with pytest.raises(KeyError):
            grid.set_line_service(1, 99, False)

    def test_edge_index_excludes_out_of_service(self):
        grid = make_simple_grid()
        grid.set_line_service(1, 2, False)
        ei = grid.edge_index()
        # Only 2 remaining in-service lines → 4 directed edges
        assert ei.shape == (2, 4)

    def test_from_dict(self):
        buses = [{"node_id": 10, "node_type": "slack"}, {"node_id": 20}]
        lines = [{"from_bus": 10, "to_bus": 20, "reactance": 0.1}]
        grid = PowerGridGraph.from_dict(buses, lines)
        assert grid.num_buses == 2
        assert grid.num_lines == 1

    def test_ieee_14_bus(self):
        grid = PowerGridGraph.ieee_14_bus()
        assert grid.num_buses == 14
        assert grid.num_lines == 20
        assert grid.is_connected()

    def test_repr(self):
        grid = make_simple_grid()
        r = repr(grid)
        assert "PowerGridGraph" in r
        assert "buses=3" in r

    def test_node_index_map_contiguous(self):
        grid = make_simple_grid()
        idx_map = grid.node_index_map()
        assert set(idx_map.values()) == {0, 1, 2}
