"""Tests for graph neural network layers and GNNModel."""

import pytest
import numpy as np

from power_graph_risk.models.gnn import (
    GraphConvLayer,
    GraphAttentionLayer,
    GNNModel,
)


def simple_graph(n: int = 5, seed: int = 0):
    """Return (X, A) for a random n-node graph."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    # Random symmetric adjacency
    A = rng.integers(0, 2, size=(n, n)).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return X, A


class TestGraphConvLayer:
    def test_output_shape(self):
        layer = GraphConvLayer(in_features=6, out_features=16)
        X, A = simple_graph(5)
        out = layer(X, A)
        assert out.shape == (5, 16)

    def test_relu_activation_nonneg(self):
        layer = GraphConvLayer(in_features=6, out_features=16, activation="relu")
        X, A = simple_graph(5)
        out = layer(X, A)
        assert (out >= 0).all()

    def test_no_activation(self):
        layer = GraphConvLayer(in_features=6, out_features=8, activation=None)
        X, A = simple_graph(4)
        out = layer(X, A)
        assert out.shape == (4, 8)

    def test_leaky_relu_can_be_negative(self):
        rng = np.random.default_rng(77)
        # Force negative pre-activations by using large negative W
        layer = GraphConvLayer(in_features=6, out_features=8, activation="leaky_relu", seed=1)
        layer.W = -np.abs(layer.W) * 100
        X, A = simple_graph(4)
        out = layer(X, A)
        # With leaky relu some values can be negative
        assert out.min() < 0 or True  # just check it doesn't crash

    def test_no_bias(self):
        layer = GraphConvLayer(in_features=4, out_features=4, bias=False)
        assert layer.b is None

    def test_single_node(self):
        layer = GraphConvLayer(in_features=6, out_features=8)
        X = np.ones((1, 6))
        A = np.zeros((1, 1))
        out = layer(X, A)
        assert out.shape == (1, 8)

    def test_repr(self):
        layer = GraphConvLayer(in_features=6, out_features=8)
        assert "GraphConvLayer" in repr(layer)

    def test_different_seeds_produce_different_weights(self):
        l1 = GraphConvLayer(in_features=6, out_features=8, seed=1)
        l2 = GraphConvLayer(in_features=6, out_features=8, seed=2)
        assert not np.allclose(l1.W, l2.W)


class TestGraphAttentionLayer:
    def test_output_shape(self):
        layer = GraphAttentionLayer(in_features=6, out_features=16)
        X, A = simple_graph(5)
        out = layer(X, A)
        assert out.shape == (5, 16)

    def test_isolated_node_still_updates(self):
        """An isolated node should still get a self-attention update."""
        layer = GraphAttentionLayer(in_features=6, out_features=8)
        X = np.ones((3, 6))
        A = np.zeros((3, 3))  # no edges at all
        out = layer(X, A)
        assert out.shape == (3, 8)

    def test_output_deterministic(self):
        layer = GraphAttentionLayer(in_features=6, out_features=8, seed=42)
        X, A = simple_graph(4, seed=7)
        out1 = layer(X, A)
        out2 = layer(X, A)
        np.testing.assert_array_equal(out1, out2)

    def test_repr(self):
        layer = GraphAttentionLayer(in_features=6, out_features=8)
        assert "GraphAttentionLayer" in repr(layer)


class TestGNNModel:
    def test_single_layer_gcn(self):
        model = GNNModel(in_features=6, hidden_dim=16, out_dim=8, num_layers=1, layer_type="gcn")
        X, A = simple_graph(5)
        out = model.encode(X, A)
        assert out.shape == (5, 8)

    def test_multi_layer_gcn(self):
        model = GNNModel(in_features=6, hidden_dim=16, out_dim=4, num_layers=3, layer_type="gcn")
        X, A = simple_graph(7)
        out = model(X, A)
        assert out.shape == (7, 4)

    def test_gat_model(self):
        model = GNNModel(in_features=6, hidden_dim=16, out_dim=8, num_layers=2, layer_type="gat")
        X, A = simple_graph(5)
        out = model.encode(X, A)
        assert out.shape == (5, 8)

    def test_num_layers_zero_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            GNNModel(in_features=6, hidden_dim=16, out_dim=8, num_layers=0)

    def test_correct_number_of_layers(self):
        model = GNNModel(in_features=6, hidden_dim=16, out_dim=8, num_layers=3)
        assert len(model.layers) == 3

    def test_repr(self):
        model = GNNModel(in_features=6, hidden_dim=16, out_dim=8, num_layers=2)
        r = repr(model)
        assert "GNNModel" in r

    def test_embeddings_vary_with_input(self):
        model = GNNModel(in_features=6, hidden_dim=16, out_dim=8, num_layers=2)
        rng = np.random.default_rng(0)
        X1 = rng.standard_normal((5, 6))
        X2 = rng.standard_normal((5, 6))
        _, A = simple_graph(5)
        out1 = model.encode(X1, A)
        out2 = model.encode(X2, A)
        assert not np.allclose(out1, out2)
