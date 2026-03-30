"""
Graph Neural Network layers and model.

This module implements GNN building blocks using plain NumPy so that the
framework runs without a deep-learning framework dependency.

Three components are provided:

* :class:`GraphConvLayer` — a graph convolutional layer (GCN-style,
  Kipf & Welling 2017).
* :class:`GraphAttentionLayer` — a simplified graph attention layer
  (GAT-style, Veličković et al. 2018).
* :class:`GNNModel` — a stacked multi-layer GNN that supports either
  layer type and produces node embeddings for downstream tasks.
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x >= 0, x, alpha * x)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / (exp.sum(axis=axis, keepdims=True) + 1e-12)


def _glorot_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class GraphConvLayer:
    """Graph convolutional layer (GCN).

    Computes :math:`H' = \\hat{A} H W + b` where
    :math:`\\hat{A} = \\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2}` is
    the symmetrically normalised adjacency with self-loops.

    Parameters
    ----------
    in_features:
        Dimensionality of the input node features.
    out_features:
        Dimensionality of the output node embeddings.
    bias:
        Whether to add a bias term.
    activation:
        Activation function to apply after the linear transform.
        ``"relu"``, ``"leaky_relu"``, or ``None`` for no activation.
    seed:
        Random seed used for weight initialisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[Literal["relu", "leaky_relu"]] = "relu",
        seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        rng = np.random.default_rng(seed)
        self.W: np.ndarray = _glorot_uniform(in_features, out_features, rng)
        self.b: Optional[np.ndarray] = np.zeros(out_features) if bias else None

    def _normalise_adj(self, A: np.ndarray) -> np.ndarray:
        """Add self-loops and symmetrically normalise."""
        A_tilde = A + np.eye(A.shape[0])
        d = A_tilde.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        X:
            Node feature matrix, shape ``(N, in_features)``.
        A:
            Adjacency matrix, shape ``(N, N)``.  Need not be normalised.

        Returns
        -------
        np.ndarray, shape ``(N, out_features)``
        """
        A_hat = self._normalise_adj(A)
        out = A_hat @ X @ self.W
        if self.b is not None:
            out = out + self.b
        if self.activation == "relu":
            out = _relu(out)
        elif self.activation == "leaky_relu":
            out = _leaky_relu(out)
        return out

    def __call__(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        return self.forward(X, A)

    def __repr__(self) -> str:
        return (
            f"GraphConvLayer(in={self.in_features}, out={self.out_features}, "
            f"activation={self.activation})"
        )


class GraphAttentionLayer:
    """Simplified graph attention layer (GAT).

    Computes multi-head attention-weighted aggregation of neighbour
    features.  For simplicity a single attention head is implemented here
    (extend to multi-head by stacking).

    Parameters
    ----------
    in_features:
        Dimensionality of the input node features.
    out_features:
        Dimensionality of the output node embeddings.
    bias:
        Whether to add a bias term after aggregation.
    activation:
        Activation function applied after the linear transform.
    seed:
        Random seed for weight initialisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[Literal["relu", "leaky_relu"]] = "leaky_relu",
        seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        rng = np.random.default_rng(seed)
        self.W: np.ndarray = _glorot_uniform(in_features, out_features, rng)
        # Attention vector: concat of two transformed feature vectors
        self.a: np.ndarray = rng.uniform(-0.1, 0.1, size=(2 * out_features,))
        self.b: Optional[np.ndarray] = np.zeros(out_features) if bias else None

    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Forward pass with additive attention.

        Parameters
        ----------
        X:
            Node feature matrix, shape ``(N, in_features)``.
        A:
            Adjacency matrix, shape ``(N, N)``.

        Returns
        -------
        np.ndarray, shape ``(N, out_features)``
        """
        N = X.shape[0]
        H = X @ self.W  # (N, out_features)

        # Compute attention coefficients
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        a_left = self.a[: self.out_features]    # (out_features,)
        a_right = self.a[self.out_features :]   # (out_features,)

        e_left = H @ a_left    # (N,)
        e_right = H @ a_right  # (N,)
        # Broadcasting: e[i,j] = e_left[i] + e_right[j]
        E = e_left[:, None] + e_right[None, :]  # (N, N)
        E = _leaky_relu(E)

        # Mask: only attend to neighbours (including self)
        mask = A + np.eye(N)
        E = np.where(mask > 0, E, -1e9)
        alpha = _softmax(E, axis=1)  # (N, N)

        # Aggregate
        out = alpha @ H  # (N, out_features)
        if self.b is not None:
            out = out + self.b
        if self.activation == "relu":
            out = _relu(out)
        elif self.activation == "leaky_relu":
            out = _leaky_relu(out)
        return out

    def __call__(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        return self.forward(X, A)

    def __repr__(self) -> str:
        return (
            f"GraphAttentionLayer(in={self.in_features}, out={self.out_features}, "
            f"activation={self.activation})"
        )


class GNNModel:
    """Stacked GNN encoder that produces node-level embeddings.

    Layers are stacked depth-first; each hidden layer transforms the
    ``in_features`` → ``hidden_dim``, and the final layer maps to
    ``out_dim``.

    Parameters
    ----------
    in_features:
        Dimension of raw node features.
    hidden_dim:
        Width of hidden layers.
    out_dim:
        Dimension of the output node embeddings.
    num_layers:
        Total number of GNN layers (must be ≥ 1).
    layer_type:
        ``"gcn"`` for :class:`GraphConvLayer`, or ``"gat"`` for
        :class:`GraphAttentionLayer`.
    seed:
        Base random seed; each layer gets ``seed + layer_index``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        layer_type: Literal["gcn", "gat"] = "gcn",
        seed: int = 42,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layer_type = layer_type

        layer_cls = GraphConvLayer if layer_type == "gcn" else GraphAttentionLayer
        self.layers: List = []

        for i in range(num_layers):
            fin = in_features if i == 0 else hidden_dim
            fout = out_dim if i == num_layers - 1 else hidden_dim
            act = "relu" if i < num_layers - 1 else None
            self.layers.append(layer_cls(fin, fout, activation=act, seed=seed + i))

    def encode(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Produce node embeddings by forward-passing through all layers.

        Parameters
        ----------
        X:
            Node feature matrix, shape ``(N, in_features)``.
        A:
            Adjacency matrix, shape ``(N, N)``.

        Returns
        -------
        np.ndarray, shape ``(N, out_dim)``
            Node-level embeddings.
        """
        H = X
        for layer in self.layers:
            H = layer(H, A)
        return H

    def __call__(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        return self.encode(X, A)

    def __repr__(self) -> str:
        layer_reprs = "\n  ".join(repr(l) for l in self.layers)
        return f"GNNModel(\n  {layer_reprs}\n)"
