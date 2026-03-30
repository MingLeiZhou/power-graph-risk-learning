"""
Risk score prediction head.

This module provides a lightweight multi-layer-perceptron (MLP) head that
maps GNN node embeddings to scalar risk scores, and a simple training loop
using mean-squared-error loss with gradient descent.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LinearLayer:
    """Fully-connected layer with optional ReLU activation.

    Parameters
    ----------
    in_dim:
        Input dimension.
    out_dim:
        Output dimension.
    activation:
        ``"relu"`` or ``None``.
    seed:
        Random seed for weight initialisation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Optional[str] = "relu",
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation
        # Cache for backward pass
        self._input: Optional[np.ndarray] = None
        self._pre_act: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        z = x @ self.W + self.b
        self._pre_act = z
        return _relu(z) if self.activation == "relu" else z

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Compute gradients and update weights in-place.

        Parameters
        ----------
        grad_out:
            Gradient of the loss w.r.t. the output of this layer.
        lr:
            Learning rate.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. the input of this layer.
        """
        if self.activation == "relu":
            grad_out = grad_out * _relu_grad(self._pre_act)
        grad_W = self._input.T @ grad_out
        grad_b = grad_out.sum(axis=0)
        grad_in = grad_out @ self.W.T
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_in


class RiskScoreHead:
    """MLP that maps node embeddings to per-node risk scores in [0, 1].

    Parameters
    ----------
    in_dim:
        Dimension of the input node embeddings.
    hidden_dims:
        Sizes of hidden layers.  Defaults to ``[64, 32]``.
    seed:
        Base random seed.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Optional[List[int]] = None,
        seed: int = 0,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [64, 32]
        dims = [in_dim] + hidden_dims + [1]
        self.layers: List[LinearLayer] = []
        for i, (fin, fout) in enumerate(zip(dims[:-1], dims[1:])):
            act = "relu" if i < len(dims) - 2 else None
            self.layers.append(LinearLayer(fin, fout, activation=act, seed=seed + i))

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute risk scores.

        Parameters
        ----------
        embeddings:
            Node embeddings, shape ``(N, in_dim)``.

        Returns
        -------
        np.ndarray, shape ``(N,)``
            Risk scores in ``[0, 1]`` (after sigmoid).
        """
        h = embeddings
        for layer in self.layers:
            h = layer.forward(h)
        return _sigmoid(h.squeeze(-1))

    def fit(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> List[float]:
        """Train the head on labelled node data using MSE loss.

        Parameters
        ----------
        embeddings:
            Node embeddings, shape ``(N, in_dim)``.
        targets:
            Target risk scores in ``[0, 1]``, shape ``(N,)``.
        epochs:
            Number of gradient-descent steps.
        lr:
            Learning rate.

        Returns
        -------
        list of float
            Loss at each epoch.
        """
        losses = []
        for _ in range(epochs):
            preds = self.forward(embeddings)
            loss = float(np.mean((preds - targets) ** 2))
            losses.append(loss)

            # Backprop — gradient of MSE through sigmoid
            N = len(targets)
            grad = 2.0 * (preds - targets) / N
            # grad through sigmoid: sigma * (1 - sigma)
            sig = preds
            grad = grad * sig * (1.0 - sig)
            grad = grad[:, None]  # (N, 1)
            for layer in reversed(self.layers):
                grad = layer.backward(grad, lr)
        return losses

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        return self.forward(embeddings)

    def __repr__(self) -> str:
        dims = [self.layers[0].W.shape[0]] + [l.W.shape[1] for l in self.layers]
        return f"RiskScoreHead(dims={dims})"
