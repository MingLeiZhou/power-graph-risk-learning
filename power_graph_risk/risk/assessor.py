"""
Risk assessor for power systems.

This module integrates the GNN encoder with the risk-score head to
provide end-to-end risk assessment of a power grid.  It also implements
a cascade-failure simulator used to derive ground-truth risk labels for
training.

Key classes
-----------
* :class:`RiskReport` — data-class holding per-bus risk scores and
  summary statistics.
* :class:`RiskAssessor` — orchestrates feature extraction, GNN encoding,
  and risk-score prediction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from power_graph_risk.data.power_grid import PowerGridGraph
from power_graph_risk.models.gnn import GNNModel
from power_graph_risk.models.risk_model import RiskScoreHead


@dataclass
class RiskReport:
    """Container for risk-assessment results.

    Attributes
    ----------
    bus_risk_scores:
        Mapping from bus_id to its risk score in ``[0, 1]``.
    system_risk:
        Overall system risk score (weighted average of bus scores by
        degree).
    high_risk_buses:
        List of bus ids with risk score above ``threshold``.
    threshold:
        The threshold used to determine *high risk* buses.
    """

    bus_risk_scores: Dict[int, float]
    system_risk: float
    high_risk_buses: List[int]
    threshold: float = 0.5

    def __repr__(self) -> str:
        return (
            f"RiskReport("
            f"system_risk={self.system_risk:.4f}, "
            f"high_risk_buses={self.high_risk_buses}, "
            f"threshold={self.threshold})"
        )


class CascadeSimulator:
    """Simulate N-1 and N-k contingency cascade failures.

    A simplified DC-flow approximation is used:
    overloaded lines are tripped one by one until no violations remain
    or the system separates.

    Parameters
    ----------
    grid:
        The power grid to simulate.
    overload_factor:
        Fraction above thermal limit that triggers a trip.
        E.g. ``0.0`` means trip at exactly the thermal limit.
    max_iterations:
        Maximum cascade depth (rounds of tripping) before halting.
    """

    def __init__(
        self,
        grid: PowerGridGraph,
        overload_factor: float = 0.0,
        max_iterations: int = 10,
    ) -> None:
        self.grid = grid
        self.overload_factor = overload_factor
        self.max_iterations = max_iterations

    def _dc_power_flow(self, A: np.ndarray, p_inject: np.ndarray) -> np.ndarray:
        """Approximate DC power flow using graph Laplacian pseudo-inverse.

        Returns estimated line flows (ordered by adjacency matrix).
        """
        n = A.shape[0]
        D = np.diag(A.sum(axis=1))
        L = D - A
        # Use pseudo-inverse to handle singular Laplacian
        L_pinv = np.linalg.pinv(L)
        # Voltage angles
        theta = L_pinv @ p_inject
        return theta

    def n_minus_1_risk(self) -> Dict[Tuple[int, int], float]:
        """Compute N-1 contingency risk for each line.

        Returns
        -------
        dict mapping ``(from_bus, to_bus)`` → cascade-depth score.
            A higher score indicates removing that line causes more
            instability.
        """
        idx_map = self.grid.node_index_map()
        sorted_buses = sorted(self.grid.buses.keys())
        p_inject = np.array(
            [self.grid.buses[b].p_inject - self.grid.buses[b].p_load
             for b in sorted_buses],
            dtype=np.float64,
        )
        risk_scores: Dict[Tuple[int, int], float] = {}
        for key, line in self.grid.lines.items():
            if not line.in_service:
                continue
            # Temporarily trip this line
            self.grid.set_line_service(key[0], key[1], False)
            A_trip = self.grid.adjacency_matrix(weighted=True)
            comps = self.grid.connected_components()
            if len(comps) > 1:
                # Network separated — worst case
                score = 1.0
            else:
                theta = self._dc_power_flow(A_trip, p_inject)
                # Estimate flow imbalance as risk proxy
                score = min(1.0, float(np.std(theta)))
            risk_scores[key] = score
            # Restore line
            self.grid.set_line_service(key[0], key[1], True)
        return risk_scores

    def bus_risk_from_n1(self) -> Dict[int, float]:
        """Aggregate N-1 line risks to per-bus risk scores.

        A bus's risk is the maximum N-1 risk of any incident line,
        normalised to ``[0, 1]``.

        Returns
        -------
        dict mapping bus_id → risk in ``[0, 1]``.
        """
        line_risks = self.n_minus_1_risk()
        bus_risk: Dict[int, float] = {b: 0.0 for b in self.grid.buses}
        for (u, v), score in line_risks.items():
            bus_risk[u] = max(bus_risk[u], score)
            bus_risk[v] = max(bus_risk[v], score)
        max_r = max(bus_risk.values()) if bus_risk else 1.0
        if max_r > 0:
            bus_risk = {b: v / max_r for b, v in bus_risk.items()}
        return bus_risk


class RiskAssessor:
    """End-to-end risk assessment pipeline.

    Combines graph feature extraction, GNN encoding, and a trained
    :class:`~power_graph_risk.models.risk_model.RiskScoreHead` to
    produce per-bus risk scores.

    Parameters
    ----------
    gnn_model:
        A :class:`~power_graph_risk.models.gnn.GNNModel` instance.
    score_head:
        A :class:`~power_graph_risk.models.risk_model.RiskScoreHead`
        that maps GNN embeddings to scalar risk scores.
    risk_threshold:
        Score cutoff above which a bus is classified as high-risk.
    normalise_features:
        Whether to standardise node features before passing them to
        the GNN.
    """

    def __init__(
        self,
        gnn_model: Optional[GNNModel] = None,
        score_head: Optional[RiskScoreHead] = None,
        risk_threshold: float = 0.5,
        normalise_features: bool = True,
    ) -> None:
        if gnn_model is None:
            from power_graph_risk.data.power_grid import PowerGridGraph
            gnn_model = GNNModel(
                in_features=PowerGridGraph.NODE_FEATURE_DIM,
                hidden_dim=32,
                out_dim=16,
                num_layers=2,
                layer_type="gcn",
            )
        if score_head is None:
            score_head = RiskScoreHead(in_dim=gnn_model.out_dim, hidden_dims=[32])

        self.gnn_model = gnn_model
        self.score_head = score_head
        self.risk_threshold = risk_threshold
        self.normalise_features = normalise_features
        self._scaler: Optional[StandardScaler] = StandardScaler() if normalise_features else None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        grid: PowerGridGraph,
        targets: Optional[np.ndarray] = None,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> List[float]:
        """Fit the risk-score head on labelled or pseudo-labelled data.

        If ``targets`` is ``None``, pseudo-labels are generated using
        :class:`CascadeSimulator` (N-1 contingency analysis).

        Parameters
        ----------
        grid:
            The training power grid.
        targets:
            Per-bus risk scores in ``[0, 1]``, shape ``(N,)``.
            Bus order must match :py:meth:`PowerGridGraph.node_index_map`.
        epochs:
            Number of training epochs.
        lr:
            Learning rate.

        Returns
        -------
        list of float
            Training loss per epoch.
        """
        X = grid.node_feature_matrix()
        A = grid.adjacency_matrix()

        if self.normalise_features:
            X = self._scaler.fit_transform(X)

        if targets is None:
            sim = CascadeSimulator(grid)
            bus_risk = sim.bus_risk_from_n1()
            idx_map = grid.node_index_map()
            targets = np.array(
                [bus_risk[bid] for bid in sorted(idx_map.keys())],
                dtype=np.float64,
            )

        embeddings = self.gnn_model.encode(X, A)
        losses = self.score_head.fit(embeddings, targets, epochs=epochs, lr=lr)
        self._fitted = True
        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def assess(self, grid: PowerGridGraph) -> RiskReport:
        """Assess the risk of each bus in the given grid.

        Parameters
        ----------
        grid:
            Power grid to evaluate (may differ from the training grid).

        Returns
        -------
        :class:`RiskReport`
        """
        X = grid.node_feature_matrix()
        A = grid.adjacency_matrix()

        if self.normalise_features:
            if not self._fitted:
                warnings.warn(
                    "RiskAssessor has not been fitted; features will not be "
                    "normalised.  Call fit() before assess().",
                    UserWarning,
                    stacklevel=2,
                )
                X_norm = X
            else:
                X_norm = self._scaler.transform(X)
        else:
            X_norm = X

        embeddings = self.gnn_model.encode(X_norm, A)
        raw_scores = self.score_head.forward(embeddings)  # (N,)

        idx_map = grid.node_index_map()
        sorted_bus_ids = sorted(idx_map.keys())
        bus_risk_scores = {
            bid: float(raw_scores[idx]) for bid, idx in idx_map.items()
        }

        # Degree-weighted system risk
        A_mat = grid.adjacency_matrix()
        degrees = A_mat.sum(axis=1) + 1.0  # avoid zero-weight
        weights = degrees / degrees.sum()
        system_risk = float(np.dot(weights, raw_scores))

        high_risk = [bid for bid in sorted_bus_ids
                     if bus_risk_scores[bid] >= self.risk_threshold]

        return RiskReport(
            bus_risk_scores=bus_risk_scores,
            system_risk=system_risk,
            high_risk_buses=high_risk,
            threshold=self.risk_threshold,
        )

    def topk_vulnerable_buses(
        self, grid: PowerGridGraph, k: int = 5
    ) -> List[Tuple[int, float]]:
        """Return the top-k most vulnerable buses.

        Parameters
        ----------
        grid:
            Grid to evaluate.
        k:
            Number of buses to return.

        Returns
        -------
        list of ``(bus_id, risk_score)`` sorted descending by score.
        """
        report = self.assess(grid)
        ranked = sorted(
            report.bus_risk_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        return ranked[:k]

    def __repr__(self) -> str:
        return (
            f"RiskAssessor("
            f"gnn={self.gnn_model.__class__.__name__}, "
            f"head={self.score_head.__class__.__name__}, "
            f"threshold={self.risk_threshold}, "
            f"fitted={self._fitted})"
        )
