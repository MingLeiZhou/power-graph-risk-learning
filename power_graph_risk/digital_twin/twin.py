"""
Digital twin for power system monitoring and simulation.

A digital twin maintains a live, data-driven replica of the physical
power grid.  It ingests real-time measurements, performs state
estimation, detects anomalies, tracks risk over time, and can run
what-if contingency simulations.

Key classes
-----------
* :class:`TwinSnapshot` — an immutable record of a single time step.
* :class:`DigitalTwin` — the main digital-twin controller.
"""

from __future__ import annotations

import copy
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from power_graph_risk.data.power_grid import BusNode, NodeType, PowerGridGraph
from power_graph_risk.risk.assessor import RiskAssessor, RiskReport


@dataclass
class TwinSnapshot:
    """Immutable record of the digital twin state at one time step.

    Attributes
    ----------
    timestamp:
        Unix timestamp of the snapshot.
    bus_states:
        Mapping from bus_id to a dict of measured bus attributes.
    risk_report:
        Risk report computed at this time step (may be ``None`` if
        the assessor has not been fitted yet).
    anomalies:
        List of bus ids flagged as anomalous at this time step.
    """

    timestamp: float
    bus_states: Dict[int, Dict[str, float]]
    risk_report: Optional[RiskReport] = None
    anomalies: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return (
            f"TwinSnapshot(time={ts}, "
            f"anomalies={self.anomalies}, "
            f"system_risk="
            f"{self.risk_report.system_risk:.4f if self.risk_report else 'N/A'})"
        )


class AnomalyDetector:
    """Simple statistical anomaly detector for bus measurements.

    Each bus is modelled independently.  Measurements are flagged as
    anomalous if they deviate more than ``threshold`` standard
    deviations from the rolling mean.

    Parameters
    ----------
    window:
        Number of historical observations to consider.
    threshold:
        Z-score threshold above which a sample is flagged.
    """

    def __init__(self, window: int = 20, threshold: float = 3.0) -> None:
        self.window = window
        self.threshold = threshold
        self._history: Dict[int, Deque[np.ndarray]] = {}

    def update(
        self, bus_id: int, feature_vector: np.ndarray
    ) -> bool:
        """Update history for a bus and return whether it is anomalous.

        Parameters
        ----------
        bus_id:
            Identifier of the bus.
        feature_vector:
            Current node feature vector.

        Returns
        -------
        bool
            ``True`` if the current measurement is anomalous.
        """
        if bus_id not in self._history:
            self._history[bus_id] = deque(maxlen=self.window)
        hist = self._history[bus_id]
        is_anomaly = False
        if len(hist) >= 5:
            history_array = np.stack(hist)
            mean = history_array.mean(axis=0)
            std = history_array.std(axis=0) + 1e-8
            z_scores = np.abs((feature_vector - mean) / std)
            if z_scores.max() > self.threshold:
                is_anomaly = True
        hist.append(feature_vector.copy())
        return is_anomaly

    def reset(self, bus_id: Optional[int] = None) -> None:
        """Clear history for one bus or all buses.

        Parameters
        ----------
        bus_id:
            Bus to reset.  If ``None``, all histories are cleared.
        """
        if bus_id is None:
            self._history.clear()
        elif bus_id in self._history:
            self._history[bus_id].clear()


class StateEstimator:
    """Weighted-least-squares state estimator for power grid measurements.

    Given noisy measurements of node features, this estimator filters
    them using a simple exponential moving average (EMA) to produce
    smoothed state estimates.

    Parameters
    ----------
    alpha:
        EMA smoothing factor in ``(0, 1]``.  Smaller values produce
        more smoothing.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None  # (N, F)

    def update(self, X_measured: np.ndarray) -> np.ndarray:
        """Incorporate a new measurement matrix and return the estimate.

        Parameters
        ----------
        X_measured:
            Measured node feature matrix, shape ``(N, F)``.

        Returns
        -------
        np.ndarray, shape ``(N, F)``
            Smoothed state estimate.
        """
        if self._state is None:
            self._state = X_measured.copy()
        else:
            self._state = self.alpha * X_measured + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        """Reset internal state."""
        self._state = None


class DigitalTwin:
    """Digital twin controller for a power grid.

    The twin maintains a reference copy of the grid, a state estimator,
    an anomaly detector, and a risk assessor.  Each time new
    measurements arrive via :py:meth:`update`, the twin:

    1. Updates the internal grid state with measured values.
    2. Runs the state estimator to obtain smoothed features.
    3. Checks for anomalies.
    4. Computes a new :class:`RiskReport` (if the assessor is fitted).
    5. Stores a :class:`TwinSnapshot`.

    Parameters
    ----------
    grid:
        Reference power grid.  A deep copy is held internally so the
        original is not mutated.
    assessor:
        A :class:`~power_graph_risk.risk.assessor.RiskAssessor`.
        If ``None`` a default assessor is created.
    history_len:
        Maximum number of :class:`TwinSnapshot` records to retain.
    anomaly_window:
        Window size for the anomaly detector.
    anomaly_threshold:
        Z-score threshold for anomaly detection.
    ema_alpha:
        EMA smoothing parameter for the state estimator.
    """

    def __init__(
        self,
        grid: PowerGridGraph,
        assessor: Optional[RiskAssessor] = None,
        history_len: int = 100,
        anomaly_window: int = 20,
        anomaly_threshold: float = 3.0,
        ema_alpha: float = 0.3,
    ) -> None:
        self._grid: PowerGridGraph = copy.deepcopy(grid)
        self.assessor: RiskAssessor = assessor or RiskAssessor()
        self._estimator = StateEstimator(alpha=ema_alpha)
        self._anomaly_detector = AnomalyDetector(
            window=anomaly_window, threshold=anomaly_threshold
        )
        self._history: Deque[TwinSnapshot] = deque(maxlen=history_len)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> PowerGridGraph:
        """Current internal grid state (read-only copy)."""
        return copy.deepcopy(self._grid)

    @property
    def history(self) -> List[TwinSnapshot]:
        """All retained snapshots (oldest first)."""
        return list(self._history)

    @property
    def latest_snapshot(self) -> Optional[TwinSnapshot]:
        """Most recent snapshot, or ``None`` if no updates yet."""
        return self._history[-1] if self._history else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        targets: Optional[np.ndarray] = None,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> List[float]:
        """Fit the risk assessor on the reference grid.

        Parameters
        ----------
        targets:
            Optional manually specified per-bus risk labels.
        epochs, lr:
            Training hyper-parameters passed to
            :py:meth:`RiskAssessor.fit`.

        Returns
        -------
        list of float
            Training losses.
        """
        return self.assessor.fit(self._grid, targets=targets, epochs=epochs, lr=lr)

    def update(
        self,
        measurements: Dict[int, Dict[str, float]],
        timestamp: Optional[float] = None,
    ) -> TwinSnapshot:
        """Ingest new measurements and advance the twin state.

        Parameters
        ----------
        measurements:
            Mapping from bus_id to a dict of field updates.  Accepted
            keys are any attribute names on :class:`BusNode` (e.g.
            ``"voltage_mag"``, ``"p_load"``).  Unknown keys are silently
            ignored.
        timestamp:
            Unix timestamp for this update.  Defaults to current time.

        Returns
        -------
        :class:`TwinSnapshot`
            The snapshot produced by this update.
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. Apply measurements to internal grid
        bus_states: Dict[int, Dict[str, float]] = {}
        for bus_id, attrs in measurements.items():
            if bus_id not in self._grid.buses:
                warnings.warn(f"Bus {bus_id} not in twin grid; skipping.", UserWarning, stacklevel=2)
                continue
            safe_attrs = {k: v for k, v in attrs.items() if hasattr(BusNode(0), k)}
            self._grid.update_bus(bus_id, **safe_attrs)
            bus_states[bus_id] = dict(attrs)

        # 2. State estimation (smooth feature matrix)
        X_raw = self._grid.node_feature_matrix()
        X_est = self._estimator.update(X_raw)

        # 3. Anomaly detection — check each updated bus
        anomalies: List[int] = []
        idx_map = self._grid.node_index_map()
        for bus_id in measurements:
            if bus_id not in idx_map:
                continue
            idx = idx_map[bus_id]
            if self._anomaly_detector.update(bus_id, X_est[idx]):
                anomalies.append(bus_id)

        # 4. Risk assessment
        risk_report: Optional[RiskReport] = None
        if self.assessor._fitted:
            try:
                risk_report = self.assessor.assess(self._grid)
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Risk assessment failed: {exc}", UserWarning, stacklevel=2)

        # 5. Record snapshot
        snapshot = TwinSnapshot(
            timestamp=timestamp,
            bus_states=bus_states,
            risk_report=risk_report,
            anomalies=anomalies,
        )
        self._history.append(snapshot)
        return snapshot

    def simulate_contingency(
        self,
        trip_lines: List[Tuple[int, int]],
    ) -> RiskReport:
        """Run a what-if simulation by tripping one or more lines.

        The simulation operates on a temporary deep copy of the internal
        grid so the live state is not modified.

        Parameters
        ----------
        trip_lines:
            List of ``(from_bus, to_bus)`` pairs to take out of service.

        Returns
        -------
        :class:`~power_graph_risk.risk.assessor.RiskReport`
            Risk report for the contingency scenario.
        """
        sim_grid = copy.deepcopy(self._grid)
        for u, v in trip_lines:
            try:
                sim_grid.set_line_service(u, v, False)
            except KeyError:
                warnings.warn(f"Line ({u}, {v}) not found; skipping.", UserWarning, stacklevel=2)
        return self.assessor.assess(sim_grid)

    def risk_trend(self) -> List[Tuple[float, float]]:
        """Return the system-risk time series from stored snapshots.

        Returns
        -------
        list of ``(timestamp, system_risk)`` tuples.
        """
        return [
            (snap.timestamp, snap.risk_report.system_risk)
            for snap in self._history
            if snap.risk_report is not None
        ]

    def __repr__(self) -> str:
        return (
            f"DigitalTwin("
            f"grid={self._grid!r}, "
            f"snapshots={len(self._history)}, "
            f"fitted={self.assessor._fitted})"
        )
