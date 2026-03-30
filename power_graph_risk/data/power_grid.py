"""
Power grid graph representation.

This module provides a graph-based model of a power system where:
  - Nodes represent buses (including generators and loads).
  - Edges represent transmission lines.

Node features capture electrical state (voltage magnitude and angle,
active/reactive power injection).  Edge features capture line
parameters (resistance, reactance, susceptance, thermal limit).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class NodeType(enum.Enum):
    """Bus types used in power-flow analysis."""

    SLACK = "slack"        # Reference bus (voltage angle fixed to 0)
    PV = "pv"              # Generator bus (voltage magnitude and P fixed)
    PQ = "pq"              # Load bus (P and Q injections fixed)


@dataclass
class BusNode:
    """Attributes attached to a bus node.

    Parameters
    ----------
    node_id:
        Unique integer identifier of the bus.
    node_type:
        Bus type (SLACK, PV, or PQ).
    voltage_mag:
        Voltage magnitude in per-unit (p.u.).  Defaults to 1.0 p.u.
    voltage_angle:
        Voltage angle in radians.  Defaults to 0.
    p_inject:
        Active power injection in MW (positive = generation).
    q_inject:
        Reactive power injection in MVAR.
    p_load:
        Active power demand in MW.
    q_load:
        Reactive power demand in MVAR.
    """

    node_id: int
    node_type: NodeType = NodeType.PQ
    voltage_mag: float = 1.0
    voltage_angle: float = 0.0
    p_inject: float = 0.0
    q_inject: float = 0.0
    p_load: float = 0.0
    q_load: float = 0.0

    def feature_vector(self) -> np.ndarray:
        """Return a fixed-length numpy feature vector for this bus.

        Features (length 6):
            [voltage_mag, voltage_angle, p_inject, q_inject, p_load, q_load]
        """
        return np.array(
            [
                self.voltage_mag,
                self.voltage_angle,
                self.p_inject,
                self.q_inject,
                self.p_load,
                self.q_load,
            ],
            dtype=np.float64,
        )


@dataclass
class LineEdge:
    """Attributes attached to a transmission line edge.

    Parameters
    ----------
    from_bus:
        Identifier of the sending-end bus.
    to_bus:
        Identifier of the receiving-end bus.
    resistance:
        Series resistance in p.u.
    reactance:
        Series reactance in p.u.
    susceptance:
        Shunt susceptance in p.u.
    thermal_limit:
        Maximum apparent power flow (MVA).
    in_service:
        Whether the line is currently energised.
    """

    from_bus: int
    to_bus: int
    resistance: float = 0.0
    reactance: float = 0.01
    susceptance: float = 0.0
    thermal_limit: float = 100.0
    in_service: bool = True

    @property
    def impedance(self) -> complex:
        """Complex series impedance (R + jX)."""
        return complex(self.resistance, self.reactance)

    @property
    def admittance(self) -> complex:
        """Complex series admittance (inverse of impedance)."""
        z = self.impedance
        if abs(z) < 1e-12:
            return complex(0.0, 0.0)
        return 1.0 / z

    def feature_vector(self) -> np.ndarray:
        """Return a fixed-length numpy feature vector for this line.

        Features (length 5):
            [resistance, reactance, susceptance, thermal_limit, in_service]
        """
        return np.array(
            [
                self.resistance,
                self.reactance,
                self.susceptance,
                self.thermal_limit,
                float(self.in_service),
            ],
            dtype=np.float64,
        )


class PowerGridGraph:
    """Graph representation of a power system.

    The underlying data structure is a :class:`networkx.Graph` (undirected
    for topology analysis; a directed view is available via
    :py:attr:`directed`).

    Node feature matrix and adjacency matrix can be exported as numpy
    arrays for use with graph-learning models.
    """

    NODE_FEATURE_DIM: int = 6
    EDGE_FEATURE_DIM: int = 5

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        # Map from node_id -> BusNode for quick lookup
        self._buses: Dict[int, BusNode] = {}
        # Map from (from_bus, to_bus) -> LineEdge (canonical order: from < to)
        self._lines: Dict[Tuple[int, int], LineEdge] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_bus(self, bus: BusNode) -> None:
        """Add a bus node to the grid.

        Parameters
        ----------
        bus:
            :class:`BusNode` to register.

        Raises
        ------
        ValueError
            If a bus with the same ``node_id`` already exists.
        """
        if bus.node_id in self._buses:
            raise ValueError(f"Bus {bus.node_id} already exists in the grid.")
        self._buses[bus.node_id] = bus
        self._graph.add_node(bus.node_id, data=bus)

    def add_line(self, line: LineEdge) -> None:
        """Add a transmission line edge to the grid.

        Parameters
        ----------
        line:
            :class:`LineEdge` to register.

        Raises
        ------
        ValueError
            If either endpoint bus has not been added yet, or if the line
            already exists.
        """
        if line.from_bus not in self._buses:
            raise ValueError(f"Bus {line.from_bus} not found in the grid.")
        if line.to_bus not in self._buses:
            raise ValueError(f"Bus {line.to_bus} not found in the grid.")
        key = (min(line.from_bus, line.to_bus), max(line.from_bus, line.to_bus))
        if key in self._lines:
            raise ValueError(
                f"Line between {line.from_bus} and {line.to_bus} already exists."
            )
        self._lines[key] = line
        self._graph.add_edge(line.from_bus, line.to_bus, data=line)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        buses: List[Dict],
        lines: List[Dict],
    ) -> "PowerGridGraph":
        """Build a :class:`PowerGridGraph` from plain Python dictionaries.

        Parameters
        ----------
        buses:
            List of dicts, each with keys matching :class:`BusNode` fields.
            ``node_type`` values should be strings (``"slack"``, ``"pv"``,
            ``"pq"``) or :class:`NodeType` members.
        lines:
            List of dicts, each with keys matching :class:`LineEdge` fields.

        Returns
        -------
        PowerGridGraph
        """
        grid = cls()
        for b in buses:
            b = dict(b)
            nt = b.get("node_type", "pq")
            if isinstance(nt, str):
                b["node_type"] = NodeType(nt)
            grid.add_bus(BusNode(**b))
        for ln in lines:
            grid.add_line(LineEdge(**ln))
        return grid

    @classmethod
    def ieee_14_bus(cls) -> "PowerGridGraph":
        """Return a simplified version of the IEEE 14-bus test system.

        The network has 14 buses and 20 transmission lines.  Values are
        representative (not a full power-flow solution) and are intended
        for demonstration and testing.
        """
        buses = [
            {"node_id": 1, "node_type": "slack", "voltage_mag": 1.06, "p_inject": 232.4, "q_inject": -16.9},
            {"node_id": 2, "node_type": "pv",    "voltage_mag": 1.045, "p_inject": 40.0, "q_inject": 42.4, "p_load": 21.7, "q_load": 12.7},
            {"node_id": 3, "node_type": "pv",    "voltage_mag": 1.01,  "p_inject": 0.0,  "q_inject": 23.4, "p_load": 94.2, "q_load": 19.0},
            {"node_id": 4, "node_type": "pq",    "p_load": 47.8, "q_load": -3.9},
            {"node_id": 5, "node_type": "pq",    "p_load": 7.6,  "q_load": 1.6},
            {"node_id": 6, "node_type": "pv",    "voltage_mag": 1.07,  "p_inject": 0.0,  "q_inject": 12.2, "p_load": 11.2, "q_load": 7.5},
            {"node_id": 7, "node_type": "pq"},
            {"node_id": 8, "node_type": "pv",    "voltage_mag": 1.09,  "p_inject": 0.0,  "q_inject": 17.4},
            {"node_id": 9, "node_type": "pq",    "p_load": 29.5, "q_load": 16.6},
            {"node_id": 10, "node_type": "pq",   "p_load": 9.0,  "q_load": 5.8},
            {"node_id": 11, "node_type": "pq",   "p_load": 3.5,  "q_load": 1.8},
            {"node_id": 12, "node_type": "pq",   "p_load": 6.1,  "q_load": 1.6},
            {"node_id": 13, "node_type": "pq",   "p_load": 13.5, "q_load": 5.8},
            {"node_id": 14, "node_type": "pq",   "p_load": 14.9, "q_load": 5.0},
        ]
        lines = [
            {"from_bus": 1,  "to_bus": 2,  "resistance": 0.01938, "reactance": 0.05917, "thermal_limit": 100.0},
            {"from_bus": 1,  "to_bus": 5,  "resistance": 0.05403, "reactance": 0.22304, "thermal_limit": 100.0},
            {"from_bus": 2,  "to_bus": 3,  "resistance": 0.04699, "reactance": 0.19797, "thermal_limit": 100.0},
            {"from_bus": 2,  "to_bus": 4,  "resistance": 0.05811, "reactance": 0.17632, "thermal_limit": 100.0},
            {"from_bus": 2,  "to_bus": 5,  "resistance": 0.05695, "reactance": 0.17388, "thermal_limit": 100.0},
            {"from_bus": 3,  "to_bus": 4,  "resistance": 0.06701, "reactance": 0.17103, "thermal_limit": 100.0},
            {"from_bus": 4,  "to_bus": 5,  "resistance": 0.01335, "reactance": 0.04211, "thermal_limit": 100.0},
            {"from_bus": 4,  "to_bus": 7,  "resistance": 0.0,     "reactance": 0.20912, "thermal_limit": 100.0},
            {"from_bus": 4,  "to_bus": 9,  "resistance": 0.0,     "reactance": 0.55618, "thermal_limit": 100.0},
            {"from_bus": 5,  "to_bus": 6,  "resistance": 0.0,     "reactance": 0.25202, "thermal_limit": 100.0},
            {"from_bus": 6,  "to_bus": 11, "resistance": 0.09498, "reactance": 0.19890, "thermal_limit": 100.0},
            {"from_bus": 6,  "to_bus": 12, "resistance": 0.12291, "reactance": 0.25581, "thermal_limit": 100.0},
            {"from_bus": 6,  "to_bus": 13, "resistance": 0.06615, "reactance": 0.13027, "thermal_limit": 100.0},
            {"from_bus": 7,  "to_bus": 8,  "resistance": 0.0,     "reactance": 0.17615, "thermal_limit": 100.0},
            {"from_bus": 7,  "to_bus": 9,  "resistance": 0.0,     "reactance": 0.11001, "thermal_limit": 100.0},
            {"from_bus": 9,  "to_bus": 10, "resistance": 0.03181, "reactance": 0.08450, "thermal_limit": 100.0},
            {"from_bus": 9,  "to_bus": 14, "resistance": 0.12711, "reactance": 0.27038, "thermal_limit": 100.0},
            {"from_bus": 10, "to_bus": 11, "resistance": 0.08205, "reactance": 0.19207, "thermal_limit": 100.0},
            {"from_bus": 12, "to_bus": 13, "resistance": 0.22092, "reactance": 0.19988, "thermal_limit": 100.0},
            {"from_bus": 13, "to_bus": 14, "resistance": 0.17093, "reactance": 0.34802, "thermal_limit": 100.0},
        ]
        return cls.from_dict(buses, lines)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_buses(self) -> int:
        """Number of buses in the grid."""
        return len(self._buses)

    @property
    def num_lines(self) -> int:
        """Number of transmission lines."""
        return len(self._lines)

    @property
    def buses(self) -> Dict[int, BusNode]:
        """Read-only view of the bus registry."""
        return dict(self._buses)

    @property
    def lines(self) -> Dict[Tuple[int, int], LineEdge]:
        """Read-only view of the line registry."""
        return dict(self._lines)

    @property
    def directed(self) -> nx.DiGraph:
        """A directed view of the topology (edges in both directions)."""
        return self._graph.to_directed()

    # ------------------------------------------------------------------
    # Graph matrix export
    # ------------------------------------------------------------------

    def node_index_map(self) -> Dict[int, int]:
        """Return a mapping from bus id to a contiguous 0-based index."""
        return {bus_id: idx for idx, bus_id in enumerate(sorted(self._buses.keys()))}

    def node_feature_matrix(self) -> np.ndarray:
        """Build and return the node feature matrix.

        Returns
        -------
        np.ndarray, shape (N, NODE_FEATURE_DIM)
            Rows are ordered by ascending bus id.
        """
        idx_map = self.node_index_map()
        n = len(idx_map)
        X = np.zeros((n, self.NODE_FEATURE_DIM), dtype=np.float64)
        for bus_id, bus in self._buses.items():
            X[idx_map[bus_id]] = bus.feature_vector()
        return X

    def adjacency_matrix(self, weighted: bool = False) -> np.ndarray:
        """Build and return the adjacency matrix.

        Parameters
        ----------
        weighted:
            If ``True``, entries contain the line admittance magnitude
            (1 / |Z|) rather than 1.

        Returns
        -------
        np.ndarray, shape (N, N)
            Symmetric adjacency matrix ordered by ascending bus id.
        """
        idx_map = self.node_index_map()
        n = len(idx_map)
        A = np.zeros((n, n), dtype=np.float64)
        for (u, v), line in self._lines.items():
            if not line.in_service:
                continue
            i, j = idx_map[u], idx_map[v]
            weight = abs(line.admittance) if weighted else 1.0
            A[i, j] = weight
            A[j, i] = weight
        return A

    def edge_index(self) -> np.ndarray:
        """Return the edge index array used in graph-learning models.

        Returns
        -------
        np.ndarray, shape (2, E)
            Each column ``[i, j]`` is an undirected edge between nodes
            with 0-based indices.  Both ``(i, j)`` and ``(j, i)`` are
            included so that message passing is bidirectional.
        """
        idx_map = self.node_index_map()
        rows, cols = [], []
        for (u, v), line in self._lines.items():
            if not line.in_service:
                continue
            i, j = idx_map[u], idx_map[v]
            rows += [i, j]
            cols += [j, i]
        return np.array([rows, cols], dtype=np.int64)

    def edge_feature_matrix(self) -> np.ndarray:
        """Return edge features aligned with :py:meth:`edge_index`.

        Returns
        -------
        np.ndarray, shape (2*E, EDGE_FEATURE_DIM)
            Each pair of rows represents one undirected line (both
            directions share the same feature vector).
        """
        feats = []
        for (u, v), line in self._lines.items():
            if not line.in_service:
                continue
            fv = line.feature_vector()
            feats.append(fv)
            feats.append(fv)
        return np.array(feats, dtype=np.float64) if feats else np.empty((0, self.EDGE_FEATURE_DIM))

    def laplacian(self, normalised: bool = True) -> np.ndarray:
        """Return the graph Laplacian matrix.

        Parameters
        ----------
        normalised:
            If ``True`` return the symmetrically normalised Laplacian
            :math:`L_{sym} = I - D^{-1/2} A D^{-1/2}`.
            Otherwise return the combinatorial Laplacian :math:`L = D - A`.

        Returns
        -------
        np.ndarray, shape (N, N)
        """
        A = self.adjacency_matrix(weighted=False)
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        if not normalised:
            return L
        # Symmetric normalisation
        with np.errstate(divide="ignore", invalid="ignore"):
            D_inv_sqrt = np.where(degrees > 0, degrees ** -0.5, 0.0)
        D_inv_sqrt_mat = np.diag(D_inv_sqrt)
        return np.eye(len(degrees)) - D_inv_sqrt_mat @ A @ D_inv_sqrt_mat

    # ------------------------------------------------------------------
    # Topology analysis
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Return ``True`` if the grid graph is connected (all buses reachable)."""
        service_graph = nx.Graph()
        service_graph.add_nodes_from(self._buses.keys())
        for (u, v), line in self._lines.items():
            if line.in_service:
                service_graph.add_edge(u, v)
        return nx.is_connected(service_graph)

    def connected_components(self) -> List[List[int]]:
        """Return connected components as lists of bus ids."""
        service_graph = nx.Graph()
        service_graph.add_nodes_from(self._buses.keys())
        for (u, v), line in self._lines.items():
            if line.in_service:
                service_graph.add_edge(u, v)
        return [sorted(c) for c in nx.connected_components(service_graph)]

    def criticality_scores(self) -> Dict[int, float]:
        """Return betweenness-centrality scores for each bus.

        High betweenness centrality indicates buses that are critical
        for routing power through the network.

        Returns
        -------
        dict mapping bus_id -> centrality score in [0, 1].
        """
        return nx.betweenness_centrality(self._graph, normalized=True)

    def update_bus(self, node_id: int, **kwargs) -> None:
        """Update attributes of an existing bus.

        Parameters
        ----------
        node_id:
            Bus to update.
        **kwargs:
            Field names and new values from :class:`BusNode`.

        Raises
        ------
        KeyError
            If ``node_id`` is not registered.
        """
        bus = self._buses[node_id]
        for key, value in kwargs.items():
            if not hasattr(bus, key):
                raise AttributeError(f"BusNode has no attribute '{key}'.")
            setattr(bus, key, value)

    def set_line_service(self, from_bus: int, to_bus: int, in_service: bool) -> None:
        """Toggle the in-service status of a transmission line.

        Parameters
        ----------
        from_bus, to_bus:
            Endpoints of the line (order does not matter).
        in_service:
            New service state.

        Raises
        ------
        KeyError
            If the line does not exist.
        """
        key = (min(from_bus, to_bus), max(from_bus, to_bus))
        if key not in self._lines:
            raise KeyError(f"Line ({from_bus}, {to_bus}) not found.")
        self._lines[key].in_service = in_service
        # Keep NetworkX edge attribute in sync
        self._graph[from_bus][to_bus]["data"].in_service = in_service

    def __repr__(self) -> str:
        return (
            f"PowerGridGraph("
            f"buses={self.num_buses}, "
            f"lines={self.num_lines}, "
            f"connected={self.is_connected()})"
        )
