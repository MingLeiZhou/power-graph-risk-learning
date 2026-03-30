# power-graph-risk-learning

A data-driven framework for **risk assessment** and **digital twin modelling** in power systems using graph-based learning.

---

## Overview

Modern power grids are large, complex networks whose failure can have far-reaching consequences.
This framework combines **graph representation learning** with **data-driven risk scoring** and a **live digital twin** to deliver:

| Capability | Module |
|---|---|
| Power grid graph model (buses, transmission lines, electrical features) | `power_graph_risk.data` |
| Graph Convolutional Network (GCN) and Graph Attention Network (GAT) layers | `power_graph_risk.models.gnn` |
| Risk-score prediction head (trainable MLP) | `power_graph_risk.models.risk_model` |
| N-1 contingency simulator + end-to-end risk assessor | `power_graph_risk.risk` |
| Digital twin with state estimation and anomaly detection | `power_graph_risk.digital_twin` |
| Evaluation metrics (MAE, RMSE, AUC, vulnerability index) | `power_graph_risk.utils` |

---

## Architecture

```
               ┌─────────────────────────────────────────────────┐
               │               PowerGridGraph                    │
               │  Bus nodes (V, θ, P, Q)  +  Line edges (R,X,B) │
               └───────────────────┬─────────────────────────────┘
                                   │  node feature matrix X  (N×6)
                                   │  adjacency matrix A     (N×N)
                                   ▼
               ┌─────────────────────────────────────────────────┐
               │              GNNModel (GCN / GAT)               │
               │  2+ stacked graph convolutional layers           │
               └───────────────────┬─────────────────────────────┘
                                   │  node embeddings H     (N×d)
                                   ▼
               ┌─────────────────────────────────────────────────┐
               │             RiskScoreHead  (MLP)                │
               │  Trained with MSE loss on pseudo/manual labels  │
               └───────────────────┬─────────────────────────────┘
                                   │  per-bus risk scores   (N,) ∈ [0,1]
                                   ▼
               ┌─────────────────────────────────────────────────┐
               │                RiskReport                       │
               │  bus_risk_scores · system_risk · high_risk_buses│
               └─────────────────────────────────────────────────┘
                                   ▲
               ┌─────────────────────────────────────────────────┐
               │              DigitalTwin                        │
               │  StateEstimator (EMA)  +  AnomalyDetector (σ)  │
               │  Contingency simulation  +  Risk time-series    │
               └─────────────────────────────────────────────────┘
```

Pseudo-labels for training are generated automatically by a **N-1 contingency cascade simulator** that trips each line in turn and measures the resulting topological/flow disturbance.

---

## Installation

```bash
git clone https://github.com/MingLeiZhou/power-graph-risk-learning.git
cd power-graph-risk-learning
pip install -e .
```

**Requirements** (installed automatically):

| Package | Version |
|---|---|
| numpy | ≥ 1.21 |
| scipy | ≥ 1.7 |
| networkx | ≥ 2.6 |
| scikit-learn | ≥ 1.0 |
| pandas | ≥ 1.3 |

---

## Quick Start

### 1 – Build a power grid graph

```python
from power_graph_risk.data import PowerGridGraph, BusNode, LineEdge, NodeType

# Use the built-in IEEE 14-bus test system
grid = PowerGridGraph.ieee_14_bus()
print(grid)
# PowerGridGraph(buses=14, lines=20, connected=True)

# Or build your own
grid = PowerGridGraph()
grid.add_bus(BusNode(node_id=1, node_type=NodeType.SLACK, voltage_mag=1.05, p_inject=200.0))
grid.add_bus(BusNode(node_id=2, node_type=NodeType.PQ,    p_load=80.0, q_load=20.0))
grid.add_line(LineEdge(from_bus=1, to_bus=2, resistance=0.02, reactance=0.06, thermal_limit=100.0))
```

### 2 – Assess risk

```python
from power_graph_risk.risk import RiskAssessor

grid = PowerGridGraph.ieee_14_bus()

assessor = RiskAssessor()
losses = assessor.fit(grid, epochs=300, lr=1e-3)  # auto pseudo-labels via N-1 simulation
report = assessor.assess(grid)

print(report)
# RiskReport(system_risk=0.3821, high_risk_buses=[4, 5, 9], threshold=0.5)
print(assessor.topk_vulnerable_buses(grid, k=5))
# [(4, 0.72), (9, 0.68), (5, 0.61), (3, 0.55), (1, 0.51)]
```

### 3 – Create a digital twin

```python
from power_graph_risk.digital_twin import DigitalTwin

twin = DigitalTwin(grid)
twin.train(epochs=300)   # fits the risk assessor internally

# Ingest live measurements
snapshot = twin.update({
    1: {"voltage_mag": 1.06, "p_inject": 215.0},
    9: {"p_load": 31.0, "q_load": 18.0},
})
print(snapshot)
# TwinSnapshot(time=2024-01-15T12:00:00Z, anomalies=[], system_risk=0.3714)

# What-if: trip line 4→9
contingency_report = twin.simulate_contingency([(4, 9)])
print(f"Contingency system risk: {contingency_report.system_risk:.4f}")

# Retrieve risk time-series
trend = twin.risk_trend()  # list of (timestamp, system_risk)
```

### 4 – Evaluate predictions

```python
import numpy as np
from power_graph_risk.utils.metrics import (
    root_mean_squared_error, risk_auc,
    normalise_risk_scores, compute_vulnerability_index,
)

y_true  = np.array([0.8, 0.2, 0.6, 0.1])
y_pred  = np.array([0.75, 0.25, 0.55, 0.15])

print(f"RMSE : {root_mean_squared_error(y_true, y_pred):.4f}")
print(f"AUC  : {risk_auc((y_true > 0.5).astype(int), y_pred):.4f}")

# Composite vulnerability: 70% risk, 30% topological centrality
bus_risk = {1: 0.72, 2: 0.31, 3: 0.55}
centrality = {1: 0.20, 2: 0.80, 3: 0.50}
vuln = compute_vulnerability_index(bus_risk, centrality, centrality_weight=0.3)
print(vuln)
```

---

## Module Reference

### `power_graph_risk.data`

| Class | Description |
|---|---|
| `PowerGridGraph` | Main graph container. Exports node-feature matrix, adjacency matrix, edge-index, and Laplacian. |
| `BusNode` | Bus (node) with electrical attributes: V, θ, P_inj, Q_inj, P_load, Q_load. |
| `LineEdge` | Transmission line (edge) with R, X, B, thermal limit, and in-service flag. |
| `NodeType` | Enum: `SLACK`, `PV`, `PQ`. |

Key methods on `PowerGridGraph`:

| Method | Returns |
|---|---|
| `node_feature_matrix()` | `(N, 6)` array |
| `adjacency_matrix(weighted)` | `(N, N)` array |
| `edge_index()` | `(2, 2E)` int array |
| `laplacian(normalised)` | `(N, N)` array |
| `criticality_scores()` | `{bus_id: betweenness}` |
| `is_connected()` | `bool` |
| `ieee_14_bus()` | pre-built IEEE 14-bus grid |

### `power_graph_risk.models`

| Class | Description |
|---|---|
| `GraphConvLayer` | GCN layer: Â = D̃⁻¹/² Ã D̃⁻¹/², output = Â X W + b |
| `GraphAttentionLayer` | GAT-style attention-weighted aggregation |
| `GNNModel` | Stacked GNN (any depth, GCN or GAT) |
| `RiskScoreHead` | MLP that maps node embeddings → risk score ∈ [0,1] |

### `power_graph_risk.risk`

| Class | Description |
|---|---|
| `CascadeSimulator` | N-1 contingency analysis; produces pseudo-labels |
| `RiskAssessor` | Fits GNN + MLP on pseudo/manual labels; outputs `RiskReport` |
| `RiskReport` | Dataclass: per-bus scores, system risk, high-risk buses |

### `power_graph_risk.digital_twin`

| Class | Description |
|---|---|
| `DigitalTwin` | Main controller: ingest measurements, detect anomalies, track risk |
| `StateEstimator` | EMA filter for smoothing noisy measurements |
| `AnomalyDetector` | Rolling z-score anomaly flagging per bus |
| `TwinSnapshot` | Immutable record of one time-step |

### `power_graph_risk.utils`

| Function | Description |
|---|---|
| `mean_absolute_error` | MAE |
| `mean_squared_error` | MSE |
| `root_mean_squared_error` | RMSE |
| `precision_recall_f1` | Binary classification metrics |
| `risk_auc` | ROC-AUC via trapezoidal rule |
| `normalise_risk_scores` | Min-max normalisation |
| `compute_vulnerability_index` | Composite risk + centrality index |

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

119 tests covering all modules.

---

## License

MIT
