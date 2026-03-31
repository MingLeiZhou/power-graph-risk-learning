# Power Graph Risk Learning: OPFData → Graph Preprocessing

## 1) Project title

**Power Graph Risk Learning**  
Research-oriented preprocessing for converting OPFData samples into graph representations suitable for graph learning and self-supervised learning in power systems.

## 2) Motivation

Modern power-system datasets such as OPFData contain rich structured information (buses, generators, loads, shunts, transmission assets, and OPF solutions).  
Transforming these samples into graph-structured data enables:

- graph neural network (GNN) modeling of power-grid topology and physics constraints,
- self-supervised pretraining on large unlabeled operating states,
- reusable representations for downstream tasks (risk analysis, forecasting, anomaly detection, contingency-aware learning).

## 3) Repository goal

This repository focuses on a practical and inspectable preprocessing pipeline:

- convert raw OPFData JSON files into graph objects,
- preserve node/edge topology and static electrical attributes,
- optionally attach OPF solution fields as dynamic supervision signals,
- provide validation artifacts to check schema consistency and information preservation.

## 4) Data sources

- **Primary source:** OPFData JSON samples (`grid`, `solution`, `metadata`)
- **Related datasets (optional, included in demo workspace):**
  - **PGLib-OPF** test cases
  - **PowerGraph** dataset resources

See:

- `download_dataset.py`
- `power_demo_work/README.md`

## 5) Graph schema

The current implementation is in:

- `power_graph_builder.py`

### 5.1 Node definition

Nodes represent four physical categories:

- `bus`
- `generator`
- `load`
- `shunt`

Global node order in each graph:

`[buses | generators | loads | shunts]`

### 5.2 Edge definition

Edges represent:

- `ac_line` (bus ↔ bus structure from OPFData sender/receiver direction)
- `transformer` (bus ↔ bus)
- `generator_link` (generator → bus)
- `load_link` (load → bus)
- `shunt_link` (shunt → bus)

Global edge order:

`[ac_line | transformer | generator_link | load_link | shunt_link]` (when links are included)

### 5.3 Node features

`x` is a shared (homogeneous) node feature matrix:

- raw feature block padded to width `NODE_RAW_MAX=11`,
- followed by node-type one-hot block (`NODE_TYPE_DIM=4`),
- total `NODE_FEAT_DIM=15`.

Also exported:

- `node_type` (integer type id per node: 0 bus, 1 generator, 2 load, 3 shunt)

### 5.4 Edge features

`edge_attr` is a shared edge feature matrix:

- raw feature block padded to width `EDGE_RAW_MAX=11`,
- followed by edge-type one-hot block (`EDGE_TYPE_DIM=5`),
- total `EDGE_FEAT_DIM=16`.

Also exported:

- `edge_type` (integer type id per edge: 0 ac_line, 1 transformer, 2 generator_link, 3 load_link, 4 shunt_link)

### 5.5 Labels and metadata

- `y`: scalar objective from `metadata.objective`
- `meta`: counts and schema metadata, including:
  - `n_bus`, `n_gen`, `n_load`, `n_shunt`, `n_nodes`, `n_edges`
  - `node_type_counts`
  - `edge_type_counts`
  - `schema_version`
  - normalization mode

### 5.6 Static vs dynamic solution features

Static graph tensors:

- `x`
- `edge_index`
- `edge_attr`

Dynamic OPF solution tensors (optional):

- `sol_node` (bus: angle/vmag, generator: P/Q)
- `sol_edge` (ac_line/transformer flow terms)

Optional merged dynamic views (`merge_solution=True`):

- `x_dyn = concat(x, sol_node)`
- `edge_attr_dyn = concat(edge_attr, sol_edge)`

### 5.7 Homogeneous vs heterogeneous design

Current design is a **homogeneous graph with typed features** (single `x` and `edge_attr`, plus explicit type indicators).  
This is convenient and compatible with common PyG workflows, but not yet a true heterogeneous graph object.

## 6) Current limitations (explicit)

1. **Padded shared feature space across node types**  
   The same column index can have different physical meaning across node types. Type indicators mitigate ambiguity, but semantic mismatch remains.

2. **Link-edge attribute sparsity**  
   Link edges currently carry type information but no rich raw electrical feature block.

3. **Static vs dynamic integration risk**  
   Merging OPF solutions into model inputs (`x_dyn`, `edge_attr_dyn`) can cause label leakage depending on task design; use carefully.

4. **Normalization design constraints**  
   Dataset-level normalization is supported and recommended, but users must fit statistics on training data only for strict experimental hygiene.

## 7) Quick start

### 7.1 Install

Core Python runtime is sufficient for dict outputs; optional packages enable PyG tensors.

```bash
python -m pip install --upgrade pip
python -m pip install requests
# optional:
python -m pip install torch torch-geometric
```

### 7.2 Download demo data

```bash
python download_dataset.py
```

### 7.3 Run preprocessing example

```bash
python example_usage.py
```

### 7.4 Run validation notebook

Open and run:

- `graph_schema_validation.ipynb`

This notebook demonstrates:

- raw OPFData structure,
- transformed graph structure,
- node/edge count consistency,
- feature dimensional consistency,
- node/edge type distributions,
- information preservation checks,
- normalization behavior (per-graph vs dataset-level),
- known schema limitations.

## 8) Repository structure

```text
power-graph-risk-learning/
├── power_graph_builder.py          # OPFData → graph transformation pipeline
├── example_usage.py                # runnable preprocessing examples
├── graph_schema_validation.ipynb   # validation report notebook
├── test.ipynb                      # legacy exploratory notebook
├── download_dataset.py             # data download helper
├── power_demo_work/                # demo data workspace
│   └── README.md
└── README.md
```

## 9) Next steps / roadmap

- Add a true heterogeneous graph export path (per-node-type / per-edge-type stores).
- Expand link-edge attributes with richer physics-aware descriptors where available.
- Add train/val/test split utilities with leakage-safe normalization fitting.
- Add unit tests for graph-schema invariants and feature-alignment checks.
- Add downstream benchmark tasks for representation quality evaluation.
