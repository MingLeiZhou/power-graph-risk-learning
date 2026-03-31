# Power Graph Risk Learning

Research-focused preprocessing for transforming OPFData samples into scalable graph-learning datasets.

## 1) Project motivation

OPFData JSON files contain rich power-grid structure (buses, generators, loads, shunts, transmission assets) and optimization results. For GNN and graph-representation research, raw JSON is not ideal for large-scale analysis, indexing, and repeatable training data builds.

This repository now provides a full pipeline:

**JSON → Parquet → DuckDB → Graph dataset**

Why this is useful:
- **JSON** preserves the original sample semantics.
- **Parquet** gives compact, columnar, scalable structured storage.
- **DuckDB** provides a lightweight query layer for fast dataset analysis.
- **Graph export** provides model-ready data (`x`, `edge_index`, `edge_attr`, `y`, metadata).

## 2) Pipeline overview

### Stage A: Raw JSON (OPFData)
Input is a directory tree of OPFData JSON samples (one file = one sample), by default:

- `power_demo_work/opfdata`

### Stage B: Structured Parquet tables
`process_opfdata_dataset.py` parses all JSON files and writes tables to:

- `data/processed/parquet/samples.parquet`
- `data/processed/parquet/nodes.parquet`
- `data/processed/parquet/edges.parquet`
- `data/processed/parquet/solution_nodes.parquet` (optional rows)
- `data/processed/parquet/solution_edges.parquet` (optional rows)

All related records include `sample_id`.

### Stage C: DuckDB query layer
`build_duckdb.py` (or integrated flags in `process_opfdata_dataset.py`) loads Parquet into:

- `data/processed/opfdata.duckdb`
- tables: `samples`, `nodes`, `edges`, `solution_nodes`, `solution_edges`

Example analytics queries are included:
- number of samples
- average nodes per sample
- average edges per sample
- node-type distribution
- edge-type distribution

### Stage D: Graph dataset export
Graph export converts structured tables into sample-wise graph objects under:

- `data/processed/graphs/`

Supported outputs:
- serialized dict graphs in `.json` (default)
- serialized dict graphs in `.pt` (requires `torch`)

Each graph contains:
- `x`
- `edge_index`
- `edge_attr`
- `y`
- `sol_node`
- `sol_edge`
- `node_type`
- `edge_type`
- `metadata`

A `manifest.json` with schema summary is also written to `data/processed/graphs/`.

## 3) Repository structure

```text
power-graph-risk-learning/
├── power_demo_work/
│   └── opfdata/                       # raw OPFData JSON tree (input)
├── data/
│   └── processed/
│       ├── parquet/                   # structured Parquet tables
│       ├── opfdata.duckdb             # DuckDB analytical store
│       └── graphs/                    # exported graph dataset
├── power_graph_builder.py             # core single-sample graph builder
├── opfdata_pipeline.py                # reusable JSON/Parquet/DuckDB/graph helpers
├── process_opfdata_dataset.py         # end-to-end pipeline CLI
├── build_duckdb.py                    # Parquet -> DuckDB CLI
├── download_dataset.py                # demo data downloader
└── README.md
```

## 4) Quick start

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install duckdb
# optional for .pt graph serialization:
python -m pip install torch
```

### 4.1 Process all JSON files into Parquet

```bash
python process_opfdata_dataset.py \
  --input-dir power_demo_work/opfdata \
  --parquet-dir data/processed/parquet
```

### 4.2 Build DuckDB and run example queries

```bash
python build_duckdb.py \
  --parquet-dir data/processed/parquet \
  --duckdb-path data/processed/opfdata.duckdb \
  --run-queries
```

### 4.3 Export graph dataset

```bash
python process_opfdata_dataset.py \
  --input-dir power_demo_work/opfdata \
  --parquet-dir data/processed/parquet \
  --export-graphs \
  --graphs-dir data/processed/graphs \
  --graph-format json
```

### 4.4 Run full pipeline in one command

```bash
python process_opfdata_dataset.py \
  --input-dir power_demo_work/opfdata \
  --parquet-dir data/processed/parquet \
  --build-duckdb \
  --duckdb-path data/processed/opfdata.duckdb \
  --run-duckdb-queries \
  --export-graphs \
  --graphs-dir data/processed/graphs
```

Useful flags:
- `--limit N` for small debugging runs
- `--log-every K` for progress interval
- `--temp-dir PATH` for intermediate NDJSON workspace
- `--keep-temp` to keep intermediate files

## 5) Data schema

### 5.1 `samples` table
One row per JSON sample.

Core fields:
- `sample_id`
- `source_file`
- `objective`
- `base_mva`
- `has_solution`
- node counts (`n_bus`, `n_generator`, `n_load`, `n_shunt`, `n_nodes`)
- edge counts (`n_edges_ac_line`, `n_edges_transformer`, `n_edges_generator_link`, `n_edges_load_link`, `n_edges_shunt_link`, `n_edges`)

### 5.2 `nodes` table
One row per node in a sample.

Core fields:
- `sample_id`
- `node_id` (global order in graph)
- `node_type` (`bus`, `generator`, `load`, `shunt`)
- `node_local_id` (type-local index)
- `raw_feature_dim`
- `raw_features` (list)

### 5.3 `edges` table
One row per edge in a sample.

Core fields:
- `sample_id`
- `edge_id` (global order in graph)
- `edge_type` (`ac_line`, `transformer`, `generator_link`, `load_link`, `shunt_link`)
- `edge_local_id` (type-local index)
- `src_node_type`, `src_node_local_id`
- `dst_node_type`, `dst_node_local_id`
- `raw_feature_dim`
- `raw_features` (list)

### 5.4 `solution_nodes` and `solution_edges`
Optional solution rows when present in JSON:
- `solution_nodes`: bus/generator solution features
- `solution_edges`: ac_line/transformer flow features

### 5.5 Graph dataset structure
Per sample graph object:
- `x`: node features (`padded raw + node-type one-hot`)
- `edge_index`: `[sources, destinations]`
- `edge_attr`: edge features (`padded raw + edge-type one-hot`)
- `y`: scalar objective (`metadata.objective`)
- `sol_node`: aligned optional node solution tensor (zeros if missing)
- `sol_edge`: aligned optional edge solution tensor (zeros if missing)
- `node_type`: integer type id per node
- `edge_type`: integer type id per edge
- `metadata`: sample id, source file, schema + feature dimensions

## 6) Current limitations

1. **Homogeneous graph representation with type indicators**  
   This pipeline exports unified `x` / `edge_attr` with type encoding, not native hetero graph objects.

2. **Padded shared feature blocks**  
   Different node/edge types use different raw semantics inside a shared padded space.

3. **Solution coverage differs by type**  
   Solution features are available only for specific types (bus/generator nodes and ac_line/transformer edges); missing parts are zero-filled in graph export.

4. **No train/val/test split utility yet**  
   Dataset split strategy and leakage-safe normalization workflows should be added separately for training pipelines.
