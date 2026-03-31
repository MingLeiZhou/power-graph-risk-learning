"""
example_usage.py
================
Runnable example: OPFData JSON → graph objects using PowerGraphBuilder.

Run from the repository root:
    python example_usage.py

If torch-geometric is installed the graphs will be PyG Data objects.
Otherwise they fall back to plain Python dicts.
"""

from __future__ import annotations

import json
from pathlib import Path

from power_graph_builder import (
    PowerGraphBuilder,
    build_graph_from_json,
    batch_process,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    SOL_NODE_DIM,
    SOL_EDGE_DIM,
)

# ---------------------------------------------------------------------------
# Locate sample data shipped with the repo
# ---------------------------------------------------------------------------

OPFDATA_DIR = Path("power_demo_work/opfdata")
SAMPLE_JSON = next(OPFDATA_DIR.rglob("*.json"), None)

if SAMPLE_JSON is None:
    raise FileNotFoundError(
        "No OPFData JSON files found under power_demo_work/opfdata/. "
        "Run download_dataset.py first."
    )

print("=" * 60)
print("PowerGraphBuilder — example usage")
print("=" * 60)

# ---------------------------------------------------------------------------
# Example 1: single file, default settings
# ---------------------------------------------------------------------------
print("\n--- Example 1: build_graph_from_json (single file) ---")
graph = build_graph_from_json(SAMPLE_JSON)

if isinstance(graph, dict):
    x         = graph["x"]
    ei        = graph["edge_index"]
    ea        = graph["edge_attr"]
    y         = graph["y"]
    sol_node  = graph["sol_node"]
    sol_edge  = graph["sol_edge"]
    meta      = graph["meta"]
    n_nodes   = len(x)
    n_edges   = len(ea)
    x_width   = len(x[0])  if x  else 0
    ea_width  = len(ea[0]) if ea else 0
    sn_width  = len(sol_node[0]) if sol_node else 0
    se_width  = len(sol_edge[0]) if sol_edge else 0
else:
    # PyG Data object
    x         = graph.x
    ei        = graph.edge_index
    ea        = graph.edge_attr
    y         = graph.y
    sol_node  = graph.sol_node
    sol_edge  = graph.sol_edge
    meta      = graph.meta
    n_nodes   = x.shape[0]
    n_edges   = ea.shape[0]
    x_width   = x.shape[1]
    ea_width  = ea.shape[1]
    sn_width  = sol_node.shape[1]
    se_width  = sol_edge.shape[1]

print(f"  Source file   : {SAMPLE_JSON.name}")
print(f"  Node count    : {n_nodes}  (expected feat dim {NODE_FEAT_DIM})")
print(f"  Edge count    : {n_edges}  (expected feat dim {EDGE_FEAT_DIM})")
print(f"  x shape       : ({n_nodes}, {x_width})")
print(f"  edge_attr dim : {ea_width}")
print(f"  sol_node dim  : {sn_width}  (angle/vmag for buses, P/Q for gens)")
print(f"  sol_edge dim  : {se_width}  (p_fr, q_fr, p_to, q_to)")
print(f"  y (objective) : {y if isinstance(y, float) else y.item():.4f}")
print(f"  meta          : {meta}")

# ---------------------------------------------------------------------------
# Example 2: batch processing a folder
# ---------------------------------------------------------------------------
print("\n--- Example 2: batch_process (folder) ---")

graphs = batch_process(OPFDATA_DIR)
print(f"  Processed {len(graphs)} graphs from {OPFDATA_DIR}")

if graphs:
    g0 = graphs[0]
    n_nodes_0 = len(g0["x"]) if isinstance(g0, dict) else g0.x.shape[0]
    n_edges_0 = len(g0["edge_attr"]) if isinstance(g0, dict) else g0.edge_attr.shape[0]
    print(f"  First graph — nodes: {n_nodes_0}, edges: {n_edges_0}")

# ---------------------------------------------------------------------------
# Example 3: PowerGraphBuilder with normalisation (no links)
# ---------------------------------------------------------------------------
print("\n--- Example 3: PowerGraphBuilder with normalisation + no links ---")

builder = PowerGraphBuilder(
    normalize_features=True,
    normalization_mode="graph",  # explicit per-graph normalization
    include_solution=True,
    include_links=False,   # only ac_line / transformer edges
)
graph_norm = builder.build_graph_from_json(SAMPLE_JSON)

if isinstance(graph_norm, dict):
    n_norm_edges = len(graph_norm["edge_attr"])
    n_norm_nodes = len(graph_norm["x"])
else:
    n_norm_edges = graph_norm.edge_attr.shape[0]
    n_norm_nodes = graph_norm.x.shape[0]

print(f"  Without links — nodes: {n_norm_nodes}, edges: {n_norm_edges}")

# ---------------------------------------------------------------------------
# Example 4: Fit shared normalisers across the whole dataset, then transform
# ---------------------------------------------------------------------------
print("\n--- Example 4: dataset-level normalisation ---")

# Step 1: collect raw graphs (no normalisation yet)
builder_raw = PowerGraphBuilder(normalize_features=False)
raw_graphs = builder_raw.batch_process(OPFDATA_DIR)

# Step 2: fit normalisers on the collected data
builder_raw.fit_normalizers(raw_graphs)

# Step 3: re-process with the fitted normalisers applied
builder_raw.normalize_features = True
builder_raw.normalization_mode = "dataset"
norm_graphs = builder_raw.batch_process(OPFDATA_DIR)

print(f"  Fitted normalisers on {len(raw_graphs)} graphs")
print(f"  Re-processed {len(norm_graphs)} normalised graphs")

# ---------------------------------------------------------------------------
# Show raw JSON → graph mapping summary
# ---------------------------------------------------------------------------
print("\n--- Mapping summary ---")
print(f"""
  Raw JSON field               →  Graph tensor / list
  ─────────────────────────────────────────────────────────────
  grid.nodes.bus   (N_bus×4)   →  x[0:N_bus,                    0:4]  + one-hot[0]
  grid.nodes.gen   (N_g ×11)   →  x[N_bus:N_bus+N_g,             0:11] + one-hot[1]
  grid.nodes.load  (N_l ×2 )   →  x[N_bus+N_g:N_bus+N_g+N_l,    0:2]  + one-hot[2]
  grid.nodes.shunt (N_s ×2 )   →  x[N_bus+N_g+N_l:N_bus+N_g+N_l+N_s, 0:2] + one-hot[3]
  grid.edges.ac_line (E_ac×9)  →  edge_attr[...,0:9]   + one-hot[ac_line]
  grid.edges.transf. (E_tr×11) →  edge_attr[...,0:11]  + one-hot[transformer]
  generator_link               →  edge_attr[...,0:0]   + one-hot[generator_link]
  load_link                    →  edge_attr[...,0:0]   + one-hot[load_link]
  shunt_link                   →  edge_attr[...,0:0]   + one-hot[shunt_link]
  metadata.objective           →  y  (regression label)
  solution.nodes.bus           →  sol_node[0:N_bus]   (angle, vmag)
  solution.nodes.generator     →  sol_node[N_bus:N_g] (P, Q)
  solution.edges.ac_line       →  sol_edge[0:E_ac]    (p_fr,q_fr,p_to,q_to)
  solution.edges.transformer   →  sol_edge[E_ac:]
""")

print("Done.")
