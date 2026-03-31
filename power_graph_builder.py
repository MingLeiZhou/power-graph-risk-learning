"""
power_graph_builder.py
======================
Research-grade preprocessing pipeline: raw OPFData JSON → graph objects.

Data source structure (OPFData, one JSON file = one power system state)
-----------------------------------------------------------------------
{
  "grid": {
    "nodes": {
      "bus":       [[...], ...],   # N_bus  x F_bus  (4 features)
      "generator": [[...], ...],   # N_gen  x F_gen  (11 features)
      "load":      [[...], ...],   # N_load x F_load (2 features)
      "shunt":     [[...], ...],   # N_shunt x F_shunt (2 features)
    },
    "edges": {
      "ac_line":       {"senders": [...], "receivers": [...], "features": [[...], ...]},
      "transformer":   {"senders": [...], "receivers": [...], "features": [[...], ...]},
      "generator_link":{"senders": [...], "receivers": [...]},  # gen → bus
      "load_link":     {"senders": [...], "receivers": [...]},  # load → bus
      "shunt_link":    {"senders": [...], "receivers": [...]},  # shunt → bus
    },
    "context": [[[base_mva]]]
  },
  "solution": {
    "nodes": {
      "bus":       [[angle, vmag], ...],
      "generator": [[p, q], ...]
    },
    "edges": {
      "ac_line":     {"senders": [...], "receivers": [...], "features": [[p_fr, q_fr, p_to, q_to], ...]},
      "transformer": {"senders": [...], "receivers": [...], "features": [[p_fr, q_fr, p_to, q_to], ...]}
    }
  },
  "metadata": {"objective": float}
}

Graph design
------------
Node ordering (global index):
  [0 .. N_bus-1]                          → buses
  [N_bus .. N_bus+N_gen-1]                → generators
  [N_bus+N_gen .. N_bus+N_gen+N_load-1]   → loads
  [N_bus+N_gen+N_load .. N-1]             → shunts

Edge ordering:
  ac_line edges first, then transformer edges, then link edges (gen/load/shunt↔bus).

Node features (x):
  Per-element raw features, zero-padded to a common width, followed by a 4-dim
  one-hot type encoding [is_bus, is_gen, is_load, is_shunt].

Edge features (edge_attr):
  Raw features for ac_line / transformer, zeros for link edges, followed by a
  5-dim one-hot type encoding
  [is_ac_line, is_transformer, is_generator_link, is_load_link, is_shunt_link].

Labels (y):
  Scalar OPF objective value from metadata (float).

Solution tensors (stored as separate keys when available):
  sol_node – solution features per node (bus: angle+vmag, gen: P+Q, zeros otherwise)
  sol_edge – solution power flows per edge (for ac_line / transformer)
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Optional heavy dependencies — the module degrades gracefully when absent
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import Tensor
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    Tensor = Any  # type: ignore[assignment,misc]

try:
    from torch_geometric.data import Data as PyGData
    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYG_AVAILABLE = False
    PyGData = None  # type: ignore[assignment,misc]

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    import array as _array_mod  # stdlib fallback used only for type hints
    _NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
GraphDict = Dict[str, Any]   # {"x", "edge_index", "edge_attr", "y", ...}
GraphResult = Union[Any, GraphDict]  # PyGData when available, else GraphDict

# ---------------------------------------------------------------------------
# Feature dimension constants (from inspecting the dataset)
# ---------------------------------------------------------------------------
# Raw feature widths per element type
BUS_FEAT_DIM    = 4   # [v_min_pu, bus_type, v_min, v_max]
GEN_FEAT_DIM    = 11  # [p_max, q_max, q_min, p_max2, p_min, cost0..2, ...]
LOAD_FEAT_DIM   = 2   # [p_d, q_d]
SHUNT_FEAT_DIM  = 2   # [g_sh, b_sh]

AC_LINE_FEAT_DIM    = 9   # [angle_min, angle_max, r, x, b, rate_a, rate_b, rate_c, ...]
TRANSFORMER_FEAT_DIM = 11  # [angle_min, angle_max, r, x, b, rate*, tap, shift, ...]
LINK_FEAT_DIM       = 0   # link edges carry no raw features

# One-hot encoding widths
NODE_TYPE_DIM = 4   # [bus, gen, load, shunt]
EDGE_TYPE_DIM = 5   # [ac_line, transformer, generator_link, load_link, shunt_link]

# Padded raw feature width (max of all raw dims)
NODE_RAW_MAX = max(BUS_FEAT_DIM, GEN_FEAT_DIM, LOAD_FEAT_DIM, SHUNT_FEAT_DIM)  # 11
EDGE_RAW_MAX = max(AC_LINE_FEAT_DIM, TRANSFORMER_FEAT_DIM, LINK_FEAT_DIM)       # 11

# Final feature dimensions
NODE_FEAT_DIM = NODE_RAW_MAX + NODE_TYPE_DIM   # 15
EDGE_FEAT_DIM = EDGE_RAW_MAX + EDGE_TYPE_DIM   # 16

# Solution feature dimensions
SOL_NODE_DIM = 2   # (angle, vmag) for bus  /  (P, Q) for gen  /  zeros otherwise
SOL_EDGE_DIM = 4   # (p_fr, q_fr, p_to, q_to) for ac_line / transformer / zeros


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    """Convert a value to float, falling back to *default* on failure."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _pad_or_truncate(feats: List[float], width: int) -> List[float]:
    """Return a list of exactly *width* floats, padding with 0 or truncating."""
    n = len(feats)
    if n >= width:
        return feats[:width]
    return feats + [0.0] * (width - n)


def _to_float_list(raw: Any) -> List[float]:
    """Convert raw JSON value (list / scalar / None) to a flat list of floats."""
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [_safe_float(raw)]
    if isinstance(raw, (list, tuple)):
        return [_safe_float(v) for v in raw]
    return []


def _one_hot(idx: int, size: int) -> List[float]:
    vec = [0.0] * size
    if 0 <= idx < size:
        vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Normalization helper  (StandardScaler-style, per-feature column)
# ---------------------------------------------------------------------------

class FeatureNormalizer:
    """
    Incremental column-wise StandardScaler that operates on plain Python lists.

    Usage::

        norm = FeatureNormalizer()
        norm.fit(list_of_feature_vectors)
        normalized = norm.transform(list_of_feature_vectors)
    """

    def __init__(self) -> None:
        self.mean_: Optional[List[float]] = None
        self.std_:  Optional[List[float]] = None

    def fit(self, matrix: List[List[float]]) -> "FeatureNormalizer":
        if not matrix:
            return self
        n_cols = len(matrix[0])
        n_rows = len(matrix)
        self.mean_ = [0.0] * n_cols
        for row in matrix:
            for j, v in enumerate(row):
                self.mean_[j] += v / n_rows
        var = [0.0] * n_cols
        for row in matrix:
            for j, v in enumerate(row):
                diff = v - self.mean_[j]
                var[j] += diff * diff / n_rows
        self.std_ = [math.sqrt(v) if v > 1e-10 else 1.0 for v in var]
        return self

    def transform(self, matrix: List[List[float]]) -> List[List[float]]:
        if self.mean_ is None:
            return matrix
        return [
            [(v - self.mean_[j]) / self.std_[j] for j, v in enumerate(row)]
            for row in matrix
        ]

    def fit_transform(self, matrix: List[List[float]]) -> List[List[float]]:
        return self.fit(matrix).transform(matrix)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

class PowerGraphBuilder:
    """
    Converts OPFData JSON samples into graph representations.

    Parameters
    ----------
    normalize_features : bool
        If True, apply per-feature StandardScaler normalisation to node and
        edge features. Default: False.
        If ``normalization_mode="dataset"``, the builder uses statistics
        fitted by ``fit_normalizers``. If no fitted statistics are available
        in dataset mode, the builder falls back to per-graph normalisation and
        emits a warning.
    include_solution : bool
        If True, attach ground-truth solution tensors (sol_node, sol_edge)
        as additional attributes.  Default: True.
    include_links : bool
        If True, add generator/load/shunt link edges to the graph.
        Default: True.
    normalization_mode : str
        Either ``"dataset"`` (recommended for reproducible training pipelines)
        or ``"graph"`` (fit/transform independently per graph).
        Default: ``"dataset"``.
    merge_solution : bool
        If True and ``include_solution=True``, merge solution tensors into
        model-ready dynamic features ``x_dyn`` and ``edge_attr_dyn`` while
        retaining ``sol_node`` and ``sol_edge`` for supervision/debugging.
        Default: False.
    """

    def __init__(
        self,
        normalize_features: bool = False,
        include_solution:   bool = True,
        include_links:      bool = True,
        normalization_mode: str = "dataset",
        merge_solution:     bool = False,
    ) -> None:
        if normalization_mode not in {"dataset", "graph"}:
            raise ValueError("normalization_mode must be either 'dataset' or 'graph'")
        self.normalize_features = normalize_features
        self.include_solution   = include_solution
        self.include_links      = include_links
        self.normalization_mode = normalization_mode
        self.merge_solution     = merge_solution

        # Shared normalizers — populated by fit_normalizers()
        self._node_norm: Optional[FeatureNormalizer] = None
        self._edge_norm: Optional[FeatureNormalizer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph_from_json(self, file_path: Union[str, Path]) -> GraphResult:
        """
        Build a graph from a single OPFData JSON file.

        Parameters
        ----------
        file_path : str or Path
            Path to a single ``example_*.json`` file.

        Returns
        -------
        PyG ``Data`` object when torch-geometric is available, otherwise a
        plain ``dict`` with keys ``x``, ``edge_index``, ``edge_attr``, ``y``,
        ``sol_node``, ``sol_edge``, ``meta``.
        """
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as fh:
            sample = json.load(fh)
        return self._build_graph(sample, source_file=str(file_path))

    def batch_process(
        self,
        folder_path: Union[str, Path],
        recursive: bool = True,
        glob_pattern: str = "*.json",
    ) -> List[GraphResult]:
        """
        Convert all JSON files in *folder_path* to graph objects.

        Parameters
        ----------
        folder_path : str or Path
            Root directory to search for JSON files.
        recursive : bool
            Search sub-directories recursively.  Default: True.
        glob_pattern : str
            File glob pattern.  Default: ``"*.json"``.

        Returns
        -------
        List of graph objects (same type as ``build_graph_from_json``).
        """
        folder_path = Path(folder_path)
        files = sorted(
            folder_path.rglob(glob_pattern) if recursive
            else folder_path.glob(glob_pattern)
        )
        if not files:
            warnings.warn(
                f"No files matching '{glob_pattern}' found in {folder_path}",
                UserWarning,
                stacklevel=2,
            )
            return []

        graphs = []
        errors = 0
        for fp in files:
            try:
                graphs.append(self.build_graph_from_json(fp))
            except Exception as exc:
                warnings.warn(f"Skipping {fp}: {exc}", UserWarning, stacklevel=2)
                errors += 1

        if errors:
            warnings.warn(
                f"{errors}/{len(files)} files failed to parse.",
                UserWarning,
                stacklevel=2,
            )
        return graphs

    def fit_normalizers(self, graphs: List[GraphDict]) -> "PowerGraphBuilder":
        """
        Fit shared column-wise normalizers from a list of graph dicts.

        After calling this, subsequent ``build_graph_from_json`` / ``batch_process``
        calls with ``normalize_features=True`` will use these fitted statistics.

        Parameters
        ----------
        graphs : list of graph dicts
            Typically the output of ``batch_process``.
        """
        all_x: List[List[float]] = []
        all_e: List[List[float]] = []
        for g in graphs:
            x = g["x"] if isinstance(g, dict) else g.x.tolist()
            e = g["edge_attr"] if isinstance(g, dict) else g.edge_attr.tolist()
            # Fit only on raw feature blocks; keep one-hot type bits untouched.
            all_x.extend([row[:NODE_RAW_MAX] for row in x])
            all_e.extend([row[:EDGE_RAW_MAX] for row in e])

        if all_x:
            self._node_norm = FeatureNormalizer().fit(all_x)
        if all_e:
            self._edge_norm = FeatureNormalizer().fit(all_e)
        return self

    def has_fitted_normalizers(self) -> bool:
        """Return True when both node and edge shared normalizers are available."""
        return self._node_norm is not None and self._edge_norm is not None

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    def _build_graph(
        self, sample: Dict[str, Any], source_file: str = ""
    ) -> GraphResult:
        """Core graph construction from a parsed JSON dict."""
        grid     = sample.get("grid",     {})
        solution = sample.get("solution", {})
        metadata = sample.get("metadata", {})

        grid_nodes = grid.get("nodes", {})
        grid_edges = grid.get("edges", {})

        # ---- 1. Build node index offsets ----------------------------------
        #  Global node ordering: bus | generator | load | shunt
        buses      = grid_nodes.get("bus",       [])
        generators = grid_nodes.get("generator", [])
        loads      = grid_nodes.get("load",      [])
        shunts     = grid_nodes.get("shunt",     [])

        n_bus  = len(buses)
        n_gen  = len(generators)
        n_load = len(loads)
        n_shunt = len(shunts)
        n_nodes = n_bus + n_gen + n_load + n_shunt

        # offset into global node array for each element type
        bus_offset   = 0
        gen_offset   = n_bus
        load_offset  = n_bus + n_gen
        shunt_offset = n_bus + n_gen + n_load

        # ---- 2. Build node feature matrix  --------------------------------
        #  Each row = zero-padded raw features + one-hot type encoding
        x_rows: List[List[float]] = []
        node_type_idx: List[int] = []

        # buses  (type idx = 0)
        for raw in buses:
            feats = _pad_or_truncate(_to_float_list(raw), NODE_RAW_MAX)
            x_rows.append(feats + _one_hot(0, NODE_TYPE_DIM))
            node_type_idx.append(0)

        # generators (type idx = 1)
        for raw in generators:
            feats = _pad_or_truncate(_to_float_list(raw), NODE_RAW_MAX)
            x_rows.append(feats + _one_hot(1, NODE_TYPE_DIM))
            node_type_idx.append(1)

        # loads (type idx = 2)
        for raw in loads:
            feats = _pad_or_truncate(_to_float_list(raw), NODE_RAW_MAX)
            x_rows.append(feats + _one_hot(2, NODE_TYPE_DIM))
            node_type_idx.append(2)

        # shunts (type idx = 3)
        for raw in shunts:
            feats = _pad_or_truncate(_to_float_list(raw), NODE_RAW_MAX)
            x_rows.append(feats + _one_hot(3, NODE_TYPE_DIM))
            node_type_idx.append(3)

        # ---- 3. Build edge index + edge feature matrix  -------------------
        #  Edge ordering: ac_line | transformer | [gen/load/shunt links]
        src_list: List[int] = []
        dst_list: List[int] = []
        ea_rows:  List[List[float]] = []
        edge_type_idx: List[int] = []

        def _add_edges(
            edge_data: Dict[str, Any],
            src_offset: int,
            dst_offset: int,
            type_idx: int,       # 0=ac_line, 1=transformer, 2=generator_link, 3=load_link, 4=shunt_link
        ) -> None:
            senders   = edge_data.get("senders",   [])
            receivers = edge_data.get("receivers", [])
            features  = edge_data.get("features",  [])
            n_edges = max(len(senders), len(receivers))
            for i in range(n_edges):
                s = senders[i]   if i < len(senders)   else 0
                r = receivers[i] if i < len(receivers) else 0
                src_list.append(int(s) + src_offset)
                dst_list.append(int(r) + dst_offset)
                raw = _to_float_list(features[i]) if i < len(features) else []
                feats = _pad_or_truncate(raw, EDGE_RAW_MAX)
                ea_rows.append(feats + _one_hot(type_idx, EDGE_TYPE_DIM))
                edge_type_idx.append(type_idx)

        # ac_line: bus → bus
        _add_edges(
            grid_edges.get("ac_line", {}),
            src_offset=bus_offset, dst_offset=bus_offset,
            type_idx=0,
        )
        # transformer: bus → bus
        _add_edges(
            grid_edges.get("transformer", {}),
            src_offset=bus_offset, dst_offset=bus_offset,
            type_idx=1,
        )

        if self.include_links:
            # generator_link: generator → bus  (senders=gen idx, receivers=bus idx)
            _add_edges(
                grid_edges.get("generator_link", {}),
                src_offset=gen_offset, dst_offset=bus_offset,
                type_idx=2,
            )
            # load_link: load → bus
            _add_edges(
                grid_edges.get("load_link", {}),
                src_offset=load_offset, dst_offset=bus_offset,
                type_idx=3,
            )
            # shunt_link: shunt → bus
            _add_edges(
                grid_edges.get("shunt_link", {}),
                src_offset=shunt_offset, dst_offset=bus_offset,
                type_idx=4,
            )

        # ---- 4. Optional normalisation  -----------------------------------
        if self.normalize_features:
            # Only normalize raw blocks, preserving one-hot type encodings.
            x_raw = [row[:NODE_RAW_MAX] for row in x_rows]
            x_type = [row[NODE_RAW_MAX:] for row in x_rows]
            e_raw = [row[:EDGE_RAW_MAX] for row in ea_rows]
            e_type = [row[EDGE_RAW_MAX:] for row in ea_rows]

            if self.normalization_mode == "dataset":
                if self._node_norm is not None and self._edge_norm is not None:
                    x_raw = self._node_norm.transform(x_raw)
                    e_raw = self._edge_norm.transform(e_raw)
                else:
                    warnings.warn(
                        "normalize_features=True with normalization_mode='dataset' but shared "
                        "normalizers are not fitted; falling back to per-graph normalization.",
                        UserWarning,
                        stacklevel=2,
                    )
                    if x_raw:
                        x_raw = FeatureNormalizer().fit_transform(x_raw)
                    if e_raw:
                        e_raw = FeatureNormalizer().fit_transform(e_raw)
            else:
                if x_raw:
                    x_raw = FeatureNormalizer().fit_transform(x_raw)
                if e_raw:
                    e_raw = FeatureNormalizer().fit_transform(e_raw)

            x_rows = [r + t for r, t in zip(x_raw, x_type)]
            ea_rows = [r + t for r, t in zip(e_raw, e_type)]

        # ---- 5. Solution tensors (per-node and per-edge)  -----------------
        sol_node_rows: List[List[float]] = [[0.0] * SOL_NODE_DIM for _ in range(n_nodes)]
        sol_edge_rows: List[List[float]] = [[0.0] * SOL_EDGE_DIM for _ in range(len(src_list))]

        if self.include_solution:
            sol_nodes = solution.get("nodes", {})
            sol_edges = solution.get("edges", {})

            # bus solution (angle, vmag)
            for i, raw in enumerate(sol_nodes.get("bus", [])):
                if i < n_bus:
                    sol_node_rows[bus_offset + i] = _pad_or_truncate(
                        _to_float_list(raw), SOL_NODE_DIM
                    )

            # generator solution (P, Q)
            for i, raw in enumerate(sol_nodes.get("generator", [])):
                if i < n_gen:
                    sol_node_rows[gen_offset + i] = _pad_or_truncate(
                        _to_float_list(raw), SOL_NODE_DIM
                    )

            # Edge solutions — we need to know how many ac_line / transformer edges we have
            n_ac   = len(grid_edges.get("ac_line",     {}).get("senders", []))
            n_traf = len(grid_edges.get("transformer", {}).get("senders", []))

            for i, raw in enumerate(sol_edges.get("ac_line", {}).get("features", [])):
                if i < n_ac:
                    sol_edge_rows[i] = _pad_or_truncate(
                        _to_float_list(raw), SOL_EDGE_DIM
                    )
            for i, raw in enumerate(sol_edges.get("transformer", {}).get("features", [])):
                if i < n_traf:
                    sol_edge_rows[n_ac + i] = _pad_or_truncate(
                        _to_float_list(raw), SOL_EDGE_DIM
                    )

        # ---- 6. Label  ----------------------------------------------------
        #  Primary label: OPF objective (regression target)
        y = _safe_float(metadata.get("objective"))

        # ---- 7. Assemble edge_index (shape 2 x E)  ------------------------
        # edge_index[0] = source nodes, edge_index[1] = destination nodes
        edge_index_pair = [src_list, dst_list]

        # ---- 8. Return result  --------------------------------------------
        meta = {
            "source_file": source_file,
            "n_bus":  n_bus,
            "n_gen":  n_gen,
            "n_load": n_load,
            "n_shunt": n_shunt,
            "n_nodes": n_nodes,
            "n_edges": len(src_list),
            "node_type_counts": {
                "bus": n_bus,
                "generator": n_gen,
                "load": n_load,
                "shunt": n_shunt,
            },
            "edge_type_counts": {
                "ac_line": sum(1 for t in edge_type_idx if t == 0),
                "transformer": sum(1 for t in edge_type_idx if t == 1),
                "generator_link": sum(1 for t in edge_type_idx if t == 2),
                "load_link": sum(1 for t in edge_type_idx if t == 3),
                "shunt_link": sum(1 for t in edge_type_idx if t == 4),
            },
            "schema_version": "v2",
            "normalization_mode": self.normalization_mode if self.normalize_features else "none",
        }

        extra: Dict[str, Any] = {
            "node_type": node_type_idx,
            "edge_type": edge_type_idx,
        }
        if self.include_solution and self.merge_solution:
            x_dyn = [x + s for x, s in zip(x_rows, sol_node_rows)]
            edge_attr_dyn = [e + s for e, s in zip(ea_rows, sol_edge_rows)]
            extra["x_dyn"] = x_dyn
            extra["edge_attr_dyn"] = edge_attr_dyn

        return _package_graph(
            x_rows, edge_index_pair, ea_rows,
            y, sol_node_rows, sol_edge_rows, meta, extra=extra,
        )


# ---------------------------------------------------------------------------
# Packaging helpers — PyG Data vs. plain dict
# ---------------------------------------------------------------------------

def _package_graph(
    x_rows:         List[List[float]],
    edge_index_pair: List[List[int]],
    ea_rows:         List[List[float]],
    y:               float,
    sol_node_rows:   List[List[float]],
    sol_edge_rows:   List[List[float]],
    meta:            Dict[str, Any],
    extra:           Optional[Dict[str, Any]] = None,
) -> GraphResult:
    """Convert Python lists into a PyG Data object or a plain dict."""

    if _PYG_AVAILABLE and _TORCH_AVAILABLE:
        x          = torch.tensor(x_rows,          dtype=torch.float32)
        edge_index = torch.tensor(edge_index_pair, dtype=torch.long)
        edge_attr  = torch.tensor(ea_rows,         dtype=torch.float32)
        y_t        = torch.tensor([y],             dtype=torch.float32)
        sol_node   = torch.tensor(sol_node_rows,   dtype=torch.float32)
        sol_edge   = torch.tensor(sol_edge_rows,   dtype=torch.float32)
        kwargs: Dict[str, Any] = {}
        if extra is not None:
            if "node_type" in extra:
                kwargs["node_type"] = torch.tensor(extra["node_type"], dtype=torch.long)
            if "edge_type" in extra:
                kwargs["edge_type"] = torch.tensor(extra["edge_type"], dtype=torch.long)
            if "x_dyn" in extra:
                kwargs["x_dyn"] = torch.tensor(extra["x_dyn"], dtype=torch.float32)
            if "edge_attr_dyn" in extra:
                kwargs["edge_attr_dyn"] = torch.tensor(extra["edge_attr_dyn"], dtype=torch.float32)
        return PyGData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_t,
            sol_node=sol_node,
            sol_edge=sol_edge,
            meta=meta,
            **kwargs,
        )

    # Fallback: plain dict (works without torch / PyG)
    out = {
        "x":          x_rows,           # list[list[float]]  shape (N, NODE_FEAT_DIM)
        "edge_index": edge_index_pair,  # list[list[int]]    shape (2, E)
        "edge_attr":  ea_rows,          # list[list[float]]  shape (E, EDGE_FEAT_DIM)
        "y":          y,                # float
        "sol_node":   sol_node_rows,    # list[list[float]]  shape (N, SOL_NODE_DIM)
        "sol_edge":   sol_edge_rows,    # list[list[float]]  shape (E, SOL_EDGE_DIM)
        "meta":       meta,
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def build_graph_from_json(
    file_path: Union[str, Path],
    normalize_features: bool = False,
    include_solution:   bool = True,
    include_links:      bool = True,
    normalization_mode: str = "dataset",
    merge_solution:     bool = False,
) -> GraphResult:
    """
    Build a single graph from an OPFData JSON file.

    This is a convenience wrapper around ``PowerGraphBuilder``.

    Parameters
    ----------
    file_path : str or Path
        Path to a single ``example_*.json`` file.
    normalize_features : bool
        Apply per-graph StandardScaler normalisation to features.
    include_solution : bool
        Attach ground-truth solution tensors as ``sol_node`` / ``sol_edge``.
    include_links : bool
        Include generator / load / shunt link edges.
    normalization_mode : str
        Either ``"dataset"`` or ``"graph"``.
    merge_solution : bool
        If True and solutions are included, return additional merged dynamic
        features ``x_dyn`` / ``edge_attr_dyn``.

    Returns
    -------
    PyG ``Data`` object or dict depending on available packages.
    """
    builder = PowerGraphBuilder(
        normalize_features=normalize_features,
        include_solution=include_solution,
        include_links=include_links,
        normalization_mode=normalization_mode,
        merge_solution=merge_solution,
    )
    return builder.build_graph_from_json(file_path)


def batch_process(
    folder_path: Union[str, Path],
    recursive:          bool = True,
    glob_pattern:       str  = "*.json",
    normalize_features: bool = False,
    include_solution:   bool = True,
    include_links:      bool = True,
    normalization_mode: str = "dataset",
    merge_solution:     bool = False,
) -> List[GraphResult]:
    """
    Convert all JSON files in *folder_path* to graph objects.

    This is a convenience wrapper around ``PowerGraphBuilder``.

    Parameters
    ----------
    folder_path : str or Path
        Root directory to search for JSON files.
    recursive : bool
        Search sub-directories recursively.  Default: True.
    glob_pattern : str
        File glob pattern.  Default: ``"*.json"``.
    normalize_features : bool
        Apply per-graph StandardScaler normalisation to features.
    include_solution : bool
        Attach ground-truth solution tensors.
    include_links : bool
        Include link edges.
    normalization_mode : str
        Either ``"dataset"`` or ``"graph"``.
    merge_solution : bool
        If True and solutions are included, return additional merged dynamic
        features ``x_dyn`` / ``edge_attr_dyn``.

    Returns
    -------
    List of graph objects.
    """
    builder = PowerGraphBuilder(
        normalize_features=normalize_features,
        include_solution=include_solution,
        include_links=include_links,
        normalization_mode=normalization_mode,
        merge_solution=merge_solution,
    )
    return builder.batch_process(
        folder_path, recursive=recursive, glob_pattern=glob_pattern
    )
