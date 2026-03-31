from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from power_graph_builder import EDGE_FEAT_DIM, EDGE_RAW_MAX, EDGE_TYPE_DIM, NODE_FEAT_DIM, NODE_RAW_MAX, SOL_EDGE_DIM, SOL_NODE_DIM

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for this pipeline. Install with: python -m pip install duckdb"
    ) from exc

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


NODE_TYPES = ("bus", "generator", "load", "shunt")
EDGE_TYPES = ("ac_line", "transformer", "generator_link", "load_link", "shunt_link")

NODE_TYPE_TO_IDX = {name: idx for idx, name in enumerate(NODE_TYPES)}
EDGE_TYPE_TO_IDX = {name: idx for idx, name in enumerate(EDGE_TYPES)}

EDGE_ENDPOINT_TYPES = {
    "ac_line": ("bus", "bus"),
    "transformer": ("bus", "bus"),
    "generator_link": ("generator", "bus"),
    "load_link": ("load", "bus"),
    "shunt_link": ("shunt", "bus"),
}


@dataclass
class ProcessSummary:
    json_files_seen: int
    json_files_parsed: int
    json_files_failed: int
    parquet_files: Dict[str, Path]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float_list(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, (list, tuple)):
        out: List[float] = []
        for v in raw:
            fv = _safe_float(v)
            out.append(0.0 if fv is None else fv)
        return out
    return []


def _nested_first_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        for item in value:
            found = _nested_first_number(item)
            if found is not None:
                return found
    return None


def _one_hot(index: int, width: int) -> List[float]:
    vec = [0.0] * width
    if 0 <= index < width:
        vec[index] = 1.0
    return vec


def _pad(values: List[float], width: int) -> List[float]:
    if len(values) >= width:
        return values[:width]
    return values + [0.0] * (width - len(values))


def _sample_id_from_path(path: Path, input_dir: Path) -> str:
    rel = path.relative_to(input_dir).as_posix()
    stem = rel[:-5] if rel.lower().endswith(".json") else rel
    return stem.replace("/", "__")


def iter_json_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.rglob("*.json"))


class _NDJSONSink:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "samples": self.base_dir / "samples.ndjson",
            "nodes": self.base_dir / "nodes.ndjson",
            "edges": self.base_dir / "edges.ndjson",
            "solution_nodes": self.base_dir / "solution_nodes.ndjson",
            "solution_edges": self.base_dir / "solution_edges.ndjson",
        }
        self.handles = {
            key: path.open("w", encoding="utf-8")
            for key, path in self.paths.items()
        }

    def write_rows(self, table_name: str, rows: Iterable[Dict[str, Any]]) -> None:
        fh = self.handles[table_name]
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for fh in self.handles.values():
            fh.close()


def extract_rows_from_sample(sample: Dict[str, Any], sample_id: str, source_file: str) -> Dict[str, List[Dict[str, Any]]]:
    grid = sample.get("grid", {}) if isinstance(sample, dict) else {}
    solution = sample.get("solution", {}) if isinstance(sample, dict) else {}
    metadata = sample.get("metadata", {}) if isinstance(sample, dict) else {}

    grid_nodes = grid.get("nodes", {}) if isinstance(grid, dict) else {}
    grid_edges = grid.get("edges", {}) if isinstance(grid, dict) else {}

    buses = grid_nodes.get("bus", []) or []
    generators = grid_nodes.get("generator", []) or []
    loads = grid_nodes.get("load", []) or []
    shunts = grid_nodes.get("shunt", []) or []

    node_counts = {
        "bus": len(buses),
        "generator": len(generators),
        "load": len(loads),
        "shunt": len(shunts),
    }

    edge_counts: Dict[str, int] = {}
    for edge_type in EDGE_TYPES:
        edge_data = grid_edges.get(edge_type, {})
        if not isinstance(edge_data, dict):
            edge_data = {}
        senders = edge_data.get("senders", []) or []
        receivers = edge_data.get("receivers", []) or []
        features = edge_data.get("features", []) or []
        edge_counts[edge_type] = max(len(senders), len(receivers), len(features))

    base_mva = _nested_first_number(grid.get("context"))

    sample_row = {
        "sample_id": sample_id,
        "source_file": source_file,
        "objective": _safe_float(metadata.get("objective")),
        "base_mva": base_mva,
        "has_solution": bool(solution),
        "n_bus": node_counts["bus"],
        "n_generator": node_counts["generator"],
        "n_load": node_counts["load"],
        "n_shunt": node_counts["shunt"],
        "n_nodes": sum(node_counts.values()),
        "n_edges_ac_line": edge_counts["ac_line"],
        "n_edges_transformer": edge_counts["transformer"],
        "n_edges_generator_link": edge_counts["generator_link"],
        "n_edges_load_link": edge_counts["load_link"],
        "n_edges_shunt_link": edge_counts["shunt_link"],
        "n_edges": sum(edge_counts.values()),
    }

    node_rows: List[Dict[str, Any]] = []
    node_id = 0
    for node_type in NODE_TYPES:
        rows = grid_nodes.get(node_type, []) or []
        for local_id, raw in enumerate(rows):
            features = _to_float_list(raw)
            node_rows.append(
                {
                    "sample_id": sample_id,
                    "node_id": node_id,
                    "node_type": node_type,
                    "node_local_id": local_id,
                    "raw_feature_dim": len(features),
                    "raw_features": features,
                }
            )
            node_id += 1

    edge_rows: List[Dict[str, Any]] = []
    edge_id = 0
    for edge_type in EDGE_TYPES:
        src_type, dst_type = EDGE_ENDPOINT_TYPES[edge_type]
        edge_data = grid_edges.get(edge_type, {})
        if not isinstance(edge_data, dict):
            edge_data = {}
        senders = edge_data.get("senders", []) or []
        receivers = edge_data.get("receivers", []) or []
        features = edge_data.get("features", []) or []
        n_edges = max(len(senders), len(receivers), len(features))
        for local_id in range(n_edges):
            raw = _to_float_list(features[local_id]) if local_id < len(features) else []
            edge_rows.append(
                {
                    "sample_id": sample_id,
                    "edge_id": edge_id,
                    "edge_type": edge_type,
                    "edge_local_id": local_id,
                    "src_node_type": src_type,
                    "dst_node_type": dst_type,
                    "src_node_local_id": _safe_int(senders[local_id]) if local_id < len(senders) else 0,
                    "dst_node_local_id": _safe_int(receivers[local_id]) if local_id < len(receivers) else 0,
                    "raw_feature_dim": len(raw),
                    "raw_features": raw,
                }
            )
            edge_id += 1

    sol_node_rows: List[Dict[str, Any]] = []
    sol_nodes = solution.get("nodes", {}) if isinstance(solution, dict) else {}
    for node_type in ("bus", "generator"):
        rows = sol_nodes.get(node_type, []) or []
        for local_id, raw in enumerate(rows):
            features = _to_float_list(raw)
            sol_node_rows.append(
                {
                    "sample_id": sample_id,
                    "node_type": node_type,
                    "node_local_id": local_id,
                    "raw_feature_dim": len(features),
                    "raw_features": features,
                }
            )

    sol_edge_rows: List[Dict[str, Any]] = []
    sol_edges = solution.get("edges", {}) if isinstance(solution, dict) else {}
    for edge_type in ("ac_line", "transformer"):
        edge_data = sol_edges.get(edge_type, {})
        if not isinstance(edge_data, dict):
            edge_data = {}
        features = edge_data.get("features", []) or []
        for local_id, raw in enumerate(features):
            values = _to_float_list(raw)
            sol_edge_rows.append(
                {
                    "sample_id": sample_id,
                    "edge_type": edge_type,
                    "edge_local_id": local_id,
                    "raw_feature_dim": len(values),
                    "raw_features": values,
                }
            )

    return {
        "samples": [sample_row],
        "nodes": node_rows,
        "edges": edge_rows,
        "solution_nodes": sol_node_rows,
        "solution_edges": sol_edge_rows,
    }


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _ndjson_to_parquet(ndjson_path: Path, parquet_path: Path, con: "duckdb.DuckDBPyConnection") -> bool:
    if not ndjson_path.exists() or ndjson_path.stat().st_size == 0:
        return False
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    query = (
        "COPY ("
        f"SELECT * FROM read_json_auto('{_sql_path(ndjson_path)}', format='newline_delimited')"
        ") TO "
        f"'{_sql_path(parquet_path)}'"
        " (FORMAT PARQUET)"
    )
    con.execute(query)
    return True


def process_json_to_parquet(
    input_dir: Path,
    parquet_dir: Path,
    temp_dir: Optional[Path] = None,
    log_every: int = 200,
    limit: Optional[int] = None,
) -> ProcessSummary:
    input_dir = input_dir.resolve()
    parquet_dir = parquet_dir.resolve()
    temp_dir = (temp_dir or parquet_dir.parent / "_tmp_ndjson").resolve()

    files = iter_json_files(input_dir)
    if limit is not None:
        files = files[:limit]

    sink = _NDJSONSink(temp_dir)
    seen = 0
    parsed = 0
    failed = 0

    try:
        for path in files:
            seen += 1
            try:
                with path.open("r", encoding="utf-8") as fh:
                    sample = json.load(fh)
                sample_id = _sample_id_from_path(path, input_dir)
                source_file = path.relative_to(input_dir).as_posix()
                rows = extract_rows_from_sample(sample, sample_id=sample_id, source_file=source_file)
                for table_name, table_rows in rows.items():
                    sink.write_rows(table_name, table_rows)
                parsed += 1
            except Exception as exc:
                failed += 1
                print(f"[WARN] Failed to parse {path}: {exc}")

            if seen % max(1, log_every) == 0:
                print(f"[INFO] Processed {seen}/{len(files)} JSON files (ok={parsed}, failed={failed})")
    finally:
        sink.close()

    parquet_paths = {
        "samples": parquet_dir / "samples.parquet",
        "nodes": parquet_dir / "nodes.parquet",
        "edges": parquet_dir / "edges.parquet",
        "solution_nodes": parquet_dir / "solution_nodes.parquet",
        "solution_edges": parquet_dir / "solution_edges.parquet",
    }

    con = duckdb.connect(database=":memory:")
    try:
        for table_name, ndjson_path in sink.paths.items():
            created = _ndjson_to_parquet(ndjson_path, parquet_paths[table_name], con)
            if created:
                print(f"[INFO] Wrote {parquet_paths[table_name]}")
            else:
                print(f"[INFO] Skipped empty table: {table_name}")
    finally:
        con.close()

    return ProcessSummary(
        json_files_seen=seen,
        json_files_parsed=parsed,
        json_files_failed=failed,
        parquet_files=parquet_paths,
    )


def build_duckdb_from_parquet(parquet_dir: Path, duckdb_path: Path) -> None:
    parquet_dir = parquet_dir.resolve()
    duckdb_path = duckdb_path.resolve()
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    table_files = {
        "samples": parquet_dir / "samples.parquet",
        "nodes": parquet_dir / "nodes.parquet",
        "edges": parquet_dir / "edges.parquet",
        "solution_nodes": parquet_dir / "solution_nodes.parquet",
        "solution_edges": parquet_dir / "solution_edges.parquet",
    }

    con = duckdb.connect(str(duckdb_path))
    try:
        for table_name, file_path in table_files.items():
            if not file_path.exists():
                print(f"[INFO] Missing parquet for table {table_name}, skipping load")
                continue
            con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{_sql_path(file_path)}')"
            )
            count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"[INFO] Loaded {table_name}: {count} rows")
    finally:
        con.close()


def run_example_duckdb_queries(duckdb_path: Path) -> None:
    duckdb_path = duckdb_path.resolve()
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        def _print_query(sql: str) -> None:
            rows = con.execute(sql).fetchall()
            if not rows:
                print("(no rows)")
                return
            for row in rows:
                print(" | ".join(str(col) for col in row))

        print("\n[QUERY] Number of samples")
        _print_query("SELECT COUNT(*) AS num_samples FROM samples")

        print("\n[QUERY] Average number of nodes per sample")
        _print_query(
            """
            SELECT AVG(node_count) AS avg_nodes_per_sample
            FROM (
                SELECT sample_id, COUNT(*) AS node_count
                FROM nodes
                GROUP BY sample_id
            )
            """
        )

        print("\n[QUERY] Average number of edges per sample")
        _print_query(
            """
            SELECT AVG(edge_count) AS avg_edges_per_sample
            FROM (
                SELECT sample_id, COUNT(*) AS edge_count
                FROM edges
                GROUP BY sample_id
            )
            """
        )

        print("\n[QUERY] Node type distribution")
        _print_query(
            "SELECT node_type, COUNT(*) AS count FROM nodes GROUP BY node_type ORDER BY count DESC"
        )

        print("\n[QUERY] Edge type distribution")
        _print_query(
            "SELECT edge_type, COUNT(*) AS count FROM edges GROUP BY edge_type ORDER BY count DESC"
        )
    finally:
        con.close()


def export_graph_dataset_from_parquet(
    parquet_dir: Path,
    graphs_dir: Path,
    graph_format: str = "json",
    log_every: int = 200,
    limit: Optional[int] = None,
) -> int:
    parquet_dir = parquet_dir.resolve()
    graphs_dir = graphs_dir.resolve()
    graphs_dir.mkdir(parents=True, exist_ok=True)

    samples_parquet = parquet_dir / "samples.parquet"
    nodes_parquet = parquet_dir / "nodes.parquet"
    edges_parquet = parquet_dir / "edges.parquet"
    sol_nodes_parquet = parquet_dir / "solution_nodes.parquet"
    sol_edges_parquet = parquet_dir / "solution_edges.parquet"

    if not samples_parquet.exists() or not nodes_parquet.exists() or not edges_parquet.exists():
        raise FileNotFoundError(
            "samples.parquet, nodes.parquet, and edges.parquet are required before graph export"
        )

    con = duckdb.connect(database=":memory:")
    exported = 0
    has_sol_nodes = sol_nodes_parquet.exists()
    has_sol_edges = sol_edges_parquet.exists()

    try:
        sample_ids = [
            row[0]
            for row in con.execute(
                f"SELECT sample_id FROM read_parquet('{_sql_path(samples_parquet)}') ORDER BY sample_id"
            ).fetchall()
        ]
        if limit is not None:
            sample_ids = sample_ids[:limit]

        for idx, sample_id in enumerate(sample_ids, start=1):
            objective_row = con.execute(
                f"SELECT objective, source_file FROM read_parquet('{_sql_path(samples_parquet)}') WHERE sample_id = ?",
                [sample_id],
            ).fetchone()
            objective = float(objective_row[0]) if objective_row and objective_row[0] is not None else 0.0
            source_file = objective_row[1] if objective_row else ""

            node_rows = con.execute(
                f"""
                SELECT node_id, node_type, node_local_id, raw_features
                FROM read_parquet('{_sql_path(nodes_parquet)}')
                WHERE sample_id = ?
                ORDER BY node_id
                """,
                [sample_id],
            ).fetchall()

            edge_rows = con.execute(
                f"""
                SELECT edge_id, edge_type, edge_local_id, src_node_type, src_node_local_id,
                       dst_node_type, dst_node_local_id, raw_features
                FROM read_parquet('{_sql_path(edges_parquet)}')
                WHERE sample_id = ?
                ORDER BY edge_id
                """,
                [sample_id],
            ).fetchall()

            global_node_index: Dict[Tuple[str, int], int] = {}
            node_type_idx: List[int] = []
            x_rows: List[List[float]] = []
            for node_id, node_type, node_local_id, raw_features in node_rows:
                node_type_idx_value = NODE_TYPE_TO_IDX[str(node_type)]
                features = _to_float_list(raw_features)
                x_rows.append(_pad(features, NODE_RAW_MAX) + _one_hot(node_type_idx_value, len(NODE_TYPES)))
                node_type_idx.append(node_type_idx_value)
                global_node_index[(str(node_type), int(node_local_id))] = int(node_id)

            src_list: List[int] = []
            dst_list: List[int] = []
            edge_type_idx: List[int] = []
            edge_local_map: Dict[Tuple[str, int], int] = {}
            edge_attr_rows: List[List[float]] = []

            for edge_id, edge_type, edge_local_id, src_type, src_local, dst_type, dst_local, raw_features in edge_rows:
                src = global_node_index.get((str(src_type), int(src_local)), 0)
                dst = global_node_index.get((str(dst_type), int(dst_local)), 0)
                src_list.append(src)
                dst_list.append(dst)

                edge_type_name = str(edge_type)
                edge_type_idx_value = EDGE_TYPE_TO_IDX[edge_type_name]
                edge_type_idx.append(edge_type_idx_value)
                edge_local_map[(edge_type_name, int(edge_local_id))] = int(edge_id)

                features = _to_float_list(raw_features)
                edge_attr_rows.append(_pad(features, EDGE_RAW_MAX) + _one_hot(edge_type_idx_value, len(EDGE_TYPES)))

            sol_node_rows = [[0.0] * SOL_NODE_DIM for _ in range(len(x_rows))]
            if has_sol_nodes:
                rows = con.execute(
                    f"""
                    SELECT node_type, node_local_id, raw_features
                    FROM read_parquet('{_sql_path(sol_nodes_parquet)}')
                    WHERE sample_id = ?
                    """,
                    [sample_id],
                ).fetchall()
                for node_type, node_local_id, raw_features in rows:
                    node_idx = global_node_index.get((str(node_type), int(node_local_id)))
                    if node_idx is None or node_idx >= len(sol_node_rows):
                        continue
                    sol_node_rows[node_idx] = _pad(_to_float_list(raw_features), SOL_NODE_DIM)

            sol_edge_rows = [[0.0] * SOL_EDGE_DIM for _ in range(len(edge_attr_rows))]
            if has_sol_edges:
                rows = con.execute(
                    f"""
                    SELECT edge_type, edge_local_id, raw_features
                    FROM read_parquet('{_sql_path(sol_edges_parquet)}')
                    WHERE sample_id = ?
                    """,
                    [sample_id],
                ).fetchall()
                for edge_type, edge_local_id, raw_features in rows:
                    edge_idx = edge_local_map.get((str(edge_type), int(edge_local_id)))
                    if edge_idx is None or edge_idx >= len(sol_edge_rows):
                        continue
                    sol_edge_rows[edge_idx] = _pad(_to_float_list(raw_features), SOL_EDGE_DIM)

            graph_dict = {
                "x": x_rows,
                "edge_index": [src_list, dst_list],
                "edge_attr": edge_attr_rows,
                "y": objective,
                "sol_node": sol_node_rows,
                "sol_edge": sol_edge_rows,
                "node_type": node_type_idx,
                "edge_type": edge_type_idx,
                "metadata": {
                    "sample_id": sample_id,
                    "source_file": source_file,
                    "schema_version": "v2",
                    "node_feature_dim": NODE_FEAT_DIM,
                    "edge_feature_dim": EDGE_FEAT_DIM,
                },
            }

            output_path = graphs_dir / f"{sample_id}.{graph_format}"
            if graph_format == "json":
                output_path.write_text(json.dumps(graph_dict, ensure_ascii=False), encoding="utf-8")
            elif graph_format == "pt":
                if not _TORCH_AVAILABLE:
                    raise RuntimeError("graph_format='pt' requires torch installed")
                torch.save(graph_dict, output_path)
            else:
                raise ValueError("graph_format must be one of: json, pt")

            exported += 1
            if idx % max(1, log_every) == 0:
                print(f"[INFO] Exported {idx}/{len(sample_ids)} graphs")
    finally:
        con.close()

    manifest = {
        "num_graphs": exported,
        "graph_format": graph_format,
        "schema": {
            "nodes": list(NODE_TYPES),
            "edges": list(EDGE_TYPES),
            "x": f"(N, {NODE_FEAT_DIM}) = padded raw node features ({NODE_RAW_MAX}) + one-hot node type ({len(NODE_TYPES)})",
            "edge_index": "(2, E) source/destination global node indices",
            "edge_attr": f"(E, {EDGE_FEAT_DIM}) = padded raw edge features ({EDGE_RAW_MAX}) + one-hot edge type ({len(EDGE_TYPES)})",
            "y": "scalar objective from samples.objective",
            "sol_node": f"(N, {SOL_NODE_DIM}) solution node features",
            "sol_edge": f"(E, {SOL_EDGE_DIM}) solution edge features",
        },
    }
    (graphs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return exported


def remove_temp_dir(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("power_demo_work/opfdata"),
        help="Directory containing OPFData JSON files",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("data/processed/parquet"),
        help="Output directory for parquet tables",
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=Path("data/processed/opfdata.duckdb"),
        help="DuckDB file path",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/processed/graphs"),
        help="Output directory for graph dataset files",
    )
    parser.add_argument("--log-every", type=int, default=200, help="Progress logging frequency")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N samples")
