from __future__ import annotations

import argparse
from pathlib import Path

from opfdata_pipeline import (
    add_common_cli_args,
    build_duckdb_from_parquet,
    export_graph_dataset_from_parquet,
    process_json_to_parquet,
    remove_temp_dir,
    run_example_duckdb_queries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process OPFData JSON files into Parquet tables, optional DuckDB tables, and graph dataset exports."
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/processed/_tmp_ndjson"),
        help="Temporary NDJSON workspace used before parquet conversion",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary NDJSON files",
    )
    parser.add_argument(
        "--build-duckdb",
        action="store_true",
        help="Build DuckDB tables from parquet outputs",
    )
    parser.add_argument(
        "--run-duckdb-queries",
        action="store_true",
        help="Run example analysis queries after building/using DuckDB",
    )
    parser.add_argument(
        "--export-graphs",
        action="store_true",
        help="Export graph dataset from parquet tables",
    )
    parser.add_argument(
        "--graph-format",
        choices=["json", "pt"],
        default="json",
        help="Graph output format for data/processed/graphs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[INFO] Input JSON directory: {args.input_dir}")
    summary = process_json_to_parquet(
        input_dir=args.input_dir,
        parquet_dir=args.parquet_dir,
        temp_dir=args.temp_dir,
        log_every=args.log_every,
        limit=args.limit,
    )

    print(
        f"[INFO] JSON scan complete: seen={summary.json_files_seen}, "
        f"parsed={summary.json_files_parsed}, failed={summary.json_files_failed}"
    )

    if args.build_duckdb:
        print(f"[INFO] Building DuckDB: {args.duckdb_path}")
        build_duckdb_from_parquet(args.parquet_dir, args.duckdb_path)

    if args.run_duckdb_queries:
        print(f"[INFO] Running DuckDB queries: {args.duckdb_path}")
        run_example_duckdb_queries(args.duckdb_path)

    if args.export_graphs:
        print(f"[INFO] Exporting graphs to: {args.graphs_dir}")
        n = export_graph_dataset_from_parquet(
            parquet_dir=args.parquet_dir,
            graphs_dir=args.graphs_dir,
            graph_format=args.graph_format,
            log_every=args.log_every,
            limit=args.limit,
        )
        print(f"[INFO] Graph export complete: {n} graphs")

    if not args.keep_temp:
        remove_temp_dir(args.temp_dir)
        print(f"[INFO] Removed temp dir: {args.temp_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
