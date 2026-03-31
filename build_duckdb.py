from __future__ import annotations

import argparse
from pathlib import Path

from opfdata_pipeline import build_duckdb_from_parquet, run_example_duckdb_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DuckDB tables from OPFData parquet outputs")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("data/processed/parquet"),
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=Path("data/processed/opfdata.duckdb"),
        help="Output DuckDB database path",
    )
    parser.add_argument(
        "--run-queries",
        action="store_true",
        help="Run example analysis queries after loading tables",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[INFO] Building DuckDB file: {args.duckdb_path}")
    build_duckdb_from_parquet(args.parquet_dir, args.duckdb_path)

    if args.run_queries:
        run_example_duckdb_queries(args.duckdb_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
