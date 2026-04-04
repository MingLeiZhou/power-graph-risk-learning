#!/usr/bin/env python3
"""Ingest full OPFData + PowerGraph metadata into DuckDB (clean rebuild)."""
from pathlib import Path
import json
import duckdb

DATA_ROOT = Path('data')
OPFDATA_DIR = DATA_ROOT / 'opfdata'
PG_META_DIR = DATA_ROOT / 'processed' / 'powergraph_graphs'
OUT_DB = DATA_ROOT / 'processed' / 'opfdata.duckdb'
BATCH = 1000


def iter_opf_rows():
    for p in OPFDATA_DIR.rglob('*.json'):
        try:
            d = json.loads(p.read_text())
            case = 'unknown'
            for part in p.parts:
                if 'case' in part.lower():
                    case = part
                    break
            grid = d.get('grid', {})
            nodes = grid.get('nodes', {})
            edges = grid.get('edges', {})
            n_nodes = sum(len(v) for v in nodes.values()) if isinstance(nodes, dict) else (len(nodes) if isinstance(nodes, list) else 0)
            n_edges = sum(len(v) for v in edges.values()) if isinstance(edges, dict) else (len(edges) if isinstance(edges, list) else 0)
            objective = d.get('metadata', {}).get('objective', None)
            yield (str(p.relative_to(DATA_ROOT)), case, int(n_nodes), int(n_edges), objective, str(p))
        except Exception:
            continue


def main():
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(OUT_DB))

    con.execute('DROP TABLE IF EXISTS opf_samples')
    con.execute('''CREATE TABLE opf_samples(
        relpath TEXT, case_name TEXT, n_nodes INTEGER, n_edges INTEGER, objective DOUBLE, fullpath TEXT
    )''')

    batch, total = [], 0
    for row in iter_opf_rows():
        batch.append(row)
        if len(batch) >= BATCH:
            con.executemany('INSERT INTO opf_samples VALUES (?,?,?,?,?,?)', batch)
            total += len(batch)
            print(f'Inserted OPF rows: {total}', end='\r')
            batch = []
    if batch:
        con.executemany('INSERT INTO opf_samples VALUES (?,?,?,?,?,?)', batch)
        total += len(batch)
    print(f'\nOPF inserted: {total}')

    con.execute('DROP TABLE IF EXISTS powergraph_files')
    con.execute('''CREATE TABLE powergraph_files(
        filename TEXT, relpath TEXT, size_mb DOUBLE, n_keys INTEGER, vars JSON
    )''')

    # only keep unique per-mat metadata (the 32 generated with __raw__ in filename)
    files = sorted([p for p in PG_META_DIR.glob('*.json') if '__raw__' in p.name])
    rows = []
    for p in files:
        m = json.loads(p.read_text())
        relpath = m.get('source', '')
        size_mb = 0.0
        src = DATA_ROOT / relpath if relpath else None
        if src and src.exists():
            size_mb = src.stat().st_size / (1024 * 1024)
        n_keys = len(m.get('vars', {}))
        rows.append((p.name, relpath, size_mb, n_keys, json.dumps(m.get('vars', {}))))

    if rows:
        con.executemany('INSERT INTO powergraph_files VALUES (?,?,?,?,?)', rows)
    print(f'PowerGraph metadata inserted: {len(rows)}')

    # sanity prints
    print('opf_samples =', con.execute('SELECT count(*) FROM opf_samples').fetchone()[0])
    print('powergraph_files =', con.execute('SELECT count(*) FROM powergraph_files').fetchone()[0])
    con.close()


if __name__ == '__main__':
    main()
