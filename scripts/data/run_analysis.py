#!/usr/bin/env python3
from pathlib import Path
import duckdb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DB_PATH = 'data/processed/opfdata.duckdb'
OUT_DIR = Path('analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

conn = duckdb.connect(DB_PATH)

# case distribution
q_cases = 'SELECT case_name, count(*) as n FROM opf_samples GROUP BY case_name ORDER BY n DESC'
df_cases = conn.execute(q_cases).df()
df_cases.to_csv(OUT_DIR / 'opf_case_distribution.csv', index=False)

# objective distribution
q_obj = "SELECT objective FROM opf_samples WHERE objective IS NOT NULL"
df_obj = conn.execute(q_obj).df()
if not df_obj.empty:
    obj_series = df_obj['objective']
    plt.figure(figsize=(6,4))
    obj_series.hist(bins=50)
    plt.title('Objective distribution (OPFData)')
    plt.xlabel('Objective')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'objective_hist.png')

# node/edge stats
q_stats = 'SELECT n_nodes, n_edges FROM opf_samples'
df_stats = conn.execute(q_stats).df()
df_stats.describe().to_csv(OUT_DIR / 'node_edge_stats.csv')

# powergraph table
df_pg = conn.execute('SELECT filename, relpath, size_mb, n_keys FROM powergraph_files ORDER BY filename').df()
df_pg.to_csv(OUT_DIR / 'powergraph_files.csv', index=False)

# sample one metadata
first = df_pg['filename'].iloc[0]
meta_path = Path('data/processed/powergraph_graphs') / first
with open(str(meta_path),'r') as f:
    meta = f.read()
with open(OUT_DIR / f'{first}_meta.json','w') as f:
    f.write(meta)

print('Wrote files to', OUT_DIR)
print('Case distribution:')
print(df_cases.head(20).to_string(index=False))
print('\nPowerGraph files count:', len(df_pg))
conn.close()
