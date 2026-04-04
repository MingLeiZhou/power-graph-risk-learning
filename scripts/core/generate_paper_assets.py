#!/usr/bin/env python3
"""Generate paper-ready tables/figures and expand PowerGraph JSON vars into structured DuckDB table."""
from pathlib import Path
import json
import re
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB = Path('data/processed/opfdata.duckdb')
ANALYSIS = Path('analysis')
FIG_DIR = ANALYSIS / 'paper_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
})


def parse_shape(v):
    shape = v.get('shape') if isinstance(v, dict) else None
    if shape is None:
        return None, None
    try:
        if isinstance(shape, str):
            # like "(57, 1)"
            nums = re.findall(r'\d+', shape)
            dims = tuple(int(x) for x in nums)
        elif isinstance(shape, (list, tuple)):
            dims = tuple(int(x) for x in shape)
        else:
            return None, None
        n_elem = int(np.prod(dims)) if len(dims) > 0 else None
        return dims, n_elem
    except Exception:
        return None, None


def split_filename(fn):
    # ieee24__ieee24__raw__Ef.mat.json
    net = fn.split('__')[0] if '__' in fn else 'unknown'
    m = re.search(r'__raw__(.+?)\.mat\.json$', fn)
    mat_type = m.group(1) if m else fn
    return net, mat_type


def main():
    con = duckdb.connect(str(DB))

    # --- 1) expand vars into structured rows ---
    pg = con.execute('SELECT filename, relpath, size_mb, n_keys, vars FROM powergraph_files ORDER BY filename').fetchdf()
    rows = []
    for _, r in pg.iterrows():
        fn = r['filename']
        rel = r['relpath']
        size_mb = float(r['size_mb']) if r['size_mb'] is not None else None
        net, mat_type = split_filename(fn)
        vars_obj = r['vars']
        if isinstance(vars_obj, str):
            try:
                vars_obj = json.loads(vars_obj)
            except Exception:
                vars_obj = {}
        if not isinstance(vars_obj, dict):
            vars_obj = {}
        for k, v in vars_obj.items():
            dtype = v.get('dtype') if isinstance(v, dict) else None
            vtype = v.get('type') if isinstance(v, dict) else None
            shape, n_elem = parse_shape(v if isinstance(v, dict) else {})
            rows.append({
                'filename': fn,
                'network': net,
                'mat_type': mat_type,
                'relpath': rel,
                'size_mb': size_mb,
                'var_name': k,
                'dtype': dtype,
                'var_type': vtype,
                'shape': str(shape) if shape is not None else None,
                'n_dims': len(shape) if shape is not None else None,
                'n_elements': n_elem,
            })

    var_df = pd.DataFrame(rows)
    con.execute('DROP TABLE IF EXISTS powergraph_variables')
    con.register('var_df_tmp', var_df)
    con.execute('CREATE TABLE powergraph_variables AS SELECT * FROM var_df_tmp')

    var_df.to_csv(ANALYSIS / 'tbl_powergraph_variables.csv', index=False)

    # --- 2) paper-style figures and tables ---
    case_df = con.execute('''
        SELECT case_name, COUNT(*) AS samples
        FROM opf_samples
        GROUP BY case_name
        ORDER BY samples DESC
    ''').fetchdf()
    case_df.to_csv(ANALYSIS / 'tbl_opf_case_distribution.csv', index=False)

    stats_df = con.execute('''
        SELECT
            MIN(n_nodes) AS min_nodes, MAX(n_nodes) AS max_nodes, AVG(n_nodes) AS avg_nodes,
            MIN(n_edges) AS min_edges, MAX(n_edges) AS max_edges, AVG(n_edges) AS avg_edges,
            MIN(objective) AS min_obj, MAX(objective) AS max_obj, AVG(objective) AS avg_obj
        FROM opf_samples
    ''').fetchdf()
    stats_df.to_csv(ANALYSIS / 'tbl_opf_summary_stats.csv', index=False)

    # Fig 1: case distribution bar
    plt.figure(figsize=(7.2, 4.2))
    xlabels = [x.replace('dataset_release_1__', '').replace('_0_extracted', '') for x in case_df['case_name']]
    plt.bar(xlabels, case_df['samples'], color='#4C78A8')
    plt.title('OPFData Sample Distribution by Network Case')
    plt.xlabel('Network case')
    plt.ylabel('Number of samples')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig01_opf_case_distribution.png')
    plt.close()

    # Fig 2: objective histogram
    obj = con.execute('SELECT objective FROM opf_samples WHERE objective IS NOT NULL').fetchdf()['objective']
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(obj, bins=70, color='#59A14F', edgecolor='white')
    plt.title('Objective Value Distribution (OPFData)')
    plt.xlabel('Objective value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig02_objective_histogram.png')
    plt.close()

    # Fig 3: objective by case (boxplot)
    obj_case = con.execute('''
        SELECT case_name, objective
        FROM opf_samples
        WHERE objective IS NOT NULL
    ''').fetchdf()
    order = case_df['case_name'].tolist()
    data = [obj_case[obj_case['case_name'] == c]['objective'].values for c in order]
    labels = [c.replace('dataset_release_1__', '').replace('_0_extracted', '') for c in order]
    plt.figure(figsize=(7.6, 4.6))
    bp = plt.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
    for b in bp['boxes']:
        b.set(facecolor='#F28E2B', alpha=0.7)
    plt.title('Objective Distribution by Network Case')
    plt.xlabel('Network case')
    plt.ylabel('Objective value')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig03_objective_boxplot_by_case.png')
    plt.close()

    # Fig 4: PowerGraph variable count per MAT type
    var_count = con.execute('''
        SELECT mat_type, COUNT(DISTINCT var_name) AS unique_vars
        FROM powergraph_variables
        GROUP BY mat_type
        ORDER BY unique_vars DESC, mat_type
    ''').fetchdf()
    var_count.to_csv(ANALYSIS / 'tbl_powergraph_varcount_by_mat_type.csv', index=False)

    plt.figure(figsize=(7.2, 4.2))
    plt.bar(var_count['mat_type'], var_count['unique_vars'], color='#E15759')
    plt.title('PowerGraph Unique Variable Count by MAT Type')
    plt.xlabel('MAT type')
    plt.ylabel('Unique variable count')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig04_powergraph_varcount_by_type.png')
    plt.close()

    # Fig 5: PowerGraph files and key counts by network
    net_keys = con.execute('''
        SELECT network, AVG(n_keys) AS avg_keys, COUNT(*) AS files
        FROM (
            SELECT filename, split_part(filename, '__', 1) AS network, n_keys
            FROM powergraph_files
        )
        GROUP BY network
        ORDER BY network
    ''').fetchdf()
    net_keys.to_csv(ANALYSIS / 'tbl_powergraph_network_summary.csv', index=False)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.bar(net_keys['network'], net_keys['files'], color='#76B7B2', label='File count')
    ax1.set_xlabel('Network')
    ax1.set_ylabel('File count', color='#76B7B2')
    ax2 = ax1.twinx()
    ax2.plot(net_keys['network'], net_keys['avg_keys'], color='#AF7AA1', marker='o', label='Avg keys')
    ax2.set_ylabel('Average keys per file', color='#AF7AA1')
    plt.title('PowerGraph Files and Metadata Density by Network')
    fig.tight_layout()
    plt.savefig(FIG_DIR / 'fig05_powergraph_network_file_key_summary.png')
    plt.close()

    # --- 3) captions file for paper ---
    captions = ANALYSIS / 'paper_figures' / 'captions.md'
    captions.write_text(
        "# Figure Captions (Paper-ready)\n\n"
        "- **Figure 1 (fig01_opf_case_distribution.png):** Distribution of OPFData samples across four network cases (14, 30, 57, and 118 bus systems).\n"
        "- **Figure 2 (fig02_objective_histogram.png):** Overall objective-value distribution across all OPFData samples.\n"
        "- **Figure 3 (fig03_objective_boxplot_by_case.png):** Per-case objective-value boxplots, showing central tendency and spread without outliers.\n"
        "- **Figure 4 (fig04_powergraph_varcount_by_type.png):** Number of unique variables found in each PowerGraph MAT file type.\n"
        "- **Figure 5 (fig05_powergraph_network_file_key_summary.png):** Number of MAT files and average metadata key count per PowerGraph network (ieee24/39/118/uk).\n"
    )

    con.close()
    print('Done: created table powergraph_variables + paper figures/tables in analysis/')


if __name__ == '__main__':
    main()
