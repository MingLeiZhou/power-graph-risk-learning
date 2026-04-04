# Preprocessing Result Report

Date: 2026-04-03

## 1) Conda runtime used
- Conda root: `~/miniforge3`
- Python runtime used for all processing: `~/miniforge3/bin/python` (base env)
- Installed/verified packages: `duckdb`, `pandas`, `matplotlib`, `numpy`, `scipy`, `h5py`, `nbconvert`, `nbformat`

## 2) Folder cleanup performed
Removed redundant/unneeded items:
- `venv/` (project-local virtualenv removed; switched to conda/miniforge runtime)
- `__pycache__/`
- old duplicate PowerGraph metadata JSONs (generic names) under `data/processed/powergraph_graphs/`
- obsolete script: `scripts/process_powergraph.py`

Kept core scripts:
- `scripts/process_all_powergraph_mat.py`
- `scripts/ingest_to_duckdb.py`
- `scripts/validate_data.py`
- `scripts/run_analysis.py`

## 3) Code checks and fixes
### Fixed items
1. **PowerGraph MAT reading**
   - Added robust loader behavior using `scipy.io.loadmat` with `h5py` fallback for MATLAB v7.3 files.
2. **PowerGraph metadata uniqueness**
   - Metadata filenames now include relative path pattern (`__raw__`) to avoid overwrite collisions.
3. **DuckDB ingestion consistency**
   - Rebuilds tables cleanly (`DROP TABLE IF EXISTS` + recreate), preventing duplicate accumulation.
   - Ingests only the 32 unique per-MAT metadata files into `powergraph_files`.
4. **OPFData feature extraction correctness**
   - Node/edge counts computed as sum of typed groups (`bus/generator/load/shunt`, `ac_line/transformer/...`) instead of top-level key count.

## 4) Final preprocessing outputs
### DuckDB
Database: `data/processed/opfdata.duckdb`

Tables:
- `opf_samples`: **60000 rows**
- `powergraph_files`: **32 rows**

OPF case distribution (rows):
- `dataset_release_1__pglib_opf_case14_ieee_0_extracted`: 15000
- `dataset_release_1__pglib_opf_case30_ieee_0_extracted`: 15000
- `dataset_release_1__pglib_opf_case57_ieee_0_extracted`: 15000
- `dataset_release_1__pglib_opf_case118_ieee_0_extracted`: 15000

### PowerGraph metadata files
- Directory: `data/processed/powergraph_graphs/`
- Total JSON files: **32** (one per `.mat` file)

### Validation reports
- `data/processed/reports/preprocessing_report.txt`
- `data/processed/reports/opf_cases.csv`
- `data/processed/reports/powergraph_files.csv`

## 5) Notebook report for paper
Created and executed notebook:
- `analysis/powergraph_analysis.ipynb` (executed with outputs embedded)

Notebook includes:
- Table 1: OPF sample counts by case (`opf_samples`)
- Table 2: PowerGraph metadata listing (`powergraph_files`)
- Figure: Objective histogram
- Figure: PowerGraph variable-count bar chart
- Exported paper-ready tables/figures in `analysis/`

## 6) Artifacts in `analysis/`
- `powergraph_analysis.ipynb` (executed)
- `tbl_opf_case_distribution.csv`
- `tbl_opf_summary_stats.csv`
- `tbl_powergraph_files.csv`
- `fig_objective_hist.png`
- `fig_powergraph_nkeys_bar.png`
- `preprocessing_result_report.md`

---
Status: ✅ Completed
