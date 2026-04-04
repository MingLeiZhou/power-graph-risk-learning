# power-graph-risk-learning

Cross-topology cascading-failure **early warning** and **risk assessment** for power networks.

This repo is trimmed to a clean, reproducible mainline:
- data prep
- strict LONO evaluation
- threshold calibration
- paper-ready outputs

Historical experiments are archived under `archive/`.

---

## Repository layout (current)

```text
scripts/
  core/   # main training/evaluation pipeline (use these first)
  data/   # data build/validation utilities
archive/
  scripts_experiments/  # old experimental branches
  docs_experiments/     # old reports/notes
analysis/training/      # final metrics/tables/notebooks
documents/              # proposal/template docs
```

---

## Final mainline (recommended)

Mainline = **v2 informative dataset + threshold calibration**.

Primary references:
- `analysis/training/paper_ready_sections.executed.ipynb`
- `analysis/training/paper_main_results_final.csv`
- `archive/docs_experiments/FINAL_CONVERGED_MAINLINE.md`

---

## Minimal workflow

### 1) Validate data
```bash
python scripts/data/validate_data.py
```

### 2) Build/refresh downstream dataset (v2)
```bash
python scripts/data/build_dataset_v2_informative.py
```

### 3) Train/evaluate with strict LONO
```bash
python scripts/core/train_compare_v2_informative.py
```

### 4) Calibrate decision threshold
```bash
python scripts/core/final_v2_threshold_sweep_fast.py
```

### 5) Export ROC/PR sample-level scores
```bash
python scripts/core/export_v2_lono_scores.py
```

### 6) Generate paper assets
```bash
python scripts/core/generate_paper_assets.py
```

---

## Core scripts

### `scripts/core/`
- `train_compare_v2_informative.py`
- `final_v2_threshold_sweep_fast.py`
- `export_v2_lono_scores.py`
- `generate_paper_assets.py`

### `scripts/data/`
- `validate_data.py`
- `build_dataset_v2_informative.py`
- `build_downstream_dataset.py`
- `ingest_to_duckdb.py`
- `process_all_powergraph_mat.py`
- `run_analysis.py`
- `build_augmented_dataset.py`

---

## Core outputs

- Dataset:
  - `data/processed/downstream/downstream_v2_informative.parquet`
- Main training/evaluation:
  - `analysis/training/v2_lono_cls.csv`
  - `analysis/training/v2_lono_reg.csv`
  - `analysis/training/v2_final_summary.json`
- Paper-ready:
  - `analysis/training/paper_main_results_final.csv`
  - `analysis/training/paper_roc_curve.csv`
  - `analysis/training/paper_pr_curve.csv`

---

## Notes

- Python env: `~/miniforge3/bin/python`
- macOS torch/OpenMP workaround:
  - `KMP_DUPLICATE_LIB_OK=TRUE`
- Billing-sensitive work: prefer local/non-premium providers when possible.
