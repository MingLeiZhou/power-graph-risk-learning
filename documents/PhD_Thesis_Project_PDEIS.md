# PhD Thesis Project (PDEIS Format)
## Cross-Topology Cascading-Failure Risk Prediction for Power Networks Using Graph Learning

**Author:** Minglei Zhou  
**Repository:** MingLeiZhou/power-graph-risk-learning  
**Date:** 2026-04-05  

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Objectives](#2-objectives)
3. [State of the Art](#3-state-of-the-art)
4. [Methodology](#4-methodology)
   - [4.1 Problem Formulation](#41-problem-formulation)
   - [4.2 Data Pipeline](#42-data-pipeline)
   - [4.3 Graph Construction](#43-graph-construction)
   - [4.4 Representation Learning](#44-representation-learning)
   - [4.5 Risk Prediction Model](#45-risk-prediction-model)
   - [4.6 Evaluation Strategy](#46-evaluation-strategy)
   - [4.7 Expected Contributions](#47-expected-contributions)
5. [Task Description](#5-task-description)
6. [Timeline](#6-timeline)
7. [References](#7-references)

---

## 1. Introduction and Motivation

Power systems are safety-critical infrastructures in which localized disturbances can propagate into cascading outages with severe social and economic consequences. In this setting, early identification of high-risk operating states is a core research and engineering challenge. The repository under study addresses this challenge by developing a machine-learning pipeline for **cascading-failure risk prediction** with an explicit focus on **cross-topology generalization**: models should remain informative when transferred to unseen network structures.

The practical motivation is straightforward: real operators and planners rarely deploy models only on data distributions that exactly match the training network. Instead, decision support is required under topology change, operating-point drift, and heterogeneous data regimes. Consequently, this project treats topology shift as a first-class concern rather than a secondary validation detail. The codebase operationalizes this via strict **Leave-One-Network-Out (LONO)** experiments, where each network is held out entirely during training and used only for testing.

From a scientific perspective, the work sits at the intersection of (i) cascading-risk modeling in power grids, (ii) graph-based representation of physical systems, and (iii) robust machine learning under distribution shift (Dobson et al., 2007; Pagani & Aiello, 2013; Gulrajani & Lopez-Paz, 2021). The repository is particularly relevant because it combines a reproducible data-engineering backbone (JSON/Parquet/DuckDB and MAT-based downstream feature extraction) with evaluation protocols aligned to out-of-domain deployment. This makes it suitable for a PhD proposal emphasizing method rigor, reproducibility, and real-world transferability.

The current mainline model family is tree-based and trained on engineered features extracted from PowerGraph matrices. While this may appear less fashionable than end-to-end deep graph models, it has two concrete advantages for doctoral work: (a) strong reproducibility and interpretability at the pipeline level, and (b) clear baselines for evaluating whether representation-learning additions produce genuine out-of-domain value. The repository also preserves archived self-supervised graph-learning branches, enabling a coherent thesis trajectory from robust baselines toward representation transfer extensions.

---

> **Figure 1. Overall research framework for cross-topology risk prediction.**
> *What it shows:* data sources, preprocessing tracks, modeling core, LONO evaluation loop, and deployment threshold calibration.
> *Suggested location:* end of Section 1.

---

## 2. Objectives

This doctoral project has five measurable objectives:

1. **Build a reproducible end-to-end data-to-model pipeline**
   Consolidate the repository's existing ingestion, validation, transformation, and training scripts into a traceable workflow that can be rerun from raw inputs to paper assets.

2. **Formulate cascading-risk prediction as dual tasks**
   Define and evaluate both classification (high-risk detection) and regression (continuous risk magnitude) targets grounded in repository labels (`y_cls_v2`, `y_reg`).

3. **Establish and justify strict cross-topology evaluation**
   Use LONO as the primary protocol and provide evidence-based argumentation for why random sample splits are insufficient for transfer claims.

4. **Investigate representation learning as a transfer enhancer**
   Evaluate whether self-supervised graph pretraining (archived GraphMAE-related pathway) can improve cross-network robustness over strong non-SSL baselines.

5. **Design deployment-oriented calibration profiles**
   Produce threshold-calibrated operating points (e.g., high-recall vs. high-precision) aligned with risk-sensitive operational use.

---

> **Table 1. Thesis objectives, measurable indicators, and acceptance criteria.**
> *Columns:* Objective ID | Target artifact | Evaluation metric(s) | Success criterion | Evidence file(s).
> *Content:* map each objective to concrete repository outputs (CSV/JSON/figures).

---

## 3. State of the Art

### 3.1 Cascading Failure and Power-System Risk Assessment

Cascading failures in power systems exhibit nontrivial propagation dynamics and network-dependent vulnerabilities, making purely local heuristics insufficient in many settings (Dobson et al., 2007; Buldyrev et al., 2010). Prior work emphasizes that both topological and electrical characteristics matter for vulnerability analysis and contingency behavior (Hines et al., 2010; Pagani & Aiello, 2013). This supports the thesis premise that predictive models should account for structural heterogeneity and be evaluated under topology transfer constraints.

### 3.2 Graph Learning for Structured System Data

Graph neural networks provide a natural abstraction for relational physical systems, with message-passing architectures such as GCN, GraphSAGE, GAT, and GIN widely used across domains (Kipf & Welling, 2017; Hamilton et al., 2017; Veličković et al., 2018; Xu et al., 2019). For power-system applications, graph formulation is attractive because buses, generators, loads, and physical links are inherently relational. However, strong graph inductive bias does not automatically solve domain shift across network scales and parameter regimes.

### 3.3 Self-Supervised Graph Representation Learning

Limited labeled fault/risk examples motivate self-supervised pretraining strategies. Existing methods include contrastive graph learning and predictive reconstruction paradigms (Veličković et al., 2019; You et al., 2020; Hu et al., 2020; Hou et al., 2022). GraphMAE-style masked reconstruction is particularly relevant because it can pretrain node/edge encoders from unlabeled structure-feature pairs. Nonetheless, whether these embeddings transfer robustly across distinct grid topologies remains an open empirical question in this project context.

### 3.4 Evaluation Under Distribution Shift

Robust performance claims require evaluation protocols aligned with deployment shift (Kohavi, 1995; Dietterich, 1998; Gulrajani & Lopez-Paz, 2021). In heavily imbalanced detection tasks, AUC alone is insufficient; PR-aware metrics and threshold-dependent F1/precision/recall analyses are essential (Fawcett, 2006; Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015). The repository's emphasis on LONO and threshold sweeps is therefore methodologically well grounded.

### 3.5 Critical Gap Motivating This Thesis

Existing literature provides strong components — graph models, SSL strategies, risk metrics — but fewer works tightly integrate:

1. **reproducible data engineering**,
2. **cross-topology split discipline**, and
3. **deployment-calibrated decision rules**

within one unified power-risk pipeline. This thesis targets that integration.

---

> **Table 2. Thematic comparison of prior approaches.**
> *Columns:* Theme | Typical method family | Label requirement | Transfer assumption | Evaluation split style | Main limitation.
> *Content:* compare physics/statistical baselines, supervised GNNs, SSL graph methods, and domain-generalization strategies.

> **Figure 2. Conceptual landscape of methods vs. generalization assumptions.**
> *What it shows:* method families on one axis and evaluation rigor on another.
> *Suggested location:* end of Section 3.

---

## 4. Methodology

### 4.1 Problem Formulation

Let $\mathcal{N}=\{n_1,\ldots,n_K\}$ denote the set of network domains (e.g., ieee24, ieee39, ieee118, uk). Each sample $i$ belongs to one domain $n(i)\in\mathcal{N}$, with feature vector $\mathbf{x}_i$, binary risk label $y_i^{cls}\in\{0,1\}$, and continuous risk target $y_i^{reg}\in\mathbb{R}_{\ge 0}$.

$$
\mathcal{D}=\{(\mathbf{x}_i,\ y_i^{cls},\ y_i^{reg},\ n(i))\}_{i=1}^{N}
$$

In the repository's v2 dataset construction, $y_i^{reg}$ is derived from downstream cascade severity (`dns_MW`), and $y_i^{cls}$ is generated by quantile-based multi-level stratification:

| Risk level | Condition |
|---|---|
| 0 | $y_i^{reg} \le q_{0.70}$ |
| 1 | $q_{0.70} < y_i^{reg} \le q_{0.90}$ |
| 2 | $q_{0.90} < y_i^{reg} \le q_{0.98}$ |
| 3 | $y_i^{reg} > q_{0.98}$ |

The binary positive label is defined as:

$$
y_i^{cls} = \mathbb{1}[\text{risk\_level}_i \ge 2]
$$

The two prediction tasks are:

**1. Classification**

$$
s_i = f_{cls}(\mathbf{x}_i) \in [0,1], \qquad \hat{y}_i = \mathbb{1}[s_i \ge \tau]
$$

**2. Regression**

$$
\hat{r}_i = f_{reg}(\mathbf{x}_i) \in \mathbb{R}_{\ge 0}
$$

where $\tau$ is a calibrated decision threshold.

Although the mainline predictor is tabular (tree ensembles), graph notation is central to data semantics and future representation-transfer extensions. A graph sample is:

$$
G = (V, E, \mathbf{X}, \mathbf{E})
$$

where $V$ are nodes, $E$ edges, $\mathbf{X}$ node attributes, and $\mathbf{E}$ edge attributes. In `power_graph_builder.py`, node types are buses/generators/loads/shunts; edge types are AC lines/transformers and element-to-bus links.

The methodological target is not only high in-domain fitting but robust cross-domain transfer:

$$
\max\ \mathbb{E}_{n \in \mathcal{N}} \left[ \text{Perf}\!\left(f;\ \mathcal{D}_{test}^{(n)}\right) \right]
$$

under train/test domain disjointness.

---

### 4.2 Data Pipeline

The repository implements a **dual-track pipeline** that jointly supports auditability and supervised downstream learning.

#### Track A: OPF JSON-centric audit and graph export

- `process_opfdata_dataset.py` orchestrates JSON scanning and conversion to Parquet.
- Optional DuckDB ingestion supports analytical queries and reproducibility.
- Optional graph export serializes graph-ready samples from Parquet.
- `ingest_to_duckdb.py` builds clean relational tables (`opf_samples`, `powergraph_files`) from processed data.
- `generate_paper_assets.py` derives paper-ready summary tables and figures from DuckDB.

This track can be summarized as:

$$
\text{JSON} \rightarrow \text{Parquet} \rightarrow \text{DuckDB} \rightarrow \text{analysis artifacts} \quad (+\text{graph export})
$$

#### Track B: MAT-to-downstream supervised dataset

- `build_downstream_dataset.py` reads PowerGraph MAT files (`Ef`, `Ef_nc`, `of_reg`), extracts statistical descriptors, and produces `downstream_full.parquet`.
- `build_dataset_v2_informative.py` upgrades to v2 with:
  - multi-level risk labeling and binary `y_cls_v2`,
  - network-level metadata features,
  - temporal-like perturbed features (`*_tshift`),
  - hard-example augmentation near boundary regions,
  - robust per-network normalization for model columns.

Output:

$$
\text{MAT} \rightarrow \text{feature engineering} \rightarrow \texttt{downstream\_v2\_informative.parquet}
$$

#### Why this pipeline is methodologically strong

| Property | Mechanism |
|---|---|
| Scalability | Parquet + DuckDB columnar access at large sample counts |
| Traceability | Deterministic scripts with intermediate CSV/JSON/Parquet artifacts |
| Reproducibility | README-defined single-command sequence from raw to paper outputs |
| Flexibility | Graph export path preserved for future SSL/GNN extension work |

---

> **Figure 3. Dual-track data pipeline (audit track + downstream modeling track).**
> *What it shows:* OPF JSON branch and PowerGraph MAT branch converging into training/evaluation artifacts.
> *Suggested location:* end of 4.2.

> **Table 3. Dataset summary by network.**
> *Columns:* Network | Samples | Positive rate (`y_cls_v2`) | Mean `y_reg` | Std `y_reg`.
> *Content source:* `analysis/training/paper_dataset_summary.csv`.

| Network | Samples | Positive rate | Mean y_reg | Std y_reg |
|---|---|---|---|---|
| ieee118 | 166,008 | 0.0404 | 0.000238 | 0.00175 |
| ieee24 | 27,802 | 0.1559 | 0.00987 | 0.03909 |
| ieee39 | 37,584 | 0.0691 | 0.00290 | 0.01553 |
| uk | 87,206 | 0.0256 | 0.00549 | 0.04028 |

---

### 4.3 Graph Construction

The graph builder (`power_graph_builder.py`) defines a typed heterogeneous-to-unified schema.

#### Node schema

| Type | Index | Raw dim | Type bits | Final dim | Source fields |
|---|---|---|---|---|---|
| Bus | 0 | 4 | [1,0,0,0] | 15 | v_min_pu, bus_type, v_min, v_max |
| Generator | 1 | 11 | [0,1,0,0] | 15 | p_max, q_max, q_min, p_min, cost0–2, … |
| Load | 2 | 2 | [0,0,1,0] | 15 | p_d, q_d |
| Shunt | 3 | 2 | [0,0,0,1] | 15 | g_sh, b_sh |

Global node indexing uses contiguous type blocks. Raw features are padded/truncated to common width (`NODE_RAW_MAX=11`), then concatenated with one-hot type bits (`NODE_TYPE_DIM=4`):

$$
\mathbf{x}_v = \left[ \mathbf{x}^{raw}_v \ \| \ \mathbf{t}^{node}_v \right] \in \mathbb{R}^{15}
$$

#### Edge schema

| Type | Index | Raw dim | Type bits | Final dim | Source |
|---|---|---|---|---|---|
| AC line | 0 | 9 | [1,0,0,0,0] | 16 | angle, r, x, b, rates |
| Transformer | 1 | 11 | [0,1,0,0,0] | 16 | angle, r, x, b, rates, tap, shift |
| Generator link | 2 | 0 | [0,0,1,0,0] | 16 | gen→bus incidence |
| Load link | 3 | 0 | [0,0,0,1,0] | 16 | load→bus incidence |
| Shunt link | 4 | 0 | [0,0,0,0,1] | 16 | shunt→bus incidence |

Edge raw feature block is padded to `EDGE_RAW_MAX=11` and concatenated with one-hot edge type (`EDGE_TYPE_DIM=5`):

$$
\mathbf{e}_{uv} = \left[ \mathbf{e}^{raw}_{uv} \ \| \ \mathbf{t}^{edge}_{uv} \right] \in \mathbb{R}^{16}
$$

#### Labels and optional solution tensors

- Primary scalar label: OPF objective (from metadata) for each graph sample.
- Optional `sol_node` (shape $N \times 2$): bus voltage angle/magnitude; generator real/reactive power.
- Optional `sol_edge` (shape $E \times 4$): active/reactive power flows on AC lines and transformers.
- Schema metadata includes counts by node/edge types and schema version tag.

#### Normalization and validation design

A notable implementation detail is **type-preserving normalization**:

- only raw feature blocks are scaled;
- one-hot type indicators remain unchanged.

This avoids semantic distortion of categorical identity bits. Two modes are implemented:

1. **dataset-level normalization** with fitted shared normalizers (recommended),
2. **graph-level normalization** fallback when fitted statistics are unavailable.

---

> **Figure 4. Graph schema of one sample (typed nodes/edges and feature blocks).**
> *What it shows:* node categories, edge categories, feature concatenation layout, label tensors.
> *Suggested location:* inside 4.3 after schema description.

---

### 4.4 Representation Learning

#### Motivation

The class distribution is highly skewed and heterogeneous across networks (positive rate ranging from ~2.6% for uk to ~15.6% for ieee24). Under such conditions, purely supervised training may overfit domain-specific shortcuts. Representation learning can mitigate this by extracting transferable structural patterns from unlabeled or weakly labeled graph data.

#### Repository context

The mainline currently prioritizes robust non-SSL baselines, but archived scripts document an SSL branch (`pretrain_ssl.py`, GraphMAE-related experiments, and fused/domain-calibration variants). This supports a valid PhD methodology: start from strong reproducible baselines, then test whether SSL pretraining improves cross-topology transfer under the same LONO discipline.

#### Proposed SSL formalization (GraphMAE-style)

Given graph $G$, mask a subset of node attributes $M \subset V$, encode corrupted input, and reconstruct original masked attributes:

$$
\min_{\theta, \phi}\ \mathcal{L}_{SSL}
= \frac{1}{|M|} \sum_{v \in M}
\left\|
\mathbf{x}_v - g_\phi\!\left( h_\theta(G, \tilde{\mathbf{X}}) \right)_v
\right\|_2^2
$$

where $h_\theta$ is a graph encoder and $g_\phi$ a decoder. Pretrained encoder embeddings are then transferred to downstream classifiers/regressors.

#### Transfer to downstream risk prediction

Two practical transfer pathways are aligned with this repository:

1. **Feature fusion:** concatenate SSL embeddings with engineered features before downstream model training.
2. **Representation replacement:** use graph embeddings as primary model inputs for downstream risk heads.

#### Methodological caution from repository evidence

Archived comparison outputs indicate that some fused SSL variants did not consistently improve threshold-dependent F1 under strict LONO. This is scientifically important: the thesis should treat SSL as a hypothesis to be tested under domain-shift-aware metrics, not as an assumed improvement.

---

> **Figure 5. Self-supervised pretraining and downstream transfer workflow.**
> *What it shows:* masked pretraining stage, encoder checkpoint, transfer stage, LONO evaluation loop.
> *Suggested location:* end of 4.4.

---

### 4.5 Risk Prediction Model

The current mainline (`train_compare_v2_informative.py`) uses two complementary predictors trained per LONO fold.

#### Classification model

- Model: `RandomForestClassifier` (scikit-learn)
- Inputs: engineered `ef_*`, `efnc_*`, `*_tshift`, and selected network meta features.
- Imbalance handling: sample-weight multiplier for positives ($w_+ = 10$).
- Inference: probability score $s_i$, then thresholding $\hat{y}_i = \mathbb{1}[s_i \ge \tau]$.

#### Regression model

- Model: `ExtraTreesRegressor` (scikit-learn)
- Target transform:

$$
z_i = \log(1 + y_i^{reg}), \qquad
\hat{r}_i = \max\!\left\{0,\ \exp(\hat{z}_i) - 1\right\}
$$

This stabilizes heavy-tailed targets while maintaining non-negativity at inference.

#### Threshold calibration

`final_v2_threshold_sweep_fast.py` evaluates candidate thresholds and chooses by utility:

$$
U(\tau) = 0.5 \cdot F1(\tau) + 0.2 \cdot AP + 0.2 \cdot AUC + 0.1 \cdot Precision(\tau)
$$

where AUC and AP are score-level ranking metrics; F1 and Precision are threshold-dependent.

Repository output indicates best balanced threshold $\tau = 0.12$.

#### Deployment profiles

The script exports profile-specific metrics across three operating points:

| Profile | Threshold | Priority |
|---|---|---|
| High recall | 0.12 | Maximize alarm coverage |
| Balanced | 0.12 | Weighted trade-off (best utility) |
| High precision | 0.25 | Minimize false alarms |

---

> **Table 5. Feature families used in the v2 informative model.**
> *Columns:* Feature group | Example columns | Construction script | Purpose.

> **Figure 6. Training and threshold-calibration framework.**
> *What it shows:* per-fold training, score generation, sweep, profile selection, final report outputs.
> *Suggested location:* end of 4.5.

---

### 4.6 Evaluation Strategy

#### Strict Leave-One-Network-Out (LONO)

For each network $n \in \mathcal{N}$:

$$
\mathcal{D}_{train}^{(n)} = \{i \mid n(i) \neq n\}, \qquad
\mathcal{D}_{test}^{(n)} = \{i \mid n(i) = n\}
$$

Train on $\mathcal{D}_{train}^{(n)}$, evaluate on $\mathcal{D}_{test}^{(n)}$, then aggregate over all held-out networks. This directly measures cross-topology transfer because the test network is completely unseen during fitting.

#### Why random split is insufficient

A random sample split can mix samples from the same topology into train and test, allowing models to exploit topology-specific signatures and yielding optimistic estimates. For transfer claims, this is a methodological mismatch. LONO enforces domain-disjoint evaluation and therefore provides stronger evidence for deployment beyond seen networks.

---

> **Figure 7. Random split vs LONO conceptual comparison.**
> *What it shows:* data partition diagrams illustrating leakage in random split vs. domain-disjoint LONO.
> *Suggested location:* inside 4.6.

---

#### Metric suite

**Classification:**

$$
Precision = \frac{TP}{TP+FP}, \qquad
Recall = \frac{TP}{TP+FN}, \qquad
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

plus AUC and Average Precision (AP) computed from score distributions.

**Regression:**

$$
MAE = \frac{1}{N}\sum_i |y_i - \hat{y}_i|, \qquad
RMSE = \sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2}
$$

with $R^2$ as explained-variance indicator.

> **Table 6. Evaluation metrics and operational interpretation.**
> *Columns:* Metric | Type | Formula | Operational interpretation.

#### Current result profile (repository evidence)

From `analysis/training/v2_final_summary.json` and `analysis/training/paper_main_results_final.csv`:

| Metric | Value |
|---|---|
| Best threshold $\tau$ | 0.12 |
| Mean AUC | 0.7444 |
| Mean AP | 0.1405 |
| Mean F1 | 0.1662 |
| Mean Precision | 0.1331 |
| Mean Recall | 0.4534 |
| Mean MAE | 0.0130 |
| Mean RMSE | 0.0313 |
| Mean $R^2$ | −41.16 |

These values reflect the difficulty of strict cross-network transfer — particularly the strongly negative $R^2$ — and motivate continued methodological work on representation robustness and calibration.

---

### 4.7 Expected Contributions

1. **Methodological contribution**
   A unified cross-topology risk-prediction framework combining dual-task modeling (classification + regression), threshold calibration, and transfer-aware validation.

2. **Data engineering contribution**
   A reproducible dual-track data stack linking OPF JSON auditability and MAT-derived downstream predictive datasets, with explicit intermediate artifacts and scripts.

3. **Evaluation contribution**
   Institutionalization of strict LONO as the principal benchmark for power-network generalization claims, with per-network reporting and deployment profiles.

4. **Representation-learning contribution**
   A principled assessment of SSL/GraphMAE transfer potential under strict domain shift, including negative or neutral findings where applicable.

5. **Reproducibility contribution**
   Paper-ready outputs (tables, figures, score exports, summaries) generated directly from versioned scripts, enabling traceable scientific reporting.

---

## 5. Task Description

The doctoral work is organized into five phases.

### Phase 1 — Data validation and reproducible infrastructure

- Validate OPFData and PowerGraph sources.
- Standardize ingestion and profiling outputs.
- **Deliverables:** validation reports, DuckDB tables, dataset summary tables.

### Phase 2 — Baseline cross-topology predictive core

- Build v2 informative dataset.
- Train and evaluate LONO classification/regression baselines.
- **Deliverables:** `v2_lono_cls.csv`, `v2_lono_reg.csv`, threshold sweep outputs, final summary JSON.

### Phase 3 — Representation-learning extension

- Implement/clean archived SSL pretraining pathways.
- Conduct controlled transfer experiments with identical LONO protocol.
- **Deliverables:** comparative result tables and ablation logs.

### Phase 4 — Robustness and decision-calibration studies

- Analyze threshold sensitivity and deployment profiles.
- Perform per-network failure-mode analysis.
- **Deliverables:** calibrated operating profiles, error analysis section, expanded figures.

### Phase 5 — Thesis integration and dissemination

- Integrate methodology, experiments, and discussion into PDEIS thesis format.
- Prepare manuscripts/presentations based on reproducible artifacts.
- **Deliverables:** thesis chapters, paper-ready tables/figures, publication drafts.

---

> **Table 7. Work packages, deliverables, and acceptance criteria.**
> *Columns:* Phase | Core tasks | Primary deliverables | Validation criteria | Planned publication output.

---

## 6. Timeline

### 2026–2029 Roadmap

| Year | Quarter | Focus | Key Milestones | Outputs |
|---|---|---|---|---|
| 2026 | Q2–Q3 | Pipeline consolidation | End-to-end reproducible run from raw to paper assets | Validated data reports, baseline scripts stabilized |
| 2026 | Q4 | Baseline LONO completion | Final v2 baseline and calibrated thresholds | Main baseline result tables/figures |
| 2027 | Q1–Q2 | SSL/representation integration | Pretraining-transfer pipeline operational | SSL comparative experiments v1 |
| 2027 | Q3–Q4 | Controlled ablations | Feature, threshold, and split-ablation studies | Ablation report + conference submission draft |
| 2028 | Q1–Q2 | Robustness deepening | Domain-shift diagnostics and per-network error taxonomy | Extended methodological chapter |
| 2028 | Q3–Q4 | Publication consolidation | Journal-level experimental package | Manuscript submissions |
| 2029 | Q1–Q2 | Thesis synthesis | Full PDEIS draft completed | Complete thesis manuscript |
| 2029 | Q3–Q4 | Finalization and defense preparation | Revisions, defense materials | Final thesis and defense package |

---

> **Figure 8. Gantt-style timeline (2026–2029).**
> *What it shows:* phases, overlaps, and milestone markers by quarter.
> *Suggested location:* after timeline table.

---

## 7. References

Amin, M. (2005). Energy infrastructure defense systems. *Proceedings of the IEEE, 93*(5), 861–875.

Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Buldyrev, S. V., Parshani, R., Paul, G., Stanley, H. E., & Havlin, S. (2010). Catastrophic cascade of failures in interdependent networks. *Nature, 464*(7291), 1025–1028.

Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233–240).

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923.

Dobson, I., Carreras, B. A., Lynch, V. E., & Newman, D. E. (2007). Complex systems analysis of series of blackouts: Cascading failure, critical points, and self-organization. *Chaos, 17*(2), 026103.

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.

Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning, 63*(1), 3–42.

Gulrajani, I., & Lopez-Paz, D. (2021). In search of lost domain generalization. In *International Conference on Learning Representations (ICLR)*.

Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. In *Advances in Neural Information Processing Systems* (pp. 1024–1034).

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.

Hines, P., Blumsack, S., Cotilla-Sanchez, E., & Barrows, C. (2010). The topological and electrical structure of power grids. In *2010 43rd Hawaii International Conference on System Sciences* (pp. 1–10). IEEE.

Hou, Z., Liu, X., Cen, Y., Dong, Y., Yang, H., Wang, C., Tang, J., & Zhang, J. (2022). GraphMAE: Self-supervised masked graph autoencoders. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 594–604).

Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2020). Strategies for pre-training graph neural networks. In *International Conference on Learning Representations (ICLR)*.

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations (ICLR)*.

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In *Proceedings of the 14th International Joint Conference on Artificial Intelligence* (pp. 1137–1143).

Pagani, G. A., & Aiello, M. (2013). The power grid as a complex network: A survey. *Physica A: Statistical Mechanics and its Applications, 392*(11), 2688–2700.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Raasveldt, M., & Mühleisen, H. (2019). DuckDB: An embeddable analytical database. In *CIDR*.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In *International Conference on Learning Representations (ICLR)*.

Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019). Deep graph infomax. In *International Conference on Learning Representations (ICLR)*.

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2021). A comprehensive survey on graph neural networks. *IEEE Transactions on Neural Networks and Learning Systems, 32*(1), 4–24.

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? In *International Conference on Learning Representations (ICLR)*.

You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020). Graph contrastive learning with augmentations. In *Advances in Neural Information Processing Systems, 33*, 5812–5823.

Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., & Sun, M. (2020). Graph neural networks: A review of methods and applications. *AI Open, 1*, 57–81.

---

## Reference Integrity Checklist

- ✅ **References verified (high confidence):** core ML, GNN, SSL, evaluation metrics, DuckDB, and foundational cascade/network-science entries listed above.
- ⚠️ **References needing manual verification:** minor bibliographic formatting details (page ranges, conference proceedings style, and accent rendering in names) should be checked before final submission.
- ➖ **Missing citation placeholders:** none required for core methodological claims in this draft.
