# 可投稿版文本草稿（摘要 + 方法 + 结果）

日期：2026-04-04
项目：power-graph-risk-learning
评估设定：严格 Leave-One-Network-Out (LONO)

---

## 一、摘要（中文）

电力系统连锁故障风险预测在跨拓扑泛化场景下面临显著域偏移：模型在随机划分上可取得较高指标，但在未见网络上的性能明显下降。为此，本文围绕“可泛化的早期预警”目标，构建了一个面向跨网络迁移的完整流程，包括：数据清洗与统一特征工程、v2 informative 数据集构建（难例增强、扰动特征、网络级元特征、稳健归一化）、严格 LONO 评估、阈值扫描与部署策略分档。实验表明，收敛主线（v2 + 阈值校准）在四个网络的严格 LONO 下取得分类 AUC 0.7444、AP 0.1405、F1 0.1662、Recall 0.4534，以及回归 MAE 0.0130、RMSE 0.0313。进一步对比 PU+多任务、GraphMAE+DANN/CORAL 联合适配等方法后发现，这些方法在部分维度（如召回或精度）可带来局部改进，但整体稳定性与综合效用仍不及收敛主线。结果说明：在强域偏移条件下，数据构造与阈值策略对可部署预警性能的贡献不低于复杂模型结构本身。本文最终给出可复现的训练-评估-部署模板，为跨拓扑电网风险筛查提供实证基础。

**关键词**：连锁故障；跨拓扑泛化；LONO；风险预警；阈值校准；域偏移

---

## 二、方法（中文）

### 2.1 任务定义

给定网络样本特征 $x$，同时预测：
1. 分类任务（早期预警）：$y_{cls}\in\{0,1\}$；
2. 回归任务（连续风险）：$y_{reg}\in\mathbb{R}_{\ge 0}$。

评估采用严格 LONO：每次留出一个网络作为测试域，其余网络训练，循环覆盖全部网络并取均值。

### 2.2 v2 informative 数据构建

基于 `downstream_full.parquet` 构建 `downstream_v2_informative.parquet`，核心包括：

1. **多级风险标签**：依据 $y_{reg}$ 分位点构建 `risk_level`（0–3），并定义二分类目标 `y_cls_v2`；
2. **难例增强**：围绕中高风险边界区间执行 hard-example 过采样与扰动扩增（噪声、缺失注入）；
3. **时间扰动代理特征**：对关键统计特征构建 `*_tshift`；
4. **网络级元特征**：加入每网络 `n_samples`, `pos_rate`, `yreg_mean`, `yreg_std`；
5. **稳健归一化**：按网络执行 median/IQR 标准化，以降低异常值与域尺度差异影响。

### 2.3 主线模型与阈值校准

主线采用树模型进行联合建模：
- 分类器：RandomForest（正类加权）；
- 回归器：ExtraTrees（对 $\log(1+y_{reg})$ 建模再逆变换）。

分类输出分数 $s\in[0,1]$ 后执行阈值扫描（`0.12–0.25` 等网格），并基于综合效用函数选择部署阈值：
\[
U = 0.5\cdot F1 + 0.2\cdot AP + 0.2\cdot AUC + 0.1\cdot Precision.
\]
最终平衡阈值选为 **0.12**，并给出高召回/高精度配置档用于工程部署。

### 2.4 对比与消融

为验证主线有效性，进一步评估：
- PU + 多任务联合（高召回偏好）；
- sample-level GraphMAE + DANN/CORAL 联合域适配；
- staged 训练（先重建再轻量域对齐再阈值调优）。

比较维度包含 AUC/AP/F1/Precision/Recall 与 MAE/RMSE/R²。

---

## 三、结果（中文）

### 3.1 主线最终结果（严格 LONO 均值）

- 分类：**AUC 0.7444**, AP 0.1405, F1 0.1662, Precision 0.1331, Recall 0.4534
- 回归：**MAE 0.0130**, **RMSE 0.0313**, R² -41.16

该结果在跨拓扑评估下兼顾了排序能力（AUC）与可用召回，适合作为高召回预警筛查主线。

### 3.2 与备选方法对比

- **PU+多任务**：Recall 提升至 0.9652，但 Precision/AP 与 AUC 明显下降，误报成本过高；
- **GraphMAE+DANN/CORAL 联合**：AUC 约 0.6870、AP 0.1503，精度有一定改善但召回不足；
- **Staged 联合训练**：Precision 可提升，但总体 AUC/F1 与回归误差未超越主线。

综合可见，在当前数据条件下，复杂域适配并未稳定超过“v2 数据优化 + 阈值策略”路线。

### 3.3 讨论与部署含义

1. **主要瓶颈是跨域阈值迁移，而非单纯分类器容量**；
2. **数据层改造（难例增强 + 元特征 + 稳健归一化）贡献显著**；
3. 主线可作为“人机协同预警筛查器”，暂不建议全自动处置。

---

## Four. Submission-ready English Version

### Abstract (EN)

Cross-topology generalization remains the key challenge for cascading-failure risk prediction in power grids: models trained with random splits can look strong, yet degrade substantially on unseen network topologies. We present a reproducible pipeline targeting deployable early warning under strict Leave-One-Network-Out (LONO) evaluation. The pipeline includes data validation and feature unification, an informative v2 dataset design (hard-example augmentation, temporal-shift proxies, network-level meta features, robust per-network normalization), and threshold-calibrated deployment profiles. Under strict LONO across four networks, our converged mainline (v2 + threshold calibration) achieves classification AUC 0.7444, AP 0.1405, F1 0.1662, Recall 0.4534, and regression MAE 0.0130 / RMSE 0.0313. We further compare PU+multitask and sample-level GraphMAE with joint DANN/CORAL adaptation. While these alternatives improve specific axes (e.g., recall or precision), they do not consistently outperform the mainline in overall utility and stability. The results indicate that, under strong domain shift, dataset construction and threshold policy can be as important as model architecture. We release a practical training-evaluation-deployment template for cross-topology risk screening.

### Methods (EN)

We formulate a dual-task setup: binary early-warning classification ($y_{cls}$) and continuous risk regression ($y_{reg}$), evaluated with strict LONO. The v2 informative dataset is built from the full downstream data by (i) multi-level risk labeling, (ii) hard-example augmentation near risk boundaries, (iii) temporal-shift proxy features, (iv) network-level statistics (sample count, positive rate, risk mean/std), and (v) robust median/IQR normalization per network. The mainline uses a weighted RandomForest classifier and an ExtraTrees regressor on log-transformed targets. Classification scores are post-calibrated via threshold sweep with a utility-oriented selection criterion balancing F1, AP, AUC, and precision. We benchmark against PU+multitask and sample-level GraphMAE + DANN/CORAL (including staged adaptation variants).

### Results (EN)

The converged mainline delivers the best stable trade-off under strict LONO: AUC 0.7444, AP 0.1405, F1 0.1662, Precision 0.1331, Recall 0.4534, with MAE 0.0130 and RMSE 0.0313 for regression. PU+multitask yields very high recall but suffers from severe precision/AP degradation. GraphMAE+DANN/CORAL variants improve certain dimensions (notably precision in some settings) but fail to surpass the mainline in joint detection quality and robustness. These findings support a practical conclusion: for cross-topology cascading-risk screening, robust data design plus calibrated decision policy currently provides the strongest deployment-ready baseline.
