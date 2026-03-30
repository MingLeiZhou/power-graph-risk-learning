"""
Evaluation metrics for risk assessment.

Provides regression and classification metrics used to evaluate the
quality of risk scores produced by :class:`RiskAssessor`, as well as
grid-level vulnerability indices.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error between predictions and targets.

    Parameters
    ----------
    y_true:
        Ground-truth risk scores, shape ``(N,)``.
    y_pred:
        Predicted risk scores, shape ``(N,)``.

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error.

    Parameters
    ----------
    y_true, y_pred:
        Arrays of shape ``(N,)``.

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y_true, y_pred:
        Arrays of shape ``(N,)``.

    Returns
    -------
    float
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def precision_recall_f1(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 for binary risk classification.

    Parameters
    ----------
    y_true_binary:
        Ground-truth binary labels (0 or 1), shape ``(N,)``.
    y_pred_binary:
        Predicted binary labels (0 or 1), shape ``(N,)``.

    Returns
    -------
    tuple of (precision, recall, f1)
    """
    y_true = np.asarray(y_true_binary, dtype=int)
    y_pred = np.asarray(y_pred_binary, dtype=int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def risk_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve computed with the trapezoidal rule.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels, shape ``(N,)``.
    y_score:
        Continuous risk scores, shape ``(N,)``.

    Returns
    -------
    float
        AUC in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    sort_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sort_idx]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    total_pos = tps[-1] if tps.size > 0 else 1
    total_neg = fps[-1] if fps.size > 0 else 1
    if total_pos == 0 or total_neg == 0:
        return 0.5
    tpr = tps / (total_pos + 1e-12)
    fpr = fps / (total_neg + 1e-12)
    # Prepend (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(tpr, fpr))


def normalise_risk_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise risk scores to ``[0, 1]``.

    Parameters
    ----------
    scores:
        Raw scores, shape ``(N,)``.

    Returns
    -------
    np.ndarray, shape ``(N,)``
    """
    scores = np.asarray(scores, dtype=np.float64)
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def compute_vulnerability_index(
    bus_risk_scores: Dict[int, float],
    centrality_scores: Optional[Dict[int, float]] = None,
    centrality_weight: float = 0.3,
) -> Dict[int, float]:
    """Compute a composite vulnerability index for each bus.

    The index combines the predicted risk score with optional
    betweenness-centrality to capture both electrical and topological
    vulnerability.

    Vulnerability(i) = (1 - α) × risk(i) + α × centrality(i)

    where α = ``centrality_weight``.

    Parameters
    ----------
    bus_risk_scores:
        Mapping from bus_id to risk score in ``[0, 1]``.
    centrality_scores:
        Mapping from bus_id to betweenness-centrality in ``[0, 1]``.
        If ``None``, only the risk score is used.
    centrality_weight:
        Weight given to centrality (0 = pure risk, 1 = pure centrality).

    Returns
    -------
    dict mapping bus_id → vulnerability index in ``[0, 1]``.
    """
    if not 0.0 <= centrality_weight <= 1.0:
        raise ValueError("centrality_weight must be in [0, 1].")

    vuln: Dict[int, float] = {}
    for bus_id, risk in bus_risk_scores.items():
        if centrality_scores is not None and bus_id in centrality_scores:
            cent = centrality_scores[bus_id]
            v = (1.0 - centrality_weight) * risk + centrality_weight * cent
        else:
            v = risk
        vuln[bus_id] = float(np.clip(v, 0.0, 1.0))
    return vuln
