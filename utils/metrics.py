"""Evaluation metrics for prognostic CKD models.

Includes standard classification metrics (AUROC, AUPRC, F1 at a threshold,
accuracy) and survival-oriented metrics (Harrell's C-index, time-dependent
AUROC at a chosen horizon bin, and a simple Brier-style calibration error).

scikit-learn is used where available; lightweight fallbacks are provided so the
core training loop never hard-fails on a metric.
"""

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score,
    )
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    if _HAS_SK:
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            out["auroc"] = float("nan")
        try:
            out["auprc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            out["auprc"] = float("nan")
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["acc"] = float(accuracy_score(y_true, y_pred))
    else:
        out["acc"] = float((y_pred == y_true).mean())
    return out


def concordance_index(event_times, predicted_risk, event_observed):
    """Harrell's C-index. Higher predicted_risk should mean earlier event.

    A pure-numpy O(n^2) implementation — fine for validation set sizes; swap for
    ``lifelines.utils.concordance_index`` on very large cohorts.
    """
    t = np.asarray(event_times).ravel()
    r = np.asarray(predicted_risk).ravel()
    e = np.asarray(event_observed).ravel().astype(bool)

    num = den = 0.0
    n = len(t)
    for i in range(n):
        if not e[i]:
            continue
        for j in range(n):
            if t[j] > t[i]:
                den += 1
                if r[i] > r[j]:
                    num += 1
                elif r[i] == r[j]:
                    num += 0.5
    return float(num / den) if den > 0 else float("nan")


def time_dependent_auroc(cif_at_horizon, label_at_horizon, eligible_mask=None):
    """AUROC of the cumulative incidence prediction at a fixed horizon bin.

    cif_at_horizon : (N,) predicted CIF = 1 - S(t_horizon)
    label_at_horizon : (N,) 1 if event by horizon, else 0
    eligible_mask : optional (N,) bool, drop subjects censored before horizon
    """
    cif = np.asarray(cif_at_horizon).ravel()
    lab = np.asarray(label_at_horizon).ravel()
    if eligible_mask is not None:
        m = np.asarray(eligible_mask).ravel().astype(bool)
        cif, lab = cif[m], lab[m]
    if not _HAS_SK or len(np.unique(lab)) < 2:
        return float("nan")
    return float(roc_auc_score(lab, cif))


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    return float(np.mean((y_prob - y_true) ** 2))
