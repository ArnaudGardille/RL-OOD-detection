"""Detection-quality metrics.

- AUROC / AUPR over a set of streams labelled OOD vs in-distribution.
- Detection delay relative to a known change-point (online setting).
- Average run length (ARL0 on nominal = how rarely it false-alarms; ARL1 on OOD = how fast).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(scores, labels) -> float:
    return float(roc_auc_score(labels, scores))


def aupr(scores, labels) -> float:
    return float(average_precision_score(labels, scores))


def detection_delay(alarm_at: int, change_index: int) -> float:
    """Steps from the change-point to the first alarm.

    NaN if never alarmed; an alarm at/before the change-point counts as delay 0.
    """
    if alarm_at < 0:
        return float("nan")
    return float(max(0, alarm_at - change_index))


def mean_detection_delay(alarm_ats, change_index: int) -> float:
    delays = [detection_delay(a, change_index) for a in alarm_ats]
    delays = [d for d in delays if not np.isnan(d)]
    return float(np.mean(delays)) if delays else float("nan")


def detection_rate(alarm_ats) -> float:
    return float(np.mean(np.asarray(alarm_ats) >= 0))


def average_run_length(alarm_ats, horizon: int) -> float:
    """Mean steps to alarm, censoring non-alarms at ``horizon``."""
    a = np.asarray(alarm_ats, dtype=float)
    a = np.where(a < 0, horizon, a)
    return float(a.mean())
