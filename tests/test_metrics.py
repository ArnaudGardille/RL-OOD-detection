import numpy as np

from ood_rl.experiment import metrics


def test_auroc_separable_and_chance():
    rng = np.random.default_rng(0)
    pos = rng.normal(5, 1, 200)
    neg = rng.normal(0, 1, 200)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(200), np.zeros(200)])
    assert metrics.auroc(scores, labels) > 0.99
    # identical distributions -> chance
    s2 = np.concatenate([rng.normal(0, 1, 200), rng.normal(0, 1, 200)])
    assert 0.4 < metrics.auroc(s2, labels) < 0.6


def test_detection_delay_and_arl():
    assert metrics.detection_delay(120, 100) == 20.0
    assert metrics.detection_delay(80, 100) == 0.0       # alarm before change -> 0
    assert np.isnan(metrics.detection_delay(-1, 100))    # never alarmed
    assert metrics.mean_detection_delay([120, 110, -1], 100) == 15.0
    assert metrics.detection_rate([5, -1, 7, -1]) == 0.5
    # ARL: non-alarms censored at horizon
    assert metrics.average_run_length([10, -1], 100) == 55.0
