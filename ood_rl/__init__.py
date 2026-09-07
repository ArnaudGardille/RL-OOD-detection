"""ood_rl — a small framework for online OOD environment detection and adaptation in RL.

Detection-first: gymnax-based parametric environments, a vmap-ready transition-collection
rollout (stationary + change-point), a KNN dynamics world-model, and a family of detectors
(conformal-martingale + CUSUM/KS baselines) compared via a parallel OOD-grid sweep and
detection metrics. Adaptation comes in later phases.
"""

from .envs.registry import make_env
from .envs.params import DEFAULTS, build_params, make_ood_grid
from .models.base import WorldModel
from .models.knn import KNNDynamics
from .detectors.base import Detector
from .detectors.conformal_martingale import ConformalMartingaleDetector
from .detectors.cusum import CUSUMDetector
from .detectors.ks_window import KSWindowDetector
from .detectors.reward_drop import RewardDropDetector
from .experiment.config import Phase0Config, make_key
from .experiment.rollout import (
    collect_transitions,
    collect_nonstationary,
    fit_world_model,
    calibrate_detector,
    fit_reference_detector,
    detect,
)
from .experiment import metrics, sweep

__all__ = [
    "make_env",
    "DEFAULTS",
    "build_params",
    "make_ood_grid",
    "WorldModel",
    "KNNDynamics",
    "Detector",
    "ConformalMartingaleDetector",
    "CUSUMDetector",
    "KSWindowDetector",
    "RewardDropDetector",
    "Phase0Config",
    "make_key",
    "collect_transitions",
    "collect_nonstationary",
    "fit_world_model",
    "calibrate_detector",
    "fit_reference_detector",
    "detect",
    "metrics",
    "sweep",
]
