# ood_rl — online OOD detection & adaptation for RL

An experimental framework to study **online out-of-distribution detection** (is the
environment's dynamics drifting?) and, in later phases, **online OOD adaptation** (recover
when it does). The compute-heavy core (environments, rollouts, parameter sweeps) is
JAX-based via [gymnax](https://github.com/RobertTLange/gymnax); the detection/metrics layer
is framework-agnostic (numpy / scikit-learn / scipy).

The reference detection method comes from the original research in this repo: learn a
dynamics model, treat its prediction errors as nonconformity scores, and run a **conformal
test-martingale** change detector. We benchmark against parametric physics shifts
(gravity, mass, length, …) inspired by
[Benchmark for OOD Detection in Deep RL](https://arxiv.org/abs/2112.02694).

## Status — Phase 0 (detection foundation) ✅

- gymnax-based parametric envs with a one-factor-at-a-time OOD grid (`ood_rl/envs/`).
- `vmap`-ready transition collection via `jax.lax.scan` (`ood_rl/experiment/rollout.py`).
- KNN dynamics world-model (`ood_rl/models/`).
- **Corrected** online conformal-martingale detector with **Ville-threshold FPR control**
  (`ood_rl/detectors/conformal_martingale.py`) — replaces the legacy detector, which
  computed an underflowing batch martingale on miscalibrated parametric p-values.

Measured on the nominal vs OOD environments (`examples/00_phase0_check.py`):

```
CartPole-v1 (stride=1):  nominal false-alarm rate 0.00;  gravity x5 -> 100% detection (delay ~290), x10 -> delay ~99
Pendulum-v1 (stride=5):  nominal false-alarm rate 0.00;  g x2 -> 100% detection (delay ~16)
```

**Documented finding:** Pendulum never terminates, so its transitions are strongly
autocorrelated — this violates the conformal exchangeability assumption and inflates the
false-alarm rate (≈0.50 at `stride=1`). Subsampling transitions (`stride≥3`) decorrelates
them and restores FPR control. Guarded by `tests/test_martingale_fpr.py`.

## Status — Phase 1 (detection benchmark) ✅

- Baseline detectors sharing the `Detector` ABC and the same calibration: CUSUM and
  sliding-window KS (`ood_rl/detectors/`), plus a reward-drop baseline driven by a trained
  policy. Running detectors are calibrated at the **stream** level (sup over nominal runs)
  so their false-alarm rate is controlled despite multiple testing.
- Parallel OOD-grid sweep via `vmap` (`ood_rl/experiment/sweep.py`) — the whole 105-config
  grid in ~2 s — plus detection metrics (AUROC/AUPR, detection delay, ARL; `metrics.py`).
- Intra-trajectory change-point rollout (`collect_nonstationary`) for online detection delay.
- A pure-JAX agent via rejax (`ood_rl/agents/train.py`) powering the reward-drop baseline.

Benchmark (`examples/01_detection_benchmark.py`), all detectors FPR-controlled at delta=0.01:

```
CartPole-v1   conformal AUROC 0.79 (delay 93) | cusum 0.87 | ks 0.94
Pendulum-v1   conformal AUROC 0.76 (delay 24) | cusum 0.88 | ks 0.83
```

**Headline result** — dynamics-based detection catches shifts that reward-based detection
misses: halving Pendulum gravity (g×0.5) barely changes the policy's reward, so reward-drop
stays silent (0.00), while the conformal detector — using *random* actions, no policy —
flags the dynamics change (0.75). Strong shifts (g×2/5/10) are caught by both.

## Install & run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,agents]"   # [agents]=rejax, for reward-drop

.venv/bin/python -m pytest -q                          # all tests
.venv/bin/python examples/00_phase0_check.py           # Phase 0: FPR control vs OOD detection
.venv/bin/python examples/01_detection_benchmark.py    # Phase 1: detector comparison
```

## Quickstart

```python
import jax
from ood_rl import make_env, Phase0Config
from ood_rl.envs.params import build_params
from ood_rl.experiment.rollout import fit_reference_detector, collect_transitions, detect

cfg = Phase0Config(env_name="CartPole-v1")
env, params, ood_configs = make_env(cfg.env_name)

# Train dynamics model + calibrate detector on the nominal env.
model, detector = fit_reference_detector(env, params, jax.random.PRNGKey(0), cfg, rng=0)

# Stream an OOD env (gravity x5) and detect.
ood = build_params(cfg.env_name, params, gravity=9.8 * 5)
X, y = collect_transitions(env, ood, jax.random.PRNGKey(1), cfg.n_test, cfg.memory_size, cfg.stride)
scores, alarm_at = detect(model, detector, X, y)   # alarm_at = first step the martingale crosses log(1/delta)
```

## Layout

```
ood_rl/
  envs/         gymnax EnvParams <-> physics knobs, OOD grid, change-point schedule
  models/       WorldModel ABC + KNNDynamics
  detectors/    Detector ABC; conformal-martingale + CUSUM / KS / reward-drop baselines
  experiment/   rollout (+ change-point), sweep (vmap grid), metrics, config
  agents/       rejax policy training (optional [agents] extra)
examples/       00_phase0_check.py, 01_detection_benchmark.py
tests/
```

## Roadmap

- **Phase 1 — detection:** ✅ baselines (CUSUM, sliding-KS, reward-drop), `vmap` sweeps over
  the full OOD grid, metrics (AUROC/AUPR, detection delay, ARL), intra-trajectory
  change-point rollout.
- **Phase 2 — adaptation:** `Adapter` interface; online system-ID of physics parameters
  (brax differentiable dynamics) and fine-tune/replan; regret + recovery-time metrics.
- **Phase 3 — research:** causal/modular world-models, Decision-Transformer in-context
  adaptation, multi-agent (opponent drift as non-stationarity).

## Legacy

The original notebooks (`RL_OOD_detection*.ipynb`, `new/`, `decision-transformers.ipynb`),
`rl_ood.py`, and `requirements.txt` are kept for reference but are **superseded** by the
`ood_rl` package above. They target the unmaintained `gym` API; do not build on them.
