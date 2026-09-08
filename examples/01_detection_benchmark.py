"""Phase 1 detection benchmark.

Compares OOD detectors on the parametric physics-shift grid:
  - dynamics-based detectors (conformal-martingale, CUSUM, sliding-KS) on the world-model
    nonconformity stream, scored over the whole grid via the vmap sweep;
  - the reward-drop baseline (needs a trained policy), shown on Pendulum.

For each detector we report nominal false-alarm rate, AUROC, and mean steps-to-alarm on
strong shifts. Run:  python examples/01_detection_benchmark.py
"""

import numpy as np
import jax

from ood_rl import (make_env, Phase0Config, ConformalMartingaleDetector,
                    CUSUMDetector, KSWindowDetector, RewardDropDetector)
from ood_rl.envs.params import build_params, DEFAULTS
from ood_rl.experiment import sweep, metrics
from ood_rl.experiment.rollout import collect_transitions


def dynamics_benchmark(env_name, stride):
    cfg = Phase0Config(env_name, n_train=8_000, n_test=600, delta=0.01, stride=stride)
    env, dp, configs = make_env(env_name)
    k_fit, k_sweep, k_nom = jax.random.split(jax.random.PRNGKey(0), 3)
    factories = {
        "conformal": lambda: ConformalMartingaleDetector(epsilon=cfg.epsilon, delta=cfg.delta, rng=0),
        "cusum": lambda: CUSUMDetector(delta=cfg.delta, seg_len=cfg.n_test),
        "ks": lambda: KSWindowDetector(window=100, delta=cfg.delta, eval_every=5),
    }
    model, dets = sweep.fit_detectors(env, dp, k_fit, cfg, factories, n_cal_streams=120)

    X, y, _ = sweep.sweep_streams(env, configs, k_sweep, cfg)
    al = sweep.alpha_streams(model, X, y, stride=cfg.stride)
    Xn, yn, _ = sweep.sweep_streams(env, [{"params": dp}] * 60, k_nom, cfg)
    aln = sweep.alpha_streams(model, Xn, yn, stride=cfg.stride)

    scales = np.array([c["scale"] for c in configs])
    labels = (np.abs(np.log10(scales)) > 1e-9).astype(int)
    strong = np.abs(np.log10(scales)) >= 0.5

    print(f"\n=== {env_name} | dynamics detectors (stride={stride}, delta=0.01, "
          f"{len(configs)} configs, {int(strong.sum())} strong) ===")
    print(f"  {'detector':10s} {'FPR':>5} {'AUROC(all)':>11} {'AUROC(strong)':>14} {'mean delay':>11}")
    for name, d in dets.items():
        grid = sweep.run_detector_on_streams(d, al)
        sup = np.array([g[1] for g in grid]); at = np.array([g[0] for g in grid])
        nom = sweep.run_detector_on_streams(d, aln)
        fpr = metrics.detection_rate([g[0] for g in nom])
        sup_nom = np.array([g[1] for g in nom])
        s = np.concatenate([sup[strong], sup_nom])
        lb = np.concatenate([np.ones(int(strong.sum())), np.zeros(len(sup_nom))])
        d_strong = at[strong][at[strong] >= 0]
        delay = f"{d_strong.mean():.0f} steps" if len(d_strong) else "n/a"
        print(f"  {name:10s} {fpr:5.2f} {metrics.auroc(sup, labels):11.2f} "
              f"{metrics.auroc(s, lb):14.2f} {delay:>11}")


def reward_drop_comparison(env_name="Pendulum-v1", knob="g", scales=(0.5, 2.0, 5.0, 10.0)):
    try:
        from ood_rl.agents.train import train_policy
    except ImportError:
        print("\n[reward-drop] skipped (install the [agents] extra: rejax)")
        return

    cfg = Phase0Config(env_name, n_train=8_000, n_test=600, delta=0.01, stride=5)
    env, dp, _ = make_env(env_name)
    policy = train_policy(env_name, total_timesteps=300_000, seed=0)

    # dynamics reference (conformal) for the same configs
    k_fit, k_nom = jax.random.split(jax.random.PRNGKey(1))
    model, dets = sweep.fit_detectors(
        env, dp, k_fit, cfg,
        {"conformal": lambda: ConformalMartingaleDetector(delta=cfg.delta, rng=0)},
        n_cal_streams=80,
    )
    conformal = dets["conformal"]

    def reward_streams(params, n, base):
        return np.array([collect_transitions(env, params, jax.random.fold_in(jax.random.PRNGKey(base), s),
                                             cfg.n_test, policy=policy, return_rewards=True, stride=1)[2]
                         for s in range(n)])

    def alpha_streams_for(params, n, base):
        out = []
        for s in range(n):
            X, y = collect_transitions(env, params, jax.random.fold_in(jax.random.PRNGKey(base), s),
                                       cfg.n_test, cfg.memory_size, cfg.stride)
            out.append(model.residual(X, y))
        return np.array(out)

    rdrop = RewardDropDetector(delta=cfg.delta, seg_len=cfg.n_test)
    rdrop.calibrate(reward_streams(dp, 60, 1))

    print(f"\n=== {env_name} | reward-drop (needs trained policy) vs conformal (random actions) ===")
    print(f"  {'shift':>8} {'policy reward':>14} {'reward-drop det':>16} {'conformal det':>14}")
    for sc in (1.0,) + tuple(scales):
        params = dp if sc == 1.0 else build_params(env_name, dp, **{knob: DEFAULTS[env_name][knob] * sc})
        rwd = reward_streams(params, 20, int(sc * 1000) + 2)
        rd_rate = metrics.detection_rate([rdrop.run(r)[1] for r in rwd])
        al = alpha_streams_for(params, 20, int(sc * 1000) + 3)
        cf_rate = metrics.detection_rate([conformal.run(a)[1] for a in al])
        tag = "nominal" if sc == 1.0 else f"x{sc}"
        print(f"  {tag:>8} {rwd.mean():14.2f} {rd_rate:16.2f} {cf_rate:14.2f}")
    print("  (reward-drop only fires when the shift hurts the policy; the conformal detector")
    print("   flags the dynamics change directly, from random actions, no policy needed.)")


if __name__ == "__main__":
    dynamics_benchmark("CartPole-v1", stride=1)
    dynamics_benchmark("Pendulum-v1", stride=5)
    reward_drop_comparison("Pendulum-v1")
