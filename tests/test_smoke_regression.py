"""Regression oracle: a short seeded CVPO run must reproduce byte-for-byte.

This is the gate for every subsequent change. Any modification that is meant to be
behaviour-neutral with its flag OFF must leave `smoke_baseline.json` untouched; if it does
not, stop and find out why before continuing.

Two properties are asserted separately, and they are not the same thing:

  determinism  -- two runs in the SAME process/version agree. Without this the oracle is
                  meaningless. Note torch was never seeded in the training path before
                  safe_rl/utils/seeding.py, so this did not hold at all.
  regression   -- a run agrees with the recorded baseline on disk.

Runs on CPU deliberately: the categorical projection uses scatter_add_, which has no
deterministic CUDA implementation, so bit-exact reproducibility is a CPU property.

Config overrides vs the shipped criticfix config are minimal and stated in SMOKE_OVERRIDES:
the replay warm-up is shortened so that 200 iterations actually exercise the update path
(the shipped `update_after: 5000` would leave the whole run in random-action warm-up and the
oracle would test nothing).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
pytest.importorskip("safety_gymnasium")

from safe_rl.envs import make_env  # noqa: E402
from safe_rl.runners import OffPolicyRunner  # noqa: E402
from safe_rl.utils.seeding import seed_everything  # noqa: E402

BASELINE = Path(__file__).parent / "smoke_baseline.json"
CONFIG = Path(__file__).parent.parent / "config" / "safety_gymnasium_cvpo_criticfix.yaml"

SEED = 20260805
ENV_ID = "SafetyPointGoal1-v0"
NUM_ENVS = 4
ITERATIONS = 200
CHUNK = 20

SMOKE_OVERRIDES = {
    "start_random_steps": 64,   # shipped 5000 -> whole run would be warm-up
    "update_after": 64,         # shipped 5000 -> no gradient step would ever run
    "log_interval": 10_000,     # keep stdout quiet; scalars are read from state, not logs
    "logger": "tensorboard",    # never wandb in tests
}

TOL = 1e-6


def _hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:32]


def run_smoke(iterations: int = ITERATIONS, seed: int = SEED) -> dict:
    """One seeded CVPO run; returns the quantities the oracle compares."""
    seed_everything(seed, deterministic=True)

    cfg = yaml.safe_load(open(CONFIG))
    runner_cfg = dict(cfg.get("runner", {}))
    runner_cfg.update(SMOKE_OVERRIDES)
    train_cfg = {"algorithm": cfg["algorithm"], "policy": cfg["policy"], "runner": runner_cfg}
    train_cfg["algorithm"] = dict(train_cfg["algorithm"])
    train_cfg["algorithm"]["cost_limits"] = [25.0]

    env = make_env(env_id=ENV_ID, num_envs=NUM_ENVS, device="cpu", cost_limits=[25.0], seed=seed)
    try:
        runner = OffPolicyRunner(env, train_cfg, log_dir=None, device="cpu")
        alg = runner.alg

        lam_traj, eqc_traj = [], []
        for _ in range(iterations // CHUNK):
            runner.learn(CHUNK)
            lam_traj.append(float(alg.lam))
            eqc_traj.append(float(getattr(alg, "_eqc", float("nan"))))

        # Fixed probe observations -> deterministic actions. Drawn from a private generator
        # so the probe cannot perturb the training RNG stream.
        g = torch.Generator().manual_seed(12345)
        obs_dim = env.num_obs if hasattr(env, "num_obs") else runner.alg.policy.num_critic_obs
        probe = torch.randn(10, obs_dim, generator=g)
        with torch.no_grad():
            actions = alg.policy.act(probe, deterministic=True)

        weights = torch.cat([p.flatten() for p in alg.policy.actor.parameters()])
        return {
            "lambda_traj": lam_traj,
            "eqc_traj": eqc_traj,
            "qc_thres": float(alg.qc_thres),
            "eta": float(alg.eta),
            "action_hash": _hash(actions),
            "actor_weight_hash": _hash(weights),
            "actions_first_row": [round(x, 8) for x in actions[0].tolist()],
        }
    finally:
        env.close()


def _compare(a: dict, b: dict, tol: float = TOL) -> list[str]:
    diffs = []
    for k in ("action_hash", "actor_weight_hash"):
        if a[k] != b[k]:
            diffs.append(f"{k}: {a[k]} != {b[k]}")
    for k in ("qc_thres", "eta"):
        if abs(a[k] - b[k]) > tol:
            diffs.append(f"{k}: {a[k]} vs {b[k]}")
    for k in ("lambda_traj", "eqc_traj", "actions_first_row"):
        va, vb = a[k], b[k]
        if len(va) != len(vb):
            diffs.append(f"{k}: length {len(va)} != {len(vb)}")
            continue
        for i, (x, y) in enumerate(zip(va, vb)):
            if x != x and y != y:      # both NaN
                continue
            if abs(x - y) > tol:
                diffs.append(f"{k}[{i}]: {x} vs {y}")
    return diffs


@pytest.mark.slow
def test_smoke_run_is_deterministic() -> None:
    """Two runs of the same code with the same seed must agree. Gate for everything else."""
    a = run_smoke()
    b = run_smoke()
    diffs = _compare(a, b)
    assert not diffs, "smoke run is NOT deterministic:\n  " + "\n  ".join(diffs[:10])


@pytest.mark.slow
def test_smoke_matches_baseline() -> None:
    """Behaviour-neutral changes (all new flags OFF) must not move this."""
    if not BASELINE.exists():
        pytest.skip(f"no baseline recorded yet; write one with: python -m tests.test_smoke_regression")
    recorded = json.loads(BASELINE.read_text())
    current = run_smoke()
    diffs = _compare(recorded, current)
    assert not diffs, (
        "smoke run diverged from tests/smoke_baseline.json:\n  " + "\n  ".join(diffs[:10])
        + "\n\nIf this change was meant to be behaviour-neutral, stop and find out why."
    )


if __name__ == "__main__":
    result = run_smoke()
    BASELINE.write_text(json.dumps(result, indent=1))
    print(f"wrote {BASELINE}")
    print(json.dumps({k: v for k, v in result.items() if k != "eqc_traj"}, indent=1))
