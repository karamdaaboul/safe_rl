#!/usr/bin/env python3
"""Prove that a concurrent edit to REPPOActorCritic is a no-op at action_scale=1.0.

Reproduces the evidence in PROVENANCE.md. Loads the frozen version of
``safe_rl/modules/reppo_actor_critic.py`` from a given commit alongside the working-tree
version, forces identical weights, and compares every action-producing call path
bitwise under fixed RNG seeds.

    python experiments/v00/check_equivalence.py --ref 7a54733
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from safe_rl.modules.reppo_actor_critic import REPPOActorCritic as NEW

KW = dict(
    num_actor_obs=6, num_critic_obs=9, num_actions=4, actor_type="stochastic",
    critic_type="reference", num_critics=1, squash="tanh", min_std=0.0,
    actor_kwargs={"hidden_dims": [32, 32], "network_type": "mlp", "norm_type": "rmsnorm",
                  "use_layer_norm": True, "activation": "swish", "log_std_squash": "clamp"},
    critic_kwargs={"num_atoms": 51, "v_min": -20.0, "v_max": 150.0, "hidden_dim": 32},
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="7a54733", help="Commit holding the frozen version.")
    args = ap.parse_args()

    src = subprocess.run(
        ["git", "show", f"{args.ref}:safe_rl/modules/reppo_actor_critic.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "old_rac.py"
        path.write_text(src)
        spec = importlib.util.spec_from_file_location("old_rac", path)
        old_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_mod)
    OLD = old_mod.REPPOActorCritic

    def build(cls):
        torch.manual_seed(1234)
        return cls(**KW)

    a, b = build(OLD), build(NEW)
    b.load_state_dict(a.state_dict())

    obs = torch.randn(32, 6)
    checks: dict[str, bool] = {}

    torch.manual_seed(7)
    oa = a.act(obs)
    torch.manual_seed(7)
    nb = b.act(obs)
    checks["act"] = torch.equal(oa, nb)
    checks["act_inference"] = torch.equal(a.act_inference(obs), b.act_inference(obs))

    torch.manual_seed(9)
    o = a.sample_with_log_prob(obs)
    torch.manual_seed(9)
    n = b.sample_with_log_prob(obs)
    checks["sample_with_log_prob"] = all(torch.equal(x, y) for x, y in zip(o, n))

    from torch.distributions import Normal
    base = Normal(torch.randn(16, 4), torch.rand(16, 4).add(0.1))
    acts = torch.tanh(torch.randn(16, 4))
    checks["squashed_log_prob"] = torch.equal(
        a.squashed(base).log_prob(a._clamp_squashed(acts)),
        b.squashed(base).log_prob(b._clamp_squashed(acts)),
    )
    checks["_clamp_squashed"] = torch.equal(a._clamp_squashed(acts * 2), b._clamp_squashed(acts * 2))

    print(f"action_scale on working tree: {getattr(b, 'action_scale', 'absent')}")
    for name, ok in checks.items():
        print(f"{name:<22} bitwise equal: {ok}")
    if all(checks.values()):
        print("\nEQUIVALENT — runs under either version are poolable.")
        return 0
    print("\nNOT EQUIVALENT — runs must be segregated by code version.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
