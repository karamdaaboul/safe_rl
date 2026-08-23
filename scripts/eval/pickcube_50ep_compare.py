"""50-episode PickCube-v1 evaluation with a sub-goal breakdown, for comparing
our REPPO-Gaussian, our REPPO-DIME and the TruDi reference on one protocol.

PROTOCOL (identical for every arm — this is the point):
  * ignore_terminations=True, so every episode runs the full 50-step horizon.
    Without this an arm that succeeds early ends its episode early and reports a
    much smaller return: in the 10M head-to-head ours read eval/episode_return
    5.95 vs the reference's 33.17 purely because our episodes ended at step ~15
    and theirs ran to 50. Same policy quality, different accumulation window.
  * success/grasp/lift/carry are ONCE-per-episode flags (did it ever happen),
    OR-accumulated across the episode, which is ManiSkill's `success_once`.
  * deterministic policy (ODE chain for DIME, mean action for Gaussian), since
    eval/success_ode_100 is the metric both codebases report.

SUB-GOALS. "lift" and "carry" are mjlab Lift-Cube reward terms and do not exist
on PickCube, but PickCube's evaluate() exposes the equivalent decomposition, so
the columns mean:
  grasp   <- info["is_grasped"]      both fingers on the cube
  lift    <- cube height > 2cm       off the table at all
  carry   <- info["is_obj_placed"]   cube within goal_thresh of the goal
  static  <- info["is_robot_static"] arm settled (PickCube requires this for success)
  success <- info["success"]         = is_obj_placed AND is_robot_static
"""
from __future__ import annotations

import argparse
import numpy as np
import torch


def build_env(num_envs: int, device: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    base = gym.make(
        "PickCube-v1",
        num_envs=num_envs,
        obs_mode="state",
        reward_mode="normalized_dense",
        reconfiguration_freq=1,
    )
    # ignore_terminations=True -> full 50-step episodes for every arm. Matches the
    # reference's own eval env construction (their envs.py builds eval_envs this way).
    return ManiSkillVectorEnv(base, num_envs, ignore_terminations=True, record_metrics=True)


@torch.inference_mode()
def evaluate(env, act_fn, episodes: int, device: str, horizon: int = 50) -> dict:
    n = env.num_envs
    tot = {k: [] for k in ("reward", "success", "grasp", "lift", "carry", "static")}
    done_eps = 0
    while done_eps < episodes:
        obs, _ = env.reset()
        ret = torch.zeros(n, device=device)
        once = {k: torch.zeros(n, dtype=torch.bool, device=device)
                for k in ("success", "grasp", "lift", "carry", "static")}
        for _ in range(horizon):
            a = act_fn(obs.to(device).float())
            obs, rew, term, trunc, info = env.step(a)
            ret = ret + rew.view(-1).float().clone()

            def flag(*keys):
                for k in keys:
                    if k in info:
                        return info[k].view(-1).bool().clone()
                return None

            for name, keys in (
                ("success", ("success",)),
                ("grasp", ("is_grasped",)),
                ("carry", ("is_obj_placed",)),
                ("static", ("is_robot_static",)),
            ):
                f = flag(*keys)
                if f is not None:
                    once[name] = once[name] | f
            # "lift": cube more than 2cm off the table. Read from sim state, since
            # PickCube's info has no height flag.
            try:
                h = env.unwrapped.cube.pose.p[:, 2]
                once["lift"] = once["lift"] | (h.view(-1) > 0.02)
            except Exception:
                pass

        take = min(n, episodes - done_eps)
        tot["reward"] += ret[:take].tolist()
        for k in ("success", "grasp", "lift", "carry", "static"):
            tot[k] += once[k][:take].float().tolist()
        done_eps += take
    return {k: float(np.mean(v)) for k, v in tot.items()} | {"episodes": done_eps}


def load_ours(kind: str, ckpt: str, cfg_path: str, env, device: str):
    import yaml
    from safe_rl.modules import REPPOActorCritic, DIMEActorCritic

    pcfg = dict(yaml.safe_load(open(cfg_path))["policy"])
    pcfg.pop("class_name", None)
    nobs = env.single_observation_space.shape[0]
    nact = env.single_action_space.shape[0]
    cls = DIMEActorCritic if kind == "dime" else REPPOActorCritic
    pol = cls(nobs, nobs, nact, **pcfg).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    pol.load_state_dict(ck["model_state_dict"])
    pol.eval()
    return pol.act_inference  # deterministic: ODE chain for DIME, mean for Gaussian


def load_reference(ckpt: str, cfg_path: str, env, device: str):
    """Score the AUTHORS' checkpoint on this same harness.

    Their actor_state_dict drops straight into our DIMEActorCritic.actor: our
    safe_rl/networks/dime/ is vendored verbatim from their code, and
    tests/test_dime_reference_parity.py proves the two are bit-identical (all
    10,398 gradients, max|diff| = 0). So the same 22 parameter tensors and the
    same EmpiricalNormalization buffers (_mean/_var/_std/count) load with
    strict=True — which is also a live re-check of that parity claim: any drift in
    our vendored copy would show up right here as a key or shape mismatch.

    This is what makes the three arms comparable: identical env, horizon, flags
    and episode count, rather than their in-training eval vs our offline one.
    """
    import yaml
    from safe_rl.modules import DIMEActorCritic

    pcfg = dict(yaml.safe_load(open(cfg_path))["policy"])
    pcfg.pop("class_name", None)
    nobs = env.single_observation_space.shape[0]
    nact = env.single_action_space.shape[0]
    pol = DIMEActorCritic(nobs, nobs, nact, **pcfg).to(device)

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    missing, unexpected = pol.actor.load_state_dict(ck["actor_state_dict"], strict=True), None
    norm = ck.get("normalizer_state_dict")
    if norm is not None and hasattr(pol.actor_obs_normalizer, "load_state_dict"):
        pol.actor_obs_normalizer.load_state_dict(norm, strict=True)
        print("  loaded their normalizer (_mean/_var/_std/count)")
    else:
        print("  WARNING: no normalizer in checkpoint — obs would be unnormalized")
    pol.eval()
    print(f"  loaded their actor: {len(ck['actor_state_dict'])} tensors, global_step={ck.get('global_step')}")
    return pol.act_inference


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["gauss", "dime", "ref"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--num_envs", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    env = build_env(a.num_envs, a.device)
    if a.arm == "ref":
        fn = load_reference(a.ckpt, a.config, env, a.device)
    else:
        fn = load_ours(a.arm, a.ckpt, a.config, env, a.device)
    res = evaluate(env, fn, a.episodes, a.device)
    print("RESULT " + " ".join(f"{k}={v}" for k, v in res.items()))
