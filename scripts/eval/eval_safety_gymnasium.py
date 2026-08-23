from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from safe_rl.common.tail_eval import episode_cost_stats, format_summary, write_episode_csv
from safe_rl.envs import make_env
from safe_rl.runners import OffPolicyRunner, OnPolicyRunner

# Algorithms driven by OffPolicyRunner. Routing these through OnPolicyRunner raises
# "Training type not found", which is what evaluating an MPO/CVPO checkpoint used to hit.
#
# DERIVED, not hand-listed. The hand-maintained set silently went stale three times over
# (FHDCMPOPerState, FHDCMPODIME, MPODIME were all missing), and the failure mode is a crash at the
# END of a multi-hour training run, when the launcher gets to its evaluation step. Every
# MPO/CVPO-lineage algorithm subclasses SAC, so membership is a property of the class hierarchy
# and can be read off it; only the standalone off-policy implementations that do NOT derive from
# SAC still need naming.
_OFF_POLICY_NOT_SAC_DERIVED = {"TD3", "FastSAC", "FastTD3"}


def _off_policy_algorithms() -> set:
    import safe_rl.algorithms as _algs
    from safe_rl.algorithms.sac import SAC as _SAC

    derived = {
        name
        for name in getattr(_algs, "__all__", [])
        if isinstance(getattr(_algs, name, None), type) and issubclass(getattr(_algs, name), _SAC)
    }
    return derived | _OFF_POLICY_NOT_SAC_DERIVED


OFF_POLICY_ALGORITHMS = _off_policy_algorithms()


def obs_shaping_env_kwargs(env_cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """The subset of ``env: kwargs`` that changes the observation WIDTH, for the evaluation env.

    Same hazard as the risk column, for the same reason: FH-DCMPO's observation carries
    finite-horizon columns, so the evaluation env must add them or the checkpoint will not match the
    observation width. ``env: kwargs`` is the training script's generic passthrough
    (``train_safety_gymnasium.py``, "env kwargs from config"), and evaluation has to honour the
    observation-shaping subset of it -- otherwise a policy trained on ``(s, u)`` is silently
    evaluated on ``s`` alone, which would read as the method not working rather than as a harness bug.

    Deliberately a whitelist, not a blanket forward: most ``env: kwargs`` are training-only
    (a vision worker's ``mp_context``, ManiSkill's eval-twin settings) and forwarding them wholesale
    would change what the reported return means.
    """
    keys = ("horizon_feature", "budget_feature", "budget_limit")
    extra = dict(env_cfg.get("kwargs") or {})
    forwarded = {k: extra[k] for k in keys if k in extra}
    if forwarded:
        print(f"[INFO] Observation-shaping env kwargs carried over from {config_path}: {forwarded}")
    return forwarded


def load_train_cfg(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    algorithm_cfg = cfg["algorithm"]
    policy_cfg = cfg["policy"]
    runner_cfg = cfg.get("runner", {})

    symmetry_cfg = algorithm_cfg.get("symmetry_cfg")
    if symmetry_cfg is not None:
        if not symmetry_cfg.get("data_augmentation_func"):
            algorithm_cfg["symmetry_cfg"] = None

    rnd_cfg = algorithm_cfg.get("rnd_cfg")
    if rnd_cfg is not None and rnd_cfg.get("weight", 0.0) == 0.0:
        algorithm_cfg["rnd_cfg"] = None

    out: Dict[str, Any] = {
        "algorithm": algorithm_cfg,
        "policy": policy_cfg,
        "num_steps_per_env": runner_cfg.get("num_steps_per_env", 24),
        "save_interval": runner_cfg.get("save_interval", 50),
        "empirical_normalization": runner_cfg.get("empirical_normalization", False),
        "logger": runner_cfg.get("logger", "tensorboard"),
        "wandb_project": runner_cfg.get("wandb_project", "safe_rl"),
    }
    # OffPolicyRunner reads a nested `runner` section (the on-policy path uses the flat keys
    # above), so evaluating an off-policy checkpoint needs both forms present.
    if algorithm_cfg.get("class_name", "") in OFF_POLICY_ALGORITHMS:
        out["runner"] = {
            "num_steps_per_env": runner_cfg.get("num_steps_per_env", 1),
            "save_interval": runner_cfg.get("save_interval", 50),
            "log_interval": runner_cfg.get("log_interval", 1),
            "empirical_normalization": runner_cfg.get("empirical_normalization", False),
            "logger": runner_cfg.get("logger", "tensorboard"),
            "wandb_project": runner_cfg.get("wandb_project", "safe_rl"),
            "wandb_entity": runner_cfg.get("wandb_entity"),
            "wandb_dir": runner_cfg.get("wandb_dir"),
            "max_size": runner_cfg.get("max_size", 1_000_000),
            "start_random_steps": runner_cfg.get("start_random_steps", 10000),
            "update_after": runner_cfg.get("update_after", 1000),
            "update_every": runner_cfg.get("update_every", 50),
            "n_step": runner_cfg.get("n_step", 1),
        }
    return out


def parse_cost_limits(cost_limits: str | None) -> list[float] | None:
    if cost_limits is None:
        return None
    return [float(value.strip()) for value in cost_limits.split(",") if value.strip()]


def _extract_video_frame(frame: Any) -> torch.Tensor:
    frame_tensor = torch.as_tensor(frame)
    if frame_tensor.ndim == 4:
        frame_tensor = frame_tensor[0]
    if frame_tensor.dtype != torch.uint8:
        frame_tensor = frame_tensor.clamp(0, 255).to(torch.uint8)
    return frame_tensor.cpu()


def save_video(frames: list[torch.Tensor], video_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames captured for video export.")
    import imageio.v2 as imageio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video_path), fps=fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame.cpu().numpy())
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Safe-RL agents on Safety-Gymnasium environments.")
    parser.add_argument("--env_id", type=str, required=True, help="Safety-Gymnasium env id (e.g. SafetyCarGoal1-v0).")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of vectorized environments.")
    parser.add_argument("--config", type=str, default="config/dummy_config.yaml", help="Path to training config.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
    parser.add_argument(
        "--policy", type=str, default="deterministic", choices=["deterministic", "stochastic"],
        help="WHICH POLICY IS ROLLED OUT, and it changes what the numbers mean. `deterministic` "
             "(default) is the deployment policy and the basis of every arm comparison here. "
             "`stochastic` samples from the actor, i.e. the behaviour distribution that fills the "
             "buffer and that the training curves (wandb Episode/cost) average over -- use it to "
             "reproduce a training number, never to compare two checkpoints.",
    )
    parser.add_argument("--cost_limits", type=str, default=None, help="Comma-separated cost limits.")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode (e.g. human, rgb_array).")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed (= the hidden-goal task to render).")
    parser.add_argument("--action_repeat", type=int, default=1, help="Apply each action for N simulator steps; MUST match the value used in training, since it is part of the policy's control rate. There is deliberately no --goal_pseudo_terminal here: that is a training-only value-bootstrap treatment and enabling it at evaluation would change what the reported return means.")
    parser.add_argument("--empirical_normalization", dest="empirical_normalization", action="store_true", default=None, help="Force observation normalization on, overriding the config. MUST match training: a policy trained with normalization sees a completely different input scale without it, and the checkpoint's statistics are silently discarded.")
    parser.add_argument("--no_empirical_normalization", dest="empirical_normalization", action="store_false", help="Force observation normalization off, overriding the config.")
    parser.add_argument(
        "--reach_filter", action="store_true",
        help="Enable the learned reachability safety filter (needs an RCPPO/ActorCriticReachQ checkpoint).",
    )
    parser.add_argument(
        "--reach_threshold", type=float, default=0.0,
        help="Reachability filter: Q_h feasibility threshold epsilon.",
    )
    parser.add_argument(
        "--reach_candidates", type=int, default=16,
        help="Reachability filter: candidate actions per intervention.",
    )
    parser.add_argument(
        "--reach_mode", type=str, default="switch", choices=["switch", "blend"],
        help="Reachability filter: replace or blend unsafe actions.",
    )
    parser.add_argument(
        "--risk_level", type=float, default=None,
        help="Risk-conditioned policy: pin the risk level in [0, 1] instead of sampling it "
             "(0 = risk-seeking, 1 = risk-averse). Values between trained modes are allowed. "
             "Required to evaluate a policy trained with `env: risk_modes`.",
    )
    parser.add_argument("--cbf", action="store_true",
                        help="Apply the CBF runtime action filter during evaluation (needs Safety-Gymnasium; "
                             "uses privileged hazard positions via SGCBFStateWrapper).")
    parser.add_argument("--cbf_alpha", type=float, default=0.5, help="CBF class-K rate; higher = less conservative.")
    parser.add_argument("--cbf_d_min", type=float, default=0.35, help="CBF minimum clearance from hazard centre.")
    parser.add_argument("--cbf_v_scale", type=float, default=1.0, help="CBF action-to-position scaling.")
    parser.add_argument("--cbf_max_iter", type=int, default=5, help="CBF projection iterations per step.")
    parser.add_argument("--cbf_legacy", action="store_true",
                        help="Use the old velocity-command barrier (action[0] treated as a velocity). "
                             "Measured to be model-mismatched on the force-actuated Point robot; kept for reproduction.")
    parser.add_argument("--cbf_a_scale", type=float, default=0.00629, help="CBF action->delta-velocity gain.")
    parser.add_argument("--eval_csv", type=str, default=None,
                        help="Write per-episode cost/reward/length to this CSV.")
    parser.add_argument("--video", action="store_true", help="Record the evaluation rollout(s) to mp4.")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to store evaluation videos.")
    parser.add_argument("--video_width", type=int, default=640, help="Rendered frame width (px).")
    parser.add_argument("--video_height", type=int, default=480, help="Rendered frame height (px).")
    parser.add_argument(
        "--camera_name",
        type=str,
        default="track",
        choices=["vision", "track", "fixednear", "fixedfar", "human"],
        help="Safety-Gymnasium camera: vision=agent first-person, track=third-person chase, fixednear/far=top-down.",
    )
    args = parser.parse_args()

    if args.video and args.num_envs != 1:
        raise ValueError("Video recording currently requires --num_envs 1.")
    if args.video and args.render_mode not in (None, "human", "rgb_array"):
        raise ValueError("Video recording only supports --render_mode human, rgb_array, or omitting the flag.")

    train_cfg = load_train_cfg(args.config)
    if args.empirical_normalization is not None:
        # Both forms: the on-policy runner reads the flat key, the off-policy runner
        # the nested one (see load_train_cfg).
        train_cfg["empirical_normalization"] = args.empirical_normalization
        if "runner" in train_cfg:
            train_cfg["runner"]["empirical_normalization"] = args.empirical_normalization
    cost_limits = parse_cost_limits(args.cost_limits)
    render_mode = "rgb_array" if args.video else args.render_mode
    env_kwargs: Dict[str, Any] = {
        "device": args.device,
        "render_mode": render_mode,
        "cost_limits": cost_limits,
        "seed": args.seed,
        # Part of the trained controller, so it carries over to evaluation. The
        # goal pseudo-terminal deliberately does not — it only ever shaped the
        # value target during training.
        "action_repeat": args.action_repeat,
    }
    # The policy's observation carries a risk column, so the env must add one or the
    # checkpoint will not match the observation width.
    import yaml as _yaml

    _env_cfg = (_yaml.safe_load(open(args.config)) or {}).get("env", {}) or {}
    _risk_modes = int(_env_cfg.get("risk_modes", 0))
    if _risk_modes:
        if args.risk_level is None:
            raise SystemExit(
                f"{args.config} trains {_risk_modes} risk modes; pass --risk_level in [0, 1] "
                "(0 = risk-seeking, 1 = risk-averse) so evaluation pins one instead of "
                "averaging a random mixture over episodes."
            )
        env_kwargs["risk_modes"] = _risk_modes
        env_kwargs["risk_fixed_level"] = args.risk_level
        print(f"[INFO] Risk-conditioned evaluation at level {args.risk_level:.2f} "
              f"(0 = seeking, 1 = averse)")
    elif args.risk_level is not None:
        raise SystemExit(f"--risk_level given but {args.config} has no `env: risk_modes`.")

    env_kwargs.update(obs_shaping_env_kwargs(_env_cfg, args.config))
    if args.video:
        env_kwargs.update(
            width=args.video_width,
            height=args.video_height,
            camera_name=args.camera_name,
        )
    # Correlated-episode guard. SafetyGymnasiumVecEnv.reset() tiles ONE seed across every
    # sub-env (deliberate, so the hidden-goal path can render a specific task), so with a
    # fixed --seed and num_envs > 1 all sub-envs run the SAME layout. A deterministic policy
    # then produces num_envs copies of one trajectory, which this script would count as
    # independent episodes -- inflating n, shrinking the SEM, and flattening exactly the cost
    # tail these statistics exist to measure. Left to `seed=None`, gymnasium seeds each
    # sub-env separately and the episodes are genuinely independent.
    if args.num_envs > 1 and args.seed is not None:
        print(
            "\n" + "!" * 78 +
            f"\n[WARNING] --num_envs {args.num_envs} together with --seed {args.seed}: the vec env"
            f"\n          gives every sub-env the SAME seed, so the {args.num_envs} sub-envs run"
            "\n          identical layouts and the episodes are NOT independent."
            f"\n          Effective sample size is about episodes/{args.num_envs}, and the reported"
            "\n          SEM / percentiles / CVaR will be optimistic."
            "\n          Use --num_envs 1 for statistics; keep the seed only for rendering a"
            "\n          specific task.\n" + "!" * 78 + "\n"
        )

    # Runtime CBF shielding at evaluation. The training path enables this via the YAML `cbf:`
    # block (on_policy_runner.py:225); exposing it here lets an ALREADY-TRAINED, reward-greedy
    # policy be measured with the filter attached, which is the cheap way to ask whether
    # shielding changes cost-per-goal rather than sliding along the reward/cost frontier.
    # Note it reads privileged hazard state via SGCBFStateWrapper, so results must be reported
    # as "with access to hazard positions".
    if args.cbf:
        env_kwargs["cbf_state"] = True

    env = make_env(env_id=args.env_id, num_envs=args.num_envs, **env_kwargs)

    cbf_filter = None
    if args.cbf:
        from safe_rl.cbf import SafetyGymnasiumCBFFilter

        cbf_filter = SafetyGymnasiumCBFFilter(
            alpha=args.cbf_alpha, d_min=args.cbf_d_min, v_scale=args.cbf_v_scale,
            max_iter=args.cbf_max_iter, device=args.device,
            velocity_aware=not args.cbf_legacy, a_scale=args.cbf_a_scale,
        )
        print(f"[INFO] CBF filter ON ({'legacy velocity-command' if args.cbf_legacy else 'velocity-aware'}): "
              f"alpha={args.cbf_alpha} d_min={args.cbf_d_min} a_scale={args.cbf_a_scale} "
              f"max_iter={args.cbf_max_iter}")

    alg_name = train_cfg.get("algorithm", {}).get("class_name", "")
    # Evaluation only needs the policy, but the off-policy runner allocates the full replay
    # buffer at construction -- ~8GB at the default 1M, enough to OOM the box beside a
    # training run. Shrink it; nothing here ever stores a transition.
    if "runner" in train_cfg:
        train_cfg["runner"]["max_size"] = 1000
    runner_cls = OffPolicyRunner if alg_name in OFF_POLICY_ALGORITHMS else OnPolicyRunner
    if runner_cls is OffPolicyRunner:
        print(f"[INFO] Using OffPolicyRunner for algorithm: {alg_name}")
    runner = runner_cls(env, train_cfg, log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    if args.policy == "stochastic":
        # The BEHAVIOUR policy: what fills the replay buffer and what the training logs (and hence
        # wandb) average over. Reported training cost and deterministic eval cost are different
        # quantities, and comparing one against the other has already produced one wrong
        # conclusion in this repo -- this flag exists so the comparison can be made properly.
        print("[INFO] Rolling the STOCHASTIC policy: comparable with the training curves, NOT with "
              "the deterministic deployment numbers.")
        runner.eval_mode()
        inner = runner.alg.policy
        norm = runner.obs_normalizer if runner.empirical_normalization else torch.nn.Identity()

        def policy(x):
            return inner.act(norm(x), deterministic=False)
    else:
        policy = runner.get_inference_policy(device=args.device)

    reach_filter = None
    if args.reach_filter:
        from safe_rl.filters import ReachabilitySafetyFilter

        reach_filter = ReachabilitySafetyFilter(
            policy=runner.alg.policy,
            threshold=args.reach_threshold,
            num_candidates=args.reach_candidates,
            mode=args.reach_mode,
            device=args.device,
        )
        print(f"[INFO] Reachability safety filter enabled (threshold={args.reach_threshold}, mode={args.reach_mode}).")

    obs, _ = env.get_observations()
    obs = obs.to(runner.device)
    video_frames: list[torch.Tensor] = []
    video_path: Path | None = None
    video_fps = 30
    if args.video:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        default_video_dir = checkpoint_path.parent / "videos" / "eval"
        video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else default_video_dir
        video_path = video_dir / f"{checkpoint_path.stem}_eval.mp4"
        video_fps = int(getattr(getattr(env, "env", None), "metadata", {}).get("render_fps", 30))
        if args.render_mode == "human":
            print("[INFO] Ignoring --render_mode human while recording video; using rgb_array instead.")
        print(f"[INFO] Recording video to {video_dir}")
        video_frames.append(_extract_video_frame(env.render()))

    ep_rewards = []
    ep_costs = []
    ep_lengths = []
    reward_buf = torch.zeros(env.num_envs, device=runner.device)
    cost_buf = torch.zeros(env.num_envs, device=runner.device)
    # Real episode lengths. The per-episode CSV used to hard-code 0 here, which is fine while every
    # Goal episode runs a full 1000 steps but hides an early termination -- and a short episode is a
    # cheap-looking low cost, so a silent 0 is exactly the wrong thing to record in a safety table.
    len_buf = torch.zeros(env.num_envs, dtype=torch.long, device=runner.device)

    rta_interventions = 0
    while len(ep_rewards) < args.episodes:
        with torch.inference_mode():
            actions = policy(obs)
            if cbf_filter is not None:
                actions = cbf_filter.filter(actions, env)
            if reach_filter is not None:
                # Critic obs == actor obs for Safety-Gymnasium; match training-time normalization.
                actions = reach_filter.filter(actions, runner.privileged_obs_normalizer(obs))
                rta_interventions += int(reach_filter.last_intervention_frac * env.num_envs)
        obs, rewards, dones, infos = env.step(actions)
        obs = obs.to(runner.device)
        rewards = rewards.to(runner.device)
        dones = dones.to(runner.device)
        if args.video:
            video_frames.append(_extract_video_frame(env.render()))

        costs = infos.get("costs", torch.zeros_like(rewards)).to(runner.device)
        reward_buf += rewards
        cost_buf += costs
        len_buf += 1

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            ep_rewards.extend(reward_buf[done_ids].cpu().tolist())
            ep_costs.extend(cost_buf[done_ids].cpu().tolist())
            ep_lengths.extend(len_buf[done_ids].cpu().tolist())
            reward_buf[done_ids] = 0.0
            cost_buf[done_ids] = 0.0
            len_buf[done_ids] = 0

    if args.video:
        save_video(video_frames, video_path, fps=video_fps)
        print(f"[INFO] Saved evaluation video to {video_path}")

    mean_reward = sum(ep_rewards[: args.episodes]) / args.episodes
    mean_cost = sum(ep_costs[: args.episodes]) / args.episodes
    print(f"Evaluation over {args.episodes} episodes")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Mean cost: {mean_cost:.3f}")

    # Tail-aware summary. A mean cannot show a risk-constraint win: two policies with the
    # same mean cost can have completely different tails, and the tail is what a CVaR
    # constraint targets (measured: mean 21.4 under a limit of 25, yet 23-31% of individual
    # episodes still exceeded it).
    limit = (cost_limits or [25.0])[0]
    costs_done = ep_costs[: args.episodes]
    rewards_done = ep_rewards[: args.episodes]
    if costs_done:
        stats = episode_cost_stats(costs_done, cost_limit=limit, rewards=rewards_done)
        print(format_summary(stats))
        if args.eval_csv:
            out = write_episode_csv(
                args.eval_csv, costs_done, rewards_done, ep_lengths[: args.episodes]
            )
            print(f"[INFO] Per-episode evaluation CSV -> {out}")
    if reach_filter is not None:
        print(f"Reachability filter interventions (env-steps): {rta_interventions}")

    env.close()


if __name__ == "__main__":
    main()
