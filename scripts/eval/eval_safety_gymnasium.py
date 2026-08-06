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
OFF_POLICY_ALGORITHMS = {"SAC", "TD3", "SafeSAC", "FastSAC", "FastTD3", "MPO", "CVPO"}


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
    parser.add_argument("--cost_limits", type=str, default=None, help="Comma-separated cost limits.")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode (e.g. human, rgb_array).")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed (= the hidden-goal task to render).")
    parser.add_argument("--hidden_goal", action="store_true", help="Hidden-goal meta-RL task (must match the trained policy's env).")
    parser.add_argument(
        "--reach_filter", action="store_true",
        help="Enable the learned reachability safety filter (needs an RCPPO/ActorCriticReach checkpoint).",
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
    cost_limits = parse_cost_limits(args.cost_limits)
    render_mode = "rgb_array" if args.video else args.render_mode
    env_kwargs: Dict[str, Any] = {
        "device": args.device,
        "render_mode": render_mode,
        "cost_limits": cost_limits,
        "seed": args.seed,
        "hidden_goal": args.hidden_goal,
    }
    if args.video:
        env_kwargs.update(
            width=args.video_width,
            height=args.video_height,
            camera_name=args.camera_name,
        )
    env = make_env(env_id=args.env_id, num_envs=args.num_envs, **env_kwargs)

    alg_name = train_cfg.get("algorithm", {}).get("class_name", "")
    runner_cls = OffPolicyRunner if alg_name in OFF_POLICY_ALGORITHMS else OnPolicyRunner
    if runner_cls is OffPolicyRunner:
        print(f"[INFO] Using OffPolicyRunner for algorithm: {alg_name}")
    runner = runner_cls(env, train_cfg, log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
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
    reward_buf = torch.zeros(env.num_envs, device=runner.device)
    cost_buf = torch.zeros(env.num_envs, device=runner.device)

    rta_interventions = 0
    while len(ep_rewards) < args.episodes:
        with torch.inference_mode():
            actions = policy(obs)
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

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            ep_rewards.extend(reward_buf[done_ids].cpu().tolist())
            ep_costs.extend(cost_buf[done_ids].cpu().tolist())
            reward_buf[done_ids] = 0.0
            cost_buf[done_ids] = 0.0

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
                args.eval_csv, costs_done, rewards_done, [0] * len(costs_done)
            )
            print(f"[INFO] Per-episode evaluation CSV -> {out}")
    if reach_filter is not None:
        print(f"Reachability filter interventions (env-steps): {rta_interventions}")

    env.close()


if __name__ == "__main__":
    main()
