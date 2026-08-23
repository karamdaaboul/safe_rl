from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _add_unitree_repo_to_path() -> None:
    explicit_repo = os.environ.get("UNITREE_RL_MJLAB_PATH")
    candidates = [explicit_repo] if explicit_repo else []
    candidates.extend(
        [
            "/opt/unitree_rl_mjlab",
            str(Path.home() / "workspaces" / "unitree_rl_mjlab"),
            str(Path(__file__).resolve().parents[3] / "unitree_rl_mjlab"),
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)
            break


_add_unitree_repo_to_path()

import mjlab  # noqa: E402
import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, list_tasks  # noqa: E402
from mjlab.tasks.tracking.mdp import MotionCommandCfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

# Optional task source — see the matching note in scripts/train/unitree_mjlab.py.
try:
    import src.tasks  # noqa: E402,F401
except Exception as exc:  # noqa: BLE001
    print(
        f"[WARN] unitree_rl_mjlab task registration failed ({type(exc).__name__}: {exc}).\n"
        "[WARN] Continuing with mjlab's built-in tasks only; Ant-*/Unitree-* ids will be unavailable.",
        file=sys.stderr,
    )

import torch  # noqa: E402
import yaml  # noqa: E402

from safe_rl.envs import make_env  # noqa: E402
from safe_rl.envs.mjlab_tasks import register_all as _register_safe_rl_mjlab_tasks  # noqa: E402
from safe_rl.runners import OffPolicyRunner, OnPolicyRunner  # noqa: E402
from safe_rl.utils.eval_utils import filter_recordable, make_q_argmax_policy  # noqa: E402

# safe_rl-owned mjlab task variants (e.g. Mjlab-Lift-Cube-Yam-Grasp).
_register_safe_rl_mjlab_tasks()


OFF_POLICY_ALGORITHMS = {"SAC", "TD3", "SafeSAC", "FastSAC", "FastTD3", "MPO", "CVPO"}



def convert_mjlab_ppo_cfg(agent_cfg: Any) -> dict[str, Any]:
    cfg = asdict(agent_cfg)
    actor_cfg = cfg["actor"]
    critic_cfg = cfg["critic"]
    algorithm_cfg = cfg["algorithm"]
    distribution_cfg = actor_cfg.get("distribution_cfg", {})

    return {
        "algorithm": {
            "class_name": "PPO",
            "normalize_advantage_per_mini_batch": algorithm_cfg.get("normalize_advantage_per_mini_batch", False),
            "value_loss_coef": algorithm_cfg["value_loss_coef"],
            "clip_param": algorithm_cfg["clip_param"],
            "use_clipped_value_loss": algorithm_cfg["use_clipped_value_loss"],
            "desired_kl": algorithm_cfg["desired_kl"],
            "entropy_coef": algorithm_cfg["entropy_coef"],
            "gamma": algorithm_cfg["gamma"],
            "lam": algorithm_cfg["lam"],
            "max_grad_norm": algorithm_cfg["max_grad_norm"],
            "learning_rate": algorithm_cfg["learning_rate"],
            "num_learning_epochs": algorithm_cfg["num_learning_epochs"],
            "num_mini_batches": algorithm_cfg["num_mini_batches"],
            "schedule": algorithm_cfg["schedule"],
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "policy": {
            "class_name": "ActorCritic",
            "actor_type": "gaussian",
            "critic_type": "standard",
            "actor_obs_normalization": actor_cfg.get("obs_normalization", False),
            "critic_obs_normalization": critic_cfg.get("obs_normalization", False),
            "actor_kwargs": {
                "hidden_dims": list(actor_cfg["hidden_dims"]),
                "activation": actor_cfg["activation"],
                "init_noise_std": distribution_cfg.get("init_std", 1.0),
                "noise_std_type": distribution_cfg.get("std_type", "scalar"),
            },
            "critic_kwargs": {
                "hidden_dims": list(critic_cfg["hidden_dims"]),
                "activation": critic_cfg["activation"],
            },
        },
        "num_steps_per_env": cfg["num_steps_per_env"],
        "save_interval": cfg["save_interval"],
        "empirical_normalization": False,
        "logger": "tensorboard",
        "wandb_project": "safe_rl",
        "wandb_entity": None,
        "run_name": getattr(agent_cfg, "run_name", ""),
    }


def load_train_cfg(task_id: str, checkpoint_path: Path, train_cfg_path: str | None) -> dict[str, Any]:
    candidate_paths: list[Path] = []
    if train_cfg_path is not None:
        candidate_paths.append(Path(train_cfg_path).expanduser().resolve())
    candidate_paths.append(checkpoint_path.parent / "params" / "agent.yaml")

    for path in candidate_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                cfg = yaml.safe_load(file)
            cfg.setdefault("algorithm", {}).setdefault("class_name", "PPO")
            cfg.setdefault("policy", {}).setdefault("class_name", "ActorCritic")
            # Raw training YAMLs nest runner keys under "runner:"; OnPolicyRunner expects
            # them at the top level. Flatten unless it's an off-policy config (which keeps
            # the nested structure deliberately).
            alg_class = cfg.get("algorithm", {}).get("class_name", "PPO")
            if "runner" in cfg and "num_steps_per_env" not in cfg and alg_class not in OFF_POLICY_ALGORITHMS:
                cfg.update(cfg.pop("runner"))
            return cfg

    agent_cfg = load_rl_cfg(task_id)
    return convert_mjlab_ppo_cfg(agent_cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate/play safe_rl PPO agents on Unitree mjlab tasks.")
    parser.add_argument("--env_id", type=str, required=True, help="Registered mjlab task id, e.g. Unitree-G1-Flat.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--config", type=str, default=None, help="Optional path to saved params/agent.yaml.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of vectorized environments. Defaults to the env cfg's value (the play cfg sets a small fixed count).")
    parser.add_argument("--play", action="store_true", help="Load the task's play env cfg (small terrain + fixed small env count, suited to interactive viewing).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for evaluation.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of completed episodes to evaluate.")
    parser.add_argument("--motion_file", type=str, default=None, help="Required for tracking tasks.")
    parser.add_argument("--headless", action="store_true", help="Run without any rendering (metrics only).")
    parser.add_argument(
        "--viewer",
        type=str,
        default="auto",
        choices=["auto", "native", "viser"],
        help="Interactive viewer backend (auto picks native if DISPLAY is set).",
    )
    parser.add_argument("--video", action="store_true", help="Record the first evaluation rollout to mp4.")
    parser.add_argument("--video_length", type=int, default=1000, help="Recorded video length in steps.")
    parser.add_argument(
        "--video_res",
        type=str,
        default="720p",
        choices=["480p", "720p", "1080p"],
        help=(
            "Recording resolution. VideoRecorder buffers EVERY frame in RAM before "
            "encoding, so this sets the memory cost directly: 1080p is 6.2 MB/frame "
            "(6.2 GB for a 1000-step episode), 720p 2.8 GB, 480p 0.9 GB. 1080p renders "
            "get OOM-killed on this box whenever another job is resident."
        ),
    )
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to store recorded evaluation videos.")
    parser.add_argument(
        "--video_decim",
        type=int,
        default=None,
        help=(
            "Smooth-video render decimation. Stock --video grabs one frame per control "
            "step (e.g. 0.2s -> 5 fps, choppy). Set this to a small divisor of the env's "
            "decimation (e.g. 4 for the nav env's 40) to render every few physics ticks "
            "and hold each policy action across them -- identical control, ~50 fps output."
        ),
    )
    parser.add_argument(
        "--episode_s",
        type=float,
        default=None,
        help=(
            "Override episode_length_s (and the pose command's resampling period). Use it "
            "to restore a finite episode in play mode, which sets episode_length_s huge so "
            "interactive viewing runs forever -- e.g. --episode_s 12 for bounded episodes."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--cmd_script",
        type=str,
        default=None,
        help=(
            "Drive the velocity command from a fixed script instead of the env's random "
            "resampling, so different policies are given the IDENTICAL task (the same "
            "--seed does NOT achieve this: the command RNG interleaves with other "
            "randomized events). Format: 'STEPS:vx,vy,wz;STEPS:vx,vy,wz;...', e.g. "
            "'200:1.0,0,0;200:0,0,1.0'. The last segment repeats if the episode is longer."
        ),
    )
    parser.add_argument(
        "--dump_traj",
        type=str,
        default=None,
        help=(
            "Write a per-step CSV of commanded vs achieved base velocity to this path "
            "(velocity-tracking tasks only). Averages like error_vel_xy hide *how* a policy "
            "misses — gain, bias, lag; this exposes it. Use with --num_envs 1."
        ),
    )
    parser.add_argument("--export_onnx", action="store_true", help="Export actor to ONNX alongside the checkpoint then exit.")
    parser.add_argument(
        "--action_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic"],
        help=(
            "Action selection at eval. 'deterministic' is tanh(mu) (act_inference), what "
            "deployment normally uses. 'stochastic' draws one squashed sample from pi(.|s) "
            "— the distribution the actor is actually trained on. Comparing the two is the "
            "exploration/deployment gap; see reports/EVAL_PROTOCOL.md. Ignored when "
            "--q_argmax > 0, which defines its own selection rule."
        ),
    )
    parser.add_argument(
        "--one_episode_per_env",
        action="store_true",
        help=(
            "Take exactly one episode from each env instead of the first --episodes "
            "episodes to finish. Without this, envs that start together and finish "
            "together are truncated to --episodes, keeping only the EARLIEST finishers "
            "— which over-represents falls and discards long clean episodes. Required by "
            "protocol E1 (use --num_envs N --episodes N --one_episode_per_env)."
        ),
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Write the evaluation summary as JSON to this path (for experiments/registry.csv).",
    )
    parser.add_argument(
        "--q_argmax",
        type=int,
        default=0,
        help=(
            "REPPO only: eval-time policy improvement. Sample this many actions from the "
            "policy (plus the mode), score each with min(Q1,Q2), and execute the argmax. "
            "0 disables (use the deterministic mode). Recovers the greedy action that the "
            "max-entropy objective hides in the policy mode."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.env_id not in list_tasks():
        raise ValueError(f"Unknown env_id '{args.env_id}'. Run with one of: {', '.join(list_tasks())}")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_cfg = load_train_cfg(args.env_id, checkpoint_path, args.config)
    env_cfg: ManagerBasedRlEnvCfg = load_env_cfg(args.env_id, play=args.play)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
    if args.episode_s is not None:
        env_cfg.episode_length_s = args.episode_s
        if "pose" in env_cfg.commands:
            env_cfg.commands["pose"].resampling_time_range = (args.episode_s, args.episode_s)

    # Smooth-video: render at a finer decimation and hold each control action across
    # the extra ticks (see --video_decim). Physics/control are unchanged; we just
    # render more often. hold == original_decimation / video_decimation.
    video_hold = 1
    if args.video and args.video_decim is not None:
        orig_decim = env_cfg.decimation
        if orig_decim % args.video_decim != 0:
            raise ValueError(
                f"--video_decim {args.video_decim} must divide the env decimation {orig_decim}."
            )
        video_hold = orig_decim // args.video_decim
        env_cfg.decimation = args.video_decim
        print(f"[INFO] smooth video: decimation {orig_decim}->{args.video_decim}, "
              f"hold each action {video_hold} steps ({1.0 / (env_cfg.sim.mujoco.timestep * args.video_decim):.0f} fps)")

    is_tracking_task = "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    if is_tracking_task:
        if not args.motion_file:
            raise ValueError("Tracking tasks require --motion_file.")
        motion_path = Path(args.motion_file).expanduser().resolve()
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        env_cfg.commands["motion"].motion_file = str(motion_path)

    configure_torch_backends()

    interactive = not args.headless and not args.video
    env_render_mode = "rgb_array" if args.video else None
    if args.video:
        # Resolution drives the RAM cost of recording (see --video_res). 720p is the
        # default: legible for inspection at ~45% of 1080p's frame buffer.
        _res = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}[args.video_res]
        env_cfg.viewer.width, env_cfg.viewer.height = _res
        print(f"[INFO] recording at {_res[0]}x{_res[1]} "
              f"(~{_res[0]*_res[1]*3*args.video_length/1e9:.2f} GB of frame buffer)")

    # Match training: build a ManagerBasedSafeRlEnv (cost manager active) for
    # cfgs that carry a cost cfg, so eval reports cost/constraint metrics.
    from src.envs import build_env

    env = build_env(env_cfg, args.device, render_mode=env_render_mode)
    if getattr(env, "cost_limits", None) is not None:
        print(f"[INFO] cost_limits from cost manager: {env.cost_limits}")
    if args.video:
        default_video_dir = checkpoint_path.parent / "videos" / "eval"
        video_dir = Path(args.video_dir).expanduser().resolve() if args.video_dir else default_video_dir
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to {video_dir}")

    agent_cfg = load_rl_cfg(args.env_id)
    vec_env = make_env(env_id=args.env_id, env=env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    alg_name = train_cfg.get("algorithm", {}).get("class_name", "PPO")
    if alg_name in OFF_POLICY_ALGORITHMS:
        print(f"[INFO] Using OffPolicyRunner for algorithm: {alg_name}")
        runner = OffPolicyRunner(vec_env, train_cfg, log_dir=None, device=args.device)
    else:
        print(f"[INFO] Using OnPolicyRunner for algorithm: {alg_name}")
        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=None, device=args.device)
    runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    if args.action_mode == "stochastic":
        # get_inference_policy() hands back act_inference, i.e. tanh(mu) — the
        # deployment action. The stochastic arm instead draws one squashed sample from
        # pi(.|s), which is the distribution the actor is actually trained against.
        # Both REPPOActorCritic.act and ActorCritic.act normalize obs internally and
        # default to sampling, so the same call covers PPO and REPPO.
        _policy_module = runner.alg.policy
        policy = lambda obs: _policy_module.act(obs)  # noqa: E731
        print("[INFO] Action mode: stochastic (one sample from pi(.|s)).")
    else:
        print("[INFO] Action mode: deterministic (tanh(mu)).")

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")

    if args.export_onnx:
        runner.export_policy_to_onnx(
            path=str(checkpoint_path.parent),
            filename=checkpoint_path.stem + ".onnx",
        )
        vec_env.close()
        return

    if interactive:
        # ViserPlayViewer calls env.get_observations() which returns (tensor, extras).
        # Unwrap the tuple so the policy receives a plain tensor.
        _base_policy = policy
        def policy(obs):
            if isinstance(obs, tuple):
                obs = obs[0]
            return _base_policy(obs)

        viewer = args.viewer
        if viewer == "auto":
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            viewer = "native" if has_display else "viser"
        if viewer == "native":
            NativeMujocoViewer(vec_env, policy).run()
        else:
            ViserPlayViewer(vec_env, policy).run()
        vec_env.close()
        return

    obs, extras = vec_env.get_observations()
    obs = obs.to(runner.device)

    q_argmax = None
    if args.q_argmax > 0:
        if alg_name != "REPPO":
            raise ValueError("--q_argmax is only supported for REPPO checkpoints.")
        q_argmax = make_q_argmax_policy(runner.alg.policy, args.q_argmax)
        print(f"[INFO] Eval-time Q-argmax over {args.q_argmax} samples (+mode).")

    def get_critic_obs(default: torch.Tensor, info: dict) -> torch.Tensor:
        return info.get("observations", {}).get("critic", default).to(runner.device)

    critic_obs = get_critic_obs(obs, extras)

    ep_rewards: list[float] = []
    ep_costs: list[float] = []
    ep_lengths: list[int] = []
    # mjlab publishes command-term metrics as "Metrics/<term>/<name>" into
    # extras["log"] at reset (managers/command_manager.py). For manipulation
    # tasks the metric of record is a success rate, not reward, so collect
    # whatever the task reports rather than hard-coding a key.
    ep_metrics: list[dict[str, float]] = []
    reward_buf = torch.zeros(vec_env.num_envs, device=runner.device)
    cost_buf = torch.zeros(vec_env.num_envs, device=runner.device)
    length_buf = torch.zeros(vec_env.num_envs, dtype=torch.long, device=runner.device)

    # Velocity-tracking diagnostics.
    #
    # WARNING about mjlab's own `Metrics/twist/error_vel_xy`: it is a CUMULATIVE sum
    # divided by a fixed constant (`resampling_time_range[1] / step_dt`, = 400 steps
    # for Go2), NOT a per-step average — see mjlab/tasks/velocity/mdp/velocity_command.py
    # `_update_metrics`. So it scales with episode length: a policy that falls at step
    # 290 scores ~3.4x "better" than an identical policy that survives 1000 steps, and
    # for a full episode it reads 2.5x the true mean. It is therefore NOT comparable
    # across policies with different survival times, nor against any implementation
    # that reports a mean. We compute the length-normalized mean here alongside it.
    traj_rows: list[tuple[float, ...]] = []
    traj_cmd_term = None
    base_env = getattr(vec_env, "env", vec_env)
    base_env = getattr(base_env, "unwrapped", base_env)
    cmd_mgr = getattr(base_env, "command_manager", None)
    if cmd_mgr is not None:
        for cand in ("twist", "base_velocity"):
            if cand in list(getattr(cmd_mgr, "active_terms", [])):
                traj_cmd_term = cand
                break
    if traj_cmd_term is not None:
        traj_robot = base_env.scene["robot"]
        traj_mgr = cmd_mgr
    elif args.dump_traj:
        print("[WARN] --dump_traj: no twist/base_velocity command term; trace disabled.")
    # Scripted command schedule: expand "STEPS:vx,vy,wz;..." into a per-step table.
    cmd_script: torch.Tensor | None = None
    if args.cmd_script:
        if traj_cmd_term is None:
            raise ValueError("--cmd_script requires a twist/base_velocity command term.")
        segments = []
        for chunk in args.cmd_script.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            steps_str, vec_str = chunk.split(":")
            vec = [float(x) for x in vec_str.split(",")]
            if len(vec) != 3:
                raise ValueError(f"--cmd_script segment '{chunk}' must give exactly vx,vy,wz")
            segments.append((int(steps_str), vec))
        if not segments:
            raise ValueError("--cmd_script parsed to zero segments")
        table = []
        for n_steps, vec in segments:
            table.extend([vec] * n_steps)
        cmd_script = torch.tensor(table, dtype=torch.float32, device=runner.device)
        cmd_term_obj = traj_mgr._terms[traj_cmd_term]
        if not hasattr(cmd_term_obj, "vel_command_b"):
            raise ValueError(
                f"--cmd_script: command term '{traj_cmd_term}' has no vel_command_b buffer to override."
            )
        print(f"[INFO] scripted commands: {len(segments)} segments, {len(table)} steps")

    def apply_cmd_script(idx: int) -> None:
        """Overwrite the command buffer so every policy is given the same task.

        Runs after env.step, so a scheduled change reaches the observation one step
        late — irrelevant for segments hundreds of steps long, and identical for
        every policy being compared.
        """
        if cmd_script is None:
            return
        row = cmd_script[min(idx, len(cmd_script) - 1)]
        buf = cmd_term_obj.vel_command_b
        buf[:] = row.to(buf.device)

    track_err_sum = torch.zeros(vec_env.num_envs, device=runner.device)
    track_yaw_sum = torch.zeros(vec_env.num_envs, device=runner.device)
    ep_track_err: list[float] = []
    ep_track_yaw: list[float] = []

    # R7: without --one_episode_per_env the loop below keeps whichever episodes finish
    # first and truncates the rest, which biases the sample toward early failures
    # because every env starts at the same step. Tracking one slot per env removes the
    # bias: each env contributes exactly its FIRST completed episode, so slow/clean
    # episodes are counted rather than discarded.
    recorded = torch.zeros(vec_env.num_envs, dtype=torch.bool, device=runner.device)

    def _done_collecting() -> bool:
        if args.one_episode_per_env:
            return bool(recorded.all())
        return len(ep_rewards) >= args.episodes

    step_idx = 0
    actions = None
    while not _done_collecting():
        # Recompute the action only every video_hold steps (== 1 unless smooth video):
        # the control period is unchanged, we just render the in-between physics ticks.
        if step_idx % video_hold == 0:
            with torch.inference_mode():
                actions = q_argmax(obs, critic_obs) if q_argmax is not None else policy(obs)
        obs, rewards, dones, infos = vec_env.step(actions)
        step_idx += 1
        obs = obs.to(runner.device)
        critic_obs = get_critic_obs(obs, infos)
        rewards = rewards.to(runner.device)
        dones = dones.to(runner.device)

        costs = infos.get("costs", torch.zeros_like(rewards)).to(runner.device)
        # The cost manager emits a per-constraint vector (num_envs, num_costs);
        # collapse to a per-env total cost for the episode summary.
        if costs.dim() > 1:
            costs = costs.sum(dim=-1)
        reward_buf += rewards
        cost_buf += costs
        length_buf += 1

        if traj_cmd_term is not None:
            with torch.inference_mode():
                apply_cmd_script(step_idx - 1)
                cmd_all = traj_mgr.get_command(traj_cmd_term)
                lin_all = traj_robot.data.root_link_lin_vel_b
                ang_all = traj_robot.data.root_link_ang_vel_b
                track_err_sum += torch.norm(cmd_all[:, :2] - lin_all[:, :2], dim=-1).to(runner.device)
                track_yaw_sum += torch.abs(cmd_all[:, 2] - ang_all[:, 2]).to(runner.device)
                if args.dump_traj:
                    # Actions are also recorded, per dimension, so the same trace
                    # supports action-distribution comparisons across policies.
                    act0 = actions[0].detach().float().cpu().tolist()
                    traj_rows.append(
                        (
                            float(step_idx),
                            float(cmd_all[0][0]), float(cmd_all[0][1]), float(cmd_all[0][2]),
                            float(lin_all[0][0]), float(lin_all[0][1]), float(ang_all[0][2]),
                            *[float(a) for a in act0],
                        )
                    )

        runner.alg.policy.reset(dones=dones)

        # Reset the per-env accumulators for EVERY finished env, but only *record*
        # the ones this protocol still wants — otherwise a second episode from a
        # fast-failing env would leak into the sample (see filter_recordable).
        done_ids, reset_ids = filter_recordable(
            (dones > 0).nonzero(as_tuple=False).squeeze(-1), recorded, args.one_episode_per_env
        )
        if done_ids.numel() == 0:
            if reset_ids.numel() > 0:
                reward_buf[reset_ids] = 0.0
                cost_buf[reset_ids] = 0.0
                length_buf[reset_ids] = 0
                if traj_cmd_term is not None:
                    track_err_sum[reset_ids] = 0.0
                    track_yaw_sum[reset_ids] = 0.0
            continue

        ep_rewards.extend(reward_buf[done_ids].detach().cpu().tolist())
        ep_costs.extend(cost_buf[done_ids].detach().cpu().tolist())
        ep_lengths.extend(length_buf[done_ids].detach().cpu().tolist())
        if traj_cmd_term is not None:
            steps = length_buf[done_ids].clamp_min(1).float()
            ep_track_err.extend((track_err_sum[done_ids] / steps).detach().cpu().tolist())
            ep_track_yaw.extend((track_yaw_sum[done_ids] / steps).detach().cpu().tolist())
            track_err_sum[reset_ids] = 0.0
            track_yaw_sum[reset_ids] = 0.0

        log = infos.get("log") or infos.get("episode") or {}
        if isinstance(log, dict):
            batch = {}
            for key, value in log.items():
                if not key.startswith("Metrics/"):
                    continue
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    value = value.item()
                if isinstance(value, (int, float)):
                    batch[key] = float(value)
            if batch:
                ep_metrics.append(batch)

        reward_buf[reset_ids] = 0.0
        cost_buf[reset_ids] = 0.0
        length_buf[reset_ids] = 0

    if not args.one_episode_per_env:
        # Legacy path only. Under E1 (--one_episode_per_env) every collected episode is
        # kept: the sample is already exactly one per env, and truncating it here would
        # reintroduce the earliest-finisher bias this flag exists to remove.
        ep_rewards = ep_rewards[: args.episodes]
        ep_costs = ep_costs[: args.episodes]
        ep_lengths = ep_lengths[: args.episodes]
        ep_track_err = ep_track_err[: args.episodes]
        ep_track_yaw = ep_track_yaw[: args.episodes]
    print(f"Evaluation over {len(ep_rewards)} episodes")
    print(f"Mean reward: {sum(ep_rewards) / len(ep_rewards):.3f}")
    print(f"Mean cost: {sum(ep_costs) / len(ep_costs):.3f}")
    print(f"Mean length: {sum(ep_lengths) / len(ep_lengths):.1f}")
    if ep_track_err:
        n = len(ep_track_err)
        print(
            f"Mean per-step |v_cmd - v_xy|: {sum(ep_track_err) / n:.4f}   "
            f"(length-normalized; USE THIS to compare policies)"
        )
        print(f"Mean per-step |w_cmd - w_z| : {sum(ep_track_yaw) / n:.4f}")
    # Survival = reached the episode cap. mjlab truncates at max_episode_length, so a
    # shorter episode means a termination condition fired (for Go2: a fall). Reported
    # alongside tracking because a policy that falls sees a DIFFERENT command sequence
    # than one that survives (see reports/EVAL_PROTOCOL.md §8.1) — tracking error is
    # only comparable between arms with comparable survival.
    max_len = int(getattr(vec_env, "max_episode_length", 0) or 0)
    survived = [ln for ln in ep_lengths if max_len and ln >= max_len]
    survival_rate = len(survived) / len(ep_lengths) if ep_lengths and max_len else float("nan")
    print(f"Survival rate (reached {max_len} steps): {survival_rate:.3f}")
    print(f"Fall rate: {1.0 - survival_rate:.3f}")

    metric_means: dict[str, float] = {}
    if ep_metrics:
        keys = sorted({k for batch in ep_metrics for k in batch})
        for key in keys:
            values = [batch[key] for batch in ep_metrics if key in batch]
            metric_means[key] = sum(values) / len(values)
            print(f"Mean {key}: {metric_means[key]:.4f}")

    if args.json_out:
        n_tr = len(ep_track_err)
        summary = {
            "checkpoint": str(checkpoint_path),
            "env_id": args.env_id,
            "algorithm": alg_name,
            "config": args.config,
            "seed": args.seed,
            "num_envs": vec_env.num_envs,
            "episodes_requested": args.episodes,
            "episodes_collected": len(ep_rewards),
            "one_episode_per_env": bool(args.one_episode_per_env),
            "action_mode": "q_argmax" if args.q_argmax > 0 else args.action_mode,
            "q_argmax_samples": args.q_argmax,
            "max_episode_length": max_len,
            "mean_reward": sum(ep_rewards) / len(ep_rewards),
            "mean_cost": sum(ep_costs) / len(ep_costs),
            "mean_length": sum(ep_lengths) / len(ep_lengths),
            "survival_rate": survival_rate,
            "tracking_error_xy": (sum(ep_track_err) / n_tr) if n_tr else None,
            "yaw_error": (sum(ep_track_yaw) / n_tr) if n_tr else None,
            "mjlab_metrics": metric_means,
            # Per-episode raw values: the program requires per-seed/per-episode raw data
            # to be stored, not just aggregates (bootstrap CIs, worst-case, IQM).
            "per_episode": {
                "reward": ep_rewards,
                "cost": ep_costs,
                "length": ep_lengths,
                "tracking_error_xy": ep_track_err,
                "yaw_error": ep_track_yaw,
            },
        }
        json_path = Path(args.json_out).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"[INFO] wrote evaluation summary to {json_path}")

    if args.dump_traj and traj_rows:
        out_path = Path(args.dump_traj).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            n_act = max(len(row) for row in traj_rows) - 7
            cols = ["step", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz"]
            cols += [f"a{i}" for i in range(n_act)]
            handle.write(",".join(cols) + "\n")
            for row in traj_rows:
                handle.write(",".join(f"{v:.6f}" for v in row) + "\n")
        print(f"[INFO] wrote {len(traj_rows)} trajectory rows to {out_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
