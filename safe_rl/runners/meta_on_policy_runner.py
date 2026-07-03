from __future__ import annotations

import copy
import os
import random
import statistics
import time
import torch
import torch.nn as nn
import torch.optim as optim

from safe_rl.algorithms.p3o import P3O
from safe_rl.runners.on_policy_runner import OnPolicyRunner
from safe_rl.utils import store_code_state


class MetaOnPolicyRunner(OnPolicyRunner):
    """Constrained MAML (cMAML) runner — Reptile outer loop over a P3O inner loop.

    Implements the framework of "Constrained Model-Agnostic Meta Learning":
    a meta-policy theta is adapted to each sampled task (= env seed) by running
    K inner steps of a constrained policy-optimization algorithm (here P3O,
    treated as a black box), and the meta-policy is then moved toward the mean
    of the adapted parameters (Reptile meta-gradient).

    Following the thesis (sec. 7.4) the three parameter groups are handled
    differently because the cost function is task-independent while the reward is
    not:
      - actor: meta-learned via Reptile interpolation;
      - cost critic: a *global* critic trained on-policy on the meta-policy's
        rollouts pooled across all tasks, then used as the per-task starting
        point. This is what lets one critic predict cost-to-go across the whole
        task distribution;
      - reward critic: not meta-learned — re-initialized per task.

    The optional eta meta-safety dual (sec. 7.5) constrains the meta-policy's own
    cost; for now eta is tracked, dual-ascended and logged (see ``_eta_update``).
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        if not isinstance(self.alg, P3O):
            raise ValueError(
                f"MetaOnPolicyRunner currently supports only P3O as the inner-loop algorithm, "
                f"got {type(self.alg).__name__}."
            )

        meta_cfg = train_cfg.get("meta", {}) or {}
        self.inner_steps = int(meta_cfg.get("inner_steps", 3))
        self.num_tasks = int(meta_cfg.get("num_tasks", 20))
        self.meta_lr = float(meta_cfg.get("meta_lr", 0.5))
        # ANIL-style feature reuse (slow trunk / fast head). The Reptile outer step
        # uses meta_lr for the actor trunk and meta_lr_head for the output head, so
        # the shared feature extractor evolves slowly while the head adapts fast.
        # Defaults to meta_lr -> identical to the old single-rate Reptile update.
        self.meta_lr_head = float(meta_cfg.get("meta_lr_head", self.meta_lr))
        self.seed_pool = list(meta_cfg.get("seed_pool", list(range(self.num_tasks))))

        # Global cost-critic training (sec. 7.4)
        self.cost_critic_epochs = int(meta_cfg.get("cost_critic_epochs", 8))
        self.cost_critic_batch_size = int(meta_cfg.get("cost_critic_batch_size", 4096))

        # eta meta-safety dual (sec. 7.5)
        self.eta_adaptive = bool(meta_cfg.get("eta_adaptive", False))
        self.eta = float(meta_cfg.get("eta_init", 0.0))
        self.eta_lr = float(meta_cfg.get("eta_lr", 0.05))
        self.eta_cost_limit = float(meta_cfg.get("cost_limit", self.env.cost_limits[0]))

        # eta-penalized meta step (sec. 7.5): a cost-reducing gradient step on the
        # meta-actor, weighted by eta, applied after the Reptile reward update. This
        # is what makes the *initialization* safe (not just fast-adapting).
        self.eta_penalized = bool(meta_cfg.get("eta_penalized", False))
        self.meta_cost_lr = float(meta_cfg.get("meta_cost_lr", 5e-4))
        self.meta_cost_epochs = int(meta_cfg.get("meta_cost_epochs", 4))
        self.meta_cost_clip = float(meta_cfg.get("meta_cost_clip", 0.2))
        self.meta_cost_target_kl = float(meta_cfg.get("meta_cost_target_kl", 0.02))
        self.meta_cost_batch_size = int(meta_cfg.get("meta_cost_batch_size", self.cost_critic_batch_size))
        # Fix A: keep the exploration std/log_std out of the cost step so it reshapes
        # only the action mean and never collapses the noise the inner loop needs.
        self.meta_cost_protect_std = bool(meta_cfg.get("meta_cost_protect_std", True))
        # eta deadband: only run the cost step while the meta-policy is actually unsafe
        # (meta_policy_cost > limit). The cost surrogate uses A_C with eta>=0, so it
        # always pushes cost down even once safe (no reward counterweight) and drags the
        # init to the do-nothing policy; gating it on the violation restores the proper
        # "only constrain when violating" Lagrangian behavior.
        self.eta_deadband = bool(meta_cfg.get("eta_deadband", True))
        # Built lazily on first use (needs the live actor params after a policy load).
        self.meta_cost_optimizer = None

        policy = self.alg.policy

        # Partition parameters by role (see class docstring).
        self.meta_actor_params: dict[str, torch.Tensor] = {}
        self.meta_cost_params: dict[str, torch.Tensor] = {}
        self.value_init_params: dict[str, torch.Tensor] = {}
        for name, p in policy.named_parameters():
            if name.startswith("critics"):
                self.value_init_params[name] = p.detach().clone()
            elif name.startswith("cost_critic"):
                self.meta_cost_params[name] = p.detach().clone()
            else:
                self.meta_actor_params[name] = p.detach().clone()

        # Output-head params of the actor (for the slow-trunk/fast-head Reptile step).
        self.head_param_names = self._identify_head_param_names()
        if self.meta_lr_head != self.meta_lr:
            print(
                f"[cMAML] slow-trunk/fast-head Reptile: meta_lr(trunk)={self.meta_lr} "
                f"meta_lr_head={self.meta_lr_head}; head params={sorted(self.head_param_names)}"
            )

        # Persistent optimizer for the global cost critic (momentum carries across
        # meta iterations). Separate from P3O's own optimizer, which is reset per
        # task; this one trains only the shared cost critic on meta-policy data.
        self.cost_critic_loss_type = getattr(policy, "cost_critic_loss_type", None)
        if self.alg.cost_critic_params:
            self.cost_optimizer = optim.Adam(self.alg.cost_critic_params, lr=self.alg.cost_critic_lr)
        else:
            self.cost_optimizer = None

        # Snapshots to reset inner-loop state between tasks so nothing leaks across
        # tasks (optimizer momentum, adaptive LR, penalty factor kappa).
        self._init_opt_state = copy.deepcopy(self.alg.optimizer.state_dict())
        self._init_kappa = list(getattr(self.alg, "kappa", []))
        self._init_lr = float(getattr(self.alg, "learning_rate"))

    # ----- inner/outer-loop helpers -------------------------------------------------

    def _load_meta_into_policy(self) -> None:
        """Reset the live policy: actor + cost critic to meta-params, reward critic
        to its (non-meta-learned) init snapshot."""
        with torch.no_grad():
            for name, p in self.alg.policy.named_parameters():
                if name in self.meta_actor_params:
                    p.copy_(self.meta_actor_params[name])
                elif name in self.meta_cost_params:
                    p.copy_(self.meta_cost_params[name])
                elif name in self.value_init_params:
                    p.copy_(self.value_init_params[name])

    def _reset_inner_state(self) -> None:
        """Reset P3O's inner optimizer/penalty state to its construction defaults."""
        self.alg.optimizer.load_state_dict(copy.deepcopy(self._init_opt_state))
        if self._init_kappa:
            self.alg.kappa = list(self._init_kappa)
        self.alg.learning_rate = self._init_lr
        self.alg.optimizer.param_groups[0]["lr"] = self._init_lr
        if hasattr(self.alg, "storage"):
            self.alg.storage.clear()

    def _capture_adapted_actor(self) -> dict[str, torch.Tensor]:
        return {
            name: p.detach().clone()
            for name, p in self.alg.policy.named_parameters()
            if name in self.meta_actor_params
        }

    def _identify_head_param_names(self) -> set[str]:
        """Names (within meta_actor_params) of the actor's output head.

        ANIL's "feature reuse" insight only needs the final policy layer to adapt
        fast. The head is the actor Linear(s) whose output width is the action dim
        (the mean head, and a log-std head if present), plus the state-independent
        std/log_std parameter. Detected structurally (not by hardcoded index) so it
        survives changes to depth or LayerNorm. Returns the empty set if nothing
        matches, in which case the Reptile step degrades to the single-rate update.
        """
        num_actions = int(getattr(self.env, "num_actions"))
        head: set[str] = set()
        for mod_name, module in self.alg.policy.named_modules():
            if (
                isinstance(module, nn.Linear)
                and module.out_features == num_actions
                and mod_name.startswith("actor")
            ):
                head.add(f"{mod_name}.weight")
                head.add(f"{mod_name}.bias")
        for name in self.meta_actor_params:
            if name.split(".")[-1] in ("std", "log_std"):
                head.add(name)
        return head & set(self.meta_actor_params.keys())

    def _reptile_meta_update(self, adapted_actors: list[dict[str, torch.Tensor]]) -> None:
        with torch.no_grad():
            for name, meta_p in self.meta_actor_params.items():
                mean_adapted = torch.stack([a[name] for a in adapted_actors], dim=0).mean(dim=0)
                beta = self.meta_lr_head if name in self.head_param_names else self.meta_lr
                meta_p.add_(beta * (mean_adapted - meta_p))

    @staticmethod
    def _explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
        """1 - Var(target - pred) / Var(target). 1.0 = perfect, 0.0 = predicts the
        mean, <0 = worse than the mean. Scale-free, so it is comparable across
        meta-iterations even as the cost-return magnitude drifts."""
        var_t = target.var()
        if var_t <= 0:
            return 0.0
        return float(1.0 - (target - pred).var() / (var_t + 1e-8))

    def _train_global_cost_critic(self, obs_cat: torch.Tensor, ret_cat: torch.Tensor) -> dict[str, float]:
        """Train the shared cost critic on the meta-policy's pooled rollouts.

        Starts from the current meta cost critic, runs several epochs of cost
        regression over data collected from pi_theta on every task this iteration,
        then writes the result back as the new meta cost critic.

        Returns diagnostics including ``pre_ev`` — the explained variance of the
        *incoming* meta critic on this iteration's fresh cross-task data, i.e. how
        well the global critic generalizes before it sees the new data. A pre_ev
        that climbs over meta-iterations is the signal that the critic is becoming
        a good global cost predictor.
        """
        if self.cost_optimizer is None or obs_cat.shape[0] == 0:
            return {"loss": 0.0, "pre_ev": 0.0, "post_ev": 0.0, "pre_rmse": 0.0, "target_mean": 0.0}

        self._load_meta_into_policy()  # cost critic <- current meta cost critic
        policy = self.alg.policy
        is_hlgauss = self.cost_critic_loss_type == "hlgauss"
        n = obs_cat.shape[0]
        bs = min(self.cost_critic_batch_size, n)

        # Generalization diagnostic: how well the incoming meta critic predicts
        # this iteration's fresh, never-trained-on cross-task data.
        with torch.no_grad():
            pre_pred = policy.evaluate_cost(obs_cat)
            pre_ev = self._explained_variance(pre_pred, ret_cat)
            pre_rmse = float((pre_pred - ret_cat).pow(2).mean().sqrt().item())

        total, count = 0.0, 0
        for _ in range(self.cost_critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                ob, tgt = obs_cat[idx], ret_cat[idx]
                obs_n = policy.critic_obs_normalizer(ob)
                if is_hlgauss:
                    logits = policy.cost_critic(obs_n)
                    loss = policy.cost_critic.loss(logits, tgt).mean()
                else:
                    loss = (policy.cost_critic(obs_n) - tgt).pow(2).mean()
                self.cost_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.alg.cost_critic_params, self.alg.max_grad_norm)
                self.cost_optimizer.step()
                total += loss.item()
                count += 1

        with torch.no_grad():
            post_pred = policy.evaluate_cost(obs_cat)
            post_ev = self._explained_variance(post_pred, ret_cat)
            for name, p in policy.named_parameters():
                if name in self.meta_cost_params:
                    self.meta_cost_params[name].copy_(p.detach())

        return {
            "loss": total / max(count, 1),
            "pre_ev": pre_ev,
            "post_ev": post_ev,
            "pre_rmse": pre_rmse,
            "target_mean": float(ret_cat.mean().item()),
        }

    def _meta_actor_std_mean(self) -> float:
        """Mean action std of the meta-actor (diagnostic: detects entropy collapse
        from the cost step). Handles scalar-std and log-std parameterizations."""
        for name, p in self.meta_actor_params.items():
            leaf = name.split(".")[-1]
            if leaf == "std":
                return float(p.mean().item())
            if leaf == "log_std":
                return float(p.exp().mean().item())
        return float("nan")

    def _eta_step_params(self) -> list[torch.Tensor]:
        """Live actor tensors the eta cost step is allowed to update.

        With ``meta_cost_protect_std`` (Fix A) the exploration std/log_std is
        excluded, so the cost step reshapes only the action mean and never
        collapses the exploration noise the inner-loop adaptation relies on.
        """
        params = []
        for name, p in self.alg.policy.named_parameters():
            if name not in self.meta_actor_params:
                continue
            if self.meta_cost_protect_std and name.split(".")[-1] in ("std", "log_std"):
                continue
            params.append(p)
        return params

    def _eta_penalized_meta_step(self, md: dict[str, torch.Tensor]) -> dict[str, float]:
        """Cost-reducing gradient step on the meta-actor, weighted by eta (sec. 7.5).

        After Reptile has pulled theta toward the reward-adapted actors, push it back
        down in cost: minimize ``eta * E[ A_C * ratio ]`` on the meta-policy's own
        pooled rollouts, where ``A_C`` is the cost advantage baselined by the freshly
        trained global cost critic. eta self-regulates the pressure — it grows while
        theta violates the limit and decays to ~0 once theta is safe — so this step
        vanishes when no longer needed. Clipped surrogate + grad-norm + a KL early-stop
        keep it from destroying the reward initialization (eta can be large).
        """
        zero = {"surrogate": 0.0, "kl": 0.0, "ratio_max": 1.0, "adv_cost_mean": 0.0, "epochs": 0.0}
        if self.eta <= 0.0 or md is None or md["actor_obs"].shape[0] == 0:
            return zero

        self._load_meta_into_policy()  # actor <- current meta actor
        policy = self.alg.policy
        if self.meta_cost_optimizer is None:
            self.meta_cost_optimizer = optim.Adam(self._eta_step_params(), lr=self.meta_cost_lr)

        actor_obs, actions = md["actor_obs"], md["actions"]
        old_logp = md["old_logp"].squeeze(-1)
        old_mu, old_sigma = md["mu"], md["sigma"]

        # Cost advantage with the updated global critic as baseline. Scale by std only
        # (never mean-center — the sign of A_C is what marks a sample as unsafe; see P3O).
        with torch.no_grad():
            cost_val = policy.evaluate_cost(md["critic_obs"]).reshape(md["cost_ret"].shape)
            adv_c = (md["cost_ret"] - cost_val).squeeze(-1)
            adv_c = adv_c / (adv_c.std() + 1e-8)
            adv_cost_mean = float(adv_c.mean().item())

        n = actor_obs.shape[0]
        bs = min(self.meta_cost_batch_size, n)
        actor_params = self._eta_step_params()
        last_kl, ratio_max, surr_acc, n_steps, epochs_done = 0.0, 1.0, 0.0, 0, 0
        stop = False
        for _ in range(self.meta_cost_epochs):
            if stop:
                break
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                policy.act(actor_obs[idx])
                logp = policy.get_actions_log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv_c[idx]
                # Pessimistic (upper-bound) clip for a term we MINIMIZE — the cost
                # analogue of P3O's reward surrogate clip.
                surr = torch.max(a * ratio, a * torch.clamp(ratio, 1.0 - self.meta_cost_clip,
                                                            1.0 + self.meta_cost_clip)).mean()
                # Normalized-Lagrangian weight eta/(1+eta) in [0, 1): the cost step can
                # shape the meta-actor but never dominate Reptile's reward gains (raw eta
                # drove cost->0 AND reward->0). Self-balancing: ->0 when safe, ->1 when eta large.
                loss = (self.eta / (1.0 + self.eta)) * surr
                self.meta_cost_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, self.alg.max_grad_norm)
                self.meta_cost_optimizer.step()

                with torch.no_grad():
                    mu, sigma = policy.action_mean, policy.action_std
                    kl = torch.sum(
                        torch.log(sigma / old_sigma[idx] + 1e-5)
                        + (old_sigma[idx].pow(2) + (old_mu[idx] - mu).pow(2)) / (2.0 * sigma.pow(2))
                        - 0.5, dim=-1).mean()
                    last_kl = float(kl.item())
                    ratio_max = max(ratio_max, float(ratio.max().item()))
                    surr_acc += float(surr.item())
                    n_steps += 1
                if last_kl > self.meta_cost_target_kl:
                    stop = True
                    break
            epochs_done += 1

        # Persist the cost-adjusted actor as the new meta-actor.
        with torch.no_grad():
            for name, p in policy.named_parameters():
                if name in self.meta_actor_params:
                    self.meta_actor_params[name].copy_(p.detach())

        return {
            "surrogate": surr_acc / max(n_steps, 1),
            "kl": last_kl,
            "ratio_max": ratio_max,
            "adv_cost_mean": adv_cost_mean,
            "epochs": float(epochs_done),
        }

    def _eta_update(self, meta_policy_costs: list[float]) -> None:
        # Dual ascent on the meta-policy's own cost constraint (sec. 7.5). eta is the
        # weight used by _eta_penalized_meta_step: it grows while the meta-policy
        # violates the limit and decays toward 0 as it becomes safe.
        violation = statistics.mean(meta_policy_costs) - self.eta_cost_limit
        self.eta = max(0.0, self.eta + self.eta_lr * violation)

    def _sample_tasks(self) -> list[int]:
        # With replacement, matching the thesis (App. A).
        return random.choices(self.seed_pool, k=self.num_tasks)

    def _collect_and_adapt_task(self, it: int):
        """Run K inner P3O updates on the current task.

        Returns (last_reward, last_cost, meta_policy_cost, meta_policy_reward,
        meta_data). ``meta_policy_reward`` is the meta-policy's pre-adaptation
        reward (k==0) — the diagnostic for whether the init itself is drifting to a
        low-reward basin. ``meta_data`` holds the meta-policy's first-rollout tensors (critic_obs,
        cost_ret, actor_obs, actions, old_logp, mu, sigma) used to train the global
        cost critic (sec. 7.4) and drive the eta-penalized meta step (sec. 7.5).
        """
        num_costs = len(self.env.cost_limits) if hasattr(self.env, "cost_limits") else 1
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)

        last_reward = float("nan")
        last_cost = float("nan")
        meta_policy_cost: float | None = None
        meta_policy_reward: float | None = None
        meta_data: dict[str, torch.Tensor] | None = None

        for k in range(self.inner_steps):
            rew_eps: list[float] = []
            cost_eps: list[float] = []
            cur_r = torch.zeros(self.env.num_envs, device=self.device)
            cur_c = torch.zeros(self.env.num_envs, device=self.device)

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    costs = infos.get("costs", infos.get("cost", torch.zeros_like(rewards))).to(self.device)
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    self.alg.process_env_step(rewards, costs, dones, infos)

                    cur_r += rewards
                    cur_c += costs if costs.dim() == 1 else costs.sum(dim=-1)
                    new_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                    if len(new_ids) > 0:
                        rew_eps.extend(cur_r[new_ids].cpu().numpy().tolist())
                        cost_eps.extend(cur_c[new_ids].cpu().numpy().tolist())
                        cur_r[new_ids] = 0
                        cur_c[new_ids] = 0

                self.alg.compute_returns(privileged_obs)
                self.alg.compute_cost_returns(privileged_obs)

                # Snapshot the meta-policy's rollout (k==0, before any adaptation).
                # critic_obs/cost_ret train the global cost critic (sec. 7.4);
                # actor_obs/actions/old_logp/mu/sigma drive the eta-penalized cost
                # policy-gradient step on the meta-actor (sec. 7.5).
                if k == 0:
                    storage = self.alg.storage
                    critic_src = storage.privileged_observations
                    if critic_src is None:
                        critic_src = storage.observations
                    meta_data = {
                        "critic_obs": critic_src.flatten(0, 1).detach().clone(),
                        "cost_ret": storage.cost_returns.flatten(0, 1).detach().clone(),
                        "actor_obs": storage.observations.flatten(0, 1).detach().clone(),
                        "actions": storage.actions.flatten(0, 1).detach().clone(),
                        "old_logp": storage.actions_log_prob.flatten(0, 1).detach().clone(),
                        "mu": storage.mu.flatten(0, 1).detach().clone(),
                        "sigma": storage.sigma.flatten(0, 1).detach().clone(),
                    }

            # Pass mean episode cost so P3O's penalty factor adapts (single-cost case).
            current_costs = [statistics.mean(cost_eps)] if (cost_eps and num_costs == 1) else None
            self.alg.update(current_costs=current_costs, iteration=it)

            if k == 0 and cost_eps:
                meta_policy_cost = statistics.mean(cost_eps)
            if k == 0 and rew_eps:
                meta_policy_reward = statistics.mean(rew_eps)
            if rew_eps:
                last_reward = statistics.mean(rew_eps)
            if cost_eps:
                last_cost = statistics.mean(cost_eps)

        return last_reward, last_cost, meta_policy_cost, meta_policy_reward, meta_data

    # ----- main loop ----------------------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "wandb":
                from safe_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'wandb' or 'tensorboard'.")

        self.train_mode()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            tasks = self._sample_tasks()
            adapted_actors: list[dict[str, torch.Tensor]] = []
            task_rewards: list[float] = []
            task_costs: list[float] = []
            meta_policy_costs: list[float] = []
            meta_policy_rewards: list[float] = []
            meta_batches: list[dict[str, torch.Tensor]] = []

            for task in tasks:
                self._load_meta_into_policy()
                self._reset_inner_state()
                self.env.set_task(task)
                self.env.reset()

                r, c, mc, mr, md = self._collect_and_adapt_task(it)
                adapted_actors.append(self._capture_adapted_actor())
                if r == r:  # not nan
                    task_rewards.append(r)
                if c == c:
                    task_costs.append(c)
                if mc is not None:
                    meta_policy_costs.append(mc)
                if mr is not None:
                    meta_policy_rewards.append(mr)
                if md is not None:
                    meta_batches.append(md)

            # Pool the meta-policy's first-rollout data across tasks.
            pooled = {k: torch.cat([b[k] for b in meta_batches], dim=0) for k in meta_batches[0]} \
                if meta_batches else None

            # Outer updates: actor via Reptile, cost critic via global on-policy training,
            # then the eta-penalized cost step that makes the meta-actor itself safer.
            self._reptile_meta_update(adapted_actors)
            cc_diag = {"loss": 0.0, "pre_ev": 0.0, "post_ev": 0.0, "pre_rmse": 0.0, "target_mean": 0.0}
            if pooled is not None:
                cc_diag = self._train_global_cost_critic(pooled["critic_obs"], pooled["cost_ret"])
            mc_diag = {"surrogate": 0.0, "kl": 0.0, "ratio_max": 1.0, "adv_cost_mean": 0.0,
                       "epochs": 0.0, "fired": 0.0}
            mean_meta_cost = statistics.mean(meta_policy_costs) if meta_policy_costs else None
            # Deadband gate: fire the cost step only while the init is unsafe. When the
            # cost can't be measured (no completed episodes) skip it — don't erode reward
            # blindly. With eta_deadband off, the step fires every iter (old behavior).
            unsafe = (not self.eta_deadband) or (
                mean_meta_cost is not None and mean_meta_cost > self.eta_cost_limit
            )
            if self.eta_penalized and pooled is not None and unsafe:
                mc_diag = self._eta_penalized_meta_step(pooled)
                mc_diag["fired"] = 1.0
            if self.eta_adaptive and meta_policy_costs:
                self._eta_update(meta_policy_costs)
            self._load_meta_into_policy()  # leave the live policy at the meta-parameters

            self.current_learning_iteration = it
            iter_time = time.time() - start

            if self.log_dir is not None and not self.disable_logs:
                self._log_meta(
                    it, tot_iter, task_rewards, task_costs, meta_policy_costs,
                    meta_policy_rewards, cc_diag, mc_diag, iter_time,
                )
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type == "wandb" and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log_meta(self, it, tot_iter, task_rewards, task_costs, meta_policy_costs,
                  meta_policy_rewards, cc_diag, mc_diag, iter_time):
        mean_r = statistics.mean(task_rewards) if task_rewards else float("nan")
        mean_c = statistics.mean(task_costs) if task_costs else float("nan")
        mean_mc = statistics.mean(meta_policy_costs) if meta_policy_costs else float("nan")
        mean_mr = statistics.mean(meta_policy_rewards) if meta_policy_rewards else float("nan")
        actor_std = self._meta_actor_std_mean()
        mean_kappa = float(torch.tensor(self.alg.kappa).mean().item()) if self.alg.kappa else 0.0

        if self.writer is not None:
            self.writer.add_scalar("Meta/adapted_task_reward", mean_r, it)
            self.writer.add_scalar("Meta/adapted_task_cost", mean_c, it)
            self.writer.add_scalar("Meta/meta_policy_cost", mean_mc, it)
            self.writer.add_scalar("Meta/meta_policy_reward", mean_mr, it)
            self.writer.add_scalar("Meta/actor_std", actor_std, it)
            self.writer.add_scalar("Meta/global_cost_critic_loss", cc_diag["loss"], it)
            self.writer.add_scalar("Meta/cost_critic_pre_ev", cc_diag["pre_ev"], it)
            self.writer.add_scalar("Meta/cost_critic_post_ev", cc_diag["post_ev"], it)
            self.writer.add_scalar("Meta/cost_critic_pre_rmse", cc_diag["pre_rmse"], it)
            self.writer.add_scalar("Meta/cost_target_mean", cc_diag["target_mean"], it)
            self.writer.add_scalar("Meta/eta", self.eta, it)
            self.writer.add_scalar("Meta/kappa", mean_kappa, it)
            self.writer.add_scalar("Meta/cost_limit", self.eta_cost_limit, it)
            self.writer.add_scalar("Meta/meta_cost_surrogate", mc_diag["surrogate"], it)
            self.writer.add_scalar("Meta/meta_cost_kl", mc_diag["kl"], it)
            self.writer.add_scalar("Meta/meta_cost_ratio_max", mc_diag["ratio_max"], it)
            self.writer.add_scalar("Meta/meta_cost_adv_mean", mc_diag["adv_cost_mean"], it)
            self.writer.add_scalar("Meta/cost_step_fired", mc_diag.get("fired", 0.0), it)

        print(
            f"[cMAML it {it}/{tot_iter}] tasks={self.num_tasks} inner={self.inner_steps} "
            f"adapted_reward={mean_r:.3f} adapted_cost={mean_c:.3f} meta_policy_cost={mean_mc:.3f} "
            f"meta_policy_reward={mean_mr:.3f} actor_std={actor_std:.3f} "
            f"cc_loss={cc_diag['loss']:.3f} cc_pre_ev={cc_diag['pre_ev']:.3f} "
            f"cc_post_ev={cc_diag['post_ev']:.3f} eta={self.eta:.3f} kappa={mean_kappa:.3f} "
            f"mc_surr={mc_diag['surrogate']:.3f} mc_kl={mc_diag['kl']:.3f} "
            f"fired={mc_diag.get('fired', 0.0):.0f} "
            f"({iter_time:.1f}s)"
        )
