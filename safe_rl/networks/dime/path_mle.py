"""Path-space MLE helpers for MPO with a DIME diffusion actor (safe_rl addition).

These live BESIDE the vendored reference integrators (`integrators.py`), never
inside them: `tests/test_dime_reference_parity.py` pins those files to the TruDi
reference bit-for-bit.

MPO's M-step needs `log pi_theta(a|s)` at actions drawn from the frozen old
policy — intractable for a diffusion actor's action marginal. The tractable
substitute is the *path* log-likelihood: roll the old actor's SDE chain and keep
the whole trajectory, then evaluate the fixed trajectory's per-step Gaussian
transition log-probs under the new chain. `sde_rollout_with_traj` mirrors the
forward-kernel arithmetic of `sde_integrator` (integrators.py) minus the
backward/log_w half; `path_logprob_and_kl` additionally accumulates the same
per-step KL(old‖new) that `logratio` uses for the trust region, so one
control-net forward per step serves both the MLE and the KL.
"""

from __future__ import annotations

import torch

from safe_rl.networks.dime.utils import log_prob_kernel

_KL_FORMS = ("simplified", "full")


def _step_kernel(diffusion_model, obs, x, step):
    """One forward-kernel evaluation: mean and per-dim noise scale at ``x``.

    Same arithmetic as ``sde_integrator``'s forward half: ``eta = dt/friction``,
    ``scale = sqrt(2*eta)``, ``mean = x + eta*(prior_score + control_net)``.
    """
    dt = diffusion_model.delta_t_fn(step)
    eta = dt / diffusion_model.friction_fn(step)
    scale = torch.sqrt(2 * eta)
    drift = diffusion_model.drift_fn(step, x)
    fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
    return fwd_mean, scale


def sde_rollout_with_traj(diffusion_model, obs):
    """Roll the SDE chain, keeping the full trajectory and kernel parameters.

    obs : [M, obs_dim]. Returns ``(traj, fwd_means, scales)`` with shapes
    ``[T+1, M, A]``, ``[T, M, A]``, ``[T, A]`` (per-dim noise scale per step).
    Gradient handling is the caller's job (run under ``torch.no_grad()`` for the
    frozen old policy).
    """
    bs = obs.shape[0]
    x = diffusion_model.prior_sampler(bs, stop_grad=True, device=obs.device)
    traj, fwd_means, scales = [x], [], []
    for step in torch.arange(0, diffusion_model.diff_steps, dtype=torch.float32):
        fwd_mean, scale = _step_kernel(diffusion_model, obs, x, step)
        x = fwd_mean + scale * torch.randn_like(fwd_mean)
        traj.append(x)
        fwd_means.append(fwd_mean)
        scales.append(scale.expand(diffusion_model.action_dim))
    return torch.stack(traj), torch.stack(fwd_means), torch.stack(scales)


def path_logprob_and_kl(diffusion_model, obs, traj, old_means, old_scales, kl_form="simplified"):
    """Path log-likelihood of a fixed trajectory + per-step KL(old‖new).

    diffusion_model : the NEW policy's ``DiffusionModel`` (gradients flow only
        through it).
    obs / traj / old_means / old_scales : as produced by ``sde_rollout_with_traj``
        on the OLD policy (already detached).
    kl_form : "simplified" charges only the squared mean difference over
        ``2*sigma_old^2`` (reference-exact, matches ``logratio``); "full" is the
        complete Gaussian KL including the log-variance terms — required when the
        new policy's noise scale is learnable (learn_dt/learn_friction).

    Returns ``(log_prob, kl)``, both ``[M]``.
    """
    if kl_form not in _KL_FORMS:
        raise ValueError(f"kl_form must be one of {_KL_FORMS}, got {kl_form!r}")
    bs = obs.shape[0]
    log_prob = torch.zeros(bs, device=obs.device, dtype=torch.float32)
    kl = torch.zeros(bs, device=obs.device, dtype=torch.float32)
    for k, step in enumerate(torch.arange(0, diffusion_model.diff_steps, dtype=torch.float32)):
        fwd_mean, scale = _step_kernel(diffusion_model, obs, traj[k], step)
        log_prob = log_prob + log_prob_kernel(traj[k + 1], fwd_mean, scale)
        sq_diff = (fwd_mean - old_means[k]) ** 2
        if kl_form == "simplified":
            kl_step = sq_diff / (2 * old_scales[k] ** 2 + 1e-8)
        else:
            kl_step = torch.log(scale / old_scales[k]) + (old_scales[k] ** 2 + sq_diff) / (2 * scale**2) - 0.5
        kl = kl + kl_step.sum(dim=-1)
    return log_prob, kl
