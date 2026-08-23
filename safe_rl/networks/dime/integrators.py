"""Euler–Maruyama integrator closures for the DIME diffusion policy.

Vendored verbatim (imports aside) from the TruDi reference
(`trudi/src/networks/reppo_dime/torch_dime_integrators.py`). Four factories:

  * `sde_integrator`         — forward EM step + the DIME ELBO weight
                               (log p_bwd − log p_fwd), used for sampling.
  * `logratio`               — the trust-region KL used by REPPO-DIME's main
                               (forward-KL) variant: per-step closed-form KL
                               between the two transition kernels, trajectory
                               rolled out under the OLD policy ⇒ path-space
                               KL(π_old ‖ π_new).
  * `sde_integrator_with_kl` — one fused rollout under the NEW policy producing
                               both the ELBO weight and a reverse KL(new‖old);
                               the cheaper `rev_kl` variant.
  * `ode_integrator`         — deterministic probability-flow step, eval only.

`logratio` deliberately keeps the reference's SIMPLIFIED per-step KL — squared
drift difference over 2·σ_old², with the log-scale-ratio and variance terms
dropped (the full formula is commented out in their source too). That is what
the published reference results ran; do not "fix" it.
"""

from __future__ import annotations

import torch

from safe_rl.networks.dime.utils import check_stop_grad, log_prob_kernel


def sde_integrator(obs, diffusion_model, stop_grad=False):
    """
    Factory function that creates an SDE integrator closure.

    Args:
        obs: Observations for the forward/backward models
        diffusion_model: The diffusion model containing drift, forward, backward functions
        stop_grad: Whether to stop gradients on the forward mean

    Returns:
        A function that performs one integration step
    """

    def integrate_EM(x, log_w, step):
        """
        Single step of Euler-Maruyama integration for SDE.

        Args:
            x: Current state
            log_w: Current log weight
            step: Current time step (tensor)

        Returns:
            Tuple of (x_new, log_w_new)
        """
        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        scaler_sigma = 1.0 / diffusion_model.friction_fn(step)
        eta = dt * scaler_sigma
        scale = torch.sqrt(2 * eta)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        fwd_mean = check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean

        # Sample new state
        eps = torch.randn_like(fwd_mean)
        x_new = fwd_mean + scale * eps

        # Backward kernel
        drift_new = diffusion_model.drift_fn(step + 1, x_new)
        bwd_mean = x_new + eta * (drift_new + diffusion_model.backward_model(step + 1, x_new, obs))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        # Update weight
        log_w_new = log_w + (bwd_log_prob - fwd_log_prob)

        return x_new, log_w_new

    return integrate_EM


def sde_integrator_with_kl(obs, diffusion_model, target_diffusion_model, stop_grad=False):
    """
    Factory function that creates an SDE integrator closure with Reverse KL computation.

    Args:
        obs: Observations for the forward/backward models
        diffusion_model: Current diffusion model (New Policy)
        target_diffusion_model: Target diffusion model (Old Policy)
        stop_grad: Whether to stop gradients on the forward mean (usually False for Rev KL)

    Returns:
        A function that performs one integration step
    """

    def integrate_EM(x, log_w, kl_log_w, step):
        """
        Single step of Euler-Maruyama integration for SDE with KL calculation.

        Args:
            x: Current state
            log_w: Current DIME log weight (Entropy Lower Bound accumulator)
            kl_log_w: Current KL log weight (Trust Region accumulator)
            step: Current time step (tensor)

        Returns:
            Tuple of (x_new, log_w_new, kl_log_w_new)
        """
        # Compute parameters for NEW policy (Theta)
        dt = diffusion_model.delta_t_fn(step)
        scaler_sigma = 1.0 / diffusion_model.friction_fn(step)
        eta = dt * scaler_sigma
        scale = torch.sqrt(2 * eta)  # Sigma (New)

        # Forward kernel mean (Mu New)
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))

        # Optionally stop gradients (usually False for Reverse KL as we optimize the trajectory)
        if stop_grad:
            fwd_mean = fwd_mean.detach()

        # Compute parameters for OLD policy (Theta')
        target_scaler_sigma = 1.0 / target_diffusion_model.friction_fn(step)
        target_eta = dt * target_scaler_sigma
        target_scale = torch.sqrt(2 * target_eta)  # Sigma' (Old)

        # Forward kernel mean (Mu' Old)
        target_drift = target_diffusion_model.drift_fn(step, x)
        old_fwd_mean = x + target_eta * (target_drift + target_diffusion_model.forward_model(step, x, obs))

        # Stop gradients flowing into the old policy (it is fixed)
        old_fwd_mean = old_fwd_mean.detach()

        # Reverse KL requires expectation over samples from the NEW policy
        # DIME also requires sampling from the current policy to estimate the entropy bound
        eps = torch.randn_like(fwd_mean)
        x_new = fwd_mean + scale * eps

        # Compute DIME Objective Terms (Entropy Bound)
        # Backward kernel for DIME (denoising step for entropy bound)
        drift_new = diffusion_model.drift_fn(step + 1, x_new)
        bwd_mean = x_new + eta * (drift_new + diffusion_model.backward_model(step + 1, x_new, obs))

        # Evaluate kernels for DIME weight (log p_bwd - log p_fwd)
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)
        log_w_new = log_w + (bwd_log_prob - fwd_log_prob)

        # Compute Closed-Form Reverse KL(New || Old)
        # Formula: KL(q||p) = log(sig'/sig) + (sig^2 + (mu-mu')^2)/(2sig'^2) - 1/2
        # Here: q = New (model), p = Old (target)
        # Note: scale is Sigma, target_scale is Sigma'
        kl_elementwise = (
            torch.log(target_scale / scale)
            + (scale**2 + (fwd_mean - old_fwd_mean) ** 2) / (2 * target_scale**2)
            - 0.5
        )

        # Sum over feature dimensions
        step_kl = torch.sum(kl_elementwise, dim=-1)

        # Accumulate Trust Region KL
        kl_log_w_new = kl_log_w + step_kl

        return x_new, log_w_new, kl_log_w_new

    return integrate_EM


def ode_integrator(obs, diffusion_model, stop_grad=False, ode_coef=1.0):
    """
    Factory function that creates an ODE integrator closure.

    Args:
        obs: Observations for the forward model
        diffusion_model: The diffusion model containing drift and forward functions
        stop_grad: Whether to stop gradients
        ode_coef: Coefficient for the forward model (default 1.0)

    Returns:
        A function that performs one integration step
    """

    def integrate_EM(x, step):
        """
        Single step of Euler-Maruyama integration for ODE (deterministic).

        Args:
            x: Current state
            step: Current time step (tensor)

        Returns:
            New state x_new
        """
        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        scaler_sigma = 1.0 / diffusion_model.friction_fn(step)
        eta = dt * scaler_sigma

        # Forward kernel (deterministic)
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + ode_coef * diffusion_model.forward_model(step, x, obs))

        # Always apply stop_grad for ODE
        x_new = check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean

        return x_new

    return integrate_EM


def logratio_with_full_kl(diffusion_model, target_diffusion_model, obs, stop_grad=False):
    """DIAGNOSTIC (safe_rl addition, not in the reference).

    Identical to `logratio` — same trajectory, same RNG draws, same returned
    `log_w` used for the loss — but additionally accumulates the FULL
    closed-form per-step Gaussian KL(old‖new):

        log(sig_new/sig_old) + (sig_old^2 + (mu_old-mu_new)^2)/(2 sig_new^2) - 1/2

    The two agree exactly when the old and new per-dim friction (hence the
    transition-noise scale) match, and diverge as friction drifts within an
    iteration. Logging both measures how much trust region the simplified form
    is silently giving away. The loss must keep using the simplified value to
    stay reference-exact.
    """

    def logratio_EM(x, log_w, full_kl, step):
        dt = diffusion_model.delta_t_fn(step)
        scaler_sigma = 1.0 / diffusion_model.friction_fn(step)
        eta = dt * scaler_sigma
        scale = torch.sqrt(2 * eta)

        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))

        target_scaler_sigma = 1.0 / target_diffusion_model.friction_fn(step)
        target_eta = dt * target_scaler_sigma
        target_scale = torch.sqrt(2 * target_eta)

        target_drift = target_diffusion_model.drift_fn(step, x)
        old_fwd_mean = x + target_eta * (target_drift + target_diffusion_model.forward_model(step, x, obs))
        old_fwd_mean = old_fwd_mean.detach()

        sq_diff = (fwd_mean - old_fwd_mean) ** 2
        log_w_new = log_w + torch.sum(sq_diff / ((2 * target_scale**2) + 1e-8), dim=-1)
        full_step = (
            torch.log(scale / target_scale)
            + (target_scale**2 + sq_diff) / (2 * scale**2)
            - 0.5
        )
        full_kl_new = full_kl + torch.sum(full_step, dim=-1)

        eps = torch.randn_like(old_fwd_mean)
        x_new = old_fwd_mean + target_scale * eps
        if stop_grad:
            x_new = x_new.detach()

        return x_new, log_w_new, full_kl_new

    return logratio_EM


def logratio(diffusion_model, target_diffusion_model, obs, stop_grad=False):
    """
    Factory function that creates a log-ratio integrator closure for Forward KL divergence.
    Computes KL(Old || New) analytically per step.

    Args:
        diffusion_model: Current diffusion model (New Policy)
        target_diffusion_model: Target diffusion model (Old Policy)
        obs: Observations
        stop_grad: Whether to stop gradients on trajectory samples

    Returns:
        A function that performs one integration step
    """

    def logratio_EM(x, log_w, step):
        """
        Single step for computing analytical log-ratio between two diffusion models.

        Args:
            x: Current state
            log_w: Current log weight (accumulated KL)
            step: Current time step (tensor)

        Returns:
            Tuple of (x_new, log_w_new)
        """
        # Compute parameters for NEW policy (Theta)
        dt = diffusion_model.delta_t_fn(step)
        scaler_sigma = 1.0 / diffusion_model.friction_fn(step)
        eta = dt * scaler_sigma
        scale = torch.sqrt(2 * eta)  # Sigma (New)

        # Forward kernel mean (Mu)
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))

        # Compute parameters for OLD policy (Theta')
        target_scaler_sigma = 1.0 / target_diffusion_model.friction_fn(step)
        target_eta = dt * target_scaler_sigma
        target_scale = torch.sqrt(2 * target_eta)  # Sigma' (Old)

        # Forward kernel mean (Mu')
        target_drift = target_diffusion_model.drift_fn(step, x)
        old_fwd_mean = x + target_eta * (target_drift + target_diffusion_model.forward_model(step, x, obs))

        # Stop gradient for old diffusion (it is the fixed target)
        old_fwd_mean = old_fwd_mean.detach()

        # Simplified per-step KL (reference-exact — see module docstring):
        # squared drift difference over 2*sigma_old^2; the full closed form
        #   log(sig/sig') + (sig'^2 + (mu-mu')^2)/(2 sig^2) - 1/2
        # is what the reference source also has commented out.
        kl_elementwise = ((fwd_mean - old_fwd_mean) ** 2) / ((2 * target_scale**2) + 1e-8)

        # Sum over feature dimensions to get KL for this step
        step_kl = torch.sum(kl_elementwise, dim=-1)

        # Update log weight (Accumulate total KL)
        log_w_new = log_w + step_kl

        # Forward KL requires expectation over samples from the OLD policy
        eps = torch.randn_like(old_fwd_mean)
        x_new = old_fwd_mean + target_scale * eps

        # Stop gradients through the state transition if requested
        # (For Forward KL, we usually stop grad on x_new as it comes from the fixed old policy)
        if stop_grad:
            x_new = x_new.detach()

        return x_new, log_w_new

    return logratio_EM
