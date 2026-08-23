"""Amortized ``lambda_psi(s)``: a regressor onto the E-step dual's KKT solution.

This is the one carve-out to CLAUDE.md rule 1 ("duals are computed, never learned"), and it is
narrow. The head has its own network, its own optimizer and its own learning rate -- which is
legal **only** because its loss is a supervised regression onto targets ``lambda*_b`` that the
convex solve has already produced exactly. It never descends ``dg/dlambda``. If you ever find
yourself backpropagating the dual through this module, you have rebuilt the PID-Lagrangian the
whole method exists to remove (chapter section 5.3-B, and CLAUDE.md's "things you will be tempted
to do that are wrong").

**Why it is not just a scalar regression.** The KKT solution has real probability mass on both
corners of the box: ``lambda_b = 0`` where the constraint is already slack, and
``lambda_b = lambda_max`` where the state's sampled actions cannot reach its target. ``log lambda``
is undefined at the first and uninformative at the second, and a plain masked scalar regression
would spend its capacity smearing the interior across two point masses. So the head predicts a
3-way class -- inactive / interior / infeasible -- and regresses the *interior* value only, on the
interior mask.

What it buys, beyond compute (which is nearly nothing -- the E-step already evaluates all K cost
values):

* generalisation to states outside the current minibatch, which smooths exactly the critic noise
  that makes an exact per-state solve brittle where the cost is nearly flat across actions;
* ``lambda(s)`` available at deployment as a boundary-proximity signal, without a solve -- the
  input a sampling-time safety filter wants (chapter section 8.2, milestone M6).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from safe_rl.networks import MLP

INACTIVE, INTERIOR, INFEASIBLE = 0, 1, 2


class LambdaHead(nn.Module):
    """Predicts the per-state multiplier's regime and, where interior, its value.

    Outputs ``[B, 4]``: three class logits, then one unconstrained scalar mapped to
    ``(0, lambda_max)`` by a sigmoid. The sigmoid, rather than a softplus or a raw linear output,
    is deliberate -- the target is bounded by construction, so the head should be too.
    """

    def __init__(
        self,
        num_obs: int,
        lambda_max: float,
        hidden_dims: tuple | list = (64, 64),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.lambda_max = float(lambda_max)
        self.net = MLP(input_dim=num_obs, output_dim=4, hidden_dims=list(hidden_dims), activation=activation)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(class_logits [B, 3], interior_value [B])``."""
        out = self.net(obs)
        return out[:, :3], self.lambda_max * torch.sigmoid(out[:, 3])

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """The multiplier the head would use, ``[B]``: the corners are snapped, not interpolated."""
        logits, interior = self.forward(obs)
        cls = logits.argmax(dim=-1)
        lam = torch.where(cls == INTERIOR, interior, torch.zeros_like(interior))
        return torch.where(cls == INFEASIBLE, torch.full_like(lam, self.lambda_max), lam)

    def set_lambda_max(self, lambda_max: float) -> None:
        """Track a ``lambda_max_mode="balanced"`` cap while it is still being measured."""
        self.lambda_max = float(lambda_max)


def kkt_classes(lam: torch.Tensor, inactive: torch.Tensor, at_cap: torch.Tensor) -> torch.Tensor:
    """Regression targets' regime labels, ``[B]`` long, from the solver's own certificates."""
    cls = torch.full_like(lam, float(INTERIOR), dtype=torch.long)
    cls = torch.where(inactive, torch.full_like(cls, INACTIVE), cls)
    return torch.where(at_cap, torch.full_like(cls, INFEASIBLE), cls)


def lambda_head_loss(
    head: LambdaHead,
    obs: torch.Tensor,
    lam_target: torch.Tensor,
    inactive: torch.Tensor,
    at_cap: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy on the regime + Huber on the interior value. Returns ``(loss, diagnostics)``.

    ``lam_target`` and the two masks come straight from the bisection, so they are the exact KKT
    solution rather than an estimate of it -- which is what makes this regression and not dual
    descent. They must arrive detached; the caller solves under ``no_grad``.
    """
    logits, interior = head(obs)
    cls = kkt_classes(lam_target, inactive, at_cap)
    ce = nn.functional.cross_entropy(logits, cls)

    mask = cls == INTERIOR
    if bool(mask.any()):
        reg = nn.functional.huber_loss(interior[mask], lam_target[mask], delta=0.1 * head.lambda_max)
    else:
        reg = torch.zeros((), dtype=logits.dtype, device=logits.device)

    with torch.no_grad():
        pred = head.predict(obs)
        err = (pred - lam_target).abs()
        var = lam_target.var(unbiased=False)
        diag = {
            "lambda_head_loss": float(ce + reg),
            "lambda_head_ce": float(ce),
            "lambda_head_reg": float(reg),
            "lambda_head_mae": float(err.mean()),
            # R^2 against the constant predictor: <= 0 means the head is not beating "predict the
            # batch mean", i.e. it has learned nothing worth using in place of the solve.
            "lambda_head_r2": float(1.0 - (err.pow(2).mean() / var.clamp_min(1e-12))),
            "lambda_head_class_acc": float((logits.argmax(dim=-1) == cls).to(logits.dtype).mean()),
            "lambda_head_frac_interior": float(mask.to(logits.dtype).mean()),
        }
    return ce + reg, diag
