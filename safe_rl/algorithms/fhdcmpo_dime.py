"""FH-DCMPO with a DIME/TruDi diffusion actor: constrained E-step, denoising M-step.

This is CLAUDE.md's M3 for the finite-horizon constrained method. It is almost entirely
*composition*, which is the point -- the two halves were designed to meet here:

* :class:`~safe_rl.algorithms.fhdcmpo.FHDCMPO` owns the **E-step**: the exactly-solved convex dual
  over ``(eta, lambda)`` and the finite-horizon conservatism statistic
  ``rho = E[Z_c] + kappa (CVaR_alpha[Z_c] - E[Z_c])`` in true episodic units.
* :class:`~safe_rl.algorithms.mpo_dime.MPODIME` owns the **M-step**: a weighted path-space MLE
  under a path-space KL trust region, with the denoising chain rolled by
  ``safe_rl.networks.dime``.

The seam is ``_estep_weights``. ``MPODIME._update_actor_and_alpha`` samples ``N`` denoising paths
per state, scores their final actions with the critics, and calls ``self._estep_weights(...)`` to
turn those scores into per-sample weights. ``FHDCMPO`` overrides exactly that method, so the
diffusion M-step inherits the constrained weights with no further plumbing. ``CVPO`` deliberately
does not override ``_update_actor_and_alpha``, which is why the MRO
``FHDCMPODIME -> FHDCMPO -> CVPO -> MPODIME -> MPO`` resolves to the path-space actor update while
keeping the constrained E-step.

**Rule 3 (CLAUDE.md) holds by construction.** The cost critic is consulted by value only, at the
final actions of the sampled chains. Nothing differentiates a cost critic through the denoiser: the
safety signal reaches the policy through the scalar weights ``w_ij``, never through a gradient
chained across ``T`` denoising steps. That is the structural advantage this design exists to buy,
and it is worth stating because it is easy to "optimise" away.

**Sequencing caveat, recorded honestly.** CLAUDE.md gates M3 on M2 passing, and at the time of
writing the Gaussian FH-DCMPO arms have NOT reached the QR-DMPO baseline (best: 38% budget
violations against 26%). A diffusion result on top of an unresolved Gaussian result cannot cleanly
attribute a difference to the policy class. Keep the Gaussian arm as the control and read any
comparison as suggestive until M2 closes.
"""

from __future__ import annotations

from typing import Any

from .fhdcmpo import FHDCMPO
from .mpo_dime import MPODIME


class FHDCMPODIME(FHDCMPO, MPODIME):
    """Constrained finite-horizon E-step + diffusion (denoising) M-step.

    Everything is inherited. The class body exists to validate the combination and to make the
    resolution order explicit, because a silent MRO surprise here would be extremely hard to see:
    the run would train, and only the constraint would quietly stop being enforced.
    """

    def __init__(self, policy, **kwargs: Any) -> None:
        super().__init__(policy, **kwargs)

        # No actor-type check here: MPODIME.__init__ already raises TypeError for a non-diffusion
        # policy, and it runs first via super(). A second check would be unreachable.

        # The constrained weights must come from FHDCMPO and the actor update from MPODIME. If a
        # future edit reorders the bases or CVPO grows an actor-update override, this run would
        # silently become unconstrained MPO-DIME -- it would train happily and ignore the cost.
        weights_owner = type(self)._estep_weights.__qualname__.split(".")[0]
        actor_owner = type(self)._update_actor_and_alpha.__qualname__.split(".")[0]
        if weights_owner != "FHDCMPO" or actor_owner != "MPODIME":
            raise RuntimeError(
                "MRO regression: expected _estep_weights from FHDCMPO and _update_actor_and_alpha "
                f"from MPODIME, got {weights_owner} and {actor_owner}. The constrained E-step "
                "would not be reaching the diffusion M-step."
            )

        print(
            "FH-DCMPO-DIME: constrained finite-horizon E-step + path-space weighted-MLE denoising "
            f"M-step (path KL budget beta = {self.eps_kl_path}, E-step KL eps = {self.eps_dual}, "
            f"qc_thres = {self.qc_thres:.3f})"
        )

    def get_penalty_info(self) -> dict[str, Any]:
        info = super().get_penalty_info()
        # Both trust regions, side by side: they are different objects and conflating them is
        # CLAUDE.md's rule 2. `eps` bounds q vs pi_old inside the dual; `beta` bounds the
        # projected diffusion policy vs pi_old in PATH space.
        info["kl_path_budget"] = float(self.eps_kl_path)
        info["alpha_path"] = float(self.alpha_path)
        return info
