"""Emit our REPPO/REPPO-DIME metrics under the TruDi reference's wandb schema.

Why: to compare our port against the authors' code you want the two runs to
overlay in one wandb workspace — same panel, same axis, no mental renaming. Their
trainer logs 25 keys under `actor/*`, `critic/*`, `train/*`; ours uses different
names. This maps ours onto theirs so both appear on the same charts.

Verified against a live reference run (project `dime_benchmark_rerun`,
`reppo-dime-2-torch-PickCube-v1`) — the key list below is read off that run, not
guessed.

Their keys, and where ours come from:

| reference key            | ours                    |
|--------------------------|-------------------------|
| actor/actor_loss         | surrogate               |
| actor/actor_grad_norm    | actor_grad_norm         |
| actor/entropy            | entropy                 |
| actor/entropy_loss       | alpha_temp_loss         |
| actor/kl                 | kl                      |
| actor/temperature        | alpha_temp              |
| actor/lagrangian         | alpha_kl                |
| actor/lagrangian_loss    | alpha_kl_loss           |
| actor/run_cost           | dime_run_cost           |
| actor/sto_cost           | (identically 0 in DIME) |
| actor/terminal_cost      | dime_terminal_cost      |
| actor/friction           | dime_friction           |
| critic/qf_loss           | value_function          |
| critic/critic_grad_norm  | critic_grad_norm        |
| critic/qf_mean           | q_value                 |
| train/return             | (runner episode stats)  |
| train/episode_len        | (runner episode stats)  |
| train/success            | (env `log` extras)      |

`train/*`, `frame`, `speed` and `charts/reset_uniformity` are runner/env-level and
are not produced here; the runner already logs episode return/length, and the
ManiSkill wrapper surfaces `success_rate`.
"""

from __future__ import annotations

# ours -> reference. Only 1:1 renames; anything absent is simply not mirrored.
TRUDI_KEY_MAP: dict[str, str] = {
    "surrogate": "actor/actor_loss",
    "actor_grad_norm": "actor/actor_grad_norm",
    "entropy": "actor/entropy",
    "alpha_temp_loss": "actor/entropy_loss",
    "kl": "actor/kl",
    "alpha_temp": "actor/temperature",
    "alpha_kl": "actor/lagrangian",
    "alpha_kl_loss": "actor/lagrangian_loss",
    "dime_run_cost": "actor/run_cost",
    "dime_terminal_cost": "actor/terminal_cost",
    "dime_friction": "actor/friction",
    "value_function": "critic/qf_loss",
    "critic_grad_norm": "critic/critic_grad_norm",
    "q_value": "critic/qf_mean",
}

# Their run carries these tags; mirroring them makes the two runs filterable together.
TRUDI_TAGS: tuple[str, ...] = ("experimental", "reppo-dime-2")


def add_trudi_aliases(metrics: dict[str, float]) -> dict[str, float]:
    """Return `metrics` plus reference-named duplicates of every mapped key.

    Additive: the original keys are preserved, so existing dashboards and any
    downstream parsing keep working. The runner logs keys containing "/" verbatim,
    so the aliases land in the reference's namespaces.
    """
    aliased = dict(metrics)
    for ours, theirs in TRUDI_KEY_MAP.items():
        if ours in metrics:
            aliased[theirs] = metrics[ours]
    # DIME's stochastic cost is identically zero by construction (see
    # safe_rl/networks/dime/models.py); the reference logs it, so mirror it.
    if "dime_run_cost" in metrics:
        aliased.setdefault("actor/sto_cost", 0.0)
    return aliased
