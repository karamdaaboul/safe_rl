# LEAP hyperparameter tuning

User authorized inspecting diagnostics, tuning SAC/MPO, and extending/repeating
promising settings across three seeds. Campaign: `logs/leap_tuning_20260917/`.
Baseline: `logs/leap_zaxis_pilot_20260916/`. All jobs use 250 environments.

## Diagnostic evidence and hypotheses

- SAC final alpha ~0.000122; alpha alone does not establish entropy collapse.
  Existing logs do not measure SAC action entropy reliably (noise std is zero).
  New evaluations measure Monte Carlo differential action entropy without changing
  action RNG draws (`torch.random.fork_rng`).
- Original MPO final E-step KL ratio 1.0, mean KL budget ratio 0.457, variance
  ratio 0.175. Mean dual ~239.67. Thus no evidence of E-step solver failure;
  late average M-step updates are constrained, though worst dimensions can exceed budgets.
- Fraction of pre-tanh means with absolute value >2.5: original MPO 0.680,
  one-step MPO 0.730. This suggests action saturation as a testable limitation,
  not proof of causation. Existing implementation already supports mean L2.
- Diagnostic event data and plots: `pilot_diagnostics.json`, `pilot_diagnostics.png`.

## Predeclared experiment protocol

Six fresh seed-1, 8M-transition screening runs. Each changes one setting or the
paired mean/variance KL budgets relative to its stated baseline:

| Config | Baseline | Change |
|---|---|---|
| sac_entropy | SAC | target_entropy_scale 1 -> 0.25 (target -16 -> -4) |
| sac_fixed_alpha | SAC | disable autotuning, fixed alpha 0.001 |
| mpo5_slow_actor | original MPO | actor_lr 3e-4 -> 1e-4 |
| mpo1_slow_actor | one-step MPO | actor_lr 3e-4 -> 1e-4 |
| mpo1_mean_l2 | one-step MPO | mean_l2_coef 0 -> 0.01 |
| mpo1_relaxed_kl | one-step MPO | mean/variance KL budgets both x4 |

Select one SAC config and one MPO config by deterministic final 8M return
(50 first episodes, eval seed10001), including existing unchanged pilot scores
as candidates. Stochastic scores are reported but not used for selection.

Selected configs run fresh at 20M transitions, training seeds1/2/3. Checkpoints
at exactly8M and20M are evaluated using separate evaluation seed20001. Existing
checkpoints do not include replay, so do not describe fresh20M runs as resumes.
Unchanged SAC and one-step MPO each get a seed1 20M control unless already selected.
Maximum additional training208M transitions; two jobs at a time, GPUs0/1.
No unbounded automated tuning; stop after this screening/validation campaign.

Frozen pilot safe_rl source reused to isolate hyperparameter effects. Controller
and configs are snapshotted in campaign; shared pinned dependencies and JAX cache.
No algorithm updates. `scripts/queue/leap_tuning.py` controls launch, queue,
selection, evaluation, report refresh. Status via `status.json`, `comparison.md`,
`comparison.json`; per-run status/logs/checkpoints under `runs/`.
Controller updates reports each queue loop (~50 sec) and on phase completion.
If any job fails, campaign becomes failed; inspect logs before recovery.

Selection regression tests passed (baseline wins vs candidates win, metric,
three seeds, duplicate controls avoided,208M cap). GPU smoke status is stored
in `smoke/status.json`; launch requires passed smoke.

## Launch

All six GPU smoke tests passed at 2026-09-17 09:05:03 UTC, including training,
checkpoint reload, both evaluation modes, finite action-entropy measurements.
Launched persistent coordinator PID2070243 after smoke success. The first two
jobs are `sac_entropy_seed1` and `sac_fixed_alpha_seed1`; MPO screens follow,
then automatic selection and validation. Read campaign status/report for progress.
