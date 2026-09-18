# Ant MPO follow-up — 2026-09-15

The user authorized automatically analysing the finished PPO/SAC/MPO comparison and
improving MPO if it falls short of SAC. A detached controller implements the first
tuning pass; this is experimental work and cannot guarantee a matching score.

## Protocol

- Wait for all nine runs and their evaluations in `logs/ant_ppo_sac_mpo_20260915`.
- Analyse three-seed returns, survival, reward components, critic loss, E-step solver
  status/residual, action sample ESS, mean saturation, and per-dimension M-step KL use.
- Practical target: MPO return at least SAC minus 5% of absolute SAC return, in both
  deterministic and stochastic evaluation. Survival may be at most 5 percentage points
  lower. These are operational thresholds, not statistical equivalence claims.
- Keep Ant-Flat, 256 environments, 16-second episode cap, gamma 0.99, architecture,
  timeout correction, and 8,192,000 transitions per run. Use final checkpoints only.
- Development evaluation: seed 10001, 50 environments, exactly one first episode each.
- Screen six individual changes on training seed 1: mean KL 0.005, hard target actor
  every 100 cycles, mean L2 coefficient 0.001, one M-step iteration, variance KL
  3.125e-6, and 128 E-step actions. Other settings remain the baseline settings.
- Try two adaptive combinations around the strongest screening result: add/strengthen
  mean regularization, and double mean KL up to 0.01. These are hypotheses, not fixes.
- Confirm up to two finalists with training seeds 2 and 3. Stop confirmation when a
  finalist meets the development target. Select using all three training seeds and
  retain the baseline if the sweep fails to produce a better eligible result.
- Freeze selection, then evaluate SAC and selected MPO across all three training
  seeds with previously unused evaluation seed 20001, both action modes. Held-out
  results never feed configuration selection in this pass.
- Maximum 12 additional training runs (98,304,000 transitions overall). Each run has
  a six-hour process timeout. Training is serial on GPU 0 after the baseline finishes.
- If all targets pass, report `target_met`; otherwise report `target_not_met` and
  preserve diagnostics for the next investigation. Infrastructure/numerical failures
  report `needs_review`. Do not claim the tuning objective succeeded merely because
  the queue finished. A further tuning pass needs a new held-out evaluation seed.

## Initial evidence (incomplete baseline)

At setup, the first MPO seed returned 25.18 deterministically, with 96% survival;
two SAC seeds averaged 32.85, with 87% survival. Tail-20 training progress reward
was 0.834 for MPO seed 1 versus 1.525 for SAC seed 1. E-step KL was 0.1 with residual
about 1e-9 and no reported solver failures. Mean saturation was about 36%, and some
individual variance KL dimensions exceeded their budgets. These motivate movement
and policy-update experiments; saturation alone has not established a causal failure.

Checked the original [MPO paper](https://arxiv.org/pdf/1806.06920),
[Acme MPO loss](https://github.com/google-deepmind/acme/blob/master/acme/tf/losses/mpo.py),
and [Acme learner](https://github.com/google-deepmind/acme/blob/master/acme/agents/tf/mpo/learning.py).
Their mean/variance constraints and target-policy choices motivate ablations, but
their hyperparameters are not assumed to transfer directly to this Ant implementation.

## Files and operation

- Controller: `scripts/queue/ant_mpo_followup.py`.
- Output: `logs/ant_mpo_followup_20260915/`.
- Live human report: `report.md`; machine state: `status.json`; protocol: `plan.json`.
- Frozen baseline code with verified hashes: `source/`; baseline configs: `baseline_configs/`.
- Run configs: `configs/`; per-trial logs/evaluations: `trials/`; checkpoints: `runs/`.
- Diagnostics: `baseline_analysis.json`; final test: `heldout_results.json`.
- Selected config: `selected_config.yaml`, which is not promoted to global defaults.
- The controller and queue helper are copied into the output so future workspace edits
  do not change a queued pass. It holds a file lock to prevent duplicate workers.
- Partial training is never overwritten or silently restarted. Evaluation can resume
  from a final checkpoint, and a frozen selection is retained after interruption.

```bash
/home/human/venvs/agx_plain/bin/python scripts/queue/ant_mpo_followup.py launch
python scripts/queue/ant_mpo_followup.py report
```

Validation before launch: 72 focused tests passed, covering the controller's baseline
gate, three-seed selection, held-out isolation, MPO correctness, and timeout handling.
All eight settings also passed full-batch CPU E/M updates with finite parameters and
E-step KL within 1e-5 of 0.1. All eight passed GPU Ant smoke training with four
environments and forced two-step timeouts, producing finite saved checkpoints. The
first checkpoint was reloaded and evaluated in both action modes. These smoke scores
are excluded from selection and the reported comparison. Logs: `smoke_cpu.log`,
`smoke_gpu.log`, and `smoke/` in the output directory. The live safe_rl source hashes
were verified to equal the frozen baseline during smoke validation.

Launched detached worker **407108**. Its state is `waiting_for_baseline`, with no tuning
trials started. The worker owns the exclusive file lock. The original SAC/MPO workers
continue their original queues; the follow-up starts training only after all nine
original runs and evaluations are complete.
