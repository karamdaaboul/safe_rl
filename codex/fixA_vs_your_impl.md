# Fix-A CVPO — exactly what I changed vs. your implementation

*Session 2026-07-22. Scope: sharpen the **M0 external reference** (FSRL CVPO) on
`SafetyPointGoal1Gymnasium-v0`. TL;DR of provenance below.*

## 0. Provenance — I edited NONE of your code

I did **not** modify a single tracked file in your repo. No changes to
`safe_rl/algorithms/cvpo.py`, `config/safety_gymnasium_cvpo.yaml`, or anything under
`safe_rl/`, `scripts/`, `config/`. Everything I did is **run-configuration** (CLI flag
overrides) on the *external* FSRL library, plus a few helper scripts under `logs/m0_fixA/`.

Files I created (all new, none overwrite yours):

| path | what |
|---|---|
| `logs/m0_fixA/run_rest.sh` | serial driver: `run_rest.sh "<seeds>" <pool>` |
| `logs/m0_fixA/RUN_CONFIG.txt` | exact overrides + why, for reproducibility |
| `logs/m0_fixA/orchestrator.out` | driver log (which seed launched/finished) |
| `logs/m0_fixA/cvpo_seed*.out` | per-seed training logs |
| `/home/human/workspaces/fsrl_m0_src` | re-cloned FSRL (persistent; old scratchpad copy was wiped) |

---

## 1. Fix-A vs the ORIGINAL M0 FSRL reference run

These are the deltas I applied as CLI flags to FSRL's `train_cvpo_agent.py`. Everything
else stays at `MujocoBaseCfg` defaults (5M steps = 250 epoch × 20 000, `training_num=20`,
`sample_act_num=16`, `gamma=0.995`, `n_step=3`).

| knob | original M0 | fix-A | why |
|---|---|---|---|
| `--cost_limit` | 25 | **22** | CVPO's constraint holds only *in expectation under q\**; realized cost runs a few points hot (weighted-ELBO bias). A 22 target lands realized cost on 25. |
| `--testing_num` (eval episodes/epoch) | 2 | **8** | The "cost oscillation" in the first report was mostly **2-episode eval noise**. 8 + tail-averaging ~20 epochs = ~160 eps → low-noise number. (Tried 20 first — it OOM-killed the box via eval-worker spike; 8 peaks ~19.7 GB, safe.) |
| `--estep_iter_num` | 1 | **1** | I initially set 5 (converge the dual per E-step) but **reverted to 1** for speed on the fast-local variant. So this is unchanged from reference. |
| seeds | 3 | **3** | fast-local variant (was going to be 5; cut for wall-clock). |
| scheduling | pool of 2 | **serial (pool 1)** | one CVPO run peaks ~20 GB of 32 GB RAM; two OOM. Confirmed twice this session. |

Exact command per seed (what the driver runs):

```bash
MUJOCO_GL=egl WANDB_MODE=offline ~/venvs/fsrl_m0/bin/python \
  /home/human/workspaces/fsrl_m0_src/examples/mlp/train_cvpo_agent.py \
  --task SafetyPointGoal1Gymnasium-v0 \
  --cost_limit 22 --testing_num 8 --estep_iter_num 1 \
  --device cuda:0 --logdir /home/human/workspaces/safe_rl/logs/m0_fixA --project m0_fixA \
  --seed <S>
```

**Net:** the only substantive knobs that differ from the stock FSRL reference are
`cost_limit 25→22` and `testing_num 2→8`. Nothing algorithmic changed.

---

## 2. FSRL reference CVPO vs YOUR in-repo `safe_rl/algorithms/cvpo.py`

For context — this is the difference between the reference I'm running and *your own*
implementation (which is NOT what these runs use; M0 exists precisely because your
`cvpo.py` has a documented negative result and can't self-certify).

| aspect | FSRL reference (what I ran) | your `safe_rl/algorithms/cvpo.py` |
|---|---|---|
| dual solve | **gradient ascent** on (η, λ): `estep_dual_lr=0.02`, `mstep_dual_lr=0.1`, 1 inner iter | two modes: `lambda_mode="grad"` (projected ascent) **or** `"dual"` (per-batch joint **SLSQP**) — the SLSQP mode matches the C-TruDi spec's "duals computed, not learned" |
| candidate actions | `sample_act_num=16` | `sample_action_num=64` (default) — wider support |
| E-step KL ε | `estep_kl=0.02` | `dual_constraint` (`eps_dual`) |
| M-step trust region | `kl_mu=0.005`, `kl_std=0.0005` | `kl_mean_constraint=0.01`, `kl_var_constraint=1e-4` |
| M-step iters | 1 | `mstep_iteration_num=5` |
| engine | tianshou 0.5.1, off-policy | your runner stack |

Key takeaway: **your `cvpo.py` is arguably closer to the C-TruDi design than the FSRL
reference** (it has an exact-SLSQP dual mode; the reference only does gradient ascent).
The reference is the trust anchor for reward-at-budget numbers, not the target design.

> **Update 2026-07-23:** the structural drift risk flagged here (SafeSAC standalone vs
> SAC) has been eliminated — `SafeSAC` now subclasses `SAC`, and safe off-policy algos
> support in-buffer n-step (costs aggregated like rewards, matching FSRL). See
> [[safe-sac-inheritance-refactor]].

---

## 3. Current run status

3 seeds (0,1,2), serial, on `SafetyPointGoal1Gymnasium-v0`. ~6 h/seed → ~18 h total.
Results (per-seed reward/cost + last-20-epoch tail averages) will be written to
`codex/m0-baselines.md` when complete.
