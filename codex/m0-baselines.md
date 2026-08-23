# M0 — Reference baselines (FSRL CVPO + SAC-Lagrangian on SafetyPointGoal1)

*C-TruDi milestone M0. Started 2026-07-12 (Claude Code). Goal: stand up the **external reference**
implementation (Liu et al.'s FSRL), run CVPO and SAC-Lagrangian to completion on
`SafetyPointGoal1Gymnasium-v0`, and confirm the reference reproduces its published numbers. No
algorithm code is written in this milestone — this is the trust anchor for M1+.*

Why the external reference and not our in-repo `cvpo.py`: our CVPO has a documented negative result
(constraint not enforced on the harder CarGoal2, see [[cvpo-negative-result.md]]), so it cannot
self-certify. M0 exists to establish a *known-good* baseline before we build on it.

## TL;DR

**Status: RUNS IN PROGRESS** (3 seeds × {CVPO, SAC-Lag}, pool of 2, full 5M-step FSRL budget).
Environment stood up and both training pipelines validated end-to-end on the local Blackwell GPU.
Gate verdict pending run completion.

## Environment (isolated — does NOT touch `agx_plain` or the repo)

FSRL pins `tianshou~=0.5.0`, which is incompatible with `agx_plain`'s stack, so M0 lives in a
**separate venv** `~/venvs/fsrl_m0` (Python 3.10.20). The resolver actually kept a modern stack that
runs on the Blackwell GPU:

| package | version | note |
|---|---|---|
| torch | 2.13.0+cu130 | CUDA OK on RTX PRO 4500 Blackwell |
| tianshou | 0.5.1 | FSRL's engine; runs fine on gymnasium 1.3.0 |
| gymnasium | 1.3.0 | resolver kept latest (tianshou 0.5.1 tolerates it) |
| safety_gymnasium | 1.2.0 | karamdaaboul fork (`/home/human/workspaces/safety-gymnasium`, editable) |
| numpy | 2.2.6 | |
| fsrl | 0.1.0 | editable from github.com/liuzuxin/FSRL |
| bullet_safety_gym | 1.1.0 | imported unconditionally by the train scripts |
| mujoco | 3.3.0 | + glfw, imageio, xmltodict, PyOpenGL, gymnasium-robotics 1.4.2 |

### Reproducible install

```bash
python3.10 -m venv ~/venvs/fsrl_m0
~/venvs/fsrl_m0/bin/pip install -U pip wheel setuptools
git clone https://github.com/liuzuxin/FSRL.git <fsrl_src>
~/venvs/fsrl_m0/bin/pip install -e <fsrl_src>            # tianshou/gymnasium/torch/bullet_safety_gym/pyrallis/wandb
~/venvs/fsrl_m0/bin/pip install --no-deps -e /home/human/workspaces/safety-gymnasium   # fork, py3.10-compatible
~/venvs/fsrl_m0/bin/pip install --no-deps glfw mujoco==3.3.0 imageio xmltodict PyOpenGL gymnasium-robotics
```
Env id is FSRL's wrapped **`SafetyPointGoal1Gymnasium-v0`** (note the `Gymnasium` infix), config
`MujocoBaseCfg`: cost_limit 25, 250 epochs × 20 000 steps = **5 M env-steps/run**, 20 train envs
(CVPO) / 10 (SAC-Lag). Runtime env vars: `MUJOCO_GL=egl WANDB_MODE=offline`.

## Run commands

```bash
cd <fsrl_src>
MUJOCO_GL=egl WANDB_MODE=offline ~/venvs/fsrl_m0/bin/python examples/mlp/train_cvpo_agent.py \
  --task SafetyPointGoal1Gymnasium-v0 --cost_limit 25 --seed <S> --device cuda:0 \
  --logdir /home/human/workspaces/safe_rl/logs/m0 --project m0
# SAC-Lag: examples/mlp/train_sacl_agent.py (same flags)
```
Pool launcher: `scratchpad/run_m0.sh` (6 jobs, `wait -n` semaphore of 2). Per-run logs at
`logs/m0/<alg>_seed<S>.out`; metrics at `logs/m0/m0/.../progress.txt` (tab-separated; key columns
`update/env_step`, `test/reward`, `test/cost`, `train/reward`, `train/cost`).

## Smoke validation (1 epoch, before committing GPU hours)

Both pipelines completed end-to-end on GPU:
- **CVPO** — collection ~600 it/s (~185 eff. w/ updates); final eval reward −27.5, cost 2.3.
- **SAC-Lag** — final eval reward −9.1, cost 8.8.

(Untrained/early numbers — the point was that train→update→eval→log all work on the Blackwell stack.)

## Gate

**Pass condition:** reference reproduces its published PointGoal1 numbers — i.e. **cost ≤ 25**
(constraint satisfied) and reward within FSRL's benchmark band. Exact published figures to be pinned
from the FSRL benchmark / CVPO paper (arXiv:2201.11927) at gate time (readthedocs benchmark values
are in images; not machine-readable).

### Results (per seed + mean±std) — TO BE FILLED

| alg | seed | reward | cost | within budget? |
|---|---|---|---|---|
| CVPO | 0 | — | — | — |
| CVPO | 1 | — | — | — |
| CVPO | 2 | — | — | — |
| SAC-Lag | 0 | — | — | — |
| SAC-Lag | 1 | — | — | — |
| SAC-Lag | 2 | — | — | — |

Plan: judge gate on 3 seeds; extend to the spec's ≥5 seeds if the frontier looks right.

## Results — original M0 (3 seeds, cost_limit 25, testing_num 2, 5M steps)

Numbers are **last-20-epoch tail averages** (per-epoch endpoints are noise; FSRL evals only
`testing_num` episodes). CSV had a mid-row column-shift artifact — cross-checked vs `.out`
`Final eval` lines.

| alg | reward | cost | osc (±sd) | within budget? |
|---|---|---|---|---|
| CVPO (mean n=3) | 20.6 ± 1.8 | 27.1 ± 1.7 | ±~16 | marginal (~2 over) |
| SAC-Lag (mean n=3) | −7.4 | 44.7 | huge | **NO — did not train** |

SAC-Lag reward collapsed to ~0/negative across seeds — non-functional, not conservative;
needs its own config pass before it can serve as a reference.

## Results — Fix-A tightened CVPO (2 seeds, cost_limit 22, testing_num 8, estep_iter_num 1)

Runs: `logs/m0_fixA/` (`cvpo_cost22.0_seed{0,1}-*`). See [[fixA_vs_your_impl]] for the exact deltas.

| seed | reward | cost (last-20) | osc (±sd) |
|---|---|---|---|
| 0 | 20.7 | 26.0 | ±6.4 |
| 1 | 18.9 | 26.4 | ±7.2 |
| **mean** | **19.8** | **26.2** | **±6.8** |

**Verdict — one win, one null:**
- ✅ Oscillation **halved** (±16 → ±6.8): most of the original "violent" swing WAS 2-episode
  eval noise (`testing_num`), not policy instability. `testing_num=8` + tail-averaging fixes it.
- ⚠️ Real ±7 oscillation remains, and mean cost barely moved (27.1 → 26.2, still ~1 over budget).
  Dropping `cost_limit` 25→22 had ~no effect on realized cost (seed0 25.9→26.0).
- **Root cause:** FSRL reference solves the dual by *gradient ascent* (1 step/update) → doesn't
  track the cost target, so moving the target doesn't move realized cost. This is precisely the
  weakness the C-TruDi exact-SLSQP dual (M1) targets. Useful negative result, not a tuning miss.
- Only 2 seeds (user stopped seed 2). Enough to see the trend; not the spec's ≥5.

Cross-refs: [[cvpo-negative-result.md]] (why the reference, not our impl); [[fixA_vs_your_impl]]
(exact fix-A deltas); [[m0-fsrl-baseline-setup]] (env recipe + testing_num OOM landmine).
