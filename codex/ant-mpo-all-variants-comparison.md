# Ant MPO aggregate comparison

The original MPO baseline and all Ant MPO follow-up experiments are aggregated by:

- Script: `scripts/analysis/ant_mpo_all_variants.py`
- Markdown report: `logs/ant_mpo_all_variants_20260916/comparison.md`
- Machine-readable results: `logs/ant_mpo_all_variants_20260916/comparison.json`

The detached watcher refreshes the report once per minute and exits after the parallel
experiments and adaptive follow-up reach terminal states. The watcher PID is stored in
`logs/ant_mpo_all_variants_20260916/watcher.pid`.

Comparison protocol:

- 8,192,000 transitions per training run.
- Training seeds 1, 2, and 3 for confirmed variants.
- 50 first episodes per training seed and action mode.
- Deterministic and stochastic policy evaluation.
- Development evaluation seed 10001.
- Adaptive follow-up selection is frozen before evaluation seed 20001 is opened.
- SAC is the performance reference from the original PPO/SAC/MPO comparison.

Interpret one-seed screening rows as preliminary. Partial rows contain two completed
training seeds and update automatically when the third seed finishes. Configurations
with the same effective algorithm settings are displayed once when a more complete
result is available; repeated nondeterministic runs are not combined as independent
training seeds.
