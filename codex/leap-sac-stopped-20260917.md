# LEAP SAC cancellation

2026-09-17: At the user's request, stopped active tuning validation SAC seeds 1
and 2 and cancelled queued SAC seed 3 and the SAC control. Preserved checkpoints
and logs. Cancellation evidence: `logs/leap_tuning_20260917/sac_cancellation.json`.

Resumed the existing MPO validation queue with coordinator PID 2462291 using
`leap_tuning_resume.py`; MPO selected seeds 1 and 2 entered training at 14:32 UTC.
The revised controller skips cancelled jobs and reuses existing selection.
These validation jobs retain the campaign's older frozen normalization code.
The separate corrected-normalization campaign remains untouched.

Corrected one-step MPO's completed seed-1 evaluation: deterministic 13.651,
stochastic 12.808, versus original one-step MPO 2.116/3.780 and PPO 11.279/11.612
at 8M transitions. This is a single-seed result, requiring replication.

User subsequently requested stopping MPO. Both active MPO validation jobs were terminated and both queued MPO jobs cancelled. Process exit was verified; checkpoints retained. Campaign status is cancelled; do not resume without a new user request. Evidence: `logs/leap_tuning_20260917/mpo_cancellation.json`.
