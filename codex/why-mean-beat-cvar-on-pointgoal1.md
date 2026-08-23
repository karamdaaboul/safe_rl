# Why the mean constraint beat the CVaR (WCSAC-style) constraint on SafetyPointGoal1

*2026-08-05. Short companion to [[cvpo-cost-critic-investigation]]. WCSAC:
https://github.com/AlgTUDelft/WCSAC (arXiv:2011.11814).*

## The result

At matched budget (15k iterations, 1 seed each):

| constraint | reward | cost (limit 25) |
|---|---|---|
| mean `E[Z_c]` (criticfix, lam_max=4) | **12.3** | 32.7 |
| CVaR_0.5 | 7.1 | 35.7 |
| CVaR_0.9 | 2.3 | 27.2 |

CVaR lost on reward and did not win on cost. Below is why — four reasons, in order of
how much they explain.

## 1. CVaR is much further from satisfiable, so lambda saturates for most of training

This is the main one. Measured on a trained agent (512 states x 64 candidate actions,
`qc_thres` = 2.50):

| signal | value | ratio to threshold |
|---|---|---|
| `E[Z_c]` | 4.45 | **1.78x** |
| `CVaR_0.5` | 6.89 | 2.75x |
| `CVaR_0.9` | 11.19 | **4.48x** |

CVaR sits 1.5-2.5x above the mean *by construction* — it is a tail average. So the same
threshold represents a far deeper violation, `lambda` integrates straight to `lambda_max`,
and stays pinned:

```
cvar05eq   Eqc: 29.0 -> 17.8 -> 12.5 -> 7.8 -> 5.7 -> 4.3 -> 3.0     (crosses at ~13k/15k)
           lam:  4.0 ->  4.0 ->  4.0 -> 4.0 -> 4.0 -> 4.0 -> 0.0
```

While `lambda` is pinned, the E-step weight `exp((Q_r - lambda*Q_c)/eta)` is effectively
"minimise cost, ignore reward". The policy spends ~13,000 of 15,000 iterations under maximum
penalty. That is the reward collapse — not a property of risk-sensitivity, but of a
controller held at its bound.

## 2. CVaR costs a lot of level for very little extra discrimination

The E-step chooses between candidate actions *at the same state*, so what matters is the
spread across actions, not the level (the level cancels in the softmax):

| signal | level vs threshold | per-state action spread |
|---|---|---|
| `E[Z_c]` | 1.78x | 0.0170 |
| `CVaR_0.5` | 2.75x | 0.0230 (1.36x) |
| `CVaR_0.9` | 4.48x | 0.0273 (1.60x) |

Going to CVaR_0.9 multiplies the *violation* by 2.5x to gain 1.6x in *discrimination*.
The exchange rate is bad: you pay in constraint pressure, which is what damages reward, and
you buy a modest improvement in the one quantity that helps.

## 3. The CVaR being constrained was itself mis-calibrated

WCSAC reads CVaR off a fitted normal (mean head + variance head). We read it exactly off the
categorical atoms — but the *distribution* is under-dispersed regardless:

```
predicted std 3.21  vs  realized std 5.55        (2.1x too narrow)
PIT KS 0.381 (95% crit 0.008), deciles U-shaped
VaR_0.9 covers 0.858 of realized returns (want 0.90)
```

So raw CVaR understates tail risk by ~25-30%. The constraint therefore pays the full price of
a strict constraint while delivering less tail protection than its name claims. (A post-hoc
quantile recalibration fixes the calibration — KS 0.381 -> 0.064 — but was not applied inside
training.)

## 4. The premise did not hold: once the critic was fixed, the mean constraint already worked

WCSAC's motivation is that a mean constraint can read as satisfied while episodes blow the
budget. That was exactly true of our *broken* critic — it under-read the level by 2.2x, so
`lambda` never engaged and cost sat at ~47 against a limit of 25.

But the fix for that was the level, not the risk measure. With n-step cost targets, a
non-negative head, and `lambda_max` capped at 4, the **mean** constraint delivers:

```
cost 25.6, reward 19.2      (both_lam4, at budget)
```

Once the mean is estimated correctly and the multiplier is bounded, there is no gap left for
the risk measure to close on this task. CVaR is a fix for a problem that the level fix already
solved.

## Caveats (the comparison is not airtight)

- **Our CVaR thresholds were mis-scaled.** They used the analytic `qc_scale` 0.1 rather than
  the measured 0.0764, so they encoded an episodic budget of 32.7, not 25. The reward damage
  is real, but the cost figures are not a clean test. A corrected rerun (`qc_thres` 2.96 /
  4.79) is the outstanding item.
- One seed per arm; 15k iterations for CVaR against 30k for the best mean arms.
- **Not a refutation of WCSAC.** This says CVaR did not help *on SafetyPointGoal1, with a
  correctly-levelled cost critic and a bounded multiplier*. On a task where the tail genuinely
  diverges from the mean, or where the mean constraint cannot be made to bind, the argument
  for a risk measure is untouched.

## Footnote: one thing worth taking from WCSAC anyway

Their `cost_constraint = cost_lim * (1 - gamma**max_ep_len)/(1 - gamma)/max_ep_len` is the
**same uniform-cost scaling** we measured as mis-specified here (0.1 analytic vs 0.0764
measured, a 31% loose budget). Adopting their formula unchanged imports that error.

Their variance loss `0.5*mean(v + v' - 2*sqrt(v*v'))` — a Wasserstein-style distance rather
than MSE on a heavy-tailed target — is well-conditioned and worth borrowing independently of
the risk constraint.
