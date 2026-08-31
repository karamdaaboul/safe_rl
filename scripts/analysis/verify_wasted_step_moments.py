"""Monte Carlo check of wasted_step_fraction_proposition.md eqs (13)-(14)."""
import numpy as np

rng = np.random.default_rng(20260831)
d = 3
g = np.array([0.7, -0.3, 0.2])
beta = 0.5

def qfun(u):                      # E[q] = 0 by construction
    return u @ g + beta * (u[..., 0] * u[..., 1])

# ---- population moments (large MC; exact values not needed in closed form)
NU = 6_000_000
u = rng.standard_normal((NU, d))
q = qfun(u)
a  = (q[:, None] * u).mean(0)
nu = (q ** 2).mean()
D  = np.einsum('n,ni,nj->ij', q ** 2, u, u) / NU
del u, q

print(f"a  = {np.round(a, 4)}   (analytic: g = {g}, since E[(u1 u2) u] = 0)")
print(f"nu = {nu:.4f}          (analytic: |g|^2 + beta^2 = {g@g + beta**2:.4f})")
print()

NB = 500_000
for M in (2, 8, 32):
    ub = rng.standard_normal((NB, M, d))
    qb = qfun(ub)
    qc = qb - qb.mean(1, keepdims=True)
    aM = np.einsum('nm,nmd->nd', qc, ub) / M          # (NB, d)

    emp_mean = aM.mean(0)
    emp_cov  = np.cov(aM.T)

    pred_mean = (1 - 1 / M) * a
    VM       = (M - 1) / M**3 * ((M - 1) * D + nu * np.eye(d) - (M - 2) * np.outer(a, a))
    VM_wrong = (M - 1) / M**3 * ((M - 1) * D + nu * np.eye(d))          # drops aa^T term

    print(f"M = {M}")
    print(f"  mean : max|emp - (1-1/M)a|          = {np.abs(emp_mean - pred_mean).max():.5f}")
    print(f"  cov  : max|emp - eq(14)|            = {np.abs(emp_cov - VM).max():.5f}"
          f"   (entries up to {np.abs(VM).max():.4f})")
    print(f"  cov  : max|emp - WRONG (no aa^T)|   = {np.abs(emp_cov - VM_wrong).max():.5f}"
          f"   (should exceed the line above for M>2)")
    print()
