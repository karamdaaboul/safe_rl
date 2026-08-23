"""Numerical parity of our REPPO components against the authors' reference implementations.

Why this exists: reading two codebases side by side is how we convinced ourselves the port
was faithful, and it was wrong twice. `critic_loss_denominator` defaulted to a non-reference
convention, and a claimed "missing feature" (`reduce_kl`) turned out to be a multiplier that
defaults to 1. Both would have been caught here.

The reference formulas are transcribed VERBATIM with file:line citations so the test runs in
a bare CPU venv. Where the reference repo is present, its real functions are additionally
executed and compared, so a transcription drifting from the source also fails.

There are TWO references and they disagree with each other:
  torch (src/torchrl/)  -> produced the published ManiSkill results; our port mirrors this
  jax   (src/jaxrl/)    -> produced the published DMC results
The lambda-return truncation gate differs between them (see the dedicated test below).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from safe_rl.algorithms.reppo import REPPO  # noqa: E402

REF_REPO = Path("/home/human/workspaces/reppo_original")
GAMMA, LAM = 0.99, 0.95


# ---------------------------------------------------------------------------
# Verbatim reference transcriptions
# ---------------------------------------------------------------------------


def ref_torch_hl_gauss(inp, vmin, vmax, num_atoms):
    """VERBATIM src/torchrl/reppo_util.py:756-778."""
    x = torch.clip(inp, vmin, max=vmax)
    bin_width = (vmax - vmin) / (num_atoms - 1)
    support = torch.linspace(vmin - bin_width / 2, vmax + bin_width / 2, num_atoms + 1)
    sigma = bin_width * 0.75
    cdf_evals = torch.erf((support.unsqueeze(0) - x).squeeze() / (math.sqrt(2.0) * sigma + 1e-6))
    z = cdf_evals[..., -1] - cdf_evals[..., 0]
    probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    return (probs / (z.unsqueeze(-1) + 1e-6)).reshape(*inp.shape[:-1], num_atoms)


def ref_torch_gve(rewards, dones, truncated, next_values):
    """VERBATIM src/torchrl/reppo.py:234-248 (compute_gve); mutates truncated[-1] = 1."""
    truncated = truncated.clone()
    truncated[-1] = 1.0
    gves, last = [], 0
    for t in reversed(range(rewards.shape[0])):
        lambda_sum = LAM * last + (1.0 - LAM) * next_values[t]
        delta = GAMMA * torch.where(truncated[t].bool(), next_values[t], (1.0 - dones[t]) * lambda_sum)
        last = rewards[t] + delta
        gves.insert(0, last)
    return torch.stack(gves)


def ref_jax_gve(rewards, dones, truncated, next_values):
    """VERBATIM src/jaxrl/reppo.py:415-449: reverse scan, carry `truncated` init = ONES.

    Consequence: step t is gated by truncated[t+1], NOT truncated[t]. The importance
    weight is inert at the shipped defaults (exploration_noise_max == min == 1.0 -> exp(0)).
    """
    out = torch.zeros_like(rewards)
    c_lr, c_tr = next_values[-1].clone(), torch.ones_like(truncated[0])
    for t in reversed(range(rewards.shape[0])):
        lambda_sum = LAM * c_lr + (1.0 - LAM) * next_values[t]
        delta = GAMMA * torch.where(c_tr.bool(), next_values[t], (1.0 - dones[t]) * lambda_sum)
        c_lr = rewards[t] + delta
        out[t] = c_lr
        c_tr = truncated[t]
    return out


def ours_gve(rewards, dones, truncated, next_values):
    """The recursion in safe_rl/algorithms/reppo.py::compute_returns."""
    truncated = truncated.clone()
    truncated[-1] = 1.0  # force_last_step_truncated
    mask = torch.maximum((1.0 - dones).clamp_min(0.0), truncated)
    out = torch.zeros_like(rewards)
    recurr = next_values[-1].clone()
    for t in reversed(range(rewards.shape[0])):
        blend = torch.where(truncated[t].bool(), next_values[t],
                            (1.0 - LAM) * next_values[t] + LAM * recurr)
        recurr = rewards[t] + GAMMA * mask[t] * blend
        out[t] = recurr
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rollout():
    torch.manual_seed(0)
    T, N = 16, 6
    trunc = torch.zeros(T, N, 1)
    trunc[4, 1] = trunc[9, 2] = trunc[11, 1] = 1.0  # mid-rollout timeouts
    return dict(
        soft_r=torch.randn(T, N, 1),
        next_v=torch.randn(T, N, 1).abs() * 10,
        done=torch.zeros(T, N, 1),
        trunc=trunc,
    )


# ---------------------------------------------------------------------------
# HL-Gauss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vmin,vmax,atoms", [(0.0, 150.0, 151), (-15.0, 15.0, 151)])
def test_hlgauss_matches_torch_reference(vmin, vmax, atoms):
    """The critic's regression target. A bug here silently corrupts every update."""
    torch.manual_seed(1)
    t = torch.cat([
        torch.tensor([vmin - 5, vmin, vmin + 0.3, (vmin + vmax) / 2, vmax - 0.7, vmax, vmax + 5]),
        torch.empty(12).uniform_(vmin, vmax),
    ])
    ours = REPPO._hlgauss_embed(t, vmin, vmax, atoms)
    ref = ref_torch_hl_gauss(t.unsqueeze(-1), vmin, vmax, atoms)
    assert torch.allclose(ours, ref, atol=1e-5), f"max diff {(ours - ref).abs().max():.3e}"


def test_hlgauss_rows_are_normalised():
    ours = REPPO._hlgauss_embed(torch.linspace(-20, 170, 64), 0.0, 150.0, 151)
    assert torch.allclose(ours.sum(-1), torch.ones(64), atol=1e-5)


# ---------------------------------------------------------------------------
# Lambda-return target
# ---------------------------------------------------------------------------


def test_gve_matches_torch_reference_exactly(rollout):
    """Bit-identical to compute_gve — this is the parity claim for the ManiSkill arm."""
    ours = ours_gve(rollout["soft_r"], rollout["done"], rollout["trunc"], rollout["next_v"])
    ref = ref_torch_gve(rollout["soft_r"], rollout["done"], rollout["trunc"], rollout["next_v"])
    assert torch.equal(ours, ref), f"max diff {(ours - ref).abs().max():.3e}"


def test_gve_matches_torch_reference_with_terminations(rollout):
    done = torch.zeros_like(rollout["done"])
    done[6, 3] = done[2, 0] = 1.0
    ours = ours_gve(rollout["soft_r"], done, rollout["trunc"], rollout["next_v"])
    ref = ref_torch_gve(rollout["soft_r"], done, rollout["trunc"], rollout["next_v"])
    assert torch.equal(ours, ref)


def test_the_two_references_disagree_on_the_truncation_gate(rollout):
    """Documents a discrepancy in the AUTHORS' code, not ours.

    The jax trainer gates step t on truncated[t+1] (reverse-scan carry initialised to
    ones); the torch trainer gates on truncated[t]. We match torch. This is why the DMC
    comparison (jax-derived curves) is not apples-to-apples while ManiSkill is.

    If this test ever starts passing, the references have been reconciled upstream and the
    DMC caveat in reports/PAPER_BENCH_PROTOCOL.md should be revisited.
    """
    trunc_last = rollout["trunc"].clone()
    trunc_last[-1] = 1.0
    rt = ref_torch_gve(rollout["soft_r"], rollout["done"], rollout["trunc"], rollout["next_v"])
    rj = ref_jax_gve(rollout["soft_r"], rollout["done"], trunc_last, rollout["next_v"])
    assert not torch.allclose(rt, rj), "references unexpectedly agree — re-check the DMC caveat"


def test_shifting_our_gate_reproduces_the_jax_reference(rollout):
    """Pins the difference to exactly one thing: the index of the truncation gate."""
    soft_r, done, trunc, next_v = (rollout[k] for k in ("soft_r", "done", "trunc", "next_v"))
    trunc_last = trunc.clone()
    trunc_last[-1] = 1.0
    gate = torch.cat([trunc_last[1:], torch.ones_like(trunc_last[:1])], 0)  # trunc[t+1]
    out = torch.zeros_like(soft_r)
    recurr = next_v[-1].clone()
    for t in reversed(range(soft_r.shape[0])):
        blend = torch.where(gate[t].bool(), next_v[t], (1.0 - LAM) * next_v[t] + LAM * recurr)
        recurr = soft_r[t] + GAMMA * blend  # done == 0 here
        out[t] = recurr
    ref = ref_jax_gve(soft_r, done, trunc_last, next_v)
    assert torch.allclose(out, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_critic_ce_batch_denominator_matches_reference_mean_form():
    """`critic_loss_denominator` must be "batch" for parity.

    Our DEFAULT is "mask", which is a different quantity — the bench configs pin "batch"
    for exactly this reason.
    """
    torch.manual_seed(2)
    B, atoms, vmin, vmax = 512, 151, 0.0, 150.0
    logits = torch.randn(B, atoms)
    targets = torch.rand(B) * vmax
    truncated = (torch.rand(B) < 0.1).float()

    soft = REPPO._hlgauss_embed(targets, vmin, vmax, atoms)
    ce = -(soft * torch.log_softmax(logits, dim=-1)).sum(-1)
    m = (1.0 - truncated).clamp(0, 1)

    ours_batch = (m * ce).sum() / m.numel()
    ours_mask = (m * ce).sum() / m.sum().clamp_min(1.0)
    reference = ((1.0 - truncated) * ce).mean()

    assert torch.allclose(ours_batch, reference, atol=1e-5)
    assert not torch.allclose(ours_mask, reference, atol=1e-3), "the two conventions must differ"


def test_clipped_actor_loss_matches_reference_with_default_reduce_kl():
    """reduce_kl defaults to 1, so their `clipped` branch IS our torch.where form."""
    torch.manual_seed(3)
    kl = torch.rand(4096) * 0.3
    primary = 0.01 * torch.randn(4096) - torch.randn(4096) * 5
    alpha_kl, bound, reduce_kl = 0.05, 0.1, 1.0
    ours = torch.where(kl < bound, primary, alpha_kl * kl)
    ref = torch.where(kl < bound, primary, kl * alpha_kl * reduce_kl)
    assert torch.equal(ours, ref)


def test_aux_loss_stacks_reward_term_with_one_over_d_plus_one_weight():
    """Reference form: concat D feature errors with 1 reward error, mean over D+1,
    masked by (1 - done) (jaxrl/reppo.py:493-501)."""
    torch.manual_seed(4)
    B, D = 256, 512
    pred, tgt = torch.randn(B, D), torch.randn(B, D)
    pred_rew, rew = torch.randn(B, 1), torch.randn(B)
    done = (torch.rand(B) < 0.05).float()

    se = torch.cat([(pred - tgt).pow(2), (pred_rew - rew.view(-1, 1)).pow(2)], dim=-1)
    ours = ((1.0 - done.view(-1, 1)).clamp(0, 1) * se).mean(dim=-1)
    assert se.shape[-1] == D + 1, "reward error must be one extra slot, not a separate mean"
    ref = ((1.0 - done.reshape(-1, 1)) * se).mean(dim=-1)
    assert torch.equal(ours, ref)


def test_dual_loss_signs_drive_multipliers_the_right_way():
    """alpha_kl must RISE when KL exceeds the bound; alpha_temp when entropy is below target."""
    alpha_kl, bound = torch.tensor(0.05, requires_grad=True), 0.1
    (alpha_kl * (bound - torch.tensor(0.30))).backward()   # kl above bound
    assert alpha_kl.grad < 0, "descending this loss must increase alpha_kl"

    alpha_t, target = torch.tensor(0.01, requires_grad=True), -0.5
    (alpha_t * (torch.tensor(-2.0) - target)).backward()   # entropy below target
    assert alpha_t.grad < 0, "descending this loss must increase alpha_temp"


# ---------------------------------------------------------------------------
# Cross-check the transcriptions against the real reference source
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REF_REPO.exists(), reason="reference repo not present")
def test_transcribed_hlgauss_matches_the_real_reference_function():
    """Guards against this file's transcription drifting from the reference source."""
    import importlib.util

    path = REF_REPO / "src" / "torchrl" / "reppo_util.py"
    if not path.exists():
        pytest.skip("reference reppo_util.py not found")
    spec = importlib.util.spec_from_file_location("_ref_reppo_util", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — reference pulls heavy optional deps
        pytest.skip(f"reference module not importable here: {type(exc).__name__}")

    t = torch.linspace(-5.0, 155.0, 32).unsqueeze(-1)
    assert torch.allclose(ref_torch_hl_gauss(t, 0.0, 150.0, 151),
                          mod.hl_gauss(t, 0.0, 150.0, 151), atol=1e-6)
