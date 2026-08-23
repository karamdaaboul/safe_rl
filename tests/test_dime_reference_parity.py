"""Numerical parity between our vendored DIME and the original TruDi source.

`tests/test_reppo_dime.py` checks that our DIME *behaves sensibly*. This file
checks something stronger and more specific: that it is **the same math** as the
authors' code. It extracts the reference implementation straight out of
`safe_rl/trudi (1).zip`, builds both actors from identical weights, and compares
every sampler/KL entry point plus the full TruDi actor loss and all of its
parameter gradients under identical RNG.

Skipped automatically when the zip is absent, so the suite still runs anywhere.
Rationale for keeping it: the vendored files are a verbatim copy of someone
else's implementation. A future refactor (renaming, "simplifying" the integrator,
`black` reflowing an expression) could silently change the math, and only a
numerical check would catch it — reading the diff would not.
"""

from __future__ import annotations

import copy
import importlib
import pathlib
import sys
import zipfile

import pytest
import torch

ZIP = pathlib.Path(__file__).resolve().parents[1] / "trudi (1).zip"
if not ZIP.exists():  # also allow the in-package location
    ZIP = pathlib.Path(__file__).resolve().parents[1] / "safe_rl" / "trudi (1).zip"

ACT, OBS, BS, STEPS = 7, 23, 16, 8
DIFF = dict(diff_steps=STEPS, init_std=2.5, friction=1.0, per_dim_friction=True,
            learn_dt=False, per_step_dt=False, learn_prior=False, learn_betas=False,
            learn_friction=True, learn_mass_matrix=False)
NET = dict(num_layers=3, num_hid=32, num_time_hid=16, num_time_out=8,
           outer_clip=1e4, inner_clip=1e2, weight_init=1e-8, bias_init=0.0,
           layer_norm=True, layer_norm_type="LayerNorm")


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Extract the reference torch DIME modules from the zip and import them."""
    if not ZIP.exists():
        pytest.skip(f"TruDi reference zip not present at {ZIP}")
    root = tmp_path_factory.mktemp("trudi_ref")
    with zipfile.ZipFile(ZIP) as z:
        members = [
            n for n in z.namelist()
            if n.startswith("trudi/src/networks/") and n.endswith(".py")
        ]
        if not members:
            pytest.skip("zip does not contain the expected trudi/src/networks tree")
        z.extractall(root, members=members)
    src_root = root / "trudi"
    sys.path.insert(0, str(src_root))
    try:
        mods = {
            "models": importlib.import_module("src.networks.reppo_dime.torch_dime_models"),
            "integ": importlib.import_module("src.networks.reppo_dime.torch_dime_integrators"),
            "sched": importlib.import_module("src.networks.reppo_dime.common.torch_schedulers"),
            "net": importlib.import_module("src.networks.reppo_dime.models.torch_control_net"),
        }
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not import reference modules: {type(exc).__name__}: {exc}")
    yield mods
    sys.path.remove(str(src_root))
    for name in [m for m in sys.modules if m.startswith("src.")]:
        sys.modules.pop(name, None)


def _build_ref(mods):
    torch.manual_seed(1234)
    net = mods["net"].ControlNetwork(action_dim=ACT, observation_dim=OBS, **NET)
    dm = mods["models"].DiffusionModel(
        action_dim=ACT, observation_dim=OBS, fwd_model=net,
        dt_schedule=mods["sched"].get_cosine_schedule(total_steps=STEPS, min=0.001, s=0.008, pow=2),
        **DIFF,
    )
    return mods["models"].DIMEActor(
        action_dim=ACT, observation_dim=OBS, diffusion_model=dm,
        sde_integrator=mods["integ"].sde_integrator,
        sde_integrator_with_kl=mods["integ"].sde_integrator_with_kl,
        ode_integrator=mods["integ"].ode_integrator,
        logratio=mods["integ"].logratio,
    )


def _build_ours():
    from safe_rl.networks.dime import (
        ControlNetwork, DIMEActor, DiffusionModel, get_cosine_schedule,
        logratio, ode_integrator, sde_integrator, sde_integrator_with_kl,
    )

    torch.manual_seed(1234)
    net = ControlNetwork(action_dim=ACT, observation_dim=OBS, **NET)
    dm = DiffusionModel(
        action_dim=ACT, observation_dim=OBS, fwd_model=net,
        dt_schedule=get_cosine_schedule(total_steps=STEPS, min=0.001, s=0.008, pow=2), **DIFF,
    )
    return DIMEActor(
        action_dim=ACT, observation_dim=OBS, diffusion_model=dm,
        sde_integrator=sde_integrator, sde_integrator_with_kl=sde_integrator_with_kl,
        ode_integrator=ode_integrator, logratio=logratio,
    )


@pytest.fixture(scope="module")
def pair(reference):
    ref, our = _build_ref(reference), _build_ours()
    our.load_state_dict(ref.state_dict())
    ref_old, our_old = copy.deepcopy(ref), copy.deepcopy(our)
    with torch.no_grad():  # make old != new so the KL and grads are nontrivial
        for a in (ref, our):
            a.diffusion_model.fwd_model.state_time_net[-1].bias.add_(0.35)
            a.diffusion_model.friction.add_(0.12)
    our.load_state_dict(ref.state_dict())
    return ref, our, ref_old, our_old, torch.randn(BS, OBS)


def _same(f_ref, f_our, seed):
    torch.manual_seed(seed); r = f_ref()
    torch.manual_seed(seed); o = f_our()
    r = r if isinstance(r, tuple) else (r,)
    o = o if isinstance(o, tuple) else (o,)
    assert len(r) == len(o)
    for i, (a, b) in enumerate(zip(r, o)):
        assert torch.equal(a, b), f"output {i} differs, max|diff|={(a - b).abs().max().item():.3e}"


def test_sde_sample_matches_reference(pair):
    ref, our, _, _, obs = pair
    _same(lambda: ref.sde_sample(obs), lambda: our.sde_sample(obs), 7)


def test_ode_sample_matches_reference(pair):
    ref, our, _, _, obs = pair
    _same(lambda: ref.ode_sample(obs), lambda: our.ode_sample(obs), 8)


@pytest.mark.parametrize("k", [1, 4])
def test_kl_div_matches_reference(pair, k):
    ref, our, ref_old, our_old, obs = pair
    _same(lambda: ref.kl_div(obs, ref_old, k), lambda: our.kl_div(obs, our_old, k), 9 + k)


def test_fused_sample_and_kl_matches_reference(pair):
    ref, our, ref_old, our_old, obs = pair
    _same(lambda: ref.sde_sample_and_kl(obs, ref_old),
          lambda: our.sde_sample_and_kl(obs, our_old), 11)


def test_actor_loss_value_and_gradients_match_reference(pair):
    """The decisive check: TruDi's actor loss and EVERY parameter gradient."""
    ref, our, ref_old, our_old, obs = pair
    kl_bound, temp, beta = 0.1, 0.03, 0.02
    torch.manual_seed(5); w_q = torch.randn(ACT, 1)  # deterministic stand-in critic

    def run(actor, old, seed):
        torch.manual_seed(seed)
        actions, run_c, sto_c, term_c = actor(obs)
        log_probs = run_c + sto_c + term_c
        entropy = -run_c.mean()
        qf = (actions @ w_q).squeeze(-1)
        torch.manual_seed(seed + 1)
        _, kl = actor.kl_div(obs, old, 1, stop_grad=False)
        loss = torch.where(kl < kl_bound, -qf + temp * log_probs, kl * beta).mean()
        target_entropy = actions.shape[-1] * 4.0 + entropy
        loss = (loss + target_entropy.detach() * temp
                + (-beta * (kl - kl_bound).mean().detach())).mean()
        actor.zero_grad()
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None])
        return loss.detach(), grads

    l_ref, g_ref = run(ref, ref_old, 42)
    l_our, g_our = run(our, our_old, 42)
    assert torch.equal(l_ref, l_our), f"loss differs: {l_ref.item()} vs {l_our.item()}"
    assert g_ref.numel() > 0
    assert torch.equal(g_ref, g_our), f"grads differ, max|diff|={(g_ref - g_our).abs().max().item():.3e}"
