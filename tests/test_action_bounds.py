"""Per-joint action bounds (safe_rl.modules.action_bounds) across every bounded actor.

CPU-only, no environment. Covers: the resolver's spellings and validation, the
StochasticActor (SAC) box + log-prob consistency, ONNX/JIT wrappers, checkpoint round trips
incl. legacy checkpoints without the buffers, the DeterministicActor / FastTD3 box, the
REPPO base per-joint box (log-prob parity, entropy shift, scalar legacy path) and DIME.
"""
from __future__ import annotations

import io
import contextlib
import math

import pytest
import torch
from torch.distributions import AffineTransform, Normal, TanhTransform, TransformedDistribution

from safe_rl.modules.action_bounds import (
    resolve_action_bounds,
    split_action_bound_kwargs,
    uniform_in_box,
)
from safe_rl.modules.actor import (
    DeterministicActor,
    StochasticActor,
    _OnnxDeterministicActor,
    _OnnxStochasticActor,
)

OBS, ACT = 5, 3
BOX = [3.0, 3.0, 1.0]


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# --------------------------------------------------------------------------- resolver
def test_resolver_spellings_agree_and_default_is_unit_box():
    b0, c0, s0 = resolve_action_bounds(ACT)
    assert not s0 and torch.equal(b0, torch.zeros(ACT)) and torch.equal(c0, torch.ones(ACT))
    b, c, s = resolve_action_bounds(ACT, action_bounds=BOX)
    assert s and torch.equal(b, torch.zeros(ACT)) and torch.equal(c, torch.tensor(BOX))
    b2, c2, _ = resolve_action_bounds(ACT, action_low=[-3, -3, -1], action_high=[3, 3, 1])
    assert torch.equal(b2, b) and torch.equal(c2, c)
    b3, c3, _ = resolve_action_bounds(ACT, action_scale=2.0)
    assert torch.equal(c3, torch.full((ACT,), 2.0)) and torch.equal(b3, torch.zeros(ACT))
    b4, c4, _ = resolve_action_bounds(ACT, action_bounds=2.0)
    assert torch.equal(c4, c3)
    b5, c5, _ = resolve_action_bounds(ACT, action_low=[0, -1, -1], action_high=[2, 1, 1])
    assert torch.equal(b5, torch.tensor([1.0, 0.0, 0.0])) and torch.equal(c5, torch.tensor([1.0, 1.0, 1.0]))


def test_resolver_rejects_ambiguous_or_invalid_boxes():
    with pytest.raises(ValueError):
        resolve_action_bounds(ACT, action_bounds=BOX, action_scale=2.0)
    with pytest.raises(ValueError):
        resolve_action_bounds(ACT, action_bounds=[3.0, 3.0])
    with pytest.raises(ValueError):
        resolve_action_bounds(ACT, action_low=[-1, -1, -1], action_high=[1, -1, 1])
    with pytest.raises(ValueError):
        resolve_action_bounds(ACT, action_low=[-1, -1, -1])
    with pytest.raises(ValueError):
        resolve_action_bounds(ACT, action_scale=-1.0)
    kw = {"hidden_dims": [8], "action_bounds": BOX, "action_scale": 2.0}
    lifted = split_action_bound_kwargs(kw)
    assert kw == {"hidden_dims": [8]} and set(lifted) == {"action_bounds", "action_scale"}


# --------------------------------------------------------------------------- SAC actor
def _stochastic(**kw):
    torch.manual_seed(0)
    return StochasticActor(OBS, ACT, hidden_dims=[16], **kw)


def test_stochastic_actor_box_range_and_logprob_round_trip():
    actor = _stochastic(action_bounds=BOX)
    obs = torch.randn(512, OBS)
    a, lp = actor.sample(obs)
    det = actor.act(obs, deterministic=True)
    for x in (a, det, actor.act(obs)):
        assert (x.abs() <= torch.tensor(BOX) + 1e-5).all()
    assert a[:, :2].abs().max() > 1.0, "joints 0-1 never left the unit range"
    assert torch.allclose(lp, actor.log_prob(obs, a), atol=1e-4)
    # closed form == TransformedDistribution(Normal, [Tanh, Affine(b, c)]) reference
    mean, log_std = actor(obs)
    td = TransformedDistribution(Normal(mean, log_std.exp()), [TanhTransform(cache_size=1), AffineTransform(loc=actor.action_b, scale=actor.action_c)])
    unit = ((a - actor.action_b) / actor.action_c).clamp(-1 + 1e-6, 1 - 1e-6)
    ref = td.log_prob(actor.action_b + actor.action_c * unit).sum(-1, keepdim=True)
    assert torch.allclose(lp, ref, atol=1e-3)


def test_stochastic_actor_unbounded_is_bit_identical_to_plain_tanh():
    actor = _stochastic()
    assert not actor.scaled_actions and float(actor.neg_log_action_scale) == 0.0
    obs = torch.randn(32, OBS)
    mean, _ = actor(obs)
    assert torch.equal(actor.act(obs, deterministic=True), torch.tanh(mean))


def test_stochastic_actor_onnx_and_jit_wrappers_match_deterministic_action():
    actor = _stochastic(action_bounds=BOX)
    obs = torch.randn(16, OBS)
    wrapper = _OnnxStochasticActor(actor, None, None, False)
    assert torch.allclose(wrapper(obs), actor.act(obs, deterministic=True), atol=1e-6)
    traced = torch.jit.trace(wrapper, wrapper.get_dummy_inputs())
    assert torch.allclose(traced(obs), actor.act(obs, deterministic=True), atol=1e-6)
    ort = pytest.importorskip("onnxruntime")
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.onnx")
        torch.onnx.export(wrapper, wrapper.get_dummy_inputs(), path, input_names=["obs"], output_names=["actions"], dynamic_axes={"obs": {0: "b"}, "actions": {0: "b"}}, opset_version=11)
        out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"obs": obs.numpy()})[0]
    assert torch.allclose(torch.from_numpy(out), actor.act(obs, deterministic=True), atol=1e-5)


def test_stochastic_actor_simba_onnx_wrapper():
    torch.manual_seed(0)
    actor = StochasticActor(OBS, ACT, network_type="simba", network_kwargs={"hidden_dim": 16, "num_blocks": 1}, action_bounds=BOX)
    obs = torch.randn(8, OBS)
    assert torch.allclose(_OnnxStochasticActor(actor, None, None, False)(obs), actor.act(obs, deterministic=True), atol=1e-6)


def test_stochastic_actor_checkpoint_pins_box_and_legacy_checkpoints_keep_config():
    bounded = _stochastic(action_bounds=BOX)
    sd = bounded.state_dict()
    # (a) same cfg
    same = _stochastic(action_bounds=BOX); same.load_state_dict(sd)
    assert torch.equal(same.action_c, torch.tensor(BOX))
    # (b) a checkpoint carrying a box pins it even when the cfg says unbounded
    other = _stochastic(); _quiet(other.load_state_dict, sd)
    assert other.scaled_actions and torch.equal(other.action_c, torch.tensor(BOX))
    # (c) legacy checkpoint without buffers: strict load succeeds, configured box kept
    legacy = {k: v for k, v in sd.items() if k not in ("action_b", "action_c", "neg_log_action_scale")}
    cfg2 = _stochastic(action_bounds=2.0); cfg2.load_state_dict(legacy, strict=True)
    assert torch.equal(cfg2.action_c, torch.full((ACT,), 2.0)) and cfg2.scaled_actions
    plain = _stochastic(); plain.load_state_dict(legacy, strict=True)
    assert not plain.scaled_actions


def test_sac_actor_critic_policy_level_bounds_and_warmup_box():
    from safe_rl.modules.sac_actor_critic import SACActorCritic
    torch.manual_seed(0)
    sac = _quiet(SACActorCritic, OBS, OBS, ACT, actor_kwargs={"hidden_dims": [16]}, critic_kwargs={"hidden_dims": [16]}, action_bounds=BOX)
    assert torch.equal(sac.actor.action_c, torch.tensor(BOX))
    r = sac.sample_random_action(2000)
    assert (r.abs() <= torch.tensor(BOX) + 1e-6).all() and r[:, 0].abs().max() > 2.5
    obs = torch.randn(8, OBS)
    assert torch.allclose(sac.as_onnx()(obs), sac.act_inference(obs), atol=1e-6)
    with pytest.raises(ValueError):
        _quiet(SACActorCritic, OBS, OBS, ACT, actor_kwargs={"hidden_dims": [16], "action_bounds": BOX}, critic_kwargs={"hidden_dims": [16]}, action_bounds=BOX)


# --------------------------------------------------------------------------- FastTD3 actor
def test_deterministic_actor_box_noise_and_export():
    torch.manual_seed(0)
    actor = DeterministicActor(OBS, ACT, hidden_dims=[16], num_envs=64, noise_std_min=0.2, noise_std_max=0.3, action_bounds=BOX)
    obs = torch.randn(64, OBS)
    actor.train()
    noisy = actor.act(obs)
    assert (noisy.abs() <= torch.tensor(BOX) + 1e-6).all()
    det = actor.act_inference(obs)
    assert torch.allclose(_OnnxDeterministicActor(actor, None, None, False)(obs), det, atol=1e-6)
    # unit-space noise: the difference scales with c_j
    unit_gap = ((noisy - det) / torch.tensor(BOX)).abs()
    assert unit_gap.max() < 1.0
    plain = DeterministicActor(OBS, ACT, hidden_dims=[16])
    assert not plain.scaled_actions and torch.equal(plain(obs), torch.tanh(plain.network(obs)))


def test_td3_actor_critic_box_and_fast_td3_target_smoothing_vectors():
    from safe_rl.modules.td3_actor_critic import TD3ActorCritic
    from safe_rl.algorithms.fast_td3 import FastTD3
    torch.manual_seed(0)
    td3 = _quiet(TD3ActorCritic, OBS, OBS, ACT, num_envs=4, actor_kwargs={"hidden_dims": [16], "noise_std_min": 0.1, "noise_std_max": 0.2}, critic_kwargs={"hidden_dims": [16]}, action_bounds=BOX)
    r = td3.sample_random_action(1000)
    assert (r.abs() <= torch.tensor(BOX) + 1e-6).all() and r[:, 1].abs().max() > 2.5
    alg = _quiet(FastTD3, td3, device="cpu")
    assert torch.allclose(alg.action_high_vec, torch.tensor(BOX)) and torch.allclose(alg.action_low_vec, -torch.tensor(BOX))
    assert torch.allclose(alg.smoothing_noise_vec, 0.2 * torch.tensor(BOX)) and torch.allclose(alg.noise_clip_vec, 0.5 * torch.tensor(BOX))
    with pytest.raises(ValueError):
        _quiet(FastTD3, td3, device="cpu", action_low=-2.0, action_high=2.0)
    plain = _quiet(TD3ActorCritic, OBS, OBS, ACT, num_envs=4, actor_kwargs={"hidden_dims": [16]}, critic_kwargs={"hidden_dims": [16]})
    alg_plain = _quiet(FastTD3, plain, device="cpu", action_low=-1.0, action_high=1.0)
    assert torch.equal(alg_plain.action_high_vec, torch.ones(ACT))


# --------------------------------------------------------------------------- REPPO base
def _reppo(**kw):
    from safe_rl.modules.reppo_actor_critic import REPPOActorCritic
    torch.manual_seed(0)
    base = dict(actor_type="stochastic", critic_type="standard", squash="tanh",
                actor_kwargs={"hidden_dims": [16], "activation": "elu"}, critic_kwargs={"hidden_dims": [16], "activation": "elu"})
    base.update(kw)
    return _quiet(REPPOActorCritic, OBS, OBS, ACT, **base)


def test_reppo_per_joint_box_range_logprob_parity_and_entropy_shift():
    from safe_rl.algorithms.reppo import REPPO
    policy = _reppo(action_bounds=BOX)
    assert policy.action_scale is None and math.isclose(policy.sum_log_action_scale, 2 * math.log(3.0), rel_tol=1e-6)
    obs = torch.randn(256, OBS)
    a = policy.act(obs)
    for x in (a, policy.act_inference(obs), policy.sample_with_log_prob(obs)[0]):
        assert (x.abs() <= torch.tensor(BOX) + 1e-5).all()
    assert a[:, :2].abs().max() > 1.0
    # log-prob parity with the closed-form StochasticActor at min_std=0 (weights shared, box copied)
    ref = StochasticActor(OBS, ACT, hidden_dims=[16], activation="elu", action_bounds=BOX)
    ref.load_state_dict({k: v for k, v in policy.actor.state_dict().items() if not k.startswith("action_") and k != "neg_log_action_scale"}, strict=False)
    assert torch.allclose(policy.get_actions_log_prob(a), ref.log_prob(obs, a).squeeze(-1), atol=1e-4)
    # reference TransformedDistribution with tensor affine
    d = policy.distribution
    td = TransformedDistribution(Normal(d.mean, d.stddev), [TanhTransform(cache_size=1), AffineTransform(loc=policy.action_b, scale=policy.action_c)])
    assert torch.allclose(td.log_prob(policy._clamp_squashed(a)).sum(-1), policy.get_actions_log_prob(a), atol=1e-5)
    alg = _quiet(REPPO, policy, num_learning_epochs=1, num_mini_batches=1, device="cpu", target_entropy=-1.0)
    assert alg.target_entropy == pytest.approx(-1.0 * ACT + 2 * math.log(3.0), rel=1e-6)
    onnx = policy.as_onnx()
    assert torch.allclose(onnx(obs), policy.act_inference(obs), atol=1e-6)
    traced = torch.jit.trace(onnx, onnx.get_dummy_inputs())
    assert torch.allclose(traced(obs), policy.act_inference(obs), atol=1e-6)


def test_reppo_scalar_action_scale_is_unchanged_and_legacy_checkpoints_load():
    from safe_rl.algorithms.reppo import REPPO
    scaled = _reppo(action_scale=3.0)
    assert scaled.action_scale == 3.0 and torch.equal(scaled.action_c, torch.full((ACT,), 3.0))
    obs = torch.randn(64, OBS)
    assert (scaled.act(obs).abs() <= 3.0 + 1e-5).all()
    alg = _quiet(REPPO, scaled, num_learning_epochs=1, num_mini_batches=1, device="cpu", target_entropy=-0.5)
    assert alg.target_entropy == pytest.approx(-0.5 * ACT + ACT * math.log(3.0), rel=1e-6)
    # legacy checkpoint (no box buffers) loads strictly and keeps the configured box
    sd = {k: v for k, v in scaled.state_dict().items() if k not in ("action_b", "action_c", "neg_log_action_scale")}
    fresh = _reppo(action_scale=3.0); fresh.load_state_dict(sd, strict=True)
    assert fresh.action_scale == 3.0
    # a per-joint checkpoint pins its box on a plain policy
    boxed = _reppo(action_bounds=BOX); plain = _reppo(); _quiet(plain.load_state_dict, boxed.state_dict())
    assert plain.scaled_actions and plain.action_scale is None and torch.equal(plain.action_c, torch.tensor(BOX))
    unit = _reppo()
    assert unit.action_scale == 1.0 and not unit.scaled_actions and unit.sum_log_action_scale == 0.0
    with pytest.raises(ValueError):
        _reppo(action_bounds=BOX, action_scale=2.0)


def test_reppo_bounds_inside_actor_kwargs_are_lifted_to_the_policy():
    policy = _reppo(actor_kwargs={"hidden_dims": [16], "activation": "elu", "action_bounds": BOX})
    assert torch.equal(policy.action_c, torch.tensor(BOX))
    assert not policy.actor.scaled_actions, "the inner actor must stay a unit tanh actor"


# --------------------------------------------------------------------------- DIME actor
DIFFUSION = {"diff_steps": 4, "score_model": {"num_layers": 2, "num_hid": 16, "num_time_hid": 8, "num_time_out": 4, "layer_norm": True}}
CRITIC = {"num_atoms": 11, "v_min": -5.0, "v_max": 5.0, "hidden_dim": 16, "encoder_layers": 1, "head_layers": 1, "pred_layers": 1, "activation": "swish", "norm": "rmsnorm", "prior_scale": 1.0}


def test_dime_per_joint_box_samples_in_box_and_running_cost_offset():
    from safe_rl.modules.dime_actor_critic import DIMEActorCritic
    kw = dict(critic_type="reference", diffusion=DIFFUSION, critic_kwargs=CRITIC)
    torch.manual_seed(0); boxed = _quiet(DIMEActorCritic, OBS, OBS, ACT, action_bounds=BOX, **kw)
    torch.manual_seed(0); plain = _quiet(DIMEActorCritic, OBS, OBS, ACT, **kw)
    assert boxed.action_scale is None and math.isclose(boxed.sum_log_action_scale, 2 * math.log(3.0), rel_tol=1e-6)
    obs = torch.randn(64, OBS)
    torch.manual_seed(1); a_b, cost_b, *_ = boxed.actor.sde_sample(obs)
    torch.manual_seed(1); a_p, cost_p, *_ = plain.actor.sde_sample(obs)
    assert (a_b.abs() <= torch.tensor(BOX) + 1e-5).all()
    assert torch.allclose(a_b, plain.actor.action_b + torch.tensor(BOX) * a_p, atol=1e-5)
    assert torch.allclose(cost_b - cost_p, torch.full_like(cost_b, -2 * math.log(3.0)), atol=1e-4)
    assert (boxed.sample_random_action(200).abs() <= torch.tensor(BOX) + 1e-6).all() if hasattr(boxed, "sample_random_action") else True
    sd = {k: v for k, v in boxed.state_dict().items() if not k.endswith(("action_b", "action_c", "neg_log_action_scale"))}
    fresh = _quiet(DIMEActorCritic, OBS, OBS, ACT, action_bounds=BOX, **kw); fresh.load_state_dict(sd, strict=True)
    assert torch.equal(fresh.actor.action_c, torch.tensor(BOX))


def test_uniform_in_box_helper():
    actor = _stochastic(action_bounds=BOX)
    u = uniform_in_box(actor, 4000, "cpu")
    assert u.shape == (4000, ACT) and (u.abs() <= torch.tensor(BOX) + 1e-6).all() and u[:, 0].abs().max() > 2.9
