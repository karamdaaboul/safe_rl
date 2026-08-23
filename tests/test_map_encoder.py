from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# Go2-style layout: actor obs = proprio(47) + trailing height_scan(187 = 11x17).
NUM_OBS = 234
NUM_ACT = 12
SCAN = {"scan_shape": (11, 17), "num_maps": 1, "embed_dim": 128, "num_heads": 4}


def test_map_attention_encoder_shapes_and_grad() -> None:
    from safe_rl.networks import MapAttentionEncoder

    enc = MapAttentionEncoder(NUM_OBS, **SCAN)
    assert enc.state_dim == 47
    assert enc.output_dim == 47 + 128

    obs = torch.randn(8, NUM_OBS)
    out = enc(obs)
    assert out.shape == (8, enc.output_dim)

    # Non-degenerate loss (the encoder ends in LayerNorm, so a bare .sum() is
    # invariant to the query and would spuriously show zero grad).
    (out**2).sum().backward()
    assert enc.q_proj.weight.grad.abs().sum() > 0
    assert enc.cnn[0].weight.grad.abs().sum() > 0


def test_build_obs_encoder_none_and_unknown() -> None:
    from safe_rl.networks import build_obs_encoder

    assert build_obs_encoder("none", NUM_OBS, None) is None
    assert build_obs_encoder(None, NUM_OBS, None) is None
    with pytest.raises(ValueError):
        build_obs_encoder("not_an_encoder", NUM_OBS, None)


def test_encoder_requires_valid_scan_layout() -> None:
    from safe_rl.networks import MapAttentionEncoder

    with pytest.raises(ValueError):  # embed_dim not divisible by num_heads
        MapAttentionEncoder(NUM_OBS, scan_shape=(11, 17), embed_dim=130, num_heads=4)
    with pytest.raises(ValueError):  # scan block does not fit
        MapAttentionEncoder(50, scan_shape=(11, 17), num_maps=1)


def test_deterministic_actor_with_encoder_and_onnx() -> None:
    from safe_rl.modules.actor import DeterministicActor

    actor = DeterministicActor(
        NUM_OBS, NUM_ACT, hidden_dims=[64, 64], encoder_type="map_attention", encoder_kwargs=SCAN
    )
    obs = torch.randn(5, NUM_OBS)
    assert actor(obs).shape == (5, NUM_ACT)
    # Encoder is included in the ONNX-export graph and matches the eager path.
    onnx = actor.as_onnx()
    torch.testing.assert_close(onnx(obs), actor(obs))


def test_default_actor_has_no_encoder() -> None:
    from safe_rl.modules.actor import DeterministicActor

    actor = DeterministicActor(NUM_OBS, NUM_ACT, hidden_dims=[64, 64])
    assert actor.obs_encoder is None
    assert actor(torch.randn(5, NUM_OBS)).shape == (5, NUM_ACT)


def test_distributional_critic_with_midvector_scan() -> None:
    from safe_rl.modules.critic import DistributionalCritic

    # Privileged critic obs: proprio(47) + height_scan(187) + privileged(27) = 261.
    critic = DistributionalCritic(
        261,
        NUM_ACT,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        network_kwargs={"hidden_dims": [64, 64]},
        encoder_type="map_attention",
        encoder_kwargs={**SCAN, "scan_start": 47},
    )
    logits = critic(torch.randn(6, 261), torch.randn(6, NUM_ACT))
    assert logits.shape == (6, 51)
    value = critic.get_value(critic.get_dist(logits))
    assert value.shape == (6,)


def test_standard_critic_with_encoder() -> None:
    from safe_rl.modules.critic import StandardCritic

    critic = StandardCritic(NUM_OBS, num_actions=NUM_ACT, encoder_type="map_attention", encoder_kwargs=SCAN)
    q = critic(torch.randn(4, NUM_OBS), torch.randn(4, NUM_ACT))
    assert q.shape == (4, 1)
