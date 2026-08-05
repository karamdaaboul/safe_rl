"""Seeding must make a run reproducible; the fingerprint must not consume randomness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from safe_rl.utils.seeding import rng_fingerprint, seed_everything  # noqa: E402


def _draws() -> tuple:
    import random

    return (
        torch.rand(4).tolist(),
        np.random.rand(4).tolist(),
        [random.random() for _ in range(4)],
    )


def test_seed_everything_makes_draws_reproducible() -> None:
    seed_everything(1234)
    first = _draws()
    seed_everything(1234)
    second = _draws()
    assert first == second


def test_different_seeds_give_different_draws() -> None:
    seed_everything(1)
    a = _draws()
    seed_everything(2)
    b = _draws()
    assert a != b


def test_seed_everything_seeds_module_init() -> None:
    """The actual defect: network weights were unseeded because torch was never seeded."""
    from safe_rl.modules import SafeSACActorCritic

    def build():
        return SafeSACActorCritic(
            num_actor_obs=8, num_critic_obs=8, num_actions=2, num_costs=1,
            actor_kwargs={"hidden_dims": [16, 16]},
            critic_kwargs={"hidden_dims": [16, 16]},
            cost_critic_kwargs={"hidden_dims": [16, 16]},
        )

    seed_everything(7)
    w1 = torch.cat([p.flatten() for p in build().actor.parameters()])
    seed_everything(7)
    w2 = torch.cat([p.flatten() for p in build().actor.parameters()])
    assert torch.equal(w1, w2)

    seed_everything(8)
    w3 = torch.cat([p.flatten() for p in build().actor.parameters()])
    assert not torch.equal(w1, w3)


def test_rng_fingerprint_does_not_consume_randomness() -> None:
    seed_everything(99)
    f1 = rng_fingerprint()
    f2 = rng_fingerprint()
    assert f1 == f2, "fingerprint must be non-consuming"
    # And it must still detect that randomness WAS consumed.
    torch.rand(1)
    assert rng_fingerprint() != f1


def test_rng_fingerprint_detects_numpy_and_python_consumption() -> None:
    import random

    seed_everything(5)
    base = rng_fingerprint()
    np.random.rand(1)
    assert rng_fingerprint() != base
    seed_everything(5)
    base = rng_fingerprint()
    random.random()
    assert rng_fingerprint() != base
