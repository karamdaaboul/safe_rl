"""Clipping on EmpiricalNormalization.

Early statistics come from few samples, so a near-constant channel can divide by a
near-zero std. Clipping is opt-in; the default path must stay bit-identical.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from safe_rl.modules import EmpiricalNormalization  # noqa: E402


def _fitted(clip=None, eps_mode="add_std"):
    """A normalizer whose statistics are mean 0, std ~1 over one nearly-constant channel."""
    norm = EmpiricalNormalization(shape=[2], eps_mode=eps_mode, clip=clip)
    # Channel 0 varies; channel 1 is nearly constant -> tiny std -> huge normalized values.
    norm.update(torch.tensor([[-1.0, 1e-4], [1.0, -1e-4]]))
    return norm


def test_clip_bounds_the_output() -> None:
    outlier = torch.tensor([[0.0, 5.0]])
    assert _fitted().normalize(outlier).abs().max() > 50.0, "precondition: unclipped blows up"
    assert _fitted(clip=50.0).normalize(outlier).abs().max() == pytest.approx(50.0)


def test_no_clip_by_default_is_unchanged() -> None:
    x = torch.tensor([[0.3, 2e-4]])
    assert torch.equal(_fitted().normalize(x), _fitted(clip=None).normalize(x))


def test_clip_does_not_touch_in_range_values() -> None:
    x = torch.tensor([[0.5, 0.0]])
    assert torch.equal(_fitted(clip=50.0).normalize(x), _fitted().normalize(x))


def test_clip_applies_through_forward() -> None:
    norm = _fitted(clip=2.0)
    norm.eval()  # forward must not update statistics here
    assert norm(torch.tensor([[0.0, 5.0]])).abs().max() == pytest.approx(2.0)


def test_clip_composes_with_add_var_eps_mode() -> None:
    out = _fitted(clip=3.0, eps_mode="add_var").normalize(torch.tensor([[0.0, 5.0]]))
    assert out.abs().max() <= 3.0


def test_clip_must_be_positive() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="clip must be positive"):
            EmpiricalNormalization(shape=[2], clip=bad)


def test_clip_survives_a_state_dict_round_trip() -> None:
    """`clip` is plain config, not a buffer — a reloaded checkpoint must still clip."""
    source = _fitted(clip=50.0)
    restored = EmpiricalNormalization(shape=[2], clip=50.0)
    restored.load_state_dict(source.state_dict())
    outlier = torch.tensor([[0.0, 5.0]])
    assert torch.equal(restored.normalize(outlier), source.normalize(outlier))
