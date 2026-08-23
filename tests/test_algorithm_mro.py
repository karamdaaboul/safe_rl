"""Pin which class provides each method CVPO depends on.

CVPO(MPO, SafeSAC) inherits its update loop and cost critic from SafeSAC, its critic and
actor updates from MPO, and its storage from SAC. Nothing selects those explicitly -- they
are decided by MRO order. Reparenting any base silently changes the winner, and the failure
mode is quiet: CVPO would stop doing the Lagrangian update or stop training the cost critic
while still producing plausible-looking numbers.

These tests make that visible. If one fails, the fix is to decide the resolution
deliberately (explicit override or delegation), not to edit the expectation.
"""
from safe_rl.algorithms.cvpo import CVPO
from safe_rl.algorithms.mpo import MPO
from safe_rl.algorithms.safe_sac import SafeSAC

# method -> the class that must provide it
EXPECTED_PROVIDER = {
    # the cost path: Lagrangian update then the cost critic, both from SafeSAC
    "update": "SafeSAC",
    "_update_extra_critics": "SafeSAC",
    "_update_cost_critic": "SafeSAC",
    "_update_cost_critic_distributional": "SafeSAC",
    # the reward path: MPO's own policy evaluation and its E/M-step actor update
    "_update_critic_distributional": "MPO",
    "_update_actor_and_alpha": "MPO",
    # shared off-policy scaffolding
    "_update_critic": "SAC",
    "init_storage": "SAC",
    "_bootstrap_discount": "SAC",
    "_bootstrap_mask": "SAC",
}


def _provider(cls, method: str) -> str:
    return getattr(cls, method).__qualname__.split(".")[0]


def test_cvpo_method_resolution_is_pinned() -> None:
    wrong = {
        m: (_provider(CVPO, m), want)
        for m, want in EXPECTED_PROVIDER.items()
        if _provider(CVPO, m) != want
    }
    assert not wrong, (
        "CVPO method resolution changed (got, expected): "
        + repr(wrong)
        + "\nA base class was reparented. Decide the resolution deliberately rather than "
        "updating this expectation -- a wrong winner here silently disables the cost path."
    )


def test_cvpo_mro_shape() -> None:
    assert [c.__name__ for c in CVPO.__mro__] == ["CVPO", "MPO", "SafeSAC", "SAC", "object"]


def test_mpo_owns_its_distributional_critic() -> None:
    """MPO must not fall back to SAC's critic: SAC's carries an entropy bonus."""
    assert MPO._update_critic_distributional is not SafeSAC._update_critic_distributional
    assert _provider(MPO, "_update_critic_distributional") == "MPO"


def test_cost_critic_keys_come_from_safe_sac() -> None:
    """_extra_critic_keys drives which losses `update` reports; SAC's is empty."""
    assert CVPO._extra_critic_keys == SafeSAC._extra_critic_keys
    assert CVPO._extra_critic_keys, "cost critic losses would go unreported"
