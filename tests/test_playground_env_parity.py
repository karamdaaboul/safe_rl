"""Our MujocoPlaygroundVecEnv must reproduce the reference env core step for step.

Both sides wrap `wrap_for_brax_training(registry.load(task), episode_length, action_repeat)`.
The reference JAX trainer takes its `truncated` flag from `env_state.info["truncation"]`
(src/env_utils/jax_wrappers.py:194); our adapter surfaces the same field as `time_outs`.

If the flags or the post-reset observations diverge, the lambda-return recursion applies the
right formula to the wrong data — which looks like an algorithm problem, not an env one. This
test pins the streams to be identical across a truncation boundary.

Runs on CPU; skipped unless mujoco_playground is importable (it lives in the reference venv).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_playground")
jax = pytest.importorskip("jax")

from safe_rl.envs.mujoco_playground_vec_env import MujocoPlaygroundVecEnv  # noqa: E402

TASK, N, EP_LEN, STEPS, SEED = "CartpoleSwingupSparse", 4, 10, 24, 0


@pytest.fixture(scope="module")
def pair():
    """(reference core state-stepper, our adapter) driven from identical PRNG keys."""
    import jax.numpy as jnp
    from mujoco_playground import registry, wrapper

    base = registry.load(TASK, config=registry.get_default_config(TASK))
    ref_env = wrapper.wrap_for_brax_training(base, episode_length=EP_LEN, action_repeat=1)

    # Exactly RSLRLBraxWrapper.__init__'s key derivation, so obs are comparable elementwise.
    key_reset, _ = jax.random.split(jax.random.PRNGKey(SEED))
    keys = jax.random.split(key_reset, N)
    reset_fn, step_fn = jax.jit(ref_env.reset), jax.jit(ref_env.step)
    ref_state = reset_fn(keys)

    ours = MujocoPlaygroundVecEnv(env_id="Mjx" + TASK, num_envs=N, device="cpu",
                                  seed=SEED, episode_length=EP_LEN)
    ours.reset()
    return dict(step_fn=step_fn, ref_state=ref_state, ours=ours, jnp=jnp)


def test_streams_are_identical_across_a_truncation_boundary(pair):
    step_fn, ref_state, ours, jnp = (pair[k] for k in ("step_fn", "ref_state", "ours", "jnp"))
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(STEPS, N, ours.num_actions)).astype(np.float32)

    truncation_steps = []
    for t in range(STEPS):
        a = actions[t]
        ref_state = step_fn(ref_state, jnp.asarray(a))
        obs, rew, done, extras = ours.step(torch.as_tensor(a))

        assert np.allclose(np.asarray(ref_state.obs), obs.numpy(), atol=1e-4), f"obs diverged at step {t+1}"
        assert np.allclose(np.asarray(ref_state.reward), rew.numpy(), atol=1e-5), f"reward diverged at step {t+1}"
        assert np.allclose(np.asarray(ref_state.done), done.numpy().astype(float), atol=1e-5), \
            f"done diverged at step {t+1}"
        assert np.allclose(np.asarray(ref_state.info["truncation"]), extras["time_outs"].numpy(), atol=1e-5), \
            f"truncation flag diverged at step {t+1}"

        if float(extras["time_outs"].max()) > 0.5:
            truncation_steps.append(t + 1)

    # The point of the test: a truncation actually happened, at the expected cadence.
    assert truncation_steps == [EP_LEN, 2 * EP_LEN], f"truncation fired at {truncation_steps}"


def test_episode_length_buf_tracks_the_env(pair):
    ours = pair["ours"]
    ours.reset()
    for _ in range(EP_LEN + 3):
        ours.step(torch.zeros(N, ours.num_actions))
    assert ours.episode_length_buf.tolist() == [3] * N, "counter must zero on truncation"
