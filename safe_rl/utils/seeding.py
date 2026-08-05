"""Global RNG seeding for reproducible training runs.

Before this module, ``--seed`` on the training scripts was passed only to ``make_env`` --
the *environment layout* was seeded, but ``torch.manual_seed`` was never called anywhere in
the training path. Network initialisation, action sampling and replay-buffer sampling all
drew from an unseeded global generator, so two invocations with identical arguments produced
different weights and no run could be replayed.

That does not invalidate seed-to-seed *variance* (those runs really are independent samples,
varying init as well as layout), but it does mean a run cannot be reproduced, which rules out
any regression oracle.

``seed_everything`` seeds python / numpy / torch / cuda from one integer. ``deterministic``
additionally requests deterministic kernels; note that some ops used here (the categorical
projection's ``scatter_add_`` on CUDA) have no deterministic GPU implementation, so bit-exact
reproducibility is a CPU property. On GPU the seeding still pins init and sampling, which
removes the dominant source of run-to-run drift.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices) from a single integer.

    Args:
        seed: the base seed.
        deterministic: also request deterministic algorithms and cuDNN behaviour. This can
            slow training down and will raise for ops lacking a deterministic implementation,
            so it is opt-in and intended for regression tests rather than production runs.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def rng_fingerprint() -> str:
    """Non-consuming snapshot of global RNG state.

    Must not draw from the generators: it is used to assert that an operation (e.g. an
    evaluation pass) left the training RNG stream untouched, and a fingerprint that itself
    consumed randomness would always differ from the one taken before it.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            h.update(state.cpu().numpy().tobytes())
    np_state = np.random.get_state()
    h.update(np.asarray(np_state[1]).tobytes())
    h.update(str(np_state[2:]).encode())
    h.update(str(random.getstate()).encode())
    return h.hexdigest()
