# Map-attention observation encoder (terrain height-scan)

*2026-07-17, Claude Code. Ported from the FastTD3 fork's Go2-parkour work: an
attention-based encoder for the terrain height-scan, wired as an optional
front-end for the actor/critic trunks.*

## TL;DR

New network `safe_rl/networks/map_encoder.py::MapAttentionEncoder` — a small
zero-padded CNN over the height-scan grid + multi-head attention whose **query is
a linear embedding of the proprioceptive state** (proprio-conditioned pooling),
concatenated back with proprio. This replaces feeding the flattened scan straight
into the MLP. Recipe from **Attention-Based Map Encoding** (Xu et al.,
arXiv:2506.09588), which reports better unseen-terrain generalization than a flat
MLP, CNN-downsample, or ViT.

It is **default-off** and opt-in via YAML — a run without `encoder_type` is
byte-for-byte unchanged.

## How to enable

Add `encoder_type` + `encoder_kwargs` inside the existing `actor_kwargs` /
`critic_kwargs` blocks (they flow straight through `off_policy_runner.py` →
actor/critic `__init__`, no runner/algorithm changes):

```yaml
policy:
  class_name: TD3ActorCritic        # or SACActorCritic
  actor_kwargs:
    hidden_dims: [512, 256, 128]
    encoder_type: map_attention
    encoder_kwargs: {scan_shape: [11, 17], num_maps: 1, embed_dim: 128, num_heads: 4}
  critic_kwargs:
    num_atoms: 251
    v_min: -10.0
    v_max: 10.0
    network_kwargs: {hidden_dims: [1024, 512, 256]}
    # critic obs has the scan mid-vector (privileged terms follow), so give scan_start:
    encoder_type: map_attention
    encoder_kwargs: {scan_shape: [11, 17], num_maps: 1, embed_dim: 128, num_heads: 4, scan_start: 47}
```

## Layout assumption (important)

The encoder slices a **contiguous scan block** out of the flat obs.
- **Actor** (mjlab Go2 `policy` group): height_scan is the **trailing** 187 dims
  (`= 11×17`), so `scan_start` defaults to `num_obs - scan_dim`. No `scan_start` needed.
- **Critic** (privileged obs): the scan sits **before** extra privileged terms, so
  you must pass `scan_start` (e.g. 47 for Go2: proprio(47) | scan(187) | priv(27) = 261).

`num_maps > 1` supports stacked grids (e.g. height + roof) as CNN channels.

## Wiring (what changed)

- `MapAttentionEncoder` + `build_obs_encoder(encoder_type, num_obs, encoder_kwargs)`
  factory, exported from `safe_rl/networks/__init__.py`.
- `encoder_type`/`encoder_kwargs` added to `DeterministicActor`, `StochasticActor`
  (`safe_rl/modules/actor.py`) and `StandardCritic`, `DistributionalCritic`
  (`safe_rl/modules/critic.py`). Each builds the encoder, sizes its trunk to
  `encoder.output_dim (+ num_actions for critics)`, and encodes obs in `forward`
  before the trunk. The two ONNX-export wrappers (`_Onnx*Actor`) include the
  encoder so exported policies match the eager path (MHA is ONNX-exportable).
- The encoder trains **inside** the actor/critic param groups (FastTD3 builds
  optimizers over `policy.actor` / `policy.critic_*` params), so no optimizer edits.

## Trade-offs / notes

- Trainable (not frozen) — distinct from the env-side `VisionFeatureWrapper` PVR path.
- Attention adds compute; on heavy-CCD mjlab parkour the sim step dominates anyway.
- Only `encoder_type: map_attention` exists; the factory is the extension point.

Tests: `tests/test_map_encoder.py` (CPU) — encoder shapes/grad, both actors, both
critics incl. mid-vector `scan_start`, default no-encoder path, and ONNX parity.
Full suite: `pytest -q` passes except two **pre-existing** `safe_rl.env`
legacy-alias import failures unrelated to this change.
