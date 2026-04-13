# JUWELS Unitree deployment

- Added a JUWELS-specific `unitree_rl_mjlab` bundle under `experiments/juwels_unitree_rl_mjlab/`.
- The bundle assumes `Apptainer` on JUWELS, builds the image on the login node, and runs jobs fully offline on compute nodes.
- Default project/account paths target `hai_1075` and can be overridden via environment variables.
- Direct SSH automation is blocked until a manual JUWELS MFA/TOTP login establishes an authenticated session.
