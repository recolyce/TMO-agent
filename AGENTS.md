# Agent constraints (read before changing this repository)

This project is a **deterministic scientific pipeline** plus a future constrained
orchestration layer. It is not an autonomous agent that may execute untrusted
network code or invent sample mappings.

## Hard rules

1. The same `experimental_unit_id` / `subject_id` / `biospecimen_id` must never
   appear in more than one of train/val/test. Repeated cross-section must also
   block experiment-batch leakage.
2. Scaler, imputer, HVG, PCA, and batch-correction parameters may be fitted on
   **train only**. Provenance must record `fit_split=train`.
3. Test labels are invisible to any optimizer. Split membership, the evaluator,
   and the primary metric are not Agent-writable.
4. Do not concatenate different animals into a longitudinal trajectory. Do not
   treat `group_level_only` modalities as sample-level pairs.
5. Report PCC together with MSE/MAE. Constant-vector PCC is NA, with `n_valid`.
6. Record hashes/versions/seeds for code, data, split, config, environment, and models.
7. External repos need a pinned commit, license, container isolation, and a
   toy-data adapter test. Never auto-run unknown install/shell scripts.
8. Attribution is not causation. Missing literature is
   「在本次检索范围内未找到直接证据」, never 「首次发现」.
9. Priors require no-prior / single / combined / degree-matched random-graph ablations.
10. Uncertain sample, time, pairing, or license fields become `needs_review`. Do not guess.

## Milestone 1 scope

Implemented: schemas, synthetic bulk generator, split guard, LastValue, Ridge,
time spline, unified evaluator, local MLflow, CLI.

Not implemented (must raise, not fake success): LLM agents, real downloads,
ODE/GRU models, Optuna, IG/ablation, literature, web UI.

Deep-learning stack choice: **pure PyTorch** (not Lightning), starting milestone 4.
