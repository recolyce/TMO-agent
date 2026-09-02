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

## Milestone 2 scope

Implemented: GEO, BioStudies/ArrayExpress, PRIDE, generic HTTPS, and local
processed-file adapters. Downloads support resume, size limits, SHA-256,
official checksums, retry, rate limits, and dry-run. Paper/HTML text is
stored as untrusted metadata and is never executed. Uncertain
sample/time/modality/biospecimen maps stay `needs_review`.

## Milestone 3 scope

Implemented: approved matrices → MuData with raw/normalized/scaled layers.
Per-assay strategies: bulk RNA counts (CPM+log1p), generic log-expression
(pass-through), protein intensity (zeros→missing, log2). Zero-fill of
protein missingness does not exist as an option. Stateless per-sample math
records `learns_statistics: false`; every fitted transformer records
`fit_split: train` and full-data fits raise. QC metrics, feature_map with
explicit one-to-many / unmapped entries, mygene.info adapter behind the
mockable HttpTransport (CI never hits the network).

## Milestone 4 scope

Implemented (pure PyTorch, no Lightning, no torchdiffeq — in-house fixed-step
RK4 with hard NaN/inf detection): `gru`, `ode_rnn`, and `latent_ode`
ModelPlugins sharing one architecture: modality encoders → gated fusion →
latent dynamics → modality decoders. Sequences carry actual delta_t, missing
masks, and condition one-hots; LayerNorm only, so batch_size=1 works. The
latent ODE is a deterministic encoder-ODE-decoder (no VAE sampling).
`device: auto` uses CUDA when present. Early stopping monitors val masked
MSE; test labels are never read by fit. A diverged solver or NaN loss raises
`OdeSolverError` / `TrainingDivergedError` — never a silent garbage report.
Dynamics models are legal only for longitudinal subject_forecast; repeated
cross-sectional data raises TaskDesignError (rule 4). Also added a sklearn
`mlp` baseline on the same tabular design matrix as ridge. torch stays an
optional extra; without it those plugins are unregistered and get_model
explains the install. No biological priors yet (deliberate).

Not implemented (must raise, not fake success): LLM agents, SRA/raw FASTQ,
raw mass-spec, Optuna, IG/ablation, literature, web UI, biological priors.
