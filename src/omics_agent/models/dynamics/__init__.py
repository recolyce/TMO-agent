"""Milestone-4 temporal dynamics models (pure PyTorch, no Lightning).

Architecture shared by all three plugins:
modality encoders → gated fusion → latent dynamics → modality decoders.
Sequences use actual delta_t, missing masks, and condition covariates.
These models are legal only for longitudinal subject_forecast; repeated
cross-sectional data would fabricate trajectories and is rejected.
"""
