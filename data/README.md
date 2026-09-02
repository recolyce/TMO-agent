# Data layout

- `raw/` is read-only. Never write processed matrices back into it.
- `interim/` holds throwaway conversions.
- `processed/` holds MuData / Parquet produced by the pipeline.
- `manifests/` holds checksum and license records.

Outputs of a run go to `outputs/` at the repository root, not here.
