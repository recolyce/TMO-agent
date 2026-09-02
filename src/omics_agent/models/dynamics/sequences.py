"""Build padded per-instance history tensors from a DataForModel.

Each forecast instance uses only its own experimental unit's observations
at times <= its history time — never another animal's rows. Values are
the scaled layer; missingness is carried as an explicit mask, and the gaps
between observations are the actual delta_t, not an assumed regular grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omics_agent.errors import SchemaError, TaskDesignError
from omics_agent.models.tasks import DataForModel
from omics_agent.schemas.enums import SamplingDesign


@dataclass
class SequenceBatch:
    """Numpy arrays ready for torch conversion. ``pad`` is True on real steps."""

    modalities: list[str]
    values: dict[str, np.ndarray]  # [B, T, F_m], missing filled with 0
    masks: dict[str, np.ndarray]  # [B, T, F_m], 1 = observed
    step_dt: np.ndarray  # [B, T], gap from the previous observation
    pad: np.ndarray  # [B, T], True = real step
    target_dt: np.ndarray  # [B], gap from the last observation to the target
    condition: np.ndarray  # [B, C] one-hot
    y_true: np.ndarray  # [B, F_target], may contain NaN
    y_mask: np.ndarray  # [B, F_target]
    lengths: np.ndarray  # [B]


def modalities_from_input_names(input_feature_names: list[str]) -> list[str]:
    """Recover the ordered modality list from 'modality:feature' names."""

    seen: list[str] = []
    for name in input_feature_names:
        modality = name.split(":", 1)[0]
        if modality not in seen:
            seen.append(modality)
    if not seen:
        raise SchemaError(
            "No input modalities found in the forecast batch.",
            how_to_fix="task.input_modalities must list at least one modality.",
        )
    return seen


def assert_longitudinal(data: DataForModel, model_name: str) -> None:
    """Sequence models must not run on repeated cross-sectional data."""

    if data.bundle.sampling_design is not SamplingDesign.LONGITUDINAL:
        raise TaskDesignError(
            f"Model '{model_name}' needs longitudinal subject histories. "
            f"This dataset is {data.bundle.sampling_design.value}: each animal is "
            "observed once, so a sequence would stitch different animals together.",
            how_to_fix=(
                "Use last_value / ridge / mlp with group_time_forecast for repeated "
                "cross-sectional data. Dynamics models are only legal per-subject."
            ),
        )


def build_sequences(
    data: DataForModel,
    *,
    condition_categories: list[str],
    model_name: str,
) -> SequenceBatch:
    """Assemble one padded batch for every forecast instance in ``data``."""

    assert_longitudinal(data, model_name)
    forecast = data.forecast
    bundle = data.bundle
    if len(forecast.meta) and not bool(forecast.meta["used_same_unit_history"].all()):
        raise TaskDesignError(
            f"Model '{model_name}' requires subject_forecast instances with same-unit "
            "histories; this batch was built for group_time_forecast.",
            how_to_fix="Set task.kind: subject_forecast for dynamics models.",
        )
    modalities = modalities_from_input_names(forecast.input_feature_names)
    for modality in modalities:
        if modality not in bundle.matrices:
            raise SchemaError(
                f"Modality '{modality}' is missing from the bundle.",
                how_to_fix="Input modalities must match the dataset manifest.",
            )
    cond_index = {name: i for i, name in enumerate(condition_categories)}
    unknown = sorted({c for c in forecast.conditions if c not in cond_index})
    if unknown:
        raise SchemaError(
            f"Conditions {unknown} were never seen during training.",
            how_to_fix=(
                "Every condition in val/test must appear in train, or the covariate "
                "encoding is undefined. Check the split assignment."
            ),
        )

    obs = bundle.observations
    unit_rows: dict[str, list[tuple[float, int]]] = {}
    for position, (_, row) in enumerate(obs.iterrows()):
        unit_rows.setdefault(str(row["experimental_unit_id"]), []).append(
            (float(row["time"]), position)
        )
    for unit in unit_rows:
        unit_rows[unit].sort(key=lambda item: item[0])

    histories: list[list[int]] = []
    step_dts: list[np.ndarray] = []
    target_dts: list[float] = []
    meta = forecast.meta
    for i in range(len(forecast.instance_ids)):
        unit = forecast.group_ids[i]
        t_hist = float(meta["history_time"].iloc[i])
        t_target = float(meta["target_time"].iloc[i])
        rows = [(t, pos) for t, pos in unit_rows.get(unit, []) if t <= t_hist]
        if not rows:
            raise TaskDesignError(
                f"Instance {forecast.instance_ids[i]} has no history rows.",
                how_to_fix="subject_forecast instances need at least one earlier observation.",
            )
        times = np.array([t for t, _ in rows], dtype=float)
        histories.append([pos for _, pos in rows])
        dt = np.zeros(len(rows), dtype=float)
        dt[1:] = np.diff(times)
        step_dts.append(dt)
        target_dts.append(t_target - times[-1])

    lengths = np.array([len(h) for h in histories], dtype=int)
    n = len(histories)
    t_max = int(lengths.max())
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for modality in modalities:
        n_feat = bundle.matrices[modality].shape[1]
        val = np.zeros((n, t_max, n_feat), dtype=np.float32)
        msk = np.zeros((n, t_max, n_feat), dtype=np.float32)
        for i, rows_i in enumerate(histories):
            block = bundle.matrices[modality][rows_i]
            observed = ~np.isnan(block)
            val[i, : len(rows_i)] = np.where(observed, block, 0.0)
            msk[i, : len(rows_i)] = observed
        values[modality] = val
        masks[modality] = msk
    step_dt = np.zeros((n, t_max), dtype=np.float32)
    pad = np.zeros((n, t_max), dtype=bool)
    for i, dt in enumerate(step_dts):
        step_dt[i, : len(dt)] = dt
        pad[i, : len(dt)] = True
    condition = np.zeros((n, len(condition_categories)), dtype=np.float32)
    for i, name in enumerate(forecast.conditions):
        condition[i, cond_index[name]] = 1.0
    return SequenceBatch(
        modalities=modalities,
        values=values,
        masks=masks,
        step_dt=step_dt,
        pad=pad,
        target_dt=np.asarray(target_dts, dtype=np.float32),
        condition=condition,
        y_true=forecast.y_true.astype(np.float32, copy=True),
        y_mask=forecast.y_mask.astype(bool, copy=True),
        lengths=lengths,
    )
