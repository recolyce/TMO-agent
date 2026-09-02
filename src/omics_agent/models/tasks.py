"""Construct leakage-safe forecast instances from a MultiOmicsBundle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omics_agent.errors import TaskDesignError
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.schemas.enums import HistoryPolicy, SamplingDesign, SplitName, TaskKind
from omics_agent.schemas.experiment import TaskConfig


@dataclass
class ForecastBatch:
    """One evaluation or training set of forecast instances.

    ``y_true`` may contain NaN. ``y_mask`` is True where the target was
    actually observed. Models must not overwrite these arrays.
    """

    instance_ids: list[str]
    group_ids: list[str]
    subject_ids: list[str]
    conditions: list[str]
    batches: list[str]
    target_times: np.ndarray
    delta_t: np.ndarray
    y_true: np.ndarray
    y_mask: np.ndarray
    feature_names: list[str]
    last_target: np.ndarray
    last_inputs: np.ndarray
    input_feature_names: list[str]
    meta: pd.DataFrame


@dataclass
class PredictionArrays:
    """Model output aligned to a ForecastBatch."""

    y_pred: np.ndarray
    extras: dict[str, np.ndarray]


@dataclass
class DataForModel:
    """What a ModelPlugin receives: the split bundle plus forecast instances."""

    bundle: MultiOmicsBundle
    forecast: ForecastBatch
    split: SplitName
    train_condition_time_target_mean: dict[tuple[str, float], np.ndarray]
    train_condition_time_input_mean: dict[tuple[str, float], np.ndarray]


def _observation_index(bundle: MultiOmicsBundle) -> dict[tuple[str, float], int]:
    index: dict[tuple[str, float], int] = {}
    for position, (_, row) in enumerate(bundle.observations.iterrows()):
        key = (str(row["experimental_unit_id"]), float(row["time"]))
        index[key] = position
    return index


def _train_group_means(
    train: MultiOmicsBundle, modality: str
) -> dict[tuple[str, float], np.ndarray]:
    means: dict[tuple[str, float], np.ndarray] = {}
    values = train.matrices[modality]
    grouped = train.observations.groupby(["condition", "time"], sort=False)
    for _, part in grouped:
        condition = str(part["condition"].iloc[0])
        time = float(part["time"].iloc[0])
        block = values[np.asarray(part.index)]
        means[(condition, time)] = np.nanmean(block, axis=0)
    return means


def _lookup_last_mean(
    means: dict[tuple[str, float], np.ndarray],
    condition: str,
    target_time: float,
    n_features: int,
) -> tuple[np.ndarray, float]:
    earlier = [(t, vec) for (cond, t), vec in means.items() if cond == condition and t < target_time]
    if not earlier:
        if means:
            # Fall back to the closest train time of the same condition, else zeros.
            same = [(t, vec) for (cond, t), vec in means.items() if cond == condition]
            pool = same or [(t, vec) for (cond, t), vec in means.items()]
            t, vec = min(pool, key=lambda item: abs(item[0] - target_time))
            filled = np.where(np.isnan(vec), 0.0, vec)
            return filled, float(target_time - t)
        return np.zeros(n_features, dtype=float), np.nan
    t, vec = max(earlier, key=lambda item: item[0])
    filled = np.where(np.isnan(vec), 0.0, vec)
    return filled, float(target_time - t)


def build_forecast_batch(bundle: MultiOmicsBundle, task: TaskConfig) -> ForecastBatch:
    """Build forecast instances that respect the sampling design.

    Longitudinal ``subject_forecast`` uses the same experimental unit's past.
    RCS ``group_time_forecast`` never reads another animal as if it were the
    same individual; last-value features come from train condition×time means
    attached later in :func:`attach_train_statistics`.
    """

    if task.kind is TaskKind.SUBJECT_FORECAST:
        if bundle.sampling_design is not SamplingDesign.LONGITUDINAL:
            raise TaskDesignError(
                "subject_forecast requires longitudinal sampling.",
                how_to_fix="Use group_time_forecast for repeated cross-section.",
            )
        return _build_subject_forecast(bundle, task)
    if task.kind is TaskKind.GROUP_TIME_FORECAST:
        return _build_group_time_forecast(bundle, task)
    raise TaskDesignError(
        f"Unsupported task kind {task.kind}.",
        how_to_fix="Use subject_forecast or group_time_forecast.",
    )


def _input_vector(bundle: MultiOmicsBundle, row: int, modalities: list[str]) -> tuple[np.ndarray, list[str]]:
    """Concatenate raw (possibly NaN) last-snapshot features.

    Missing feature values are filled later from **train** column means in
    :func:`attach_train_statistics`. Filling from the current split would leak.
    """

    blocks: list[np.ndarray] = []
    names: list[str] = []
    for modality in modalities:
        blocks.append(bundle.matrices[modality][row].copy())
        names.extend([f"{modality}:{name}" for name in bundle.feature_names[modality]])
    return np.concatenate(blocks), names


def _build_subject_forecast(bundle: MultiOmicsBundle, task: TaskConfig) -> ForecastBatch:
    target = task.target_modality
    index = _observation_index(bundle)
    instance_ids: list[str] = []
    group_ids: list[str] = []
    subject_ids: list[str] = []
    conditions: list[str] = []
    batches: list[str] = []
    target_times: list[float] = []
    delta_ts: list[float] = []
    y_rows: list[np.ndarray] = []
    y_masks: list[np.ndarray] = []
    last_targets: list[np.ndarray] = []
    last_inputs: list[np.ndarray] = []
    input_names: list[str] = []
    meta_rows: list[dict[str, object]] = []

    for unit, grp in bundle.observations.groupby("experimental_unit_id"):
        times = sorted(float(t) for t in grp["time"].unique())
        if len(times) < 2:
            continue
        for step_i, t_target in enumerate(times):
            if step_i < task.horizon_steps:
                continue
            t_last = times[step_i - task.horizon_steps]
            if task.history is HistoryPolicy.PREVIOUS_ALL:
                # Milestone 1 still uses the last snapshot as the feature vector;
                # earlier times are retained only for LastValue fallback.
                pass
            key_target = (str(unit), t_target)
            key_last = (str(unit), t_last)
            if key_target not in index or key_last not in index:
                continue
            i_target = index[key_target]
            i_last = index[key_last]
            y = bundle.matrices[target][i_target]
            last_y = bundle.matrices[target][i_last].copy()
            if np.isnan(last_y).all():
                # Walk further back; if none, leave NaN and let LastValue use train mean.
                for t_prev in reversed(times[:step_i]):
                    prev = bundle.matrices[target][index[(str(unit), t_prev)]]
                    if not np.isnan(prev).all():
                        last_y = prev.copy()
                        t_last = t_prev
                        i_last = index[(str(unit), t_prev)]
                        break
            features, names = _input_vector(bundle, i_last, task.input_modalities)
            input_names = names
            last_y_filled = last_y.copy()
            last_y_filled = np.where(np.isnan(last_y_filled), np.nanmean(last_y_filled), last_y_filled)
            last_y_filled = np.where(np.isnan(last_y_filled), 0.0, last_y_filled)
            row = bundle.observations.iloc[i_target]
            instance_ids.append(f"{unit}:{t_last}->{t_target}")
            group_ids.append(str(unit))
            subject_ids.append(str(row["subject_id"]))
            conditions.append(str(row["condition"]))
            batches.append(str(row["batch"]))
            target_times.append(t_target)
            delta_ts.append(t_target - t_last)
            y_rows.append(y)
            y_masks.append(~np.isnan(y))
            last_targets.append(last_y_filled)
            last_inputs.append(features)
            meta_rows.append(
                {
                    "instance_id": instance_ids[-1],
                    "experimental_unit_id": str(unit),
                    "history_time": t_last,
                    "target_time": t_target,
                    "used_same_unit_history": True,
                }
            )
    if not instance_ids:
        raise TaskDesignError(
            "No longitudinal forecast instances could be constructed.",
            how_to_fix="Need units with at least horizon_steps+1 distinct times and an aligned target modality.",
        )
    return ForecastBatch(
        instance_ids=instance_ids,
        group_ids=group_ids,
        subject_ids=subject_ids,
        conditions=conditions,
        batches=batches,
        target_times=np.asarray(target_times, dtype=float),
        delta_t=np.asarray(delta_ts, dtype=float),
        y_true=np.vstack(y_rows),
        y_mask=np.vstack(y_masks),
        feature_names=list(bundle.feature_names[target]),
        last_target=np.vstack(last_targets),
        last_inputs=np.vstack(last_inputs),
        input_feature_names=input_names,
        meta=pd.DataFrame(meta_rows),
    )


def _build_group_time_forecast(bundle: MultiOmicsBundle, task: TaskConfig) -> ForecastBatch:
    if task.target_time_min is None:
        raise TaskDesignError(
            "group_time_forecast requires target_time_min.",
            how_to_fix="Set task.target_time_min in the experiment YAML.",
        )
    target = task.target_modality
    instance_ids: list[str] = []
    group_ids: list[str] = []
    subject_ids: list[str] = []
    conditions: list[str] = []
    batches: list[str] = []
    target_times: list[float] = []
    delta_ts: list[float] = []
    y_rows: list[np.ndarray] = []
    y_masks: list[np.ndarray] = []
    last_targets: list[np.ndarray] = []
    last_inputs: list[np.ndarray] = []
    input_names = [f"{m}:{n}" for m in task.input_modalities for n in bundle.feature_names[m]]
    n_in = len(input_names)
    n_out = len(bundle.feature_names[target])
    meta_rows: list[dict[str, object]] = []

    for position, (_, row) in enumerate(bundle.observations.iterrows()):
        t = float(row["time"])
        if t < task.target_time_min:
            continue
        y = bundle.matrices[target][position]
        instance_ids.append(f"{row['experimental_unit_id']}:{t}")
        group_ids.append(str(row["experimental_unit_id"]))
        subject_ids.append(str(row["subject_id"]))
        conditions.append(str(row["condition"]))
        batches.append(str(row["batch"]))
        target_times.append(t)
        delta_ts.append(np.nan)
        y_rows.append(y)
        y_masks.append(~np.isnan(y))
        last_targets.append(np.full(n_out, np.nan))
        last_inputs.append(np.zeros(n_in, dtype=float))
        meta_rows.append(
            {
                "instance_id": instance_ids[-1],
                "experimental_unit_id": str(row["experimental_unit_id"]),
                "history_time": np.nan,
                "target_time": t,
                "used_same_unit_history": False,
            }
        )
    if not instance_ids:
        raise TaskDesignError(
            f"No RCS instances with time >= {task.target_time_min}.",
            how_to_fix="Lower target_time_min or generate data with later time points.",
        )
    return ForecastBatch(
        instance_ids=instance_ids,
        group_ids=group_ids,
        subject_ids=subject_ids,
        conditions=conditions,
        batches=batches,
        target_times=np.asarray(target_times, dtype=float),
        delta_t=np.asarray(delta_ts, dtype=float),
        y_true=np.vstack(y_rows),
        y_mask=np.vstack(y_masks),
        feature_names=list(bundle.feature_names[target]),
        last_target=np.vstack(last_targets),
        last_inputs=np.vstack(last_inputs),
        input_feature_names=input_names,
        meta=pd.DataFrame(meta_rows),
    )


def attach_train_statistics(
    bundle: MultiOmicsBundle,
    forecast: ForecastBatch,
    task: TaskConfig,
    train: MultiOmicsBundle,
) -> DataForModel:
    """Attach train-only condition×time means. Used by LastValue and Ridge on RCS."""

    target_means = _train_group_means(train, task.target_modality)
    # Concatenate input-modality means in the same order as input_feature_names.
    input_means: dict[tuple[str, float], np.ndarray] = {}
    keys: set[tuple[str, float]] = set()
    per_mod = [ _train_group_means(train, mod) for mod in task.input_modalities ]
    for mapping in per_mod:
        keys.update(mapping)
    for key in keys:
        parts = []
        for mapping, mod in zip(per_mod, task.input_modalities, strict=True):
            n = len(train.feature_names[mod])
            parts.append(mapping.get(key, np.zeros(n)))
        input_means[key] = np.concatenate(parts)

    last_target = forecast.last_target.copy()
    last_inputs = forecast.last_inputs.copy()
    delta_t = forecast.delta_t.copy()
    train_col_means = []
    for modality in task.input_modalities:
        train_col_means.append(np.nanmean(train.matrices[modality], axis=0))
    train_input_mean = np.concatenate(train_col_means) if train_col_means else np.zeros(0)
    if last_inputs.size:
        nan_in = np.isnan(last_inputs)
        last_inputs[nan_in] = np.take(train_input_mean, np.where(nan_in)[1])
        last_inputs = np.where(np.isnan(last_inputs), 0.0, last_inputs)
    if task.kind is TaskKind.GROUP_TIME_FORECAST:
        for i, (cond, t) in enumerate(zip(forecast.conditions, forecast.target_times, strict=True)):
            vec, dt = _lookup_last_mean(target_means, cond, float(t), last_target.shape[1])
            last_target[i] = vec
            in_vec, _ = _lookup_last_mean(input_means, cond, float(t), last_inputs.shape[1])
            last_inputs[i] = in_vec
            delta_t[i] = dt
    else:
        train_target_mean = np.nanmean(train.matrices[task.target_modality], axis=0)
        nan_t = np.isnan(last_target)
        last_target[nan_t] = np.take(train_target_mean, np.where(nan_t)[1])
        last_target = np.where(np.isnan(last_target), 0.0, last_target)
    forecast = ForecastBatch(
        instance_ids=forecast.instance_ids,
        group_ids=forecast.group_ids,
        subject_ids=forecast.subject_ids,
        conditions=forecast.conditions,
        batches=forecast.batches,
        target_times=forecast.target_times,
        delta_t=delta_t,
        y_true=forecast.y_true,
        y_mask=forecast.y_mask,
        feature_names=forecast.feature_names,
        last_target=last_target,
        last_inputs=last_inputs,
        input_feature_names=forecast.input_feature_names,
        meta=forecast.meta,
    )
    split_value = bundle.observations["split"].iloc[0] if "split" in bundle.observations else "train"
    return DataForModel(
        bundle=bundle,
        forecast=forecast,
        split=SplitName(str(split_value)),
        train_condition_time_target_mean=target_means,
        train_condition_time_input_mean=input_means,
    )


def prepare_split_data(
    *,
    full: MultiOmicsBundle,
    train: MultiOmicsBundle,
    split_bundle: MultiOmicsBundle,
    split: SplitName,
    task: TaskConfig,
) -> DataForModel:
    """Build forecast instances for one split and attach train statistics."""

    del full
    forecast = build_forecast_batch(split_bundle, task)
    data = attach_train_statistics(split_bundle, forecast, task, train)
    data.split = split
    return data
