"""MLP baseline and torch-free sequence-builder tests."""

from __future__ import annotations

import numpy as np
import pytest

from omics_agent.errors import SchemaError
from omics_agent.models import get_model
from omics_agent.models.dynamics.sequences import build_sequences, modalities_from_input_names
from omics_agent.schemas.experiment import ModelParams
from tests.unit.dyn_helpers import make_bundle, make_split_data, make_task


def test_mlp_fit_predict_shapes_and_explain() -> None:
    train, val = make_split_data(make_bundle(), make_task())
    model = get_model("mlp")
    fit = model.fit(train, val, ModelParams(name="mlp", params={"max_iter": 200, "seed": 3}))
    assert fit.n_parameters > 0
    pred = model.predict(val)
    assert pred.y_pred.shape == val.forecast.y_true.shape
    assert np.isfinite(pred.y_pred).all()
    table = model.explain(val, [])
    assert "not causation" in table.caveat


def test_mlp_rejects_bad_hidden_sizes() -> None:
    train, val = make_split_data(make_bundle(), make_task())
    model = get_model("mlp")
    with pytest.raises(SchemaError, match="hidden_layer_sizes"):
        model.fit(train, val, ModelParams(name="mlp", params={"hidden_layer_sizes": "big"}))


def test_modalities_recovered_from_input_names() -> None:
    names = ["rna:G1", "rna:G2", "protein:P1"]
    assert modalities_from_input_names(names) == ["rna", "protein"]


def test_build_sequences_uses_actual_gaps_masks_and_padding() -> None:
    train, _ = make_split_data(make_bundle(), make_task())
    seq = build_sequences(train, condition_categories=["control", "disease"], model_name="test")
    assert seq.modalities == ["rna", "protein"]
    # Times are (0, 1, 2, 4). The longest history (target t=4) has gaps 0,1,1
    # and a 2-hour jump to the target — the actual delta_t, not a unit grid.
    longest = int(np.argmax(seq.lengths))
    n = int(seq.lengths[longest])
    assert n == 3
    assert np.allclose(seq.step_dt[longest, :n], [0.0, 1.0, 1.0])
    assert seq.target_dt[longest] == pytest.approx(2.0)
    assert not seq.pad[longest, n:].any() if n < seq.pad.shape[1] else True
    # Shorter histories are padded with pad=False steps.
    shortest = int(np.argmin(seq.lengths))
    assert not seq.pad[shortest, int(seq.lengths[shortest]) :].any()
    # The injected NaN protein value is masked out and zero-filled.
    assert seq.masks["protein"].min() == 0.0
    nan_free = np.isfinite(np.concatenate([seq.values[m].ravel() for m in seq.modalities]))
    assert nan_free.all()


def test_build_sequences_rejects_unseen_condition() -> None:
    train, _ = make_split_data(make_bundle(), make_task())
    with pytest.raises(SchemaError, match="never seen during training"):
        build_sequences(train, condition_categories=["control"], model_name="test")
