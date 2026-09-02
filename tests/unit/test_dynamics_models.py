"""Milestone-4 dynamics models: GRU, ODE-RNN, latent ODE.

Covers solver failure, constant series, irregular delta_t, batch_size=1,
and rejection of repeated cross-sectional data.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from omics_agent.errors import (  # noqa: E402
    OdeSolverError,
    TaskDesignError,
    TrainingDivergedError,
)
from omics_agent.models import get_model  # noqa: E402
from omics_agent.models.dynamics.odeint import odeint_rk4  # noqa: E402
from omics_agent.models.dynamics.plugin import _masked_mse  # noqa: E402
from omics_agent.models.tasks import prepare_split_data  # noqa: E402
from omics_agent.preprocessing.bundle import MultiOmicsBundle  # noqa: E402
from omics_agent.schemas.enums import SplitName, TaskKind  # noqa: E402
from omics_agent.schemas.experiment import ModelParams, TaskConfig  # noqa: E402
from tests.unit.dyn_helpers import make_bundle, make_split_data, make_task  # noqa: E402

DYNAMICS = ["gru", "ode_rnn", "latent_ode"]
FAST = {
    "epochs": 6,
    "batch_size": 16,
    "device": "cpu",
    "hidden_dim": 16,
    "emb_dim": 12,
    "patience": 2,
    "seed": 11,
}


def _params(name: str, **extra: object) -> ModelParams:
    return ModelParams(name=name, params={**FAST, **extra})


@pytest.mark.parametrize("name", DYNAMICS)
def test_fit_predict_save(name: str, tmp_path: Path) -> None:
    train, val = make_split_data(make_bundle(), make_task())
    model = get_model(name)
    fit = model.fit(train, val, _params(name))
    assert fit.n_parameters > 0
    assert fit.extras["mode"] == name
    assert fit.extras["device"] == "cpu"
    pred = model.predict(val)
    assert pred.y_pred.shape == val.forecast.y_true.shape
    assert np.isfinite(pred.y_pred).all()
    model.save(tmp_path / name)
    assert (tmp_path / name / "model.pt").is_file()
    card = json.loads((tmp_path / name / "card.json").read_text(encoding="utf-8"))
    assert card["fit_split"] == "train"
    table = model.explain(val, [])
    assert "not causation" in table.caveat


@pytest.mark.parametrize("name", ["gru", "latent_ode"])
def test_constant_series_trains_and_predicts_finite(name: str) -> None:
    train, val = make_split_data(make_bundle(constant=True), make_task())
    model = get_model(name)
    model.fit(train, val, _params(name))
    pred = model.predict(val)
    assert np.isfinite(pred.y_pred).all()
    # Constant-vector PCC is the evaluator's job (reported as NA); the model
    # must simply not crash or emit NaN here.


@pytest.mark.parametrize("name", DYNAMICS)
def test_actual_delta_t_changes_the_prediction(name: str) -> None:
    train, val = make_split_data(make_bundle(), make_task())
    model = get_model(name)
    model.fit(train, val, _params(name))
    base = model.predict(val).y_pred
    stretched = copy.deepcopy(val)
    stretched.forecast.meta["target_time"] = stretched.forecast.meta["target_time"] + 5.0
    shifted = model.predict(stretched).y_pred
    # Same histories, same conditions — only the gap to the target changed.
    assert not np.allclose(base, shifted)


def test_batch_size_one() -> None:
    train, val = make_split_data(make_bundle(n_units=4), make_task())
    model = get_model("ode_rnn")
    model.fit(train, val, _params("ode_rnn", batch_size=1, epochs=2))
    pred = model.predict(val)
    assert np.isfinite(pred.y_pred).all()


def test_odeint_nan_raises_solver_error() -> None:
    exploding = torch.nn.Linear(4, 4)
    with torch.no_grad():
        exploding.weight.fill_(1e30)
        exploding.bias.fill_(0.0)
    state = torch.ones(2, 4)
    dt = torch.ones(2, 1)
    with pytest.raises(OdeSolverError, match="NaN/inf"):
        odeint_rk4(exploding, state, dt, substeps=2)


def test_odeint_rejects_bad_substeps() -> None:
    with pytest.raises(OdeSolverError, match="substeps"):
        odeint_rk4(lambda s: s, torch.ones(1, 2), torch.ones(1, 1), substeps=0)


def test_nan_loss_stops_training_with_typed_error() -> None:
    train, val = make_split_data(make_bundle(), make_task())
    model = get_model("gru")
    with pytest.raises((TrainingDivergedError, OdeSolverError)):
        model.fit(train, val, _params("gru", lr=1e12, epochs=5))


def test_masked_mse_ignores_nan_targets() -> None:
    pred = torch.zeros(2, 2)
    target = torch.tensor([[1.0, float("nan")], [3.0, 2.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    value = float(_masked_mse(pred, target, mask))
    assert value == pytest.approx((1.0 + 9.0 + 4.0) / 3.0)


def test_repeated_cross_section_is_rejected(rcs_bundle: MultiOmicsBundle) -> None:
    units = sorted(rcs_bundle.observations["experimental_unit_id"].unique())
    full = rcs_bundle.with_split(pd.DataFrame({"experimental_unit_id": units, "split": "train"}))
    train_b = full.subset(SplitName.TRAIN)
    task = TaskConfig(
        kind=TaskKind.GROUP_TIME_FORECAST,
        target_modality="protein",
        input_modalities=["rna", "protein"],
        target_time_min=float(train_b.observations["time"].min()),
    )
    data = prepare_split_data(
        full=full, train=train_b, split_bundle=train_b, split=SplitName.TRAIN, task=task
    )
    with pytest.raises(TaskDesignError, match="longitudinal"):
        get_model("gru").fit(data, None, _params("gru"))
