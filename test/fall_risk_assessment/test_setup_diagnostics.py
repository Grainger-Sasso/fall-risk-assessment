"""Tests for the setup-diagnostics helpers.

The guarantees the systematic review relies on: clinical sheets parse into
the right labels and numbers, oriented AUCs respect the a-priori direction,
the family-wise screen does not flag noise but does flag a planted signal,
and the fixed-model evaluator reports chance on noise and signal on signal.
"""

import numpy as np
import pandas as pd
import pytest

from src.fall_risk_assessment.experiments.setup_diagnostics import (
    evaluate_fixed_model,
    max_stat_screen,
    normalize_participant_id,
    oriented_auc,
    parse_clinical_sheets,
)


def _sheets():
    controls = pd.DataFrame(
        {
            "#": ["CO-001", "CO-002"],
            "Gender(1-female, 0-male)": [1, 0],
            "Age": [75.0, 82.0],
            "Year Fall ": [1, 0],
            "TUG": [8.0, "N/A"],
            "BERG": [52, 54],
        }
    )
    fallers = pd.DataFrame(
        {
            "#": ["FL-001"],
            "Gender(0-male,1-female)": [0],
            "Age": [79.0],
            "Year Fall ": [5],
            "TUG": [11.0],
            "BERG": [48],
        }
    )
    return {"Controls": controls, "Fallers": fallers, "Legend": pd.DataFrame()}


def test_normalize_participant_id():
    assert normalize_participant_id(" co-001 ") == "CO001"
    assert normalize_participant_id("FL001") == "FL001"


def test_parse_clinical_sheets_labels_and_numbers():
    table = parse_clinical_sheets(_sheets())
    assert list(table.index) == ["CO001", "CO002", "FL001"]
    assert table["group_faller"].tolist() == [0, 0, 1]
    assert table["female"].tolist() == [1, 0, 0]
    assert np.isnan(table.loc["CO002", "tug_s"])  # "N/A" -> NaN
    assert table.loc["FL001", "falls_past_year"] == 5
    assert np.isnan(table.loc["FL001", "dgi"])  # absent column -> NaN


def test_oriented_auc_respects_direction_and_missing_values():
    y = np.array([0, 0, 1, 1])
    x = np.array([1.0, 2.0, 3.0, np.nan])
    assert oriented_auc(y, x, +1) == {"auc": 1.0, "n": 3}
    assert oriented_auc(y, x, -1)["auc"] == 0.0


def test_screen_does_not_flag_pure_noise():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 30)
    X = rng.normal(size=(60, 20))
    screen = max_stat_screen(X, y, [f"f{i}" for i in range(20)], n_permutations=500)
    assert screen["familywise_p_of_max"] > 0.05


def test_screen_flags_planted_signal_familywise():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 30)
    X = rng.normal(size=(60, 20))
    X[:, 7] += 1.5 * y
    screen = max_stat_screen(X, y, [f"f{i}" for i in range(20)], n_permutations=500)
    top = screen["features"][0]
    assert top["feature"] == "f7"
    assert top["p_familywise"] < 0.01


def test_nested_best_feature_finds_planted_signal_and_not_noise():
    from functools import partial

    from src.fall_risk_assessment.experiments.setup_diagnostics import (
        select_k_best_oof,
        selection_frequencies,
    )

    rng = np.random.default_rng(3)
    y = np.array([0, 1] * 30)
    X = rng.normal(size=(60, 30))
    noise = evaluate_fixed_model(
        X, y, n_permutations=0, n_bootstrap=200, oof_fn=partial(select_k_best_oof, k=1)
    )
    assert 0.25 <= noise["auc"] <= 0.75

    X[:, 11] += 2.0 * y
    signal = evaluate_fixed_model(
        X, y, n_permutations=0, n_bootstrap=200, oof_fn=partial(select_k_best_oof, k=1)
    )
    assert signal["auc"] > 0.85
    picks = selection_frequencies(X, y, [f"f{i}" for i in range(30)], k=1)
    assert next(iter(picks)) == "f11"


@pytest.mark.parametrize("signal, low, high", [(0.0, 0.3, 0.7), (2.0, 0.85, 1.0)])
def test_fixed_model_reports_chance_on_noise_and_signal_on_signal(signal, low, high):
    rng = np.random.default_rng(2)
    y = np.array([0, 1] * 25)
    X = rng.normal(size=(50, 3))
    X[:, 0] += signal * y
    result = evaluate_fixed_model(X, y, n_permutations=20, n_bootstrap=200, n_jobs=1)
    assert low <= result["auc"] <= high
    assert result["ci_lower"] <= result["auc"] <= result["ci_upper"]
    if signal:
        assert result["permutation_p"] < 0.1
