"""Tests for the aggregation experiment's incremental save / resume logic.

An interrupted multi-hour run must keep every finished variant, and resuming
must never mix results produced under different holdouts or CV settings.
"""

import json

import numpy as np
import pytest

from src.fall_risk_assessment.experiments.aggregation_comparison import (
    _AGGREGATORS,
    load_resumable_report,
    parse_variants,
    write_report,
)

HOLDOUTS = ["CO002", "FL004"]
CV = {"n_splits": 5, "n_repeats": 5, "fast_mode": False, "random_state": 42}


def _report(variants_done):
    return {
        "holdout_participant_ids": HOLDOUTS,
        "cv": dict(CV),
        "requested_variants": list(_AGGREGATORS),
        "variants": {name: {"nested_oof_roc_auc": 0.5} for name in variants_done},
    }


def test_parse_variants_accepts_known_subset_in_given_order():
    assert parse_variants("mean_std_iqr, mean") == ["mean_std_iqr", "mean"]


@pytest.mark.parametrize("raw", ["", "median", "mean,bogus"])
def test_parse_variants_rejects_unknown_or_empty(raw):
    with pytest.raises(SystemExit):
        parse_variants(raw)


def test_write_report_marks_partial_until_all_requested_variants_exist(tmp_path):
    path = tmp_path / "report.json"
    report = _report(["mean"])
    write_report(path, report, list(_AGGREGATORS))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "partial"
    assert list(saved["variants"]) == ["mean"]
    assert not (tmp_path / "report.json.tmp").exists()

    report["variants"]["mean_std"] = {"nested_oof_roc_auc": 0.5}
    report["variants"]["mean_std_iqr"] = {"nested_oof_roc_auc": 0.5}
    write_report(path, report, list(_AGGREGATORS))
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "complete"


def test_resume_keeps_finished_variants(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, _report(["mean"]), list(_AGGREGATORS))
    resumed = load_resumable_report(path, HOLDOUTS, CV)
    assert list(resumed["variants"]) == ["mean"]


def test_resume_refuses_different_holdouts(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, _report(["mean"]), list(_AGGREGATORS))
    with pytest.raises(SystemExit):
        load_resumable_report(path, ["CO010"], CV)


def test_resume_refuses_different_cv_settings(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, _report(["mean"]), list(_AGGREGATORS))
    with pytest.raises(SystemExit):
        load_resumable_report(path, HOLDOUTS, {**CV, "n_repeats": 3})


def test_aggregators_produce_expected_widths():
    block = np.arange(12, dtype=float).reshape(4, 3)
    widths = {
        name: np.concatenate([agg(block) for agg in aggs]).shape[0]
        for name, aggs in _AGGREGATORS.items()
    }
    assert widths == {"mean": 3, "mean_std": 6, "mean_std_iqr": 9}
