import pandas as pd
import pytest

from dengai import tuning
from dengai.forecast import make_submission
from dengai.pipeline import run


def test_submission_matches_template_order_and_is_integer(raw):
    preds = raw.test_features[["city", "year", "weekofyear"]].iloc[::-1].assign(pred=2.4)  # shuffled on purpose
    submission = make_submission(preds, raw.submission_format)
    assert list(submission.columns) == list(raw.submission_format.columns)
    assert submission[["city", "year", "weekofyear"]].equals(raw.submission_format[["city", "year", "weekofyear"]])
    assert submission.total_cases.eq(2).all() and str(submission.total_cases.dtype).startswith("int")


def test_submission_rejects_missing_predictions(raw):
    preds = raw.test_features[["city", "year", "weekofyear"]].iloc[:-1].assign(pred=1.0)
    with pytest.raises(ValueError, match="no prediction"):
        make_submission(preds, raw.submission_format)


def test_end_to_end_writes_a_valid_submission(raw, tmp_path, monkeypatch):
    """Smoke test on synthetic data with a tiny tuning grid (the real grid is exercised by `python -m dengai run`)."""
    monkeypatch.setattr(tuning, "NB_LAMBDAS", [0.1, 1])
    monkeypatch.setattr(tuning, "NB_ALPHAS", ["moment", 1.0])
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw.train_features.to_csv(raw_dir / "dengue_features_train.csv", index=False)
    raw.train_labels.to_csv(raw_dir / "dengue_labels_train.csv", index=False)
    raw.test_features.to_csv(raw_dir / "dengue_features_test.csv", index=False)
    raw.submission_format.to_csv(raw_dir / "submission_format.csv", index=False)

    summary = run(raw_dir, tmp_path / "out", compare=False)

    submission = pd.read_csv(tmp_path / "out" / "submission.csv")
    assert len(submission) == len(raw.submission_format) == summary["n_submission_rows"]
    assert submission.total_cases.ge(0).all()
    for name in ("report_window_score.csv", "split_audit.csv", "run_summary.json"):
        assert (tmp_path / "out" / name).is_file()
