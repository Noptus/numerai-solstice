"""Real parquet/CPU integration tests; public downloads and uploads are never used."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.config import Experiment, ModelSpec, load_experiment
from research.data import atomic_json, download_snapshot, file_hash, load_training_frame, validate_keys
from research.runner import Runner
from research.validation import inner_split, prior_training_mask, sample_training, training_weights


def test_dates_purge_target_horizon_and_inner_stopping():
    dates = pd.date_range("2016-01-01", "2022-01-01", freq="W-FRI")
    fit, stop = inner_split(dates, 26, 90)
    assert dates[fit].max() + pd.offsets.BDay(90) < dates[stop].min()
    assert not (fit & stop).any()
    mask = prior_training_mask(dates, "2021-06-04", 90)
    assert (dates[mask] + pd.offsets.BDay(90) < pd.Timestamp("2021-06-04")).all()
    with pytest.raises(ValueError, match="purge"):
        Experiment(models=(ModelSpec("x", target="target_jupiter_60"),), purge_business_days=60).validate()


def test_sampling_is_order_independent_and_label_independent():
    frame = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2020-01-01", periods=3), 200),
            "numerai_ticker": [f"t{i}" for i in range(600)],
            "target": np.arange(600),
        }
    )
    original = sample_training(frame, 20, 42)
    changed = frame.sample(frac=1, random_state=3).assign(target=-100)
    pd.testing.assert_frame_equal(
        original[["date", "numerai_ticker"]].reset_index(drop=True),
        sample_training(changed, 20, 42)[["date", "numerai_ticker"]].reset_index(drop=True),
    )
    assert len(set(original.numerai_ticker) & set(sample_training(frame, 20, 123).numerai_ticker)) < 40
    weights = training_weights(frame.date, equal_date=True, half_life_weeks=1)
    assert weights[-1] > weights[0]
    assert weights.mean() == pytest.approx(1)


def test_keys_are_normalized_before_duplicate_checks():
    frame = pd.DataFrame({"date": ["2020-01-01", "2020-01-01"], "numerai_ticker": ["a", "a"]})
    with pytest.raises(ValueError, match="duplicated"):
        validate_keys(frame)


@pytest.fixture
def snapshot(tmp_path):
    rng = np.random.default_rng(233)
    root = tmp_path / "data"
    root.mkdir()
    dates = pd.date_range("2003-01-03", "2026-07-31", freq="2W-FRI")
    rows = []
    neutral = []
    benchmark = []
    for date in dates:
        x = rng.integers(0, 5, size=(45, 2))
        y = pd.Series(x[:, 1] + rng.normal(size=45)).rank(pct=True).to_numpy()
        for i in range(45):
            key = {"date": str(date.date()), "numerai_ticker": f"stock{i}"}
            rows.append(
                {
                    **key,
                    "data_type": "train" if date.year < 2019 else "validation",
                    "f0": x[i, 0],
                    "f1": x[i, 1],
                    "feature_country": "US",
                    "target_jupiter_60": y[i],
                }
            )
            neutral.append({**key, **{f"neutralizer_column_{j}": rng.normal() for j in range(3)}})
            benchmark.append({**key, "lgbm_jupiter_60": x[i, 0] + rng.normal()})
    frame = pd.DataFrame(rows)
    # This row would contaminate all partitions if data_type filtering regressed.
    hidden = (
        frame.loc[frame.data_type == "validation"]
        .iloc[[0]]
        .assign(numerai_ticker="DO_NOT_TRAIN", data_type="test", target_jupiter_60=0.5)
    )
    train = frame.loc[frame.data_type == "train"]
    validation = pd.concat([frame.loc[frame.data_type == "validation"], hidden], ignore_index=True)
    for name, data in (
        ("train.parquet", train),
        ("validation.parquet", validation),
        ("validation_neutralizer.parquet", pd.DataFrame(neutral)),
        ("validation_benchmark_models.parquet", pd.DataFrame(benchmark)),
    ):
        data.to_parquet(root / name, index=False)
    atomic_json(
        root / "features.json",
        {
            "feature_sets": {
                "small": ["f0", "f1", "feature_country"],
                "medium": ["f0", "f1", "feature_country"],
            }
        },
    )
    live = (
        frame.loc[frame.date == frame.date.max()]
        .drop(columns=["target_jupiter_60"])
        .assign(date="2026-10-20", data_type="live")
    )
    live.to_parquet(root / "live.parquet", index=False)
    pd.DataFrame(neutral).iloc[-45:].assign(date="2026-10-20").to_parquet(
        root / "live_neutralizer.parquet", index=False
    )
    pd.DataFrame(benchmark).iloc[-45:].assign(date="2026-10-20").to_parquet(
        root / "live_benchmark_models.parquet", index=False
    )
    files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in root.iterdir()}
    atomic_json(root / "snapshot.json", {"dataset_version": "v6.0", "round": 123, "files": files})
    config = tmp_path / "test.toml"
    config.write_text("""name = "integration"
threads = 1
max_train_rows_per_date = 20
early_stopping_dates = 6
early_stopping_rounds = 5
folds = [["2021-06-04", "2022-06-03"]]
max_ensemble_models = 2
ensemble_steps = 2
neutralization_proportions = [0.0, 0.5]
[[models]]
name = "baseline"
kind = "ridge"
feature_set = "small"
[models.params]
alpha = 100.0
[[models]]
name = "trees"
[models.params]
n_estimators = 20
num_leaves = 5
min_child_samples = 20
learning_rate = 0.1
""")
    return root, config


def test_real_cpu_pipeline_seals_labels_and_resumes(snapshot, tmp_path, monkeypatch):
    root, config = snapshot
    output = tmp_path / "run"
    import research.runner as runner_module

    original = runner_module.load_training_frame
    accesses = []

    def checked_load(*args, **kwargs):
        accesses.append(kwargs["max_date"])
        if kwargs["max_date"] == "2100-01-01":
            assert (output / "selection.json").is_file()
            assert (output / "holdout_access.json").is_file()
        frame, sets = original(*args, **kwargs)
        assert "DO_NOT_TRAIN" not in set(frame.numerai_ticker)
        return frame, sets

    monkeypatch.setattr(runner_module, "load_training_frame", checked_load)
    runner = Runner(config, root, output)
    result = runner.run()
    assert result["holdout"]["selected_ensemble"]["ncorr"]["eligible"]
    assert result["paired_ncorr_gain_vs_baseline"]["eligible"]
    assert accesses.index("2100-01-01") > accesses.index("2024-12-27")
    for path in output.glob("*/*/trees.json"):
        metadata = json.loads(path.read_text())
        assert pd.Timestamp(metadata["last_date"]) + pd.offsets.BDay(90) < pd.Timestamp(
            metadata["evaluation_first_date"]
        )
        assert pd.Timestamp(metadata["inner_fit_last_date"]) + pd.offsets.BDay(90) < pd.Timestamp(
            metadata["inner_stop_first_date"]
        )
    # Resume returns the saved result without fitting or re-opening labels.
    monkeypatch.setattr(runner, "train", lambda *_: pytest.fail("completed run retrained"))
    assert runner.run() == result
    from scripts.check_score_parity import check

    baseline_predictions = next(output.glob("cv/*/baseline.parquet"))
    first_date = pd.read_parquet(baseline_predictions).date.iloc[0]
    parity = check(root, baseline_predictions, first_date)
    assert parity["max_absolute_difference"] < 1e-8
    from scripts.benchmark_report import compare

    comparison = compare(config, root, output)
    assert comparison["models"]["lgbm_jupiter_60"]["eligible"]
    assert comparison["paired_gain_vs_public_benchmark"]["eligible"]
    from scripts.export_live import export

    exported = export(config, root, output, output / "live")
    assert exported["rows"] == 45
    assert exported["date"] == "2026-10-20"
    assert list(pd.read_csv(output / "live/predictions.csv")) == ["numerai_ticker", "date", "prediction"]
    assert export(config, root, output, output / "live") == exported
    from scripts.download_live import download
    from scripts.predict_bundle import predict

    class LiveClient:
        def get_current_round(self):
            return 124

        def list_datasets(self, round_num):
            assert round_num == 124
            return ["v6.0/live.parquet"]

        def download_dataset(self, name, path, round_num):
            assert name == "v6.0/live.parquet" and round_num == 124
            Path(path).write_bytes((root / "live.parquet").read_bytes())

    live_only = tmp_path / "next-live"
    assert download(live_only, client=LiveClient())["round"] == 124
    assert len(list(live_only.glob("*.parquet"))) == 1
    inference_path = tmp_path / "predictions.csv"
    inferred = predict(output / "live", live_only, inference_path)
    assert inferred["round"] == 124 and inferred["rows"] == 45
    np.testing.assert_allclose(
        pd.read_csv(inference_path).prediction,
        pd.read_csv(output / "live/predictions.csv").prediction,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="another snapshot"):
        download(live_only, round_num=125, client=LiveClient())
    with pytest.raises(ValueError, match="immutable"):
        predict(output / "live", live_only, output / "live/predictions.csv")
    from scripts.package_report import package
    from scripts.verify_report import verify

    evidence = tmp_path / "portable-evidence"
    package(output, evidence)
    assert verify(evidence)["verified_files"] >= 14
    from scripts.download_reference import restore

    class CachedClient:
        def list_datasets(self, round_num):
            assert round_num == 123
            return ["v6.0/" + name for name in json.loads((root / "snapshot.json").read_text())["files"]]

        def download_dataset(self, *args, **kwargs):
            pytest.fail("cached reference sources unexpectedly downloaded")

    assert restore(evidence, root, client=CachedClient())["verified_files"] == 8
    from scripts.write_report import write

    narrative = tmp_path / "report.md"
    write(evidence, narrative)
    assert "paired mean NCORR difference" in narrative.read_text()
    (evidence / "cv_scores.csv").write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        verify(evidence)
    (output / "live/predictions.csv").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        export(config, root, output, output / "live")
    checkpoint = next(output.glob("cv/*/trees.parquet"))
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        runner._check_checkpoint(checkpoint.with_suffix(""))


def test_snapshot_and_plan_changes_are_rejected(snapshot, tmp_path):
    root, config = snapshot
    output = tmp_path / "run"
    Runner(config, root, output)
    config.write_text(config.read_text().replace('name = "integration"', 'name = "different"'))
    with pytest.raises(ValueError, match="identity changed"):
        Runner(config, root, output)
    (root / "features.json").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        Runner(config, root, tmp_path / "new")


def test_unavailable_download_fails_before_partial_sources(tmp_path):
    class Client:
        def get_current_round(self):
            return 1

        def list_datasets(self, round_num):
            return ["v5.2/train.parquet"]

        def download_dataset(self, *args, **kwargs):
            pytest.fail("download attempted despite missing v6 catalog")

    with pytest.raises(ValueError, match="unavailable"):
        download_snapshot(tmp_path, client=Client())
    assert not (tmp_path / "train.parquet").exists()


def test_holdout_cannot_open_before_selection(snapshot, tmp_path):
    root, config = snapshot
    runner = Runner(config, root, tmp_path / "run")
    with pytest.raises(FileNotFoundError):
        runner.frame("holdout")
    frame, _ = load_training_frame(root, load_experiment(config), max_date="2023-12-31")
    assert frame.date.max() <= pd.Timestamp("2023-12-31")
