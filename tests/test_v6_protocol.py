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
refit_every_dates = 13
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
    config.write_text(
        'historical_evaluation_status = "reused_exploratory"\nprior_experiment_fingerprint = "'
        + "a" * 64
        + '"\n'
        + config.read_text()
    )
    runner = Runner(config, root, output)
    result = runner.run()
    assert result["evaluation_status"] == "reused_exploratory"
    assert "prior experiment" in result["caveats"][0]
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
    for metadata_path in output.glob("*/*/trees.json"):
        meta = json.loads(metadata_path.read_text())
        assert meta["refit_count"] >= 2
        for refit in meta["refits"]:
            assert pd.Timestamp(refit["last_date"]) + pd.offsets.BDay(90) < pd.Timestamp(
                refit["evaluation_first_date"]
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
    assert "descriptive interval on reused historical data" in narrative.read_text()
    assert "Reused historical evaluation" in narrative.read_text()
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


def test_streaming_preserves_each_seed_sample_and_all_evaluation_rows(snapshot):
    from dataclasses import replace

    root, config_path = snapshot
    c = load_experiment(config_path)
    c = replace(c, models=(*c.models, replace(c.models[0], name="other-seed", seed=123)))
    streamed, _ = load_training_frame(root, c, max_date="2024-12-27")
    complete = pd.concat([pd.read_parquet(root / name) for name in ("train.parquet", "validation.parquet")])
    complete = complete.loc[complete.data_type.isin(["train", "validation"])].copy()
    complete.date = pd.to_datetime(complete.date)
    complete = complete.loc[complete.date <= "2024-12-27"]
    for left, right in (*c.folds, (c.calibration_start, c.calibration_end)):
        a = complete.loc[complete.date.between(left, right)].sort_values(["date", "numerai_ticker"])
        b = streamed.loc[streamed.date.between(left, right)].sort_values(["date", "numerai_ticker"])
        assert list(zip(a.numerai_ticker, a.date, strict=True)) == list(
            zip(b.numerai_ticker, b.date, strict=True)
        )
        eligible_a = complete.loc[prior_training_mask(complete.date, left, c.purge_business_days)]
        eligible_b = streamed.loc[prior_training_mask(streamed.date, left, c.purge_business_days)]
        for seed in (42, 123):
            sampled_a = sample_training(eligible_a, c.max_train_rows_per_date, seed)
            sampled_b = sample_training(eligible_b, c.max_train_rows_per_date, seed)
            for col in ["numerai_ticker", "date", "f0", "f1", c.primary_target]:
                np.testing.assert_array_equal(sampled_a[col].to_numpy(), sampled_b[col].to_numpy())
    assert len(streamed) < len(complete)


def test_inplace_ridge_preprocessing_preserves_predictions_and_input():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(12)
    x = rng.normal(size=(1200, 30)).astype(np.float32)
    x[rng.random(x.shape) < 0.08] = np.nan
    y = rng.normal(size=1200).astype(np.float32)
    predictions = []
    for copy in (True, False):
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True, copy=copy),
            StandardScaler(copy=copy),
            Ridge(alpha=100),
        )
        model.fit(x.copy(), y)
        predictions.append(model.predict(x.copy()))
    np.testing.assert_allclose(*predictions, atol=1e-7)


def test_refit_partitions_depend_only_on_dates():
    from research.validation import refit_windows

    dates = pd.date_range("2025-01-03", periods=59, freq="W-FRI")
    chunks = refit_windows(dates.repeat(3), 13)
    assert len(chunks) == 5
    covered = sum((int(dates.to_series().between(left, right).sum()) for left, right in chunks))
    assert covered == len(dates)
    assert refit_windows(dates[::-1], 13) == chunks
    assert refit_windows(dates, 0) == [(dates[0], dates[-1])]
    with pytest.raises(ValueError, match="cadence"):
        Experiment(models=(ModelSpec("bad", refit_every_dates=-1),)).validate()


def test_fixed_tree_budget_does_not_use_stopping_labels():
    from research.models import fit_model

    dates = pd.date_range("2018-01-05", periods=60, freq="W-FRI").repeat(10)
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {"date": dates, "f": rng.normal(size=len(dates)), "target_jupiter_60": rng.random(len(dates))}
    )
    spec = ModelSpec("fixed", early_stopping=False, params={"n_estimators": 7, "min_child_samples": 5})
    c = Experiment(threads=1, models=(spec,))
    fitted = fit_model(frame, ["f"], spec, c)
    assert fitted.metadata["tree_selection"] == "predeclared_fixed"
    assert fitted.metadata["selected_trees"] == 7
    assert "inner_stop_first_date" not in fitted.metadata


def test_prospective_journal_rejects_replacement_and_waits_for_complete_labels(
    snapshot, tmp_path, monkeypatch
):
    from datetime import datetime, timezone

    from scripts import prospective
    from scripts.export_live import export

    root, config = snapshot
    output = tmp_path / "research"
    Runner(config, root, output).run()
    bundle = output / "live"
    export(config, root, output, bundle)
    journal = tmp_path / "prospective"
    monkeypatch.setattr(prospective, "utc_now", lambda: datetime(2026, 10, 19, 12, tzinfo=timezone.utc))
    plan = prospective.freeze(bundle, journal)
    assert plan["first_prediction_date"] == "2026-10-20"
    with pytest.raises(ValueError, match="already exists"):
        prospective.freeze(bundle, journal)
    monkeypatch.setattr(prospective, "utc_now", lambda: datetime(2026, 10, 20, 12, tzinfo=timezone.utc))
    captured = prospective.collect(bundle, root, journal)
    assert captured["rows"] == 45
    with pytest.raises(ValueError, match="immutable"):
        prospective.collect(bundle, root, journal)
    pending = prospective.evaluate(journal, root)
    assert pending["mature_dates"] == 0
    assert pending["pending_dates"] == ["2026-10-20"]
    original = pd.read_parquet(root / "validation.parquet")
    future = original.loc[original.date == original.date.max()].copy().assign(date="2026-10-20")
    pd.concat([original, future]).to_parquet(root / "validation.parquet", index=False)
    for name in ("neutralizer", "benchmark_models"):
        path = root / f"validation_{name}.parquet"
        frame = pd.read_parquet(path)
        pd.concat([frame, pd.read_parquet(root / f"live_{name}.parquet")]).to_parquet(path, index=False)
    result = prospective.evaluate(journal, root)
    assert result["mature_dates"] == 1
    assert result["status"] == "interim; insufficient dates"
    assert result["paired_ncorr_gain_vs_public_benchmark"] is None
    prediction = journal / captured["prediction_file"]
    before = prediction.read_bytes()
    # Newly observed labels can affect scores, never the already committed predictions.
    future["target_jupiter_60"] = 1 - future.target_jupiter_60
    pd.concat([original, future]).to_parquet(root / "validation.parquet", index=False)
    assert prospective.evaluate(journal, root)["scores"] != result["scores"]
    assert prediction.read_bytes() == before
    # An incomplete scoring universe fails instead of selecting the favorable subset.
    pd.concat([original, future.iloc[:-1]]).to_parquet(root / "validation.parquet", index=False)
    with pytest.raises(ValueError, match="universe differs"):
        prospective.evaluate(journal, root)
    prediction.write_text("modified")
    with pytest.raises(ValueError, match="modified"):
        prospective.evaluate(journal, root)
    prediction.write_bytes(before)
    monkeypatch.setattr(prospective, "utc_now", lambda: datetime(2026, 10, 23, 12, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="fresh live date"):
        prospective.collect(bundle, root, journal)
