"""Write a compact, evidence-derived reference report without changing selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.verify_report import verify


def write(root, output, figure="holdout.svg"):
    root, output = Path(root), Path(output)
    verify(root)

    def read(name):
        return json.loads((root / name).read_text())

    plan, result, comparison = read("plan.json"), read("result.json"), read("benchmark_comparison.json")
    c, selection = plan["experiment"], result["selection"]
    exploratory = result.get("evaluation_status") == "reused_exploratory"
    title = "Numerai v6 exploratory follow-up" if exploratory else "Numerai v6 reference experiment"
    evaluation_heading = "Reused historical evaluation" if exploratory else "Reserved historical evaluation"
    reuse_note = (
        "This evaluation period was already inspected in the first study. Results below are exploratory, and their intervals do not establish independent improvement."
        if exploratory
        else ""
    )
    scores = pd.read_csv(root / "holdout_scores.csv")
    names = ["selected_ensemble", result["baseline"], c["benchmark"]]
    labels = ["Frozen selection", f"Control `{result['baseline']}`", "Public `lgbm_jupiter_60`"]
    rows = []
    for name, label in zip(names, labels, strict=True):
        stats = comparison["models"][name]
        lower, upper = stats["mean_ci95"]
        contribution = (
            result["holdout"][name]["benchmark_contribution"]["mean"] if name in result["holdout"] else 0.0
        )
        rows.append(f"| {label} | {stats['mean']:.6f} | [{lower:.6f}, {upper:.6f}] | {contribution:.6f} |")
    gain = result["paired_ncorr_gain_vs_baseline"]
    benchmark_gain = comparison["paired_gain_vs_public_benchmark"]

    def conclusion(metric, label):
        low, high = metric["mean_ci95"]
        if exploratory:
            statement = "This is a descriptive interval on reused historical data, not an independent improvement test."
        elif low > 0:
            statement = "The historical paired gain is positive under this approximate interval."
        elif high < 0:
            statement = "The selection underperforms under this approximate interval."
        else:
            statement = "The interval includes zero; a historical improvement is not established."
        return (
            f"Against {label}, the paired mean NCORR difference is **{metric['mean']:+.6f}** "
            f"(95% block-bootstrap interval [{low:+.6f}, {high:+.6f}]). {statement}"
        )

    cv = read("cv_summary.json")
    cv_order = sorted(
        cv,
        key=lambda name: cv[name]["ncorr"]["mean"] - c["selection_risk_penalty"] * cv[name]["ncorr"]["std"],
        reverse=True,
    )
    fits = read("model_fits.json")
    cv_rows = []
    for name in cv_order:
        seconds = sum(
            f["fit_seconds"] for f in fits if f["name"] == name and f["checkpoint"].startswith("cv/")
        )
        cv_rows.append(
            f"| `{name}` | {cv[name]['ncorr']['mean']:.6f} | {cv[name]['ncorr']['std']:.6f} | "
            f"{cv[name]['benchmark_contribution']['mean']:.6f} | {seconds / 60:.2f} |"
        )
    events = read("execution.json") if (root / "execution.json").exists() else []
    starts = [datetime.fromisoformat(e["utc"]) for e in events if e["event"] == "run_ready"]
    ends = [datetime.fromisoformat(e["utc"]) for e in events if e["event"] == "research_complete"]
    wall = f"{(ends[-1] - starts[0]).total_seconds() / 60:.1f} minutes" if starts and ends else "not captured"
    hardware = read("hardware.json") if (root / "hardware.json").exists() else {}
    machine = hardware.get("cpu", plan["platform"])
    memory = f"{hardware['memory_gib']} GiB RAM" if "memory_gib" in hardware else "memory not captured"
    completed = ends[-1].date().isoformat() if ends else "completion date not captured"
    source_commit = hardware.get("training_source_commit", "see source hashes in plan.json")
    parity = read("score_parity.json") if (root / "score_parity.json").exists() else None
    parity_note = (
        f"Official-metric parity on {parity['rows']:,} real stocks and {parity['neutralizer_columns']} neutralizer columns: "
        f"maximum absolute difference **{parity['max_absolute_difference']:.2e}**, recorded in the parity check."
        if parity
        else "Numerical parity tests are included in the test suite; no separate real-data check was packaged."
    )
    weights = ", ".join(f"`{name}` {weight:.1%}" for name, weight in selection["weights"].items())
    path = root.name
    live = read("live_export.json") if (root / "live_export.json").exists() else None
    live_note = (
        f"The frozen selection was also refitted for the pinned **{live['date']}** snapshot: "
        f"**{live['rows']:,}** ticker/date predictions, with model and CSV hashes in "
        f"[`live_export.json`]({path}/live_export.json). This dated local export was not uploaded, staked or traded."
        if live
        else "No dated prediction export is included in this evidence package."
    )
    output.write_text(f"""# {title}

Completed {completed} on **{machine}**, {memory}, using **{c["threads"]} CPU threads**. Public snapshot round **{plan["round"]}**; **{len(c["models"])}** predeclared candidates and **{len(c["folds"])}** purged walk-forward folds. AI-assisted implementation.

## {evaluation_heading}

{reuse_note}

**{scores.date.min()} through {scores.date.max()}**, {scores.date.nunique()} evaluation dates. All eligible stocks are scored on the same dates. Scores are not trading returns.

| Model | Mean NCORR | 95% interval for mean NCORR | Mean fixed-benchmark contribution |
| --- | ---: | --- | ---: |
{chr(10).join(rows)}

{conclusion(gain, "the predeclared control")}

{conclusion(benchmark_gain, "the public benchmark")}

![Historical NCORR paths]({figure})

Intervals use {c["bootstrap_block_dates"]}-date circular blocks and 2,000 draws. They approximate serial dependence from overlapping 60-day outcomes; they do not correct for every research decision or establish a live edge. Contribution is measured against the fixed public benchmark, whose contribution to itself is zero, rather than Numerai's changing meta-model.

## Selection and evaluation access

CV ranking: mean NCORR minus {c["selection_risk_penalty"]} × between-date standard deviation. The shortlist was **{", ".join(selection["shortlist"])}**. Calibration used {c["calibration_start"]} through {c["calibration_end"]} with **{selection["calibration_trials"]} recorded trials**; the later evaluation starts only after a further maturity gap.

Selected weights: **{weights}**. Feature-neutralization proportion: **{selection["neutralization"]:.2f}**, against the top 50 contemporaneously exposed medium-set numeric features, chosen without targets.

The [selection]({path}/selection.json), [complete calibration trial log]({path}/calibration_trials.json) and [holdout access journal]({path}/holdout_access.json) record this boundary. The historical holdout is now consumed. Changing the model after reading these results requires fresh prospective evidence; public historical data is not guaranteed never to have informed earlier work.

## All walk-forward candidates

These are the complete CV results, including losing candidates. Candidates marked `early_stopping=true` select tree counts using purged inner blocks; `false` uses its predeclared fixed tree budget. Each refit sees only labels mature before its own prediction boundary. The control is the first candidate in the frozen plan, not an exact reproduction of Numerai's example or benchmark.

| Candidate | Mean NCORR | Date-score std | Mean benchmark contribution | CV fit minutes |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(cv_rows)}

## Measured compute and data

- Research wall time: **{wall}**, including sampling, prediction, scoring and calibration; excludes download and later export/reporting.
- Model fits: **{len(fits)}**, totaling **{sum(f["fit_seconds"] for f in fits) / 60:.1f} fit minutes**. [Fit metadata]({path}/model_fits.json) includes row counts, dates, tree counts, seeds and hashes.
- Largest fit: **{max(f["rows"] for f in fits):,}** training observations; deterministic cap of **{c["max_train_rows_per_date"]:,}** per date. Evaluation stocks were not sampled.
- Snapshot files: **{sum(v["bytes"] for v in plan["dataset"].values()) / 2**30:.2f} GiB**, including supplied neutralizers and the dated live preview.
- Source, exact dependency versions, complete configuration and dataset hashes: [frozen plan]({path}/plan.json). Training source commit: `{source_commit}`. Use the source hashes recorded with this experiment.
- {parity_note}

{live_note}

## Reproduce and inspect

From the repository root, after `uv sync --frozen --extra dev --no-editable`:

```bash
# Offline consistency checks; no dataset download or training.
uv run --frozen --no-editable python -m scripts.verify_report reports/{path}

# Restore the exact recorded public round, while it remains available.
uv run --frozen --no-editable python -m scripts.download_reference \\
  reports/{path} --data data/reference-v6-{plan["round"]}
uv run --frozen --no-editable solstice run \\
  --config configs/{"rolling" if "rolling" in c["name"] else "wide" if "wide" in c["name"] else "mac"}.toml \\
  --data data/reference-v6-{plan["round"]} --output runs/reproduction
```

Small floating-point differences across hardware and libraries are possible. A changed provider file fails the reference hash check rather than silently becoming the same experiment. [The evidence manifest]({path}/evidence_manifest.json) verifies the portable score files; this is internal consistency, not an independent replication.

To regenerate the plot, install the optional `reports` dependency group, then run `python -m scripts.plot_report reports/{path} --output reports/holdout.svg`. The [research protocol](../docs/RESEARCH.md) lists technique sources and the untested follow-up experiments.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", default="holdout.svg")
    args = parser.parse_args()
    write(args.root, args.output, args.figure)


if __name__ == "__main__":
    main()
