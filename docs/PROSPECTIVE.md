# Prospective evidence

Historical results informed development. Another historical score does not make the old evaluation period unseen. The next defensible step is a publicly timestamped plan followed by predictions made before their outcomes exist.

`scripts.prospective` supports a conservative **frozen-model** test. It does not refit quarterly during this evaluation. Historical rolling-policy results and a prospective fixed-bundle test are different experiments and must be described separately. The dated export refreshes selected members on the same eligible history. Identical deterministic tree recipes can then share one fitted model, even when their historical refresh cadences differed. Export metadata records these aliases; frozen weights and prediction values are preserved. Inference loads and evaluates each identical model binary once.

## Freeze once

After completing a study and exporting its selected bundle:

```bash
uv run --frozen --no-editable python -m scripts.prospective freeze \
  --bundle runs/mac-rolling-v6-1381/live-export --journal runs/prospective-v6
```

Publish the resulting `plan.json` in ordinary Git history before collecting future predictions. It fixes the bundle hash, source/dependency identities, target, public benchmark, date-selection policy and reporting threshold. The earliest accepted prediction date is the next business day after freezing. The plan cannot be overwritten.

This test uses the first captured complete live date per ISO week. Capture consistently each week; missing weeks must be disclosed. It accepts only fresh snapshots and rejects a second capture in the same week, even if the new model score might later be better. The model bundle cannot change. A model change starts a separately reported experiment; it cannot silently replace this one.

## Collect future observations

```bash
uv run --frozen --no-editable python -m scripts.download_live --data data/next-live
uv run --frozen --no-editable python -m scripts.prospective collect \
  --bundle runs/mac-rolling-v6-1381/live-export --data data/next-live \
  --journal runs/prospective-v6
```

Use a fresh snapshot directory each round. Raw prediction CSVs remain local under the ignored journal. Publish each small `commitments/YYYY-Www.json` file before target maturity. It contains prediction, bundle and live-data hashes, capture time and row count. Keep the raw CSVs so their hashes and complete stock universe can later be checked.

**A local timestamp alone proves nothing about timeliness.** Ordinary public GitHub commit history supplies an external timestamp anchor; do not backdate commits, amend old observations or treat an unanchored local file as independently timestamped evidence. The helper deliberately does not certify its own timestamps. No automated collection or publishing schedule is installed.

## Score after labels mature

Download a new full snapshot into a fresh directory, preserving the earlier input snapshots:

```bash
uv run --frozen --no-editable solstice download --data data/prospective-scoring
uv run --frozen --no-editable python -m scripts.prospective score \
  --journal runs/prospective-v6 --data data/prospective-scoring \
  --output runs/prospective-v6/interim-score.json
```

The evaluator verifies every committed CSV hash, waits for complete targets, joins the exact ticker/date universe and scores the model alongside `lgbm_jupiter_60` with Numerai's neutralizers. A stock-universe mismatch fails instead of selecting a favorable subset. It reports per-date scores, a paired NCORR difference, pending dates, missing ISO weeks between captures, all unrecorded weeks since freezing, and scoring-file hashes. Outcomes cannot alter committed predictions.

Fewer than 52 matured weekly dates is explicitly interim. Fifty-two weeks is only a reporting threshold; overlapping 60-day outcomes, model aging, missing weeks and serial dependence can still make inference weak. Inspect the paired block interval and contribution separately. There is no automated benchmark-win declaration. Scores are not returns, and no submissions, staking or trading are performed.
