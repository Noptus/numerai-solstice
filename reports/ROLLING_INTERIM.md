# Rolling study: incomplete, inspectable results

The local Mac experiment was stopped on October 22, 2026 at the user's requested wrap-up, after **29 completed candidate refits**. The stop was not triggered by a score threshold. All completed checkpoints, including losing candidates, were retained. The study is incomplete: **no calibration, final model selection, later historical evaluation or prospective tracking was performed**.

The [predeclared design](../docs/ROLLING_STUDY.md) remains unchanged. It tests four candidates on all 1,039 numeric features, at most 2,000 training rows per date, one seed, six CPU threads and 90-business-day purges before outer and inner evaluation. Completed checkpoints were hash-checked against their local prediction/model files before packaging. Raw data and trusted pickles remain local.

## First completed development window

All four candidates and the supplied benchmark were scored on the same 53 dates, June 7, 2019 through June 5, 2020, using every eligible stock per date. These dates differ from the two earlier studies' later evaluation period, so their score levels are not directly comparable.

| Candidate | Mean NCORR | Difference from public benchmark |
| --- | ---: | ---: |
| `lgbm_jupiter_60` | 0.03802 | +0.00000 |
| `quarterly-bagged` | 0.01472 | -0.02330 |
| `quarterly-fixed-budget` | 0.03304 | -0.00498 |
| `quarterly-full` | 0.02327 | -0.01475 |
| `static-full-control` | 0.02425 | -0.01377 |

The fixed-budget candidate has the highest custom mean in this window and **still trails the public benchmark**. It was not promoted or selected. Quarterly refitting alone did not improve on the static control in this window. Only that pair isolates one changed factor; the other candidates change multiple parameters, so these results cannot attribute gains to stopping or bagging alone. No independent significance claim or live-edge claim follows from one development window.

Fold 2 is partially trained and fold 3 has not started. Partial windows are omitted from this score table rather than comparing unequal date coverage. Their completed fits remain in the ledger. The full CV-based shortlist and subsequent selection were not reached. Prior historical data already informed development; resuming would remain exploratory.

## Evidence and verification

The [portable archive](rolling-interim-v6-1381/) includes the frozen plan, source/dependency/data identities, every completed refit's metadata and artifact hashes, per-date first-window scores, benchmark comparison, hardware description and real-data scoring parity. Metric parity on 6,092 stocks and 300 neutralizers differed from `numerai-tools` by at most 6.86e-10. SHA-256 manifests establish consistency, not independent replication or external timestamp attestation.

```bash
uv run --frozen --no-editable python -m scripts.verify_interim reports/rolling-interim-v6-1381
```

On the original machine, the exact run can resume with its retained, integrity-checked checkpoints:

```bash
.venv/bin/python -m research run --config configs/rolling.toml --output runs/mac-rolling-v6-1381
```

A fresh clone needs the pinned public datasets and will retrain; model binaries are not in Git. Changing the training source, dependencies, configuration or dataset identity requires a fresh run directory. The source-aware package-cache check and duplicate-recipe inference checks are engineering validations, not evidence of predictive performance. The [prospective recording tools](../docs/PROSPECTIVE.md) have not been activated for this incomplete study.
