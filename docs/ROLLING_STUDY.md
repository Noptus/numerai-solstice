# Predeclared rolling and capacity study

Planned October 22, 2026, before this study's fits or scores. Configuration: `configs/rolling.toml`. All results on the already inspected 2025–2026 evaluation period are **exploratory**. The prior study fingerprint is linked in the configuration. No historical result from this study will be described as an independent benchmark win.

## Scientific questions and finite budget

1. Does raising the label-independent sample cap from 750 to 2,000 observations per date help a full-feature model? Cross-study differences are confounded; the new static model is a within-study control for the other questions.
2. Does refreshing an otherwise identical model every 13 evaluation dates help compared with keeping it fixed for the year?
3. Do row bagging and more features per tree help, with conservative leaves and L2 regularization?
4. Does a predeclared 2,500-tree budget avoid underfitting from noisy inner RMSE stopping? This candidate deliberately gives up adaptive inner stopping; it cannot use evaluation labels to select its tree count.

**Attribution limit:** only `static-full-control` versus `quarterly-full` changes one factor. The bagged and fixed-budget candidates change parameter bundles (including feature fraction, leaves or bagging), so their score differences cannot establish that early stopping or bagging alone caused an improvement.

Four candidates, three forward CV windows, one calibration year, one reused historical window. Six CPU threads; one fit at a time, 2,000 training observations per date, all 1,039 numeric features, one common seed. No adaptive expansion after inspecting scores. At most three CV-shortlisted models enter four greedy blend steps, an equal-weight comparison and neutralization strengths 0 and 0.5: at most 15 logged calibration trials.

At each 13-date boundary, eligible training dates are strictly earlier than that boundary minus 90 business days. A previous evaluation date may enter later training only once this gap has elapsed. All feature/target rows after that cutoff remain excluded from fitting. Every eligible evaluation stock is predicted once. Each refit is checkpointed and records training, stopping and prediction dates. Scores are aggregated only after predictions are complete.

The first three candidates use the existing purged 26-date inner stopping block. The fourth uses its fixed budget. Development CV diagnostics also compare all four candidates with the public benchmark, without changing the candidate list or selection rule. Per-window and pooled CV means are development summaries, without independent significance claims. The supplied benchmark is scored on identical stocks/dates; its own data budget and training policy are not replicated. Quarterly refits, bagging, larger samples and different stopping are hypotheses, not established advantages.

## Why these techniques

- [LightGBM's tuning guide](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html) documents larger training sets, row/feature bagging, conservative leaves and small learning rates as useful accuracy/overfitting controls. It does not establish that any particular setting wins on Numerai.
- [Numerai's data contract](https://docs.numer.ai/numerai-tournament/data) explains overlapping target horizons and unstable feature predictiveness. This motivates purged chronological evaluation and testing model refresh rather than random row splits.
- [Numerai's benchmark documentation](https://docs.numer.ai/numerai-tournament/models) recommends comparison with supplied predictions and distinguishes fixed-benchmark contribution from the changing meta-model score.

Neural TabM remains a separate, untested direction. Adding a new model family to an already broad search is not a substitute for more credible evaluation.

## What would count as credible new evidence

The historical CV, calibration and later evaluation windows have informed development. Their intervals are descriptive and do not correct for the full research process. We will publish losing candidates and retain the frozen selection even if a different candidate looks better on the later period.

A prospective check must freeze the actual model, source, dependencies and prediction rule before collecting new predictions; timestamp and hash every complete live-date prediction file before its targets mature; then join newly matured targets and public benchmark predictions by exact ticker/date keys. No changing the model or scoring subset after reading outcomes. Use paired per-date NCORR differences and block intervals, and report contribution separately. A handful of overlapping dates cannot substantiate a benchmark-beating claim. A minimum 52 weekly dates is a reporting threshold, not a guarantee of power or independence. No submission, staking or trading is required to keep this local research record.

## Execution status

The local run was stopped at the user's requested wrap-up on October 22, 2026. The predeclared design above is unchanged. See the [interim evidence](../reports/ROLLING_INTERIM.md) for completed checkpoints and the one fully scored development window. Calibration, selection and later evaluation were not reached. Completed fits remain resumable; no best candidate was promoted to a deployed model. The [prospective recording tools](PROSPECTIVE.md) are available, but no prospective plan was activated for this incomplete study.
