# Research protocol and sources

Checked October 22, 2026. Implementation choices below are hypotheses with a finite local compute budget. They do not establish a universally best Numerai model.

## Data and metrics

[Numerai's v6 data contract](https://docs.numer.ai/numerai-tournament/data) changes the grouping and identity columns to `date` and `numerai_ticker`. The small and medium sets contain 100 and 350 numeric features after excluding text country. Targets are named explicitly rather than using the mutable `target` alias. Sixty-day outcomes overlap across weekly dates.

The [official v6 example](https://github.com/numerai/example-scripts/blob/master/numerai/v6_example_model.ipynb) uses supplied neutralizer matrices for **NCORR** and neutral contribution against `lgbm_jupiter_60`. Our fast scorer is checked against [`numerai-tools` 0.7.2](https://github.com/numerai/numerai-tools/blob/master/numerai_tools/scoring.py). It uses tie-preserving Gaussian ranks and the same relative least-squares cutoff. A per-date SVD is reused across candidates to avoid repeatedly factoring the same 300-column matrix.

Contribution to a fixed public benchmark is not a historical payout against Numerai's changing meta-model. NCORR is not Spearman correlation; neither score is a trading return. The reported mean/std ratio is not an annualized investment Sharpe, and cumulative-score drawdown is not monetary loss.

## First finite candidate search (v0.6.0 reference)

| Hypothesis | Candidate | Reason and limitation |
| --- | --- | --- |
| A simple model is a useful control | `baseline-small` | Small-feature trees; inner stopping and fixed training cap are part of this baseline |
| Linear regularization may generalize | `ridge-medium` | Training-only median imputation and standardization; no claim of nonlinear expressiveness |
| More conservative trees can reduce noise | `balanced-medium`, `slow-medium` | Smaller learning rates, larger leaves and L2 regularization |
| Random thresholds can diversify | `extra-medium` | LightGBM extra trees; different seed also changes the fixed training sample |
| Regimes change | `recent-medium` | Equal date weighting plus a four-year recency half-life; can discard useful old regimes |
| Related targets can diversify errors | `factor-target`, `camille-target` | Fixed 60-day alternatives; evaluated against the same Jupiter target |
| Difficult dates deserve extra weight | `era-boost-ablation` | Two in-sample worst-half-date reweighting steps; explicitly an ablation |

LightGBM controls follow its [parameter documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html). CPU determinism and column-wise histograms are requested; exact bitwise results across hardware or library versions are not promised. `max_bin=31` is a speed/memory choice, not an independently established accuracy improvement. Six threads prevent the experiment from oversubscribing the reference Mac.

The original [era-boosting research](https://forum.numer.ai/t/era-boosted-models/189) explicitly distinguishes impressive in-sample gains from uncertain out-of-sample improvement. It is included to test that distinction. Similarly, [the original feature-neutralization experiment](https://forum.numer.ai/t/feature-neutralization-increases-bias-and-reduces-variance/7486) motivates a bias/variance tradeoff, not a universal neutralization strength.

## Selection and access boundaries

1. Evaluate all nine predeclared candidates on three disjoint forward windows. Before each window, training ends strictly before its start minus 90 business days. Earlier public validation dates can enter later training only after this maturity gap.
2. Inside each training partition, reserve its final 26 dates for early stopping and purge a further 90 business days before that block. Choose a tree count there, then refit on all eligible sampled training rows.
3. Rank candidates by mean date NCORR minus 0.25 times its standard deviation. Keep at most five; retain the baseline separately for comparison.
4. On the 2024 calibration window only, search a nonnegative greedy blend for 12 steps, compare an equal-weight blend, then compare neutralization strengths 0, 0.25, 0.5 and 0.75. All trials are written to `calibration_trials.json`, including losing trials. The top 50 contemporaneously exposed numeric features are chosen without targets for each date's neutralization transform.
5. Hash and freeze the selected weights and transform. Write a holdout access journal before loading the reserved 2025 onward labels. Compare the selected ensemble with the simple baseline on identical observations and dates.
6. Report the paired date-score difference with a circular block-bootstrap interval. The interval does not adjust for all research decisions or prove a live edge. Public historical data can already have informed prior work.

The source and dependency fingerprint, full experiment plan, snapshot hashes, per-date scores, model hashes and fit metadata are saved. Resume rejects a changed identity. Only complete, finite-date candidates are eligible. There is no automatic upload or financial side effect in the v6 path.

## Deliberately unclaimed

This run does not exhaust model families, all feature subsets, sample sizes or hyperparameters. It does not estimate transaction costs, turnover or capacity from anonymized predictions. It cannot establish suitability for a hedge-fund role by itself.

[TabM's original paper](https://arxiv.org/abs/2410.24210) provides evidence for efficient neural ensembling on broad tabular benchmarks. It does not establish superiority on Numerai v6. A future neural experiment should compare TabM and a small MLP against these trees under the same purged protocol, with a separately recorded training budget and fresh prospective evaluation. Installing a transformer alone would not be evidence of improvement.

Other useful next experiments are full-feature versus medium-feature trees, 20-day versus 60-day auxiliary targets, multiple-seed averaging, longer training samples and prospective weekly tracking of a frozen model. Choose them from CV or future evidence; do not tune them against the consumed historical holdout.

## Full-feature follow-up (v0.6.1)

The first blend beat its small-feature control and lagged the supplied public benchmark. That result motivated [`configs/wide.toml`](../configs/wide.toml): four small/medium controls and three full-feature tree candidates, using all 1,039 numeric columns. A 750-row per-date training cap keeps the wider matrices within local memory. The configuration and every candidate remain fixed before this run's scores are inspected.

The same historical evaluation period was already inspected. Its configuration records `reused_exploratory` and the first experiment's fingerprint, and the result and access journal carry that status. Any follow-up intervals on this period are descriptive. They cannot establish an independently validated improvement. New claims require prospective observations.

The loader reads narrow keys first, retains every evaluation observation and preselects the union of every candidate seed's training sample. It then scans feature batches and gathers only those keys. A regression test compares both seeds' samples and every evaluation observation against full-history loading. Ridge preprocessing uses owned NumPy arrays in place; another test checks prediction equivalence. These changes reduce unnecessary full-matrix copies without changing the sampling rule.

Within this follow-up the candidates share the same row budget and date windows. Comparing it with the first study also changes the row cap and candidate parameters; a cross-study score difference cannot be attributed solely to feature count. The fixed public benchmark's training budget is not replicated or controlled to match ours.
