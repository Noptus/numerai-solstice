# Reference experiment

A finite, nine-candidate v6 experiment is running locally on October 22, 2026 against public snapshot round **1381**. The configuration is frozen in [`configs/mac.toml`](../configs/mac.toml). No performance improvement is claimed while the comparison is incomplete.

The final evidence will include all three walk-forward folds, every calibration trial, the frozen selection, per-date historical holdout scores, a public-benchmark comparison and measured fit times. Raw data and model binaries are excluded from this repository.

## Metric verification already completed

[`score_parity.json`](score_parity.json) compares the fast scorer with `numerai-tools==0.7.2` on **6,092 stocks and 300 neutralizer columns** for June 7, 2019. The maximum absolute score difference is **3.24e-10**.

Reproduce this check after the baseline checkpoint exists:

```bash
uv run --frozen --no-editable python -m scripts.check_score_parity \
  --predictions runs/mac-v6/cv/fold1/baseline-small.parquet \
  --date 2019-06-07
```
