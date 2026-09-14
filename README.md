# Hypergradient learning-rate control for XGBoost

A small exploratory experiment asking whether XGBoost's shrinkage parameter (`eta`) can be adapted online with a hypergradient rather than held fixed.

## Idea

For boosting round `t`, let the ensemble margin change by

```
delta_margin = margin_after - margin_before
```

If the tree was added with shrinkage `eta_t`, then, holding that tree's structure and leaf values fixed,

```
d margin / d eta ~= delta_margin / eta_t
```

For binary log-loss, `dL/dmargin = sigmoid(margin) - y`, so the validation-loss hypergradient is approximated as

```
dL/deta = mean((sigmoid(margin_after) - y_val) * delta_margin / eta_t)
```

and the next round uses

```
eta_{t+1} = clip(eta_t - hyper_lr * dL/deta, eta_min, eta_max)
```

This is deliberately simple: it ignores the dependence of future tree structures on today's eta.

## Dataset

`sklearn.datasets.load_breast_cancer`, the Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 real-valued features, binary classification). It ships with scikit-learn and is derived from the UCI repository.

Each seed uses a stratified 60/20/20 train/validation/test split.

## Methods

- **fixed_tuned**: 80 boosting rounds; choose final-validation-loss winner among eta = 0.03, 0.1, 0.2.
- **hyper_tuned**: initialize eta = 0.1; choose final-validation-loss winner among hyper learning rates = 0.05, 0.2, 1.0, 5.0. Eta is constrained to [0.005, 0.5].
- **val_linesearch**: at every boosting round, fit the candidate tree at eta=1, then scalar-minimize validation log-loss over eta in [0.001, 0.5], then refit that round with the selected eta.

All methods use depth-3 XGBoost histogram trees with L2 regularization 1.0 and no row/column subsampling.

## Results

Five independent train/validation/test splits:

| method | mean test log-loss | sd | mean test AUC | mean val log-loss |
|---|---:|---:|---:|---:|
| fixed_tuned | **0.1041** | 0.0281 | **0.9909** | 0.1466 |
| hyper_tuned | 0.1072 | 0.0398 | 0.9899 | **0.1476** |
| val_linesearch | 0.1244 | 0.0453 | 0.9876 | 0.1482 |

Hypergradient eta beat tuned fixed eta on test log-loss in 3 of 5 splits, but its average test log-loss was worse by ~0.0031. With only five splits on a small dataset, this is not evidence for a meaningful difference in either direction.

The per-step validation line search performed worst. In several runs it drove eta toward the lower bound late in training, which is consistent with greedy validation optimization becoming overly conservative once the ensemble is already strong.

## Interpretation

The simple one-step hypergradient signal is real and behaves sensibly, but this experiment does **not** show that it is better than ordinary fixed-eta tuning. That is arguably the useful outcome: merely making shrinkage differentiable/adaptive is not an obvious free win.

A more interesting follow-up would test the original long-horizon idea: learn or estimate the effect of eta / optimizer controls on validation loss K boosting rounds in the future, rather than minimizing the immediate validation loss. Tree boosting is a clean test bed because the action space can be one-dimensional while the underlying model is non-differentiable in tree structure.

## Caveats

- Very small dataset and only five splits.
- Hyperparameters are selected separately on each split's validation set.
- The hypergradient treats each fitted tree direction as fixed with respect to eta; it does not differentiate through how today's eta changes later residuals and later trees.
- The line-search baseline is intentionally strong but validation-intensive and computes a temporary extra tree each round.
- This is exploratory code, not a claim of a new optimizer.

## Run

```bash
python experiment.py
```

Results are written under `/mnt/data` in the current script version; if turning this into a polished benchmark, the first cleanup should be parameterizing the output directory and dataset from the CLI.
