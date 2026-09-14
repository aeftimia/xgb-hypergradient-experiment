# Results notes

## Covertype falsification

The first large Covertype rollout-oracle run is archived here as `covtype_confounded_run.*` and is **invalid for the eta-control hypothesis**.

It appeared to improve test log-loss from 0.183317 (best fixed eta=0.3) to 0.177824 (rollout oracle), but each eta branch used a different random seed with row/column subsampling enabled. The oracle therefore selected over stochastic tree realizations as well as eta. Its trajectory chose eta=0.3 at every block, which exposed the confound.

The corrected experiment in `covtype_falsification.py` removes row/column subsampling, uses identical deterministic training conditions across candidate etas, expands the eta grid through 0.5, and runs for 750 boosting rounds. The corrected local run found best fixed eta=0.4 with test log-loss 0.152865; the K=25 rollout oracle used mostly eta=0.5 before switching partly to 0.4, but finished worse at 0.153930 test log-loss. This rejects the predeclared gate for the state-dependent rollout-oracle hypothesis.

## Online eta discovery

`covtype_online_eta.py` tests a different and more practical hypothesis: given only a reasonable prior range for eta, can an online learner improve the expected result versus choosing eta once and leaving it fixed?

The experiment samples 20 starting etas from a log-uniform prior on `[0.03, 0.80]`. For each draw it runs a paired comparison:

- fixed: train all 750 trees at the sampled `eta0`;
- adaptive: start from the exact same `eta0`, then update `log(eta)` every 10 trees using an AdaGrad-normalized multiclass validation hypergradient.

The adaptive method gets no counterfactual branches and fits exactly 750 main-path trees, the same tree-fit budget as the paired fixed baseline. The untouched test split is only evaluated after training. The primary estimand is the mean paired test-logloss difference over starting eta drawn from the prior.

The predeclared gate is: mean paired test-logloss improvement of at least 0.005 and adaptive wins on at least 70% of sampled starting etas. An exact prior-median start, `sqrt(0.03 * 0.80)`, is also reported separately.

## Local runs

```bash
python covtype_falsification.py --nthread 32
python covtype_online_eta.py --nthread 32
```

`covtype_online_eta.py` writes:

- `results/covtype_online_eta_pairs.csv`
- `results/covtype_online_eta_median.csv`
- `results/covtype_online_eta_history.json`
- `results/covtype_online_eta_summary.json`
