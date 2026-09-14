# Results notes

## Covertype falsification

The first large Covertype rollout-oracle run is archived here as `covtype_confounded_run.*` and is **invalid for the eta-control hypothesis**.

It appeared to improve test log-loss from 0.183317 (best fixed eta=0.3) to 0.177824 (rollout oracle), but each eta branch used a different random seed with row/column subsampling enabled. The oracle therefore selected over stochastic tree realizations as well as eta. Its trajectory chose eta=0.3 at every block, which exposed the confound.

The corrected experiment in `covtype_falsification.py` removes row/column subsampling, uses identical deterministic training conditions across candidate etas, expands the eta grid through 0.5, and runs for 750 boosting rounds. Only that corrected run should be used to assess whether state-dependent long-horizon eta control has signal.

The corrected GitHub Actions run was still in progress when these notes were committed.

## Local run

```bash
python covtype_falsification.py
```

The script downloads UCI Covertype through scikit-learn and writes its CSV, JSON verdict, and oracle history under `results/`.
