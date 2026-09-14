"""Paired falsification test for online discovery of XGBoost shrinkage.

Question
--------
Given only a reasonable prior range for eta, does adapting eta online improve
expected final performance relative to picking one eta once and keeping it
fixed?

Design
------
* UCI Covertype, same deterministic 250k sample/split as covtype_falsification.py.
* Prior: log-uniform eta in [0.03, 0.80].
* Draw 20 eta_0 values once from that prior with a fixed RNG seed.
* For every eta_0, train a paired comparison:
    1) fixed: keep eta=eta_0 for all 750 trees;
    2) adaptive: start at the same eta_0 and update log(eta) online.
* The adaptive method gets NO counterfactual tree branches. It trains exactly
  the same number of main-path trees as the fixed method. It uses a held-out
  control split to estimate a hypergradient from the latest fitted tree.
* Test labels/predictions are used only after each 750-tree run is complete.

Adaptive rule
-------------
Let z = log(eta). For the newly added tree, holding that tree fixed,

    d margin / d eta ~= (margin_after - margin_before) / eta.

For multiclass log-loss,

    dL / d margin_ic = p_ic - 1[y_i=c].

Therefore

    dL / d z = eta * dL/deta
             = mean_i sum_c (p_ic - onehot_ic) * delta_margin_ic.

We update z every ADAPT_EVERY trees with an AdaGrad-normalized step. This is a
pre-specified one-trajectory online learner: there is no eta grid search and no
extra tree-fitting compute for adaptation.

Primary estimand
----------------
Mean paired test-logloss difference over eta_0 ~ LogUniform(0.03, 0.80):

    mean(adaptive_test_logloss - fixed_test_logloss).

Negative is better for adaptation. We also report the fixed-prior expectation,
adaptive expectation, win rate, and a paired run starting exactly at the prior
median sqrt(eta_min * eta_max).

Predeclared success gate
------------------------
The experiment passes if BOTH:
1) mean paired test-logloss improvement is at least 0.005; and
2) adaptive beats fixed for at least 70% of sampled starting etas.

This is deliberately a practical test against initial hyperparameter ignorance,
not a claim that adaptation must beat an oracle-tuned fixed eta.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

OUT = Path("results")
OUT.mkdir(exist_ok=True)

SEED = 20260914
N_SAMPLE = 250_000
N_ROUNDS = 750
N_STARTS = 20
ETA_MIN = 0.03
ETA_MAX = 0.80
ADAPT_EVERY = 10
LOG_STEP = 0.25  # first normalized step in natural-log eta units (~28% multiplicative)
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--nthread",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="XGBoost CPU threads (default: all visible logical CPUs)",
    )
    p.add_argument("--n-starts", type=int, default=N_STARTS)
    p.add_argument("--n-rounds", type=int, default=N_ROUNDS)
    return p.parse_args()


def params(nthread: int, eta: float) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": 7,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "nthread": int(nthread),
        "seed": SEED,
        "eta": float(eta),
    }


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "logloss": float(log_loss(y, p, labels=np.arange(7))),
        "accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
    }


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def load_data():
    print("loading Covertype...", flush=True)
    X, y = fetch_covtype(return_X_y=True)
    y = y.astype(np.int32) - 1

    if len(y) > N_SAMPLE:
        X, _, y, _ = train_test_split(
            X, y, train_size=N_SAMPLE, stratify=y, random_state=SEED
        )

    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    Xctrl, Xtest, yctrl, ytest = train_test_split(
        Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED + 1
    )
    print(
        f"sample={len(y):,} train={len(ytr):,} control={len(yctrl):,} test={len(ytest):,}",
        flush=True,
    )
    return (
        xgb.DMatrix(Xtr, label=ytr),
        xgb.DMatrix(Xctrl, label=yctrl),
        xgb.DMatrix(Xtest, label=ytest),
        yctrl,
        ytest,
    )


def fixed_run(dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds):
    b = xgb.train(
        params(nthread, eta0),
        dtrain,
        num_boost_round=n_rounds,
        verbose_eval=False,
    )
    ctrl = metrics(yctrl, b.predict(dctrl))
    test = metrics(ytest, b.predict(dtest))
    return {
        "eta0": float(eta0),
        "fixed_final_eta": float(eta0),
        "fixed_control_logloss": ctrl["logloss"],
        "fixed_test_logloss": test["logloss"],
        "fixed_test_accuracy": test["accuracy"],
    }


def adaptive_run(dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds):
    # Zero-round booster gives us XGBoost's exact initial/base margins.
    booster = xgb.train(params(nthread, eta0), dtrain, num_boost_round=0)
    eta = float(eta0)
    log_eta = math.log(eta)
    grad_sq_sum = 0.0
    history = []

    for t in range(1, n_rounds + 1):
        margin_before = booster.predict(dctrl, output_margin=True)
        booster = xgb.train(
            params(nthread, eta),
            dtrain,
            num_boost_round=1,
            xgb_model=booster,
            verbose_eval=False,
        )
        margin_after = booster.predict(dctrl, output_margin=True)

        # For z=log(eta), eta cancels:
        # dL/dz ~= mean sum_c (p-y_onehot) * (margin_after-margin_before).
        if t % ADAPT_EVERY == 0:
            p = softmax(margin_after)
            residual = p.copy()
            residual[np.arange(len(yctrl)), yctrl] -= 1.0
            delta_margin = margin_after - margin_before
            g_log_eta = float(np.mean(np.sum(residual * delta_margin, axis=1)))

            grad_sq_sum += g_log_eta * g_log_eta
            normalized_step = LOG_STEP * g_log_eta / math.sqrt(grad_sq_sum + EPS)
            log_eta = float(
                np.clip(
                    log_eta - normalized_step,
                    math.log(ETA_MIN),
                    math.log(ETA_MAX),
                )
            )
            eta = math.exp(log_eta)

            ctrl_loss = metrics(yctrl, softmax(margin_after))["logloss"]
            history.append(
                {
                    "round": t,
                    "eta_after_update": eta,
                    "g_log_eta": g_log_eta,
                    "normalized_log_step": normalized_step,
                    "control_logloss": ctrl_loss,
                }
            )

    ctrl = metrics(yctrl, booster.predict(dctrl))
    test = metrics(ytest, booster.predict(dtest))
    return {
        "adaptive_final_eta": float(eta),
        "adaptive_control_logloss": ctrl["logloss"],
        "adaptive_test_logloss": test["logloss"],
        "adaptive_test_accuracy": test["accuracy"],
        "history": history,
    }


def paired_run(dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds, label):
    print(f"\n[{label}] eta0={eta0:.6f} fixed", flush=True)
    f = fixed_run(dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds)
    print(
        f"[{label}] fixed test={f['fixed_test_logloss']:.6f}; adaptive...",
        flush=True,
    )
    a = adaptive_run(dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds)
    delta = a["adaptive_test_logloss"] - f["fixed_test_logloss"]
    print(
        f"[{label}] adaptive test={a['adaptive_test_logloss']:.6f} "
        f"eta_final={a['adaptive_final_eta']:.5f} delta={delta:+.6f}",
        flush=True,
    )
    row = {**f, **{k: v for k, v in a.items() if k != "history"}}
    row["label"] = label
    row["paired_test_logloss_delta"] = float(delta)
    row["adaptive_wins"] = bool(delta < 0)
    return row, a["history"]


def main():
    args = parse_args()
    t0 = time.time()
    print(f"nthread={args.nthread} n_starts={args.n_starts} n_rounds={args.n_rounds}", flush=True)
    dtrain, dctrl, dtest, yctrl, ytest = load_data()

    rng = np.random.default_rng(SEED)
    log_starts = rng.uniform(math.log(ETA_MIN), math.log(ETA_MAX), size=args.n_starts)
    starts = np.exp(log_starts)
    prior_median = math.sqrt(ETA_MIN * ETA_MAX)

    rows = []
    histories = {}
    for i, eta0 in enumerate(starts):
        label = f"prior_draw_{i:02d}"
        row, history = paired_run(
            dtrain, dctrl, dtest, yctrl, ytest,
            float(eta0), args.nthread, args.n_rounds, label,
        )
        rows.append(row)
        histories[label] = history

    # Exact prior-median pair is reported separately and is not included in the
    # Monte Carlo estimate of the prior expectation.
    median_row, median_history = paired_run(
        dtrain, dctrl, dtest, yctrl, ytest,
        prior_median, args.nthread, args.n_rounds, "prior_median",
    )
    histories["prior_median"] = median_history

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "covtype_online_eta_pairs.csv", index=False)
    pd.DataFrame([median_row]).to_csv(OUT / "covtype_online_eta_median.csv", index=False)
    with open(OUT / "covtype_online_eta_history.json", "w") as f:
        json.dump(histories, f, indent=2)

    deltas = df["paired_test_logloss_delta"].to_numpy(float)
    mean_delta = float(np.mean(deltas))
    sem = float(stats.sem(deltas)) if len(deltas) > 1 else float("nan")
    if len(deltas) > 1:
        ci_low, ci_high = stats.t.interval(
            0.95, df=len(deltas) - 1, loc=mean_delta, scale=sem
        )
    else:
        ci_low = ci_high = float("nan")

    summary = {
        "dataset": "UCI Covertype",
        "sample_size": N_SAMPLE,
        "n_rounds": args.n_rounds,
        "n_prior_draws": args.n_starts,
        "eta_prior": {
            "distribution": "log-uniform",
            "min": ETA_MIN,
            "max": ETA_MAX,
            "median": prior_median,
        },
        "adaptive_rule": {
            "space": "log_eta",
            "update_every_trees": ADAPT_EVERY,
            "adagrad_log_step": LOG_STEP,
            "counterfactual_tree_fits": 0,
        },
        "expected_fixed_test_logloss": float(df["fixed_test_logloss"].mean()),
        "expected_adaptive_test_logloss": float(df["adaptive_test_logloss"].mean()),
        "mean_paired_test_logloss_delta": mean_delta,
        "mean_paired_improvement": float(-mean_delta),
        "paired_delta_95pct_t_interval": [float(ci_low), float(ci_high)],
        "adaptive_win_rate": float(df["adaptive_wins"].mean()),
        "median_start": median_row,
        "passes_gate": bool(mean_delta <= -0.005 and df["adaptive_wins"].mean() >= 0.70),
        "gate_definition": "mean paired test-logloss improvement >=0.005 and adaptive win rate >=70%",
        "tree_fit_budget_per_pair": {
            "fixed": args.n_rounds,
            "adaptive": args.n_rounds,
        },
        "test_used_for_adaptation": False,
        "elapsed_seconds": time.time() - t0,
    }
    with open(OUT / "covtype_online_eta_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
