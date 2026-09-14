"""Paired test for online discovery of XGBoost shrinkage.

Question
--------
Given only a reasonable prior range for eta, does adapting eta online improve
expected performance relative to picking one eta once and keeping it fixed?

Design
------
* UCI Covertype, deterministic 250k sample/split.
* Prior: log-uniform eta in [0.03, 0.80].
* Draw eta_0 values once from that prior with a fixed RNG seed.
* For every eta_0, compare:
    1) fixed: keep eta=eta_0 for all trees;
    2) adaptive: start at the same eta_0 and update log(eta) online.
* Adaptive gets NO counterfactual tree branches: equal main-path tree budget.
* Test data never affects adaptation or model selection.
* After BOTH runs in a pair finish, export test/control loss versus tree count,
  plus cumulative wall time, so convergence speed can be analyzed directly.

Primary estimand
----------------
Mean paired final test-logloss difference over eta_0 ~ LogUniform(0.03, 0.80).
Negative is better for adaptation.

Compute-efficiency exports
--------------------------
results/covtype_online_eta_convergence.csv contains, for each pair/method and
checkpoint: tree count, control/test log-loss, test accuracy, eta, and measured
cumulative training wall time.

results/covtype_online_eta_compute_summary.csv contains pair-level quantities:
* normalized AUC (mean) of test loss over the training trajectory;
* earliest adaptive tree/time reaching the fixed run's final test loss;
* earliest fixed tree/time reaching the adaptive run's final test loss;
* final wall time and paired final-loss delta.

Wall time is intentionally measured using each method's actual implementation:
fixed XGBoost trains in one native call, while adaptive performs online updates
and therefore pays its Python/prediction overhead. Tree count is the cleaner
algorithmic-compute comparison; wall time is the practical comparison.
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
CHECKPOINT_EVERY = 25
LOG_STEP = 0.25
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
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
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


class TimingCallback(xgb.callback.TrainingCallback):
    def __init__(self):
        self.start = None
        self.times = []

    def before_training(self, model):
        self.start = time.perf_counter()
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.times.append(time.perf_counter() - self.start)
        return False


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
    timer = TimingCallback()
    b = xgb.train(
        params(nthread, eta0),
        dtrain,
        num_boost_round=n_rounds,
        verbose_eval=False,
        callbacks=[timer],
    )
    ctrl = metrics(yctrl, b.predict(dctrl))
    test = metrics(ytest, b.predict(dtest))
    return {
        "eta0": float(eta0),
        "fixed_final_eta": float(eta0),
        "fixed_control_logloss": ctrl["logloss"],
        "fixed_test_logloss": test["logloss"],
        "fixed_test_accuracy": test["accuracy"],
        "booster": b,
        "wall_times": timer.times,
    }


def adaptive_run(dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds):
    booster = xgb.train(params(nthread, eta0), dtrain, num_boost_round=0)
    eta = float(eta0)
    log_eta = math.log(eta)
    grad_sq_sum = 0.0
    history = []
    wall_times = []
    start = time.perf_counter()

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
        wall_times.append(time.perf_counter() - start)

    ctrl = metrics(yctrl, booster.predict(dctrl))
    test = metrics(ytest, booster.predict(dtest))
    return {
        "adaptive_final_eta": float(eta),
        "adaptive_control_logloss": ctrl["logloss"],
        "adaptive_test_logloss": test["logloss"],
        "adaptive_test_accuracy": test["accuracy"],
        "history": history,
        "booster": booster,
        "wall_times": wall_times,
    }


def eta_at_round(method, eta0, adaptive_history, tree_count):
    if method == "fixed":
        return float(eta0)
    eta = float(eta0)
    for h in adaptive_history:
        if h["round"] <= tree_count:
            eta = float(h["eta_after_update"])
        else:
            break
    return eta


def convergence_curve(
    label, method, eta0, booster, wall_times, adaptive_history,
    dctrl, dtest, yctrl, ytest, n_rounds, checkpoint_every,
):
    checkpoints = list(range(checkpoint_every, n_rounds + 1, checkpoint_every))
    if not checkpoints or checkpoints[-1] != n_rounds:
        checkpoints.append(n_rounds)

    rows = []
    for k in checkpoints:
        pctrl = booster.predict(dctrl, iteration_range=(0, k))
        ptest = booster.predict(dtest, iteration_range=(0, k))
        cm = metrics(yctrl, pctrl)
        tm = metrics(ytest, ptest)
        rows.append({
            "label": label,
            "method": method,
            "eta0": float(eta0),
            "trees": int(k),
            "eta": eta_at_round(method, eta0, adaptive_history, k),
            "control_logloss": cm["logloss"],
            "test_logloss": tm["logloss"],
            "test_accuracy": tm["accuracy"],
            "training_wall_seconds": float(wall_times[k - 1]),
        })
    return rows


def first_reach(curve, target):
    for r in curve:
        if r["test_logloss"] <= target:
            return r["trees"], r["training_wall_seconds"]
    return None, None


def curve_mean_loss(curve):
    # Checkpoints are equally spaced except possibly the final partial interval;
    # use trapezoidal integration over tree count and normalize by span.
    x = np.asarray([0] + [r["trees"] for r in curve], dtype=float)
    y0 = curve[0]["test_logloss"]
    y = np.asarray([y0] + [r["test_logloss"] for r in curve], dtype=float)
    return float(np.trapezoid(y, x) / x[-1])


def paired_run(
    dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds,
    checkpoint_every, label,
):
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

    # Test convergence is evaluated only now, after both training runs finish.
    fixed_curve = convergence_curve(
        label, "fixed", eta0, f["booster"], f["wall_times"], [],
        dctrl, dtest, yctrl, ytest, n_rounds, checkpoint_every,
    )
    adaptive_curve = convergence_curve(
        label, "adaptive", eta0, a["booster"], a["wall_times"], a["history"],
        dctrl, dtest, yctrl, ytest, n_rounds, checkpoint_every,
    )

    a_to_f_trees, a_to_f_seconds = first_reach(adaptive_curve, f["fixed_test_logloss"])
    f_to_a_trees, f_to_a_seconds = first_reach(fixed_curve, a["adaptive_test_logloss"])
    compute = {
        "label": label,
        "eta0": float(eta0),
        "fixed_final_test_logloss": f["fixed_test_logloss"],
        "adaptive_final_test_logloss": a["adaptive_test_logloss"],
        "paired_final_test_logloss_delta": float(delta),
        "fixed_trajectory_mean_test_logloss": curve_mean_loss(fixed_curve),
        "adaptive_trajectory_mean_test_logloss": curve_mean_loss(adaptive_curve),
        "trajectory_mean_test_logloss_delta": float(
            curve_mean_loss(adaptive_curve) - curve_mean_loss(fixed_curve)
        ),
        "adaptive_trees_to_fixed_final_loss": a_to_f_trees,
        "adaptive_seconds_to_fixed_final_loss": a_to_f_seconds,
        "fixed_trees_to_adaptive_final_loss": f_to_a_trees,
        "fixed_seconds_to_adaptive_final_loss": f_to_a_seconds,
        "fixed_final_training_wall_seconds": float(f["wall_times"][-1]),
        "adaptive_final_training_wall_seconds": float(a["wall_times"][-1]),
    }

    row = {
        k: v for k, v in f.items() if k not in {"booster", "wall_times"}
    }
    row.update({
        k: v for k, v in a.items() if k not in {"history", "booster", "wall_times"}
    })
    row["label"] = label
    row["paired_test_logloss_delta"] = float(delta)
    row["adaptive_wins"] = bool(delta < 0)
    return row, a["history"], fixed_curve + adaptive_curve, compute


def main():
    args = parse_args()
    t0 = time.time()
    print(
        f"nthread={args.nthread} n_starts={args.n_starts} "
        f"n_rounds={args.n_rounds} checkpoint_every={args.checkpoint_every}",
        flush=True,
    )
    dtrain, dctrl, dtest, yctrl, ytest = load_data()

    rng = np.random.default_rng(SEED)
    log_starts = rng.uniform(math.log(ETA_MIN), math.log(ETA_MAX), size=args.n_starts)
    starts = np.exp(log_starts)
    prior_median = math.sqrt(ETA_MIN * ETA_MAX)

    rows = []
    histories = {}
    convergence_rows = []
    compute_rows = []
    for i, eta0 in enumerate(starts):
        label = f"prior_draw_{i:02d}"
        row, history, curves, compute = paired_run(
            dtrain, dctrl, dtest, yctrl, ytest,
            float(eta0), args.nthread, args.n_rounds,
            args.checkpoint_every, label,
        )
        rows.append(row)
        histories[label] = history
        convergence_rows.extend(curves)
        compute_rows.append(compute)

    median_row, median_history, median_curves, median_compute = paired_run(
        dtrain, dctrl, dtest, yctrl, ytest,
        prior_median, args.nthread, args.n_rounds,
        args.checkpoint_every, "prior_median",
    )
    histories["prior_median"] = median_history
    convergence_rows.extend(median_curves)
    compute_rows.append(median_compute)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "covtype_online_eta_pairs.csv", index=False)
    pd.DataFrame([median_row]).to_csv(OUT / "covtype_online_eta_median.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(
        OUT / "covtype_online_eta_convergence.csv", index=False
    )
    pd.DataFrame(compute_rows).to_csv(
        OUT / "covtype_online_eta_compute_summary.csv", index=False
    )
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

    prior_compute = pd.DataFrame(compute_rows[:-1])
    summary = {
        "dataset": "UCI Covertype",
        "sample_size": N_SAMPLE,
        "n_rounds": args.n_rounds,
        "n_prior_draws": args.n_starts,
        "checkpoint_every": args.checkpoint_every,
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
        "expected_fixed_trajectory_mean_test_logloss": float(
            prior_compute["fixed_trajectory_mean_test_logloss"].mean()
        ),
        "expected_adaptive_trajectory_mean_test_logloss": float(
            prior_compute["adaptive_trajectory_mean_test_logloss"].mean()
        ),
        "mean_trajectory_test_logloss_delta": float(
            prior_compute["trajectory_mean_test_logloss_delta"].mean()
        ),
        "mean_fixed_training_wall_seconds": float(
            prior_compute["fixed_final_training_wall_seconds"].mean()
        ),
        "mean_adaptive_training_wall_seconds": float(
            prior_compute["adaptive_final_training_wall_seconds"].mean()
        ),
        "median_start": median_row,
        "median_start_compute": median_compute,
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
