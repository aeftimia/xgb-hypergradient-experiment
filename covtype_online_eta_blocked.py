"""Blocked online-eta experiment for XGBoost on UCI Covertype.

This is a runtime-oriented follow-up to covtype_online_eta.py. The original
experiment is preserved unchanged for reproducibility.

Key difference
--------------
Instead of calling xgb.train once per tree, the adaptive method trains
ADAPT_BLOCK trees at the current eta in one native XGBoost call. It then
computes one validation hypergradient from the change in control-set margins
across the whole block and updates log(eta).

This reduces Python / Booster reconstruction / prediction overhead from 750
training calls to 75 calls when ADAPT_BLOCK=10. It also changes the estimator
slightly: the hypergradient now asks how scaling the latest *block* of trees
would have changed current validation loss, rather than using only the final
tree of each 10-tree interval.

Design
------
* UCI Covertype, deterministic 250k sample/split.
* eta_0 ~ LogUniform(0.03, 0.80), 20 fixed draws plus exact prior median.
* fixed and adaptive get exactly the same main-path tree budget.
* no counterfactual branches.
* test data never affects adaptation.
* exports final, trajectory, and wall-clock comparisons separately from the
  original one-tree implementation.
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
ADAPT_BLOCK = 10
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
    p.add_argument("--adapt-block", type=int, default=ADAPT_BLOCK)
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
    def __init__(self, global_start=None):
        self.start = global_start
        self.times = []

    def before_training(self, model):
        if self.start is None:
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
    booster = xgb.train(
        params(nthread, eta0),
        dtrain,
        num_boost_round=n_rounds,
        verbose_eval=False,
        callbacks=[timer],
    )
    ctrl = metrics(yctrl, booster.predict(dctrl))
    test = metrics(ytest, booster.predict(dtest))
    return {
        "eta0": float(eta0),
        "fixed_final_eta": float(eta0),
        "fixed_control_logloss": ctrl["logloss"],
        "fixed_test_logloss": test["logloss"],
        "fixed_test_accuracy": test["accuracy"],
        "booster": booster,
        "wall_times": timer.times,
    }


def adaptive_run(
    dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds, adapt_block,
):
    if adapt_block < 1:
        raise ValueError("adapt_block must be >= 1")

    booster = xgb.train(params(nthread, eta0), dtrain, num_boost_round=0)
    eta = float(eta0)
    log_eta = math.log(eta)
    grad_sq_sum = 0.0
    history = []
    wall_times = []
    start = time.perf_counter()
    rounds = 0

    while rounds < n_rounds:
        n = min(adapt_block, n_rounds - rounds)
        margin_before = booster.predict(dctrl, output_margin=True)

        timer = TimingCallback(global_start=start)
        booster = xgb.train(
            params(nthread, eta),
            dtrain,
            num_boost_round=n,
            xgb_model=booster,
            verbose_eval=False,
            callbacks=[timer],
        )
        wall_times.extend(timer.times)
        rounds += n

        margin_after = booster.predict(dctrl, output_margin=True)
        p = softmax(margin_after)
        residual = p.copy()
        residual[np.arange(len(yctrl)), yctrl] -= 1.0
        delta_margin = margin_after - margin_before

        # Derivative with respect to z=log(eta), treating the fitted block's
        # tree structures as fixed. delta_margin is the effect of the whole
        # block at the eta used for that block.
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

        ctrl_loss = metrics(yctrl, p)["logloss"]
        history.append(
            {
                "round": rounds,
                "block_size": n,
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
    x = np.asarray([0] + [r["trees"] for r in curve], dtype=float)
    y0 = curve[0]["test_logloss"]
    y = np.asarray([y0] + [r["test_logloss"] for r in curve], dtype=float)
    return float(np.trapezoid(y, x) / x[-1])


def paired_run(
    dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds,
    adapt_block, checkpoint_every, label,
):
    print(f"\n[{label}] eta0={eta0:.6f} fixed", flush=True)
    fixed = fixed_run(dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds)
    print(
        f"[{label}] fixed test={fixed['fixed_test_logloss']:.6f}; blocked adaptive...",
        flush=True,
    )
    adaptive = adaptive_run(
        dtrain, dctrl, dtest, yctrl, ytest, eta0, nthread, n_rounds, adapt_block
    )
    delta = adaptive["adaptive_test_logloss"] - fixed["fixed_test_logloss"]
    print(
        f"[{label}] adaptive test={adaptive['adaptive_test_logloss']:.6f} "
        f"eta_final={adaptive['adaptive_final_eta']:.5f} delta={delta:+.6f} "
        f"wall={adaptive['wall_times'][-1]:.1f}s",
        flush=True,
    )

    fixed_curve = convergence_curve(
        label, "fixed", eta0, fixed["booster"], fixed["wall_times"], [],
        dctrl, dtest, yctrl, ytest, n_rounds, checkpoint_every,
    )
    adaptive_curve = convergence_curve(
        label, "adaptive_blocked", eta0, adaptive["booster"], adaptive["wall_times"],
        adaptive["history"], dctrl, dtest, yctrl, ytest, n_rounds, checkpoint_every,
    )

    a_to_f_trees, a_to_f_seconds = first_reach(
        adaptive_curve, fixed["fixed_test_logloss"]
    )
    f_to_a_trees, f_to_a_seconds = first_reach(
        fixed_curve, adaptive["adaptive_test_logloss"]
    )
    compute = {
        "label": label,
        "eta0": float(eta0),
        "fixed_final_test_logloss": fixed["fixed_test_logloss"],
        "adaptive_final_test_logloss": adaptive["adaptive_test_logloss"],
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
        "fixed_final_training_wall_seconds": float(fixed["wall_times"][-1]),
        "adaptive_final_training_wall_seconds": float(adaptive["wall_times"][-1]),
        "adaptive_to_fixed_wall_ratio": float(
            adaptive["wall_times"][-1] / fixed["wall_times"][-1]
        ),
    }

    row = {
        k: v for k, v in fixed.items() if k not in {"booster", "wall_times"}
    }
    row.update({
        k: v for k, v in adaptive.items()
        if k not in {"history", "booster", "wall_times"}
    })
    row["label"] = label
    row["paired_test_logloss_delta"] = float(delta)
    row["adaptive_wins"] = bool(delta < 0)
    return row, adaptive["history"], fixed_curve + adaptive_curve, compute


def main():
    args = parse_args()
    if args.n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")

    t0 = time.time()
    print(
        f"nthread={args.nthread} n_starts={args.n_starts} n_rounds={args.n_rounds} "
        f"adapt_block={args.adapt_block} checkpoint_every={args.checkpoint_every}",
        flush=True,
    )
    dtrain, dctrl, dtest, yctrl, ytest = load_data()

    rng = np.random.default_rng(SEED)
    starts = np.exp(
        rng.uniform(math.log(ETA_MIN), math.log(ETA_MAX), size=args.n_starts)
    )
    prior_median = math.sqrt(ETA_MIN * ETA_MAX)

    rows = []
    histories = {}
    convergence_rows = []
    compute_rows = []

    for i, eta0 in enumerate(starts):
        label = f"prior_draw_{i:02d}"
        row, history, curves, compute = paired_run(
            dtrain, dctrl, dtest, yctrl, ytest, float(eta0), args.nthread,
            args.n_rounds, args.adapt_block, args.checkpoint_every, label,
        )
        rows.append(row)
        histories[label] = history
        convergence_rows.extend(curves)
        compute_rows.append(compute)

    median_row, median_history, median_curves, median_compute = paired_run(
        dtrain, dctrl, dtest, yctrl, ytest, prior_median, args.nthread,
        args.n_rounds, args.adapt_block, args.checkpoint_every, "prior_median",
    )
    histories["prior_median"] = median_history
    convergence_rows.extend(median_curves)
    compute_rows.append(median_compute)

    prefix = f"covtype_online_eta_blocked{args.adapt_block}"
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{prefix}_pairs.csv", index=False)
    pd.DataFrame([median_row]).to_csv(OUT / f"{prefix}_median.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(
        OUT / f"{prefix}_convergence.csv", index=False
    )
    pd.DataFrame(compute_rows).to_csv(
        OUT / f"{prefix}_compute_summary.csv", index=False
    )
    with open(OUT / f"{prefix}_history.json", "w") as f:
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
            "trees_per_native_train_call": args.adapt_block,
            "update_every_trees": args.adapt_block,
            "gradient_scope": "whole_latest_block",
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
        "mean_adaptive_to_fixed_wall_ratio": float(
            prior_compute["adaptive_to_fixed_wall_ratio"].mean()
        ),
        "median_start": median_row,
        "median_start_compute": median_compute,
        "passes_original_gate": bool(
            mean_delta <= -0.005 and df["adaptive_wins"].mean() >= 0.70
        ),
        "original_gate_definition": (
            "mean paired test-logloss improvement >=0.005 and adaptive win rate >=70%"
        ),
        "tree_fit_budget_per_pair": {
            "fixed": args.n_rounds,
            "adaptive": args.n_rounds,
        },
        "test_used_for_adaptation": False,
        "elapsed_seconds": time.time() - t0,
    }
    with open(OUT / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
