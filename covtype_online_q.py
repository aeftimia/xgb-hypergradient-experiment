"""Conservative online long-horizon Q controller for XGBoost eta.

This is the next experiment after covtype_online_eta_blocked.py.

Question
--------
Can a learned value model improve on the strong 10-tree blocked hypergradient
controller by reasoning about delayed consequences of eta choices?

Design
------
* Same deterministic UCI Covertype 250k sample/split and same 20 eta starts.
* Main-path tree budget remains 750 trees. No counterfactual tree branches.
* XGBoost trains in 10-tree native blocks.
* After each block, compute the same blocked hypergradient used by the baseline.
  That produces a local hypergradient proposal for the NEXT block's eta.
* Q learns from delayed realized outcomes:
      Q(s_t, eta_{t+1}) ~= L_ctrl,t+H - L_ctrl,t
  with H=5 blocks = 50 trees by default.
* To avoid the previous surrogate-exploitation failure, Q cannot optimize eta
  arbitrarily. It only ranks a small local candidate set around the current
  hypergradient proposal.
* The first 20 blocks explore that local candidate set to create action support.
  Afterwards Q acts greedily most of the time with 10% continued exploration.
* Q is a small bootstrap ensemble of regularized ridge models. Candidate score
  is pessimistic mean + beta * bootstrap std; lower is better.
* Test data is never used for training, adaptation, or action selection.

Important interpretation
------------------------
The delayed label is a Monte-Carlo outcome under the realized continuation
policy. It is not a counterfactual oracle label for every candidate eta. This
experiment asks whether online value learning from one trajectory is already
useful when constrained to supported local actions.

Baseline
--------
The script reads results/covtype_online_eta_blocked10_pairs.csv and compares Q
against the already-completed blocked-hypergradient run for the exact same eta
starts. This avoids spending another 15,000 baseline trees.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("results")
OUT.mkdir(exist_ok=True)

SEED = 20260914
N_SAMPLE = 250_000
N_ROUNDS = 750
N_STARTS = 20
ETA_MIN = 0.03
ETA_MAX = 0.80
BLOCK = 10
HORIZON_BLOCKS = 5  # 50 trees
WARMUP_BLOCKS = 20
EPSILON_EXPLORE = 0.10
LOG_STEP = 0.25
ACTION_LOG_OFFSETS = np.asarray([-0.20, -0.10, 0.0, 0.10, 0.20], dtype=float)
MIN_Q_LABELS = 15
Q_ENSEMBLE = 32
Q_RIDGE_ALPHA = 1.0
Q_PESSIMISM_BETA = 0.5
EPS = 1e-12
BASELINE_PAIRS = OUT / "covtype_online_eta_blocked10_pairs.csv"


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
    p.add_argument("--block", type=int, default=BLOCK)
    p.add_argument("--horizon-blocks", type=int, default=HORIZON_BLOCKS)
    p.add_argument("--warmup-blocks", type=int, default=WARMUP_BLOCKS)
    p.add_argument("--checkpoint-every", type=int, default=50)
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


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def entropy_from_probs(p: np.ndarray) -> float:
    return float(np.mean(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "logloss": float(log_loss(y, p, labels=np.arange(7))),
        "accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
    }


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


def state_vector(
    progress: float,
    ctrl_loss: float,
    recent_delta: float,
    recent_accel: float,
    current_eta: float,
    g_log_eta: float,
    mean_abs_margin: float,
    entropy: float,
    proposal_eta: float,
) -> dict[str, float]:
    return {
        "progress": float(progress),
        "ctrl_loss": float(ctrl_loss),
        "recent_delta": float(recent_delta),
        "recent_accel": float(recent_accel),
        "log_current_eta": float(math.log(current_eta)),
        "g_log_eta": float(g_log_eta),
        "mean_abs_margin": float(mean_abs_margin),
        "entropy": float(entropy),
        "log_proposal_eta": float(math.log(proposal_eta)),
    }


def q_features(state: dict[str, float], candidate_eta: float) -> np.ndarray:
    lc = math.log(float(candidate_eta))
    lcur = state["log_current_eta"]
    lprop = state["log_proposal_eta"]
    # A small hand-built feature map keeps the online learner data-efficient.
    return np.asarray(
        [
            state["progress"],
            state["ctrl_loss"],
            state["recent_delta"],
            state["recent_accel"],
            lcur,
            state["g_log_eta"],
            state["mean_abs_margin"],
            state["entropy"],
            lprop,
            lc,
            lc - lcur,
            lc - lprop,
            lc * lc,
            lc * state["progress"],
            lc * state["g_log_eta"],
            lc * state["ctrl_loss"],
        ],
        dtype=float,
    )


def candidate_etas(proposal_eta: float) -> list[float]:
    vals = np.exp(math.log(proposal_eta) + ACTION_LOG_OFFSETS)
    vals = np.clip(vals, ETA_MIN, ETA_MAX)
    # Preserve order but remove clipping-induced duplicates.
    out = []
    for v in vals:
        fv = float(v)
        if not out or all(abs(fv - z) > 1e-12 for z in out):
            out.append(fv)
    return out


def fit_q_ensemble(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    models = []
    n = len(y)
    for _ in range(Q_ENSEMBLE):
        idx = rng.integers(0, n, size=n)
        m = make_pipeline(StandardScaler(), Ridge(alpha=Q_RIDGE_ALPHA))
        m.fit(X[idx], y[idx])
        models.append(m)
    return models


def q_scores(models, state, candidates):
    Xc = np.stack([q_features(state, eta) for eta in candidates])
    preds = np.stack([m.predict(Xc) for m in models], axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0, ddof=1) if preds.shape[0] > 1 else np.zeros(len(candidates))
    score = mean + Q_PESSIMISM_BETA * std
    return mean, std, score


def q_run(
    dtrain,
    dctrl,
    dtest,
    yctrl,
    ytest,
    eta0,
    nthread,
    n_rounds,
    block,
    horizon_blocks,
    warmup_blocks,
    checkpoint_every,
    label,
    run_seed,
):
    if n_rounds % block != 0:
        raise ValueError("n_rounds must be divisible by block")
    n_blocks = n_rounds // block
    rng = np.random.default_rng(run_seed)

    booster = xgb.train(params(nthread, eta0), dtrain, num_boost_round=0)
    current_eta = float(eta0)
    grad_sq_sum = 0.0

    margin_before = booster.predict(dctrl, output_margin=True)
    p0 = softmax(margin_before)
    initial_loss = metrics(yctrl, p0)["logloss"]
    prev_loss = initial_loss
    prev_delta = 0.0

    # Matured supervised data: each row is state/action, each target is the
    # realized control-loss change after horizon_blocks future blocks.
    X_q: list[np.ndarray] = []
    y_q: list[float] = []
    pending: list[dict] = []
    history = []
    wall_by_block = []
    start = time.perf_counter()

    for bidx in range(1, n_blocks + 1):
        block_margin_before = margin_before
        booster = xgb.train(
            params(nthread, current_eta),
            dtrain,
            num_boost_round=block,
            xgb_model=booster,
            verbose_eval=False,
        )
        margin_after = booster.predict(dctrl, output_margin=True)
        p = softmax(margin_after)
        ctrl_loss = metrics(yctrl, p)["logloss"]
        recent_delta = ctrl_loss - prev_loss
        recent_accel = recent_delta - prev_delta

        delta_margin = margin_after - block_margin_before
        residual = p.copy()
        residual[np.arange(len(yctrl)), yctrl] -= 1.0
        g_log_eta = float(np.mean(np.sum(residual * delta_margin, axis=1)))
        grad_sq_sum += g_log_eta * g_log_eta
        normalized_step = LOG_STEP * g_log_eta / math.sqrt(grad_sq_sum + EPS)
        proposal_log_eta = float(
            np.clip(
                math.log(current_eta) - normalized_step,
                math.log(ETA_MIN),
                math.log(ETA_MAX),
            )
        )
        proposal_eta = math.exp(proposal_log_eta)

        # Mature delayed outcomes whose horizon ends at this block.
        still_pending = []
        for rec in pending:
            if rec["future_block"] == bidx:
                X_q.append(rec["features"])
                y_q.append(float(ctrl_loss - rec["base_loss"]))
            else:
                still_pending.append(rec)
        pending = still_pending

        state = state_vector(
            progress=bidx / n_blocks,
            ctrl_loss=ctrl_loss,
            recent_delta=recent_delta,
            recent_accel=recent_accel,
            current_eta=current_eta,
            g_log_eta=g_log_eta,
            mean_abs_margin=float(np.mean(np.abs(margin_after))),
            entropy=entropy_from_probs(p),
            proposal_eta=proposal_eta,
        )
        candidates = candidate_etas(proposal_eta)

        # No next action is needed after the final block.
        chosen_eta = current_eta
        action_source = "terminal"
        candidate_details = []
        if bidx < n_blocks:
            enough_data = len(y_q) >= MIN_Q_LABELS
            in_warmup = bidx <= warmup_blocks
            explore = in_warmup or (rng.random() < EPSILON_EXPLORE)

            if explore or not enough_data:
                chosen_eta = float(rng.choice(candidates))
                action_source = "warmup_random" if in_warmup else "epsilon_random"
            else:
                Xarr = np.stack(X_q)
                yarr = np.asarray(y_q, dtype=float)
                models = fit_q_ensemble(Xarr, yarr, rng)
                means, stds, scores = q_scores(models, state, candidates)
                best = int(np.argmin(scores))
                chosen_eta = float(candidates[best])
                action_source = "q_greedy"
                candidate_details = [
                    {
                        "eta": float(e),
                        "pred_mean_delta": float(mu),
                        "pred_std": float(sd),
                        "pessimistic_score": float(sc),
                    }
                    for e, mu, sd, sc in zip(candidates, means, stds, scores)
                ]

            pending.append(
                {
                    "future_block": bidx + horizon_blocks,
                    "base_loss": float(ctrl_loss),
                    "features": q_features(state, chosen_eta),
                }
            )
            # Records whose horizon would extend beyond training can never mature.
            pending = [r for r in pending if r["future_block"] <= n_blocks]

        history.append(
            {
                "block": bidx,
                "trees": bidx * block,
                "eta_used": float(current_eta),
                "hypergrad_proposal_eta": float(proposal_eta),
                "chosen_next_eta": float(chosen_eta),
                "action_source": action_source,
                "g_log_eta": float(g_log_eta),
                "normalized_log_step": float(normalized_step),
                "control_logloss": float(ctrl_loss),
                "recent_control_delta": float(recent_delta),
                "recent_control_accel": float(recent_accel),
                "mean_abs_margin": state["mean_abs_margin"],
                "entropy": state["entropy"],
                "n_q_labels": len(y_q),
                "candidate_details": candidate_details,
            }
        )
        wall_by_block.append(time.perf_counter() - start)

        prev_delta = recent_delta
        prev_loss = ctrl_loss
        margin_before = margin_after
        current_eta = chosen_eta

    ctrl = metrics(yctrl, booster.predict(dctrl))
    test = metrics(ytest, booster.predict(dtest))

    checkpoints = list(range(checkpoint_every, n_rounds + 1, checkpoint_every))
    if not checkpoints or checkpoints[-1] != n_rounds:
        checkpoints.append(n_rounds)
    convergence = []
    for k in checkpoints:
        pctrl = booster.predict(dctrl, iteration_range=(0, k))
        ptest = booster.predict(dtest, iteration_range=(0, k))
        cm = metrics(yctrl, pctrl)
        tm = metrics(ytest, ptest)
        b_for_time = int(math.ceil(k / block))
        eta_k = history[b_for_time - 1]["eta_used"]
        convergence.append(
            {
                "label": label,
                "method": "q_controller",
                "eta0": float(eta0),
                "trees": int(k),
                "eta": float(eta_k),
                "control_logloss": cm["logloss"],
                "test_logloss": tm["logloss"],
                "test_accuracy": tm["accuracy"],
                "training_wall_seconds": float(wall_by_block[b_for_time - 1]),
            }
        )

    return {
        "label": label,
        "eta0": float(eta0),
        "q_final_eta": float(history[-1]["eta_used"]),
        "q_control_logloss": ctrl["logloss"],
        "q_test_logloss": test["logloss"],
        "q_test_accuracy": test["accuracy"],
        "q_final_wall_seconds": float(wall_by_block[-1]),
        "q_labels_matured": len(y_q),
        "q_greedy_blocks": int(sum(h["action_source"] == "q_greedy" for h in history)),
        "q_random_blocks": int(
            sum("random" in h["action_source"] for h in history)
        ),
    }, history, convergence


def main():
    args = parse_args()
    if not BASELINE_PAIRS.exists():
        raise FileNotFoundError(
            f"Missing {BASELINE_PAIRS}. Pull the committed blocked-hypergradient "
            "results before running this experiment."
        )
    if args.n_rounds != N_ROUNDS:
        raise ValueError(
            "This comparison reuses the committed 750-tree hypergradient baseline; "
            "run with --n-rounds 750."
        )

    baseline = pd.read_csv(BASELINE_PAIRS)
    baseline = baseline[baseline["label"].str.startswith("prior_draw_")].copy()
    baseline = baseline.sort_values("label").head(args.n_starts)
    if len(baseline) != args.n_starts:
        raise ValueError(f"Requested {args.n_starts} starts but baseline has {len(baseline)}")

    print(
        f"nthread={args.nthread} n_starts={args.n_starts} n_rounds={args.n_rounds} "
        f"block={args.block} horizon={args.horizon_blocks} blocks "
        f"warmup={args.warmup_blocks} blocks",
        flush=True,
    )
    dtrain, dctrl, dtest, yctrl, ytest = load_data()
    t0 = time.time()

    rows = []
    histories = {}
    convergence_rows = []
    for i, br in baseline.reset_index(drop=True).iterrows():
        label = str(br["label"])
        eta0 = float(br["eta0"])
        print(
            f"\n[{label}] eta0={eta0:.6f} hypergrad baseline="
            f"{float(br['adaptive_test_logloss']):.6f}; Q...",
            flush=True,
        )
        qrow, history, curve = q_run(
            dtrain,
            dctrl,
            dtest,
            yctrl,
            ytest,
            eta0,
            args.nthread,
            args.n_rounds,
            args.block,
            args.horizon_blocks,
            args.warmup_blocks,
            args.checkpoint_every,
            label,
            run_seed=SEED + 1000 + i,
        )
        qrow["hypergrad_test_logloss"] = float(br["adaptive_test_logloss"])
        qrow["hypergrad_control_logloss"] = float(br["adaptive_control_logloss"])
        qrow["hypergrad_final_eta"] = float(br["adaptive_final_eta"])
        qrow["q_minus_hypergrad_test_logloss"] = float(
            qrow["q_test_logloss"] - qrow["hypergrad_test_logloss"]
        )
        qrow["q_beats_hypergrad"] = bool(qrow["q_minus_hypergrad_test_logloss"] < 0)
        rows.append(qrow)
        histories[label] = history
        convergence_rows.extend(curve)
        print(
            f"[{label}] Q test={qrow['q_test_logloss']:.6f} "
            f"eta_final={qrow['q_final_eta']:.5f} "
            f"delta_vs_hg={qrow['q_minus_hypergrad_test_logloss']:+.6f} "
            f"wall={qrow['q_final_wall_seconds']:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "covtype_online_q_pairs.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(
        OUT / "covtype_online_q_convergence.csv", index=False
    )
    with open(OUT / "covtype_online_q_history.json", "w") as f:
        json.dump(histories, f, indent=2)

    deltas = df["q_minus_hypergrad_test_logloss"].to_numpy(float)
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
        "baseline": "10-tree blocked hypergradient, previously committed results",
        "q_controller": {
            "block_trees": args.block,
            "horizon_blocks": args.horizon_blocks,
            "horizon_trees": args.block * args.horizon_blocks,
            "warmup_blocks": args.warmup_blocks,
            "epsilon_explore": EPSILON_EXPLORE,
            "candidate_log_offsets_around_hypergrad_proposal": ACTION_LOG_OFFSETS.tolist(),
            "model": "bootstrap ridge ensemble",
            "ensemble_size": Q_ENSEMBLE,
            "ridge_alpha": Q_RIDGE_ALPHA,
            "pessimism_beta": Q_PESSIMISM_BETA,
            "counterfactual_tree_fits": 0,
            "target": "realized future control-logloss change",
        },
        "expected_hypergrad_test_logloss": float(df["hypergrad_test_logloss"].mean()),
        "expected_q_test_logloss": float(df["q_test_logloss"].mean()),
        "mean_q_minus_hypergrad_test_logloss": mean_delta,
        "mean_q_improvement_over_hypergrad": float(-mean_delta),
        "paired_delta_95pct_t_interval": [float(ci_low), float(ci_high)],
        "q_win_rate_vs_hypergrad": float(df["q_beats_hypergrad"].mean()),
        "mean_q_wall_seconds": float(df["q_final_wall_seconds"].mean()),
        "mean_q_labels_matured": float(df["q_labels_matured"].mean()),
        "mean_q_greedy_blocks": float(df["q_greedy_blocks"].mean()),
        "predeclared_q_gate": {
            "definition": "mean Q improvement over blocked hypergradient >=0.002 and Q win rate >=60%",
            "passes": bool(mean_delta <= -0.002 and df["q_beats_hypergrad"].mean() >= 0.60),
        },
        "main_path_tree_budget": {
            "hypergradient": args.n_rounds,
            "q_controller": args.n_rounds,
        },
        "test_used_for_adaptation": False,
        "elapsed_seconds": time.time() - t0,
    }
    with open(OUT / "covtype_online_q_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
