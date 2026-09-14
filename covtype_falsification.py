"""Falsification test for long-horizon online control of XGBoost shrinkage.

Question: is there enough useful state-dependent structure in the learning-rate
schedule for a K-step rollout oracle to beat a well-tuned fixed-eta baseline?

If not, a learned Q(s, eta) controller has little reason to help when eta is the
only action. The test uses UCI Covertype, a large multiclass tabular benchmark.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_covtype
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split

OUT = Path("results")
OUT.mkdir(exist_ok=True)

SEED = 20260914
N_SAMPLE = 250_000
N_ROUNDS = 500
BLOCK = 25
ETAS = [0.025, 0.05, 0.10, 0.20, 0.30]

PARAMS = {
    "objective": "multi:softprob",
    "num_class": 7,
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "nthread": 2,
    "seed": SEED,
}


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "logloss": float(log_loss(y, p, labels=np.arange(7))),
        "accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
    }


def predict(b: xgb.Booster, d: xgb.DMatrix, end: int | None = None) -> np.ndarray:
    if end is None:
        return b.predict(d)
    return b.predict(d, iteration_range=(0, end))


def add_rounds(booster: xgb.Booster | None, dtrain: xgb.DMatrix, eta: float,
               n: int, seed: int) -> xgb.Booster:
    p = dict(PARAMS)
    p["eta"] = eta
    p["seed"] = seed
    return xgb.train(
        p,
        dtrain,
        num_boost_round=n,
        xgb_model=booster,
        verbose_eval=False,
    )


def fixed_run(dtrain, dctrl, dtest, yctrl, ytest, eta: float) -> dict:
    p = dict(PARAMS)
    p["eta"] = eta
    evals_result = {}
    b = xgb.train(
        p,
        dtrain,
        num_boost_round=N_ROUNDS,
        evals=[(dctrl, "control")],
        evals_result=evals_result,
        verbose_eval=False,
    )
    curve = np.asarray(evals_result["control"]["mlogloss"], dtype=float)
    best_round = int(np.argmin(curve)) + 1
    ctrl = metrics(yctrl, predict(b, dctrl, best_round))
    test = metrics(ytest, predict(b, dtest, best_round))
    return {
        "method": f"fixed_eta_{eta}",
        "eta": eta,
        "best_round": best_round,
        "control_logloss": ctrl["logloss"],
        "test_logloss": test["logloss"],
        "test_accuracy": test["accuracy"],
    }


def scheduled_run(dtrain, dctrl, dtest, yctrl, ytest) -> dict:
    """A conventional hand schedule: 0.20 -> 0.10 -> 0.05 -> 0.025."""
    schedule = [(0, 125, 0.20), (125, 250, 0.10), (250, 375, 0.05), (375, 500, 0.025)]
    booster = None
    best_loss = np.inf
    best_round = 0
    history = []
    rounds = 0
    for start, stop, eta in schedule:
        for _ in range(start, stop, BLOCK):
            n = min(BLOCK, stop - rounds)
            booster = add_rounds(booster, dtrain, eta, n, SEED + rounds)
            rounds += n
            c = metrics(yctrl, predict(booster, dctrl))["logloss"]
            history.append({"round": rounds, "eta": eta, "control_logloss": c})
            if c < best_loss:
                best_loss, best_round = c, rounds
    ctrl = metrics(yctrl, predict(booster, dctrl, best_round))
    test = metrics(ytest, predict(booster, dtest, best_round))
    return {
        "method": "hand_decay_schedule",
        "eta": None,
        "best_round": best_round,
        "control_logloss": ctrl["logloss"],
        "test_logloss": test["logloss"],
        "test_accuracy": test["accuracy"],
        "history": history,
    }


def rollout_oracle(dtrain, dctrl, dtest, yctrl, ytest) -> dict:
    """At every BLOCK rounds, fork the booster over ETAS and keep the branch
    with minimum held-out control loss after BLOCK more boosting rounds.

    This deliberately gives the idea an *upper bound*: it pays ~len(ETAS)x
    training compute to observe the real K-step consequence of each action.
    If this cannot beat fixed eta on untouched test data, learning Q(s, eta)
    is not promising in this action space.
    """
    booster = None
    rounds = 0
    history = []
    best_loss = np.inf
    best_round = 0

    while rounds < N_ROUNDS:
        n = min(BLOCK, N_ROUNDS - rounds)
        candidates = []
        for j, eta in enumerate(ETAS):
            b = add_rounds(booster, dtrain, eta, n, SEED + 100_000 + rounds * 10 + j)
            c = metrics(yctrl, predict(b, dctrl))["logloss"]
            candidates.append((c, eta, b))
        candidates.sort(key=lambda z: z[0])
        c, eta, booster = candidates[0]
        rounds += n
        history.append({
            "round": rounds,
            "chosen_eta": eta,
            "control_logloss": c,
            "candidate_losses": {str(e): float(loss) for loss, e, _ in candidates},
        })
        if c < best_loss:
            best_loss, best_round = c, rounds
        print(f"oracle round={rounds:4d} eta={eta:.3f} ctrl={c:.6f}", flush=True)

    ctrl = metrics(yctrl, predict(booster, dctrl, best_round))
    test = metrics(ytest, predict(booster, dtest, best_round))
    return {
        "method": "rollout_oracle_k25",
        "eta": None,
        "best_round": best_round,
        "control_logloss": ctrl["logloss"],
        "test_logloss": test["logloss"],
        "test_accuracy": test["accuracy"],
        "history": history,
    }


def main():
    t0 = time.time()
    X, y = fetch_covtype(return_X_y=True)
    y = y.astype(np.int32) - 1

    # Large but bounded run for GitHub-hosted CPU. Sampling is stratified and
    # deterministic; all methods see exactly the same rows and splits.
    if len(y) > N_SAMPLE:
        X, _, y, _ = train_test_split(
            X, y, train_size=N_SAMPLE, stratify=y, random_state=SEED
        )

    # 70% train, 15% controller/validation, 15% untouched final test.
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    Xctrl, Xtest, yctrl, ytest = train_test_split(
        Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED + 1
    )

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dctrl = xgb.DMatrix(Xctrl, label=yctrl)
    dtest = xgb.DMatrix(Xtest, label=ytest)

    rows = []
    for eta in ETAS:
        r = fixed_run(dtrain, dctrl, dtest, yctrl, ytest, eta)
        rows.append(r)
        print(r, flush=True)

    s = scheduled_run(dtrain, dctrl, dtest, yctrl, ytest)
    rows.append({k: v for k, v in s.items() if k != "history"})
    print({k: v for k, v in s.items() if k != "history"}, flush=True)

    o = rollout_oracle(dtrain, dctrl, dtest, yctrl, ytest)
    rows.append({k: v for k, v in o.items() if k != "history"})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "covtype_falsification.csv", index=False)

    best_fixed = min((r for r in rows if str(r["method"]).startswith("fixed_eta_")),
                     key=lambda r: r["control_logloss"])
    oracle = next(r for r in rows if r["method"] == "rollout_oracle_k25")
    verdict = {
        "dataset": "UCI Covertype",
        "sample_size": int(len(y)),
        "train_size": int(len(ytr)),
        "control_size": int(len(yctrl)),
        "test_size": int(len(ytest)),
        "n_rounds": N_ROUNDS,
        "block": BLOCK,
        "etas": ETAS,
        "best_fixed": best_fixed,
        "oracle": oracle,
        "oracle_minus_fixed_test_logloss": float(oracle["test_logloss"] - best_fixed["test_logloss"]),
        "oracle_relative_test_logloss_change": float((oracle["test_logloss"] / best_fixed["test_logloss"]) - 1),
        "passes_gate": bool(oracle["test_logloss"] < best_fixed["test_logloss"] - 0.001),
        "gate_definition": "oracle test logloss at least 0.001 lower than tuned fixed eta",
        "elapsed_seconds": time.time() - t0,
    }
    with open(OUT / "covtype_falsification.json", "w") as f:
        json.dump(verdict, f, indent=2)
    with open(OUT / "covtype_falsification_history.json", "w") as f:
        json.dump({"schedule": s["history"], "oracle": o["history"]}, f, indent=2)

    print("\nVERDICT")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
