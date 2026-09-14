"""Falsification test for state-dependent XGBoost shrinkage control.

Primary question: can a true K-step rollout oracle over eta beat a well-tuned
fixed eta on untouched test data? If not, a learned Q(s, eta) controller has
little reason to help when eta is the only action.

Important design constraints:
- deterministic boosting (no row/column subsampling), so the oracle cannot
  accidentally select lucky random seeds;
- same seed for all candidate branches;
- wider eta grid and longer horizon than the first screening run;
- fixed-eta baseline is allowed to choose its best iteration on control data;
- final comparison uses a separate untouched test split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

OUT = Path("results")
OUT.mkdir(exist_ok=True)

SEED = 20260914
N_SAMPLE = 250_000
N_ROUNDS = 750
BLOCK = 25
ETAS = [0.10, 0.20, 0.30, 0.40, 0.50]

PARAMS = {
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
    "nthread": 32,
    "seed": SEED,
}


def metrics(y, p):
    return {
        "logloss": float(log_loss(y, p, labels=np.arange(7))),
        "accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
    }


def predict(b, d, end=None):
    return b.predict(d) if end is None else b.predict(d, iteration_range=(0, end))


def add_rounds(booster, dtrain, eta, n, seed):
    p = dict(PARAMS)
    p["eta"] = float(eta)
    p["seed"] = int(seed)
    return xgb.train(p, dtrain, num_boost_round=n, xgb_model=booster, verbose_eval=False)


def fixed_run(dtrain, dctrl, dtest, yctrl, ytest, eta):
    p = dict(PARAMS)
    p["eta"] = float(eta)
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


def rollout_oracle(dtrain, dctrl, dtest, yctrl, ytest):
    booster = None
    rounds = 0
    history = []
    best_loss = np.inf
    best_round = 0

    while rounds < N_ROUNDS:
        n = min(BLOCK, N_ROUNDS - rounds)
        # Same seed for every candidate at this state. With subsample and
        # colsample_bytree both 1.0 this should be deterministic anyway.
        branch_seed = SEED + rounds
        candidates = []
        for eta in ETAS:
            b = add_rounds(booster, dtrain, eta, n, branch_seed)
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
        print(f"oracle round={rounds:4d} eta={eta:.2f} ctrl={c:.6f}", flush=True)

    ctrl = metrics(yctrl, predict(booster, dctrl, best_round))
    test = metrics(ytest, predict(booster, dtest, best_round))
    return {
        "method": f"rollout_oracle_k{BLOCK}",
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

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dctrl = xgb.DMatrix(Xctrl, label=yctrl)
    dtest = xgb.DMatrix(Xtest, label=ytest)

    rows = []
    for eta in ETAS:
        r = fixed_run(dtrain, dctrl, dtest, yctrl, ytest, eta)
        rows.append(r)
        print(r, flush=True)

    o = rollout_oracle(dtrain, dctrl, dtest, yctrl, ytest)
    rows.append({k: v for k, v in o.items() if k != "history"})
    pd.DataFrame(rows).to_csv(OUT / "covtype_falsification.csv", index=False)

    best_fixed = min(
        (r for r in rows if str(r["method"]).startswith("fixed_eta_")),
        key=lambda r: r["control_logloss"],
    )
    oracle = next(r for r in rows if str(r["method"]).startswith("rollout_oracle_"))
    chosen = [h["chosen_eta"] for h in o["history"]]
    verdict = {
        "dataset": "UCI Covertype",
        "sample_size": int(len(y)),
        "train_size": int(len(ytr)),
        "control_size": int(len(yctrl)),
        "test_size": int(len(ytest)),
        "n_rounds": N_ROUNDS,
        "block": BLOCK,
        "etas": ETAS,
        "deterministic_no_subsampling": True,
        "best_fixed": best_fixed,
        "oracle": oracle,
        "oracle_eta_path": chosen,
        "oracle_unique_etas": sorted(set(chosen)),
        "oracle_minus_fixed_test_logloss": float(oracle["test_logloss"] - best_fixed["test_logloss"]),
        "oracle_relative_test_logloss_change": float(oracle["test_logloss"] / best_fixed["test_logloss"] - 1),
        "passes_gate": bool(oracle["test_logloss"] < best_fixed["test_logloss"] - 0.001),
        "gate_definition": "oracle test logloss at least 0.001 lower than tuned fixed eta",
        "elapsed_seconds": time.time() - t0,
    }
    with open(OUT / "covtype_falsification.json", "w") as f:
        json.dump(verdict, f, indent=2)
    with open(OUT / "covtype_falsification_history.json", "w") as f:
        json.dump({"oracle": o["history"]}, f, indent=2)

    print("\nVERDICT")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
