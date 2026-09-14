import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def metrics(y, margin):
    p = sigmoid(margin)
    return float(log_loss(y, p, labels=[0, 1])), float(roc_auc_score(y, p))


BASE_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    max_depth=3,
    min_child_weight=1.0,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist",
    nthread=2,
    seed=0,
)


def add_round(booster, dtrain, eta, seed):
    p = BASE_PARAMS.copy()
    p["eta"] = float(eta)
    p["seed"] = int(seed)
    return xgb.train(p, dtrain, num_boost_round=1, xgb_model=booster, verbose_eval=False)


def pred_margin(booster, dmat):
    if booster is None:
        return np.zeros(dmat.num_row(), dtype=float)
    return booster.predict(dmat, output_margin=True)


def setup(seed):
    ds = load_breast_cancer()
    X, y = ds.data, ds.target
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    Xv, Xte, yv, yte = train_test_split(
        Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed + 1000
    )
    return (
        xgb.DMatrix(Xtr, label=ytr),
        xgb.DMatrix(Xv, label=yv),
        xgb.DMatrix(Xte, label=yte),
        ytr,
        yv,
        yte,
    )


def state_action_features(t, total_rounds, val_loss, recent_delta, train_loss,
                          previous_eta, mean_abs_val_margin, candidate_eta):
    return [
        t / total_rounds,
        val_loss,
        recent_delta,
        train_loss,
        previous_eta,
        mean_abs_val_margin,
        candidate_eta,
    ]


def train_fixed(seed, eta, n_rounds=80):
    dtr, dv, dte, ytr, yv, yte = setup(seed)
    booster = None
    for t in range(n_rounds):
        booster = add_round(booster, dtr, eta, seed + t)
    vloss, _ = metrics(yv, pred_margin(booster, dv))
    tloss, tauc = metrics(yte, pred_margin(booster, dte))
    return vloss, dict(seed=seed, method=f"fixed_{eta}", test_logloss=tloss, test_auc=tauc)


def train_q_controller(seed, horizon=10, n_rounds=80, warmup=20, epsilon=0.1):
    """Online critic over (training state, candidate eta) -> K-step validation loss change.

    A decision made at round t is not labeled until round t+K, so the controller never
    trains on future information that would be unavailable online. During warmup it
    explores a broad eta grid to obtain action coverage. Once enough delayed labels
    mature, an MLP predicts K-step validation-loss change for each candidate eta and
    chooses the minimum predicted value. A small epsilon keeps collecting off-policy
    data and limits critic collapse.

    This first implementation deliberately searches a 1-D grid instead of differentiating
    through Q. In one dimension the grid is cheap and separates the value-model question
    from surrogate-gradient pathologies.
    """
    dtr, dv, dte, ytr, yv, yte = setup(seed)
    rng = np.random.default_rng(seed + 999)
    eta_grid = np.linspace(0.02, 0.30, 29)

    booster = None
    eta = 0.1
    previous_delta = 0.0
    pending = []
    replay_X, replay_y = [], []
    model = None
    history = []

    for t in range(n_rounds):
        val_margin = pred_margin(booster, dv)
        train_margin = pred_margin(booster, dtr)
        val_loss, _ = metrics(yv, val_margin)
        train_loss, _ = metrics(ytr, train_margin)
        mean_abs_margin = float(np.mean(np.abs(val_margin)))

        still_pending = []
        for due_round, features, starting_loss in pending:
            if due_round <= t:
                replay_X.append(features)
                replay_y.append(val_loss - starting_loss)
            else:
                still_pending.append((due_round, features, starting_loss))
        pending = still_pending

        if t < warmup or len(replay_y) < 10:
            eta_next = float(rng.choice(eta_grid))
        else:
            if (t - warmup) % 5 == 0 or model is None:
                model = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(32, 32),
                        activation="tanh",
                        solver="lbfgs",
                        alpha=1e-3,
                        max_iter=300,
                        random_state=seed,
                    ),
                )
                model.fit(np.asarray(replay_X), np.asarray(replay_y))

            if rng.random() < epsilon:
                eta_next = float(rng.choice(eta_grid))
            else:
                candidates = np.asarray([
                    state_action_features(
                        t, n_rounds, val_loss, previous_delta, train_loss,
                        eta, mean_abs_margin, candidate_eta
                    )
                    for candidate_eta in eta_grid
                ])
                predicted_delta = model.predict(candidates)
                eta_next = float(eta_grid[np.argmin(predicted_delta)])

        chosen_features = state_action_features(
            t, n_rounds, val_loss, previous_delta, train_loss,
            eta, mean_abs_margin, eta_next
        )
        pending.append((t + horizon, chosen_features, val_loss))

        booster = add_round(booster, dtr, eta_next, seed + t)
        next_val_loss, _ = metrics(yv, pred_margin(booster, dv))
        previous_delta = next_val_loss - val_loss
        eta = eta_next
        history.append((t + 1, eta, next_val_loss, len(replay_y)))

    test_loss, test_auc = metrics(yte, pred_margin(booster, dte))
    return dict(
        seed=seed,
        method=f"q_K{horizon}",
        test_logloss=test_loss,
        test_auc=test_auc,
    ), history


def run(seeds=range(7), horizons=(5, 10, 20), n_rounds=80):
    rows = []
    for seed in seeds:
        fixed = []
        for eta in (0.03, 0.1, 0.2):
            val_loss, result = train_fixed(seed, eta, n_rounds=n_rounds)
            fixed.append((val_loss, result))
        best_fixed = min(fixed, key=lambda x: x[0])[1].copy()
        best_fixed["method"] = "fixed_tuned"
        rows.append(best_fixed)

        for horizon in horizons:
            result, _ = train_q_controller(
                seed, horizon=horizon, n_rounds=n_rounds
            )
            rows.append(result)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print(
        df.groupby("method").agg(
            test_logloss_mean=("test_logloss", "mean"),
            test_logloss_sd=("test_logloss", "std"),
            test_auc_mean=("test_auc", "mean"),
        ).to_string()
    )
    df.to_csv("results/long_horizon_q.csv", index=False)


if __name__ == "__main__":
    run()
