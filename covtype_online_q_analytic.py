"""Analytic continuous-quadratic long-horizon Q controller.

Follow-up to covtype_online_q_continuous.py.

The learned action-value surface is already exactly quadratic in
    x = log(eta_next) - log(eta_hypergrad_proposal)
for fixed state. This version therefore removes both ingredients that are not
needed for action selection:

1) no bounded scalar search;
2) no pessimistic mean + beta * bootstrap-std objective.

The bootstrap ensemble is retained only as bagging / a variance diagnostic.
Actions minimize the ensemble-mean quadratic exactly inside the same trust
region. If the learned mean quadratic is convex, use its analytic vertex. If it
is linear/concave, the exact bounded minimum is one of the two endpoints.
Uncertainty never changes the chosen action.

Everything else is held fixed against the previous continuous-Q experiment:
same data split, 20 eta starts, 750 trees, 10-tree blocks, 50-tree delayed
Monte-Carlo target, continuous warmup/exploration, trust radius, and run seeds.
This makes the experiment primarily an ablation of the optimizer/objective.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import covtype_online_q_continuous as cq

OUT = Path("results")
OUT.mkdir(exist_ok=True)
SEED = cq.SEED
BASELINE = OUT / "covtype_online_eta_blocked10_pairs.csv"
DISCRETE = OUT / "covtype_online_q_pairs.csv"
PREVIOUS_CONTINUOUS = OUT / "covtype_online_q_continuous_pairs.csv"
CURVATURE_EPS = 1e-10


def mean_q(models, state, x: float) -> float:
    """Ensemble-mean Q. Std is deliberately excluded from the action rule."""
    return float(cq.q_stats(models, state, float(x))[0])


def quadratic_coefficients(models, state):
    """Recover Qbar(x) = a*x^2 + b*x + c exactly from three evaluations."""
    q0 = mean_q(models, state, 0.0)
    qp = mean_q(models, state, 1.0)
    qm = mean_q(models, state, -1.0)
    a = 0.5 * (qp + qm) - q0
    b = 0.5 * (qp - qm)
    c = q0
    return float(a), float(b), float(c)


def choose_analytic(models, state, proposal, radius):
    """Exactly minimize the learned mean quadratic over the trust interval."""
    lo, hi = cq.bounds(proposal, radius)
    a, b, c = quadratic_coefficients(models, state)

    if hi - lo < 1e-12:
        x = float(lo)
        case = "degenerate_bound"
        unconstrained = x
    elif a > CURVATURE_EPS:
        unconstrained = float(-b / (2.0 * a))
        x = float(np.clip(unconstrained, lo, hi))
        case = "convex_interior" if lo < unconstrained < hi else "convex_clipped"
    else:
        # For a linear or concave quadratic, the exact minimum on a compact
        # interval is attained at an endpoint. This is analytic, not a search.
        qlo = a * lo * lo + b * lo + c
        qhi = a * hi * hi + b * hi + c
        x = float(lo if qlo <= qhi else hi)
        unconstrained = None
        case = "linear_or_concave_endpoint"

    eta = cq.to_eta(proposal, x)
    x = cq.to_x(proposal, eta)  # exact after global eta clipping
    mu, sd, _ = cq.q_stats(models, state, x)
    return eta, {
        "x": float(x),
        "pred_mean_delta": float(mu),
        "pred_std_diagnostic_only": float(sd),
        "bounds": [float(lo), float(hi)],
        "quadratic_a": a,
        "quadratic_b": b,
        "quadratic_c": c,
        "unconstrained_vertex_x": unconstrained,
        "optimizer_case": case,
        "uncertainty_used_for_action": False,
        "scalar_search_used": False,
    }


# cq.run resolves this name in its module globals. Monkey-patching lets us keep
# every other experimental detail identical to the previous continuous run.
cq.choose_continuous = choose_analytic


def _rename_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        out[("aq_" + k[3:]) if k.startswith("cq_") else k] = v
    return out


def main():
    args = cq.args_parse()
    if not BASELINE.exists():
        raise FileNotFoundError(f"Missing {BASELINE}")
    if args.n_rounds != cq.N_ROUNDS:
        raise ValueError("Use 750 rounds to reuse the committed hypergradient baseline")

    baseline = pd.read_csv(BASELINE)
    baseline = (
        baseline[baseline.label.str.startswith("prior_draw_")]
        .sort_values("label")
        .head(args.n_starts)
    )
    if len(baseline) != args.n_starts:
        raise ValueError("Not enough baseline starts")

    if DISCRETE.exists():
        discrete = pd.read_csv(DISCRETE)[["label", "q_test_logloss"]]
        baseline = baseline.merge(discrete, on="label", how="left")
    if PREVIOUS_CONTINUOUS.exists():
        prev = pd.read_csv(PREVIOUS_CONTINUOUS)[["label", "cq_test_logloss"]]
        baseline = baseline.merge(prev, on="label", how="left")

    print(
        f"nthread={args.nthread} n_starts={args.n_starts} n_rounds={args.n_rounds} "
        f"block={args.block} horizon={args.horizon_blocks} warmup={args.warmup_blocks} "
        f"trust={args.trust_log_radius} optimizer=analytic_mean_quadratic",
        flush=True,
    )
    dtrain, dctrl, dtest, yctrl, ytest = cq.base.load_data()
    t0 = time.time()

    rows = []
    histories = {}
    curves = []
    for i, r in baseline.reset_index(drop=True).iterrows():
        label = str(r.label)
        eta0 = float(r.eta0)
        print(
            f"\n[{label}] eta0={eta0:.6f} HG={r.adaptive_test_logloss:.6f}; analytic-Q...",
            flush=True,
        )

        # Same seed as the previous continuous-Q run: warmup and epsilon
        # exploration streams are matched as closely as possible.
        raw, history, curve = cq.run(
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
            args.trust_log_radius,
            args.checkpoint_every,
            label,
            SEED + 2000 + i,
        )
        row = _rename_row(raw)
        for h in history:
            if h.get("action_source") == "q_continuous":
                h["action_source"] = "q_analytic"
        for c in curve:
            c["method"] = "analytic_quadratic_q"

        row["hypergrad_test_logloss"] = float(r.adaptive_test_logloss)
        row["aq_minus_hypergrad_test_logloss"] = (
            row["aq_test_logloss"] - row["hypergrad_test_logloss"]
        )
        row["aq_beats_hypergrad"] = row["aq_minus_hypergrad_test_logloss"] < 0

        if "q_test_logloss" in r and pd.notna(r.q_test_logloss):
            row["discrete_q_test_logloss"] = float(r.q_test_logloss)
            row["aq_minus_discrete_q_test_logloss"] = (
                row["aq_test_logloss"] - row["discrete_q_test_logloss"]
            )
            row["aq_beats_discrete_q"] = row["aq_minus_discrete_q_test_logloss"] < 0
        if "cq_test_logloss" in r and pd.notna(r.cq_test_logloss):
            row["previous_continuous_q_test_logloss"] = float(r.cq_test_logloss)
            row["aq_minus_previous_continuous_test_logloss"] = (
                row["aq_test_logloss"] - row["previous_continuous_q_test_logloss"]
            )
            row["aq_beats_previous_continuous"] = (
                row["aq_minus_previous_continuous_test_logloss"] < 0
            )

        rows.append(row)
        histories[label] = history
        curves.extend(curve)
        print(
            f"[{label}] AQ={row['aq_test_logloss']:.6f} "
            f"eta_final={row['aq_final_eta']:.5f} "
            f"delta_vs_HG={row['aq_minus_hypergrad_test_logloss']:+.6f} "
            f"wall={row['aq_final_wall_seconds']:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "covtype_online_q_analytic_pairs.csv", index=False)
    pd.DataFrame(curves).to_csv(
        OUT / "covtype_online_q_analytic_convergence.csv", index=False
    )
    with open(OUT / "covtype_online_q_analytic_history.json", "w") as f:
        json.dump(histories, f, indent=2)

    mean_delta, lo, hi = cq.ci(df.aq_minus_hypergrad_test_logloss)
    summary = {
        "dataset": "UCI Covertype",
        "sample_size": cq.base.N_SAMPLE,
        "n_rounds": args.n_rounds,
        "n_prior_draws": args.n_starts,
        "baseline": "10-tree blocked hypergradient, committed results",
        "analytic_q_controller": {
            "block_trees": args.block,
            "horizon_blocks": args.horizon_blocks,
            "horizon_trees": args.block * args.horizon_blocks,
            "warmup_blocks": args.warmup_blocks,
            "epsilon_explore": cq.EPSILON_EXPLORE,
            "action": "continuous log-eta offset from HG proposal",
            "trust_log_radius": args.trust_log_radius,
            "model": "bootstrap ridge ensemble, quadratic in continuous action",
            "ensemble_role": "bagged mean for actions; std recorded only as diagnostic",
            "optimizer": "closed-form bounded minimization of ensemble-mean quadratic",
            "pessimism_beta": 0.0,
            "scalar_search": False,
            "counterfactual_tree_fits": 0,
            "target": "realized 50-tree future control-logloss change",
        },
        "expected_hypergrad_test_logloss": float(df.hypergrad_test_logloss.mean()),
        "expected_analytic_q_test_logloss": float(df.aq_test_logloss.mean()),
        "mean_aq_minus_hypergrad_test_logloss": mean_delta,
        "mean_aq_improvement_over_hypergrad": -mean_delta,
        "paired_delta_95pct_t_interval_vs_hypergrad": [lo, hi],
        "analytic_q_win_rate_vs_hypergrad": float(df.aq_beats_hypergrad.mean()),
        "mean_analytic_q_wall_seconds": float(df.aq_final_wall_seconds.mean()),
        "mean_boundary_eta_fraction": float(df.aq_boundary_eta_fraction.mean()),
        "predeclared_primary_gate": {
            "definition": "mean improvement over HG >=0.002 and win rate >=60%",
            "passes": bool(
                mean_delta <= -0.002 and df.aq_beats_hypergrad.mean() >= 0.60
            ),
        },
        "main_path_tree_budget": {
            "hypergradient": args.n_rounds,
            "analytic_q": args.n_rounds,
        },
        "test_used_for_adaptation": False,
        "elapsed_seconds": time.time() - t0,
    }

    if "discrete_q_test_logloss" in df:
        md, dlo, dhi = cq.ci(df.aq_minus_discrete_q_test_logloss)
        summary["secondary_vs_discrete_q"] = {
            "expected_discrete_q_test_logloss": float(df.discrete_q_test_logloss.mean()),
            "mean_aq_minus_discrete_q_test_logloss": md,
            "paired_delta_95pct_t_interval": [dlo, dhi],
            "analytic_q_win_rate_vs_discrete_q": float(df.aq_beats_discrete_q.mean()),
        }
    if "previous_continuous_q_test_logloss" in df:
        md, clo, chi = cq.ci(df.aq_minus_previous_continuous_test_logloss)
        summary["secondary_vs_previous_continuous_q"] = {
            "expected_previous_continuous_q_test_logloss": float(
                df.previous_continuous_q_test_logloss.mean()
            ),
            "mean_aq_minus_previous_continuous_test_logloss": md,
            "paired_delta_95pct_t_interval": [clo, chi],
            "analytic_q_win_rate_vs_previous_continuous": float(
                df.aq_beats_previous_continuous.mean()
            ),
        }

    with open(OUT / "covtype_online_q_analytic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
