"""Continuous quadratic long-horizon Q controller for XGBoost eta.

Steelman of covtype_online_q.py: same Covertype split, same 20 starts, same
750-tree main-path budget, same 10-tree hypergradient proposal and 50-tree
Monte-Carlo target, but the action is continuous rather than one of five bins.
For fixed state s, each ridge model is exactly quadratic in
x = log(eta_next) - log(eta_hypergrad_proposal). A bootstrap ensemble supplies
uncertainty; bounded scalar search minimizes mean(Q)+0.5*std(Q) inside a local
trust region. Warmup/exploration samples x continuously. No counterfactual tree
fits and test data never affects actions.
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import covtype_online_q as base

OUT = Path("results"); OUT.mkdir(exist_ok=True)
SEED = base.SEED; N_ROUNDS = 750; N_STARTS = 20
ETA_MIN = base.ETA_MIN; ETA_MAX = base.ETA_MAX
BLOCK = 10; HORIZON_BLOCKS = 5; WARMUP_BLOCKS = 20
EPSILON_EXPLORE = 0.10; LOG_STEP = 0.25; TRUST_LOG_RADIUS = 0.35
MIN_Q_LABELS = 15; Q_ENSEMBLE = 32; Q_RIDGE_ALPHA = 1.0
Q_PESSIMISM_BETA = 0.5; EPS = 1e-12
BASELINE = OUT / "covtype_online_eta_blocked10_pairs.csv"
DISCRETE = OUT / "covtype_online_q_pairs.csv"


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--nthread", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--n-starts", type=int, default=N_STARTS)
    p.add_argument("--n-rounds", type=int, default=N_ROUNDS)
    p.add_argument("--block", type=int, default=BLOCK)
    p.add_argument("--horizon-blocks", type=int, default=HORIZON_BLOCKS)
    p.add_argument("--warmup-blocks", type=int, default=WARMUP_BLOCKS)
    p.add_argument("--trust-log-radius", type=float, default=TRUST_LOG_RADIUS)
    p.add_argument("--checkpoint-every", type=int, default=50)
    return p.parse_args()


def state_arr(progress, loss, delta, accel, eta, grad, margin, entropy, proposal):
    return np.asarray([progress, loss, delta, accel, math.log(eta), grad,
                       margin, entropy, math.log(proposal)], dtype=float)


def features(s, x):
    """Linear basis in [s,x,x²,sx,sx²] => quadratic Q in x for fixed s."""
    x=float(x); x2=x*x
    return np.concatenate([s, [x,x2], s*x, s*x2])


def bounds(proposal, radius):
    lp=math.log(proposal)
    return (max(-radius, math.log(ETA_MIN)-lp),
            min(radius, math.log(ETA_MAX)-lp))


def to_eta(proposal, x):
    return float(np.clip(proposal*math.exp(float(x)), ETA_MIN, ETA_MAX))


def to_x(proposal, eta):
    return float(math.log(float(eta))-math.log(float(proposal)))


def fit_ensemble(X, y, rng):
    out=[]; n=len(y)
    for _ in range(Q_ENSEMBLE):
        idx=rng.integers(0,n,size=n)
        m=make_pipeline(StandardScaler(), Ridge(alpha=Q_RIDGE_ALPHA))
        m.fit(X[idx], y[idx]); out.append(m)
    return out


def q_stats(models, s, x):
    z=features(s,x)[None,:]
    pred=np.asarray([m.predict(z)[0] for m in models], float)
    mu=float(pred.mean()); sd=float(pred.std(ddof=1)) if len(pred)>1 else 0.0
    return mu, sd, mu+Q_PESSIMISM_BETA*sd


def choose_continuous(models, s, proposal, radius):
    lo,hi=bounds(proposal,radius)
    if hi-lo < 1e-12:
        x=lo; mu,sd,score=q_stats(models,s,x)
        return to_eta(proposal,x), {"x":x,"pred_mean_delta":mu,"pred_std":sd,
            "pessimistic_score":score,"bounds":[lo,hi],"optimizer_success":True}
    objective=lambda x:q_stats(models,s,float(x))[2]
    opt=minimize_scalar(objective,bounds=(lo,hi),method="bounded",
                        options={"xatol":1e-4,"maxiter":80})
    probes=[lo,hi,float(np.clip(0.0,lo,hi))]
    if np.isfinite(opt.x): probes.append(float(np.clip(opt.x,lo,hi)))
    scored=[(x,*q_stats(models,s,x)) for x in probes]
    x,mu,sd,score=min(scored,key=lambda z:z[3])
    eta=to_eta(proposal,x); x=to_x(proposal,eta); mu,sd,score=q_stats(models,s,x)
    return eta,{"x":x,"pred_mean_delta":mu,"pred_std":sd,
        "pessimistic_score":score,"bounds":[lo,hi],
        "optimizer_success":bool(opt.success),"optimizer_nfev":int(opt.nfev)}


def run(dtrain,dctrl,dtest,yctrl,ytest,eta0,nthread,n_rounds,block,horizon,
        warmup,radius,checkpoint,label,run_seed):
    if n_rounds % block: raise ValueError("n_rounds must be divisible by block")
    n_blocks=n_rounds//block; rng=np.random.default_rng(run_seed)
    booster=xgb.train(base.params(nthread,eta0),dtrain,num_boost_round=0)
    eta=float(eta0); grad_sq=0.0
    margin_before=booster.predict(dctrl,output_margin=True)
    prev_loss=base.metrics(yctrl,base.softmax(margin_before))["logloss"]; prev_delta=0.0
    Xq=[]; yq=[]; pending=[]; history=[]; wall=[]; start=time.perf_counter()

    for b in range(1,n_blocks+1):
        before=margin_before
        booster=xgb.train(base.params(nthread,eta),dtrain,num_boost_round=block,
                          xgb_model=booster,verbose_eval=False)
        after=booster.predict(dctrl,output_margin=True); p=base.softmax(after)
        loss=base.metrics(yctrl,p)["logloss"]; delta=loss-prev_loss; accel=delta-prev_delta
        dm=after-before; resid=p.copy(); resid[np.arange(len(yctrl)),yctrl]-=1.0
        g=float(np.mean(np.sum(resid*dm,axis=1))); grad_sq += g*g
        step=LOG_STEP*g/math.sqrt(grad_sq+EPS)
        proposal=math.exp(float(np.clip(math.log(eta)-step,
                         math.log(ETA_MIN),math.log(ETA_MAX))))

        keep=[]
        for rec in pending:
            if rec["future_block"]==b:
                Xq.append(rec["features"]); yq.append(float(loss-rec["base_loss"]))
            else: keep.append(rec)
        pending=keep
        s=state_arr(b/n_blocks,loss,delta,accel,eta,g,float(np.mean(np.abs(after))),
                    base.entropy_from_probs(p),proposal)
        chosen=eta; x=0.0; source="terminal"; detail={}
        if b<n_blocks:
            lo,hi=bounds(proposal,radius); enough=len(yq)>=MIN_Q_LABELS
            explore=b<=warmup or rng.random()<EPSILON_EXPLORE
            if explore or not enough:
                x=float(rng.uniform(lo,hi)) if hi>lo else lo
                chosen=to_eta(proposal,x); x=to_x(proposal,chosen)
                source="warmup_continuous" if b<=warmup else "epsilon_continuous"
                detail={"bounds":[lo,hi]}
            else:
                models=fit_ensemble(np.stack(Xq),np.asarray(yq,float),rng)
                chosen,detail=choose_continuous(models,s,proposal,radius)
                x=float(detail["x"]); source="q_continuous"
            pending.append({"future_block":b+horizon,"base_loss":loss,
                            "features":features(s,x)})
            pending=[r for r in pending if r["future_block"]<=n_blocks]
        history.append({"block":b,"trees":b*block,"eta_used":eta,
            "hypergrad_proposal_eta":proposal,"chosen_next_eta":chosen,
            "chosen_log_offset_from_proposal":x,"action_source":source,
            "g_log_eta":g,"normalized_log_step":step,"control_logloss":loss,
            "recent_control_delta":delta,"recent_control_accel":accel,
            "n_q_labels":len(yq),"q_detail":detail})
        wall.append(time.perf_counter()-start)
        prev_delta=delta; prev_loss=loss; margin_before=after; eta=chosen

    ctrl=base.metrics(yctrl,booster.predict(dctrl)); test=base.metrics(ytest,booster.predict(dtest))
    checks=list(range(checkpoint,n_rounds+1,checkpoint))
    if not checks or checks[-1]!=n_rounds: checks.append(n_rounds)
    curve=[]
    for k in checks:
        pc=booster.predict(dctrl,iteration_range=(0,k)); pt=booster.predict(dtest,iteration_range=(0,k))
        cm=base.metrics(yctrl,pc); tm=base.metrics(ytest,pt); bi=int(math.ceil(k/block))
        curve.append({"label":label,"method":"continuous_quadratic_q","eta0":eta0,
            "trees":k,"eta":history[bi-1]["eta_used"],"control_logloss":cm["logloss"],
            "test_logloss":tm["logloss"],"test_accuracy":tm["accuracy"],
            "training_wall_seconds":wall[bi-1]})
    return {"label":label,"eta0":eta0,"cq_final_eta":history[-1]["eta_used"],
        "cq_control_logloss":ctrl["logloss"],"cq_test_logloss":test["logloss"],
        "cq_test_accuracy":test["accuracy"],"cq_final_wall_seconds":wall[-1],
        "cq_labels_matured":len(yq),
        "cq_greedy_blocks":sum(h["action_source"]=="q_continuous" for h in history),
        "cq_random_blocks":sum(h["action_source"] in {"warmup_continuous","epsilon_continuous"} for h in history),
        "cq_boundary_eta_fraction":float(np.mean([(h["eta_used"]<=ETA_MIN+1e-9) or
                                      (h["eta_used"]>=ETA_MAX-1e-9) for h in history]))}, history, curve


def ci(v):
    v=np.asarray(v,float); m=float(v.mean())
    if len(v)<2:return m,float("nan"),float("nan")
    lo,hi=stats.t.interval(.95,df=len(v)-1,loc=m,scale=float(stats.sem(v)))
    return m,float(lo),float(hi)


def main():
    a=args_parse()
    if not BASELINE.exists(): raise FileNotFoundError(f"Missing {BASELINE}")
    if a.n_rounds!=N_ROUNDS: raise ValueError("Use 750 rounds to reuse committed baseline")
    b=pd.read_csv(BASELINE); b=b[b.label.str.startswith("prior_draw_")].sort_values("label").head(a.n_starts)
    if len(b)!=a.n_starts: raise ValueError("Not enough baseline starts")
    if DISCRETE.exists():
        d=pd.read_csv(DISCRETE)[["label","q_test_logloss"]]
        b=b.merge(d,on="label",how="left")
    print(f"nthread={a.nthread} n_starts={a.n_starts} n_rounds={a.n_rounds} block={a.block} "
          f"horizon={a.horizon_blocks} warmup={a.warmup_blocks} trust={a.trust_log_radius}",flush=True)
    dtrain,dctrl,dtest,yctrl,ytest=base.load_data(); t0=time.time()
    rows=[]; histories={}; curves=[]
    for i,r in b.reset_index(drop=True).iterrows():
        label=str(r.label); eta0=float(r.eta0)
        print(f"\n[{label}] eta0={eta0:.6f} HG={r.adaptive_test_logloss:.6f}; continuous-Q...",flush=True)
        row,h,c=run(dtrain,dctrl,dtest,yctrl,ytest,eta0,a.nthread,a.n_rounds,a.block,
                    a.horizon_blocks,a.warmup_blocks,a.trust_log_radius,a.checkpoint_every,
                    label,SEED+2000+i)
        row["hypergrad_test_logloss"]=float(r.adaptive_test_logloss)
        row["cq_minus_hypergrad_test_logloss"]=row["cq_test_logloss"]-row["hypergrad_test_logloss"]
        row["cq_beats_hypergrad"]=row["cq_minus_hypergrad_test_logloss"]<0
        if "q_test_logloss" in r and pd.notna(r.q_test_logloss):
            row["discrete_q_test_logloss"]=float(r.q_test_logloss)
            row["cq_minus_discrete_q_test_logloss"]=row["cq_test_logloss"]-row["discrete_q_test_logloss"]
            row["cq_beats_discrete_q"]=row["cq_minus_discrete_q_test_logloss"]<0
        rows.append(row); histories[label]=h; curves.extend(c)
        print(f"[{label}] CQ={row['cq_test_logloss']:.6f} eta_final={row['cq_final_eta']:.5f} "
              f"delta_vs_HG={row['cq_minus_hypergrad_test_logloss']:+.6f} wall={row['cq_final_wall_seconds']:.1f}s",flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/"covtype_online_q_continuous_pairs.csv",index=False)
    pd.DataFrame(curves).to_csv(OUT/"covtype_online_q_continuous_convergence.csv",index=False)
    json.dump(histories,open(OUT/"covtype_online_q_continuous_history.json","w"),indent=2)
    md,lo,hi=ci(df.cq_minus_hypergrad_test_logloss)
    summary={"dataset":"UCI Covertype","sample_size":base.N_SAMPLE,"n_rounds":a.n_rounds,
      "n_prior_draws":a.n_starts,"baseline":"10-tree blocked hypergradient, committed results",
      "continuous_q_controller":{"block_trees":a.block,"horizon_blocks":a.horizon_blocks,
        "horizon_trees":a.block*a.horizon_blocks,"warmup_blocks":a.warmup_blocks,
        "epsilon_explore":EPSILON_EXPLORE,"action":"continuous log-eta offset from HG proposal",
        "trust_log_radius":a.trust_log_radius,
        "model":"bootstrap ridge ensemble, quadratic in continuous action",
        "ensemble_size":Q_ENSEMBLE,"ridge_alpha":Q_RIDGE_ALPHA,"pessimism_beta":Q_PESSIMISM_BETA,
        "optimizer":"bounded scalar minimization of mean+beta*std","counterfactual_tree_fits":0,
        "target":"realized 50-tree future control-logloss change"},
      "expected_hypergrad_test_logloss":float(df.hypergrad_test_logloss.mean()),
      "expected_continuous_q_test_logloss":float(df.cq_test_logloss.mean()),
      "mean_cq_minus_hypergrad_test_logloss":md,"mean_cq_improvement_over_hypergrad":-md,
      "paired_delta_95pct_t_interval_vs_hypergrad":[lo,hi],
      "continuous_q_win_rate_vs_hypergrad":float(df.cq_beats_hypergrad.mean()),
      "mean_continuous_q_wall_seconds":float(df.cq_final_wall_seconds.mean()),
      "mean_boundary_eta_fraction":float(df.cq_boundary_eta_fraction.mean()),
      "predeclared_primary_gate":{"definition":"mean improvement over HG >=0.002 and win rate >=60%",
        "passes":bool(md<=-.002 and df.cq_beats_hypergrad.mean()>=.60)},
      "main_path_tree_budget":{"hypergradient":a.n_rounds,"continuous_q":a.n_rounds},
      "test_used_for_adaptation":False,"elapsed_seconds":time.time()-t0}
    if "discrete_q_test_logloss" in df:
        md2,lo2,hi2=ci(df.cq_minus_discrete_q_test_logloss)
        summary["secondary_vs_discrete_q"]={"expected_discrete_q_test_logloss":float(df.discrete_q_test_logloss.mean()),
          "mean_cq_minus_discrete_q_test_logloss":md2,"mean_cq_improvement_over_discrete_q":-md2,
          "paired_delta_95pct_t_interval":[lo2,hi2],
          "continuous_q_win_rate_vs_discrete_q":float(df.cq_beats_discrete_q.mean())}
    json.dump(summary,open(OUT/"covtype_online_q_continuous_summary.json","w"),indent=2)
    print("\nSUMMARY\n"+json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
