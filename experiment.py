import json, math, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from scipy.optimize import minimize_scalar


def sigmoid(z):
    z=np.clip(z,-40,40)
    return 1/(1+np.exp(-z))

def metrics(y, margin):
    p=sigmoid(margin)
    return {'logloss': float(log_loss(y,p,labels=[0,1])), 'auc': float(roc_auc_score(y,p))}

def dmat(X,y): return xgb.DMatrix(X,label=y)

BASE_PARAMS=dict(
    objective='binary:logistic', eval_metric='logloss', max_depth=3,
    min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0,
    reg_lambda=1.0, reg_alpha=0.0, tree_method='hist', nthread=2,
    seed=0,
)

def add_round(booster, dtrain, eta, seed):
    p=BASE_PARAMS.copy(); p['eta']=float(eta); p['seed']=int(seed)
    return xgb.train(p,dtrain,num_boost_round=1,xgb_model=booster,verbose_eval=False)

def pred_margin(booster, d):
    if booster is None:
        # logistic base_score 0.5 => margin 0
        return np.zeros(d.num_row(),dtype=float)
    return booster.predict(d, output_margin=True)

def train_fixed(dtr,dv,dt,yv,yt,eta,n_rounds,seed):
    booster=None; hist=[]
    for t in range(n_rounds):
        booster=add_round(booster,dtr,eta,seed+t)
        if (t+1)%10==0 or t==0:
            hist.append((t+1,eta,metrics(yv,pred_margin(booster,dv))['logloss']))
    return booster,hist

def train_hyper(dtr,dv,dt,yv,yt,eta0,hyper_lr,n_rounds,seed,eta_min=0.005,eta_max=0.5):
    booster=None; eta=float(eta0); hist=[]
    prev_v=pred_margin(None,dv)
    for t in range(n_rounds):
        booster=add_round(booster,dtr,eta,seed+t)
        new_v=pred_margin(booster,dv)
        direction=(new_v-prev_v)/max(eta,1e-12)
        p=sigmoid(new_v)
        hg=float(np.mean((p-yv)*direction))
        eta_next=float(np.clip(eta-hyper_lr*hg,eta_min,eta_max))
        if (t+1)%10==0 or t<5:
            hist.append((t+1,eta,hg,metrics(yv,new_v)['logloss']))
        eta=eta_next; prev_v=new_v
    return booster,hist

def train_linesearch(dtr,dv,dt,yv,yt,n_rounds,seed,eta_max=0.5):
    booster=None; hist=[]
    cur_v=pred_margin(None,dv)
    for t in range(n_rounds):
        # Candidate full-strength tree determines structure/direction.
        tmp=add_round(booster,dtr,1.0,seed+t)
        tmp_v=pred_margin(tmp,dv)
        direction=tmp_v-cur_v
        def obj(eta): return log_loss(yv,sigmoid(cur_v+eta*direction),labels=[0,1])
        res=minimize_scalar(obj,bounds=(0.001,eta_max),method='bounded',options={'xatol':1e-4})
        eta=float(res.x)
        booster=add_round(booster,dtr,eta,seed+t)
        cur_v=pred_margin(booster,dv)
        if (t+1)%10==0 or t<5:
            hist.append((t+1,eta,metrics(yv,cur_v)['logloss']))
    return booster,hist

def one_seed(seed,n_rounds=80):
    ds=load_breast_cancer(); X=ds.data; y=ds.target
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.4,stratify=y,random_state=seed)
    Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,stratify=ytmp,random_state=seed+1000)
    dtr,dv,dte=dmat(Xtr,ytr),dmat(Xv,yv),dmat(Xte,yte)
    rows=[]; hists={}
    # fixed grid; choose best by validation final loss, report all and winner marker later
    fixed=[]
    for eta in [0.03,0.1,0.2]:
        b,h=train_fixed(dtr,dv,dte,yv,yte,eta,n_rounds,seed)
        vm=metrics(yv,pred_margin(b,dv)); tm=metrics(yte,pred_margin(b,dte))
        r={'seed':seed,'method':f'fixed_{eta}','eta0':eta,'hyper_lr':None,'val_logloss':vm['logloss'],'test_logloss':tm['logloss'],'test_auc':tm['auc']}
        rows.append(r); fixed.append(r); hists[r['method']]=h
    best_fixed=min(fixed,key=lambda r:r['val_logloss'])
    rows.append({**best_fixed,'method':'fixed_tuned'})
    # hypergradient grid; select by val
    hypers=[]
    for eta0 in [0.1]:
      for hlr in [0.05,0.2,1.0,5.0]:
        b,h=train_hyper(dtr,dv,dte,yv,yte,eta0,hlr,n_rounds,seed)
        vm=metrics(yv,pred_margin(b,dv)); tm=metrics(yte,pred_margin(b,dte))
        r={'seed':seed,'method':f'hyper_e{eta0}_h{hlr}','eta0':eta0,'hyper_lr':hlr,'val_logloss':vm['logloss'],'test_logloss':tm['logloss'],'test_auc':tm['auc'],'eta_final':h[-1][1] if h else None}
        rows.append(r); hypers.append(r); hists[r['method']]=h
    best_h=min(hypers,key=lambda r:r['val_logloss'])
    rows.append({**best_h,'method':'hyper_tuned'})
    # line-search on validation
    b,h=train_linesearch(dtr,dv,dte,yv,yte,n_rounds,seed)
    vm=metrics(yv,pred_margin(b,dv)); tm=metrics(yte,pred_margin(b,dte))
    rows.append({'seed':seed,'method':'val_linesearch','eta0':None,'hyper_lr':None,'val_logloss':vm['logloss'],'test_logloss':tm['logloss'],'test_auc':tm['auc'],'eta_final':h[-1][1]})
    hists['val_linesearch']=h
    return rows,hists

if __name__=='__main__':
    allrows=[]; allh={}; t0=time.time()
    for seed in range(5):
        rows,h=one_seed(seed)
        allrows.extend(rows); allh[str(seed)]=h
        print('seed',seed,'done',round(time.time()-t0,1),'s')
    df=pd.DataFrame(allrows)
    df.to_csv('/mnt/data/xgb_hypergrad_results.csv',index=False)
    with open('/mnt/data/xgb_hypergrad_histories.json','w') as f: json.dump(allh,f)
    sel=df[df.method.isin(['fixed_tuned','hyper_tuned','val_linesearch'])]
    summary=sel.groupby('method').agg(test_logloss_mean=('test_logloss','mean'),test_logloss_sd=('test_logloss','std'),test_auc_mean=('test_auc','mean'),val_logloss_mean=('val_logloss','mean')).reset_index()
    print(summary.to_string(index=False))
    summary.to_csv('/mnt/data/xgb_hypergrad_summary.csv',index=False)
