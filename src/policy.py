"""Cross-fitted doubly robust policy evaluation with overlap-aware MILP allocation.
Training and evaluation use disjoint samples; benchmark alone may inspect DGP truth.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

@dataclass
class Trial:
    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    propensity: np.ndarray
    potential: np.ndarray # simulation-only truth, prohibited in estimator
    segment: np.ndarray

def generate_trial(seed=7,n=5000)->Trial:
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,4))
    seg=np.digitize(x[:,0],[-.7,0,.7]);
    p=np.tile([.58,.36,.06],(n,1)).astype(float)
    # Sparse treatment-2 support for some groups; known randomized design.
    p[x[:,0]<-.7]=[.78,.20,.02]
    action=(rng.uniform(size=n)[:,None]>np.cumsum(p,axis=1)).sum(axis=1)
    logits=-1.2+.5*x[:,0]-.3*x[:,1]
    odds=lambda z:1/(1+np.exp(-z))
    potential=np.stack([odds(logits),odds(logits+.7-.35*x[:,0]),
                        odds(logits+.9+.5*x[:,1]-.1*x[:,0]**2)],axis=1)
    y=rng.binomial(1,potential[np.arange(n),action])
    return Trial(x,action,y,p,potential,seg)

def fit_dr(trial:Trial, folds=4, seed=11)->np.ndarray:
    """Honest nuisance OOF; randomization propensities are externally known."""
    n=len(trial.y);k=trial.propensity.shape[1]
    preds=np.zeros((n,k))
    for train,test in KFold(n_splits=folds,shuffle=True,random_state=seed).split(trial.x):
        for arm in range(k):
            mask=train[trial.t[train]==arm]
            if len(mask)<12 or len(np.unique(trial.y[mask]))<2:
                preds[test,arm]=np.mean(trial.y[mask]) if len(mask) else .5
            else:
                model=LogisticRegression(max_iter=300,C=1.)
                model.fit(trial.x[mask],trial.y[mask]);preds[test,arm]=model.predict_proba(trial.x[test])[:,1]
    # Clip propensities to avoid catastrophic weights while retaining ESS gate.
    p=np.clip(trial.propensity,.01,1)
    return preds + (trial.t[:,None]==np.arange(k)[None,:])*(trial.y[:,None]-preds)/p

def effective_sample_size(t:np.ndarray,p:np.ndarray,arm:int)->float:
    weights=(t==arm)/np.clip(p[:,arm],.01,1)
    return float(weights.sum()**2/(np.sum(weights*weights)+1e-12))

def allocate(trial:Trial, dr:np.ndarray,budget_per_person=.65, costs=(0.,1.,2.),
             min_ess=30,z=1.0)->dict:
    """One action per segment, no extrapolation if ESS is too low.
    Costs and budget are expected per person; LCB is a conservative risk score,
    not a formal finite-sample safety certificate.
    """
    labels=np.unique(trial.segment);n=len(trial.y);choices=[];utility=[];costs_var=[];ess_evidence=[]
    for group in labels:
        mask=trial.segment==group;size=np.sum(mask)
        for arm in range(dr.shape[1]):
            ess=effective_sample_size(trial.t[mask],trial.propensity[mask],arm)
            ess_evidence.append({'segment':int(group),'arm':arm,'ess':ess})
            observations=dr[mask,arm];estimate=float(np.mean(observations))
            stderr=float(np.std(observations,ddof=1)/np.sqrt(size))
            safe=arm==0 or ess>=min_ess
            choices.append((int(group),arm,safe))
            utility.append((estimate-z*stderr)*size/n if safe else -1e6)
            costs_var.append(costs[arm]*size/n)
    mat=np.zeros((len(labels),len(choices)))
    for i,(grp,_,_) in enumerate(choices): mat[np.where(labels==grp)[0][0],i]=1
    cons=[LinearConstraint(mat,np.ones(len(labels)),np.ones(len(labels))),
          LinearConstraint(np.array(costs_var)[None,:],-np.inf,budget_per_person)]
    ans=milp(c=-np.asarray(utility),integrality=np.ones(len(choices)),
             bounds=Bounds(np.zeros(len(choices)),np.ones(len(choices))),constraints=cons,
             options={'time_limit':10.})
    if not ans.success:raise RuntimeError(f'MILP failed: {ans.message}')
    chosen={str(grp):arm for (grp,arm,_),v in zip(choices,ans.x) if v>.5}
    return {'policy':chosen,'predicted_lcb_per_person':float(-ans.fun),
            'expected_cost_per_person':float(np.dot(costs_var,ans.x)),
            'ess':ess_evidence,'unsupported_arms':[(g,a) for g,a,ok in choices if not ok]}

def ground_truth_policy_value(trial:Trial, policy:dict)->float:
    return float(np.mean([trial.potential[i,policy[str(int(g))]] for i,g in enumerate(trial.segment)]))

if __name__=='__main__':
    import json
    d=generate_trial();out=allocate(d,fit_dr(d))
    print(json.dumps({**out,'synthetic_truth_policy_value':ground_truth_policy_value(d,out['policy'])},indent=2))
