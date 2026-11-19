"""Cross-fitted DR -> overlap gate -> finite-mixture-shift robust MILP.

Research prototype for distribution shift in *segment prevalences*.  The scenarios
are declared ex ante, not learned from held-out potential outcomes.  LCB is a
heuristic unless simultaneous coverage assumptions are separately justified.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from src.policy import effective_sample_size

def shift_scenarios(weights:np.ndarray, shift:float=.12)->np.ndarray:
    w=np.asarray(weights,dtype=float)
    if np.any(w<=0) or not np.isclose(w.sum(),1):raise ValueError('positive normalized weights')
    result=[w.copy()]
    for i in range(len(w)):
        for j in range(len(w)):
            if i==j:continue
            a=min(shift,w[j]*.8)
            cand=w.copy();cand[i]+=a;cand[j]-=a
            result.append(cand)
    return np.stack(result)

def fit_robust_policy(trial,dr,scenarios,budget=.65,costs=(0.,1.,2.),min_ess=30,z=1.,robust=True):
    labels=np.unique(trial.segment)
    scenarios=np.asarray(scenarios,dtype=float)
    g=len(labels); arms=dr.shape[1]; nvar=g*arms
    if scenarios.ndim!=2 or scenarios.shape[1]!=g or np.any(scenarios<0) or not np.allclose(scenarios.sum(axis=1),1):
        raise ValueError('scenarios must be probability vectors ordered by segment labels')
    means=np.zeros((g,arms));penalties=np.zeros((g,arms));support=np.ones((g,arms),bool)
    for row,segment in enumerate(labels):
        mask=trial.segment==segment;n=mask.sum()
        for a in range(arms):
            mean=dr[mask,a].mean();se=dr[mask,a].std(ddof=1)/np.sqrt(n)
            means[row,a]=mean;penalties[row,a]=z*se
            support[row,a]=(a==0 or effective_sample_size(trial.t[mask],trial.propensity[mask],a)>=min_ess)
    # LCB may be negative; it remains a utility rather than a formal certified bound.
    values=(means-penalties).flatten(); costs=np.asarray(costs,float)
    if robust: weights_for_value=scenarios
    else:weights_for_value=scenarios[:1]
    # Each candidate action column (g,a) has its own group probability.
    values_by_scenario=np.stack([np.repeat(w,arms)*values for w in weights_for_value])
    all_scenario_values=np.stack([np.repeat(w,arms)*values for w in scenarios])
    cost_by_scenario=np.stack([np.repeat(w,arms)*np.tile(costs,g) for w in scenarios])
    # A continuous variable t in [-1,1] is the min scenario value.
    obj=np.r_[np.zeros(nvar), -1.]
    integrality=np.r_[np.ones(nvar),0.]
    lb=np.r_[np.zeros(nvar),-1000.];ub=np.r_[support.astype(float).flatten(),1000.]
    A=[];lo=[];hi=[]
    for j in range(g):
        row=np.zeros(nvar+1);row[j*arms:(j+1)*arms]=1
        A.append(row);lo.append(1);hi.append(1)
    for row in cost_by_scenario:
        A.append(np.r_[row,0]);lo.append(-np.inf);hi.append(budget)
    for row in values_by_scenario:
        A.append(np.r_[row,-1]);lo.append(0);hi.append(np.inf)
    result=milp(c=obj,integrality=integrality,bounds=Bounds(lb,ub),
                constraints=LinearConstraint(np.asarray(A),lo,hi),options={'time_limit':15})
    if not result.success:raise RuntimeError(str(result.message))
    x=result.x[:nvar].reshape(g,arms)
    policy={str(int(segment)):int(np.argmax(x[i])) for i,segment in enumerate(labels)}
    return {'policy':policy,'worst_case_lcb':float(result.x[-1]),
            'predicted_scenario_lcb':[float(row@result.x[:nvar]) for row in all_scenario_values],
            'scenario_cost':[float(row@result.x[:nvar]) for row in cost_by_scenario],
            'unsupported_count':int(np.size(support)-np.count_nonzero(support)),
            'robust':robust}

def synthetic_true_worst_value(holdout,policy,scenarios):
    labels=np.unique(holdout.segment)
    per_group=np.asarray([holdout.potential[holdout.segment==g,policy[str(int(g))]].mean() for g in labels])
    return float(np.min(np.asarray(scenarios)@per_group))

def evaluate(seed=7,n=2600):
    from src.policy import generate_trial,fit_dr
    train=generate_trial(seed,n)
    obs=np.asarray([(train.segment==v).mean() for v in sorted(set(train.segment))])
    scenarios=shift_scenarios(obs,.15)
    dr=fit_dr(train)
    nom=fit_robust_policy(train,dr,scenarios,robust=False)
    robust=fit_robust_policy(train,dr,scenarios,robust=True)
    holdout=generate_trial(seed+901,n)
    return {'seed':seed,'n_train':n,'n_independent_truth':n,'scenarios':len(scenarios),
            'nominal_policy':nom['policy'],'robust_policy':robust['policy'],
            'nominal_true_worst':synthetic_true_worst_value(holdout,nom['policy'],scenarios),
            'robust_true_worst':synthetic_true_worst_value(holdout,robust['policy'],scenarios),
            'nominal_worst_lcb_on_scenarios':float(min(nom['predicted_scenario_lcb'])),
            'robust_optimization_lcb':robust['worst_case_lcb'],
            'nominal_max_budget_cost':max(nom['scenario_cost']),
            'robust_max_budget_cost':max(robust['scenario_cost']),
            'unsupported_arms':robust['unsupported_count']}
if __name__=='__main__':
    import json
    print(json.dumps([evaluate(seed) for seed in (4,5,6,7,8)],indent=2))
