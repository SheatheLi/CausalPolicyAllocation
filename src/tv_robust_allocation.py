"""Continuous total-variation ambiguity set for budgeted policy allocation.

Cross-domain method: constraint generation (operations research) + causal DR
pseudo-outcome LCB. Worst-case population shift is solved by HiGHS LP, and
master policy uses HiGHS MILP. Not a frequentist coverage guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog,milp, Bounds,LinearConstraint

@dataclass
class Certificate:
    actions:tuple[int,...]
    objective:float
    true_worst_lcb:float
    worst_budget:float
    cuts:int
    converged:bool
    lower_bound:float
    upper_bound:float

def adversary(p0,values,radius, maximize=False):
    """Find extremal distribution with ||p-p0||_1 <= 2*radius."""
    p0=np.asarray(p0,float); values=np.asarray(values,float)
    g=len(p0)
    if not (g and values.shape==(g,) and np.all(p0>=0) and np.isclose(p0.sum(),1)
            and 0<=radius<=1):raise ValueError('invalid ambiguity set')
    # Variables: p_g and auxiliary absolute deviations u_g.
    A=[];b=[]
    for i in range(g):
        row=np.zeros(2*g);row[i]=1;row[g+i]=-1;A.append(row);b.append(p0[i])
        row=np.zeros(2*g);row[i]=-1;row[g+i]=-1;A.append(row);b.append(-p0[i])
    row=np.r_[np.zeros(g),np.ones(g)];A.append(row);b.append(2*radius)
    r=linprog(np.r_[(-1 if maximize else 1)*values,np.zeros(g)],
              A_ub=A,b_ub=b,A_eq=[np.r_[np.ones(g),np.zeros(g)]],b_eq=[1.],
              bounds=[(0,1)]*g+[(0,None)]*g,method='highs')
    if not r.success:raise RuntimeError(r.message)
    return float(r.x[:g]@values),np.asarray(r.x[:g])

def optimize(reward_lcb, p0, radius, costs, budget, supported=None,
             max_iterations=64, tolerance=1e-8):
    reward_lcb=np.asarray(reward_lcb,float);p0=np.asarray(p0,float)
    costs=np.asarray(costs,float)
    if reward_lcb.ndim != 2:
        raise ValueError('reward must be a 2D group by arm matrix')
    g,arms=reward_lcb.shape
    if (g == 0 or arms == 0 or costs.shape!=(arms,) or p0.shape!=(g,)
        or not np.isfinite(p0).all() or np.any(p0<=0) or not np.isclose(p0.sum(),1.)
        or not np.isfinite(costs).all() or np.any(costs<0)
        or not np.isfinite(reward_lcb).all() or not np.isfinite(budget)
        or budget < 0 or not np.isfinite(radius) or not 0 <= radius <= 1
        or max_iterations < 1 or tolerance <= 0):
        raise ValueError('invalid population, radius, cost, budget or optimization settings')
    support=np.ones_like(reward_lcb,dtype=bool) if supported is None else np.asarray(supported,bool)
    if support.shape!=reward_lcb.shape or not np.all(support.any(axis=1)):
        raise ValueError('support infeasible')
    n=g*arms
    # Adversarially generated weights are constraints, not additional data.
    gain_cuts=[p0.copy()]; cost_cuts=[p0.copy()]
    best=None
    for turn in range(max_iterations):
        rows=[];lower=[];upper=[]
        for j in range(g):
            a=np.zeros(n+1);a[j*arms:(j+1)*arms]=1
            rows.append(a);lower.append(1);upper.append(1)
        for p in gain_cuts:
            # t <= sum p_j reward_j(action_j)
            a=np.r_[-(p[:,None]*reward_lcb).reshape(-1),1.]
            rows.append(a);lower.append(-np.inf);upper.append(0.)
        for p in cost_cuts:
            a=np.r_[(p[:,None]*np.broadcast_to(costs,(g,arms))).reshape(-1),0.]
            rows.append(a);lower.append(-np.inf);upper.append(budget)
        res=milp(np.r_[np.zeros(n),-1.],integrality=np.r_[np.ones(n),0],
                bounds=Bounds(np.r_[np.zeros(n),-1e5],np.r_[support.astype(float).reshape(-1),1e5]),
                constraints=LinearConstraint(np.array(rows),lower,upper),
                options={'time_limit':20.})
        if not res.success:raise RuntimeError('master MILP '+str(res.message))
        assignment=tuple(int(i) for i in np.argmax(res.x[:n].reshape(g,arms),axis=1))
        selected=np.asarray([reward_lcb[j,a] for j,a in enumerate(assignment)])
        incurred=np.asarray([costs[a] for a in assignment])
        worst,pworst=adversary(p0,selected,radius)
        high_cost,pcost=adversary(p0,incurred,radius,maximize=True)
        upper_bound=float(res.x[-1]);lower_bound=worst
        best=Certificate(assignment,upper_bound,worst,high_cost,turn+1,
                         bool(high_cost<=budget+tolerance and upper_bound-worst<=tolerance),
                         lower_bound,upper_bound)
        if best.converged:return best
        if high_cost>budget+tolerance:cost_cuts.append(pcost)
        if upper_bound-worst>tolerance:gain_cuts.append(pworst)
        # If no useful violation exists, numerical convergence is attained.
        if high_cost<=budget+tolerance and upper_bound-worst<=tolerance*10:
            best.converged=True;return best
    # A policy that has not closed the objective gap or feasibility separation
    # is not an executable decision. Never silently turn it into a certificate.
    raise RuntimeError(
        f'robust optimization not certified after {max_iterations} iterations: '
        f'worst_budget={best.worst_budget:.9g}, budget={budget:.9g}, '
        f'objective_gap={best.upper_bound - best.lower_bound:.9g}'
    )


def optimize_crossfit_trial(trial, dr, radius=.12, costs=(0.,1.,2.),
                            budget=.65, min_ess=30, z=1.):
    """Connect OOF doubly-robust pseudo-outcomes to continuous-shift solver.

    Does NOT use `trial.potential`: the simulation oracle remains test-only.
    """
    from .policy import effective_sample_size
    labels=np.unique(trial.segment)
    dr=np.asarray(dr,float)
    if dr.shape!=(len(trial.y),len(costs)):raise ValueError('DR dimension mismatch')
    group_weights=np.array([(trial.segment==v).mean() for v in labels])
    reward=np.zeros((len(labels),len(costs)))
    support=np.ones_like(reward,dtype=bool)
    for j,segment in enumerate(labels):
        mask=trial.segment==segment
        n=int(mask.sum())
        for a in range(len(costs)):
            observations=dr[mask,a]
            reward[j,a]=float(observations.mean()-z*observations.std(ddof=1)/np.sqrt(n))
            support[j,a]=(a==0 or effective_sample_size(trial.t[mask],trial.propensity[mask],a)>=min_ess)
    certificate=optimize(reward,group_weights,radius,costs,budget,support)
    return {'certificate':certificate,'policy':{str(int(seg)):int(a) for seg,a in zip(labels,certificate.actions)},
            'estimated_prevalence':group_weights.tolist(),'supported_actions':support.tolist(),
            'lcb_matrix':reward.tolist()}


def optimize_finite_scenarios(reward_lcb,p0,scenarios,costs,budget,supported=None):
    """Comparable finite stress-set baseline with same objective and budget.

    Its guarantee is restricted to explicit scenarios; it does not certify
    any distribution outside them.
    """
    reward=np.asarray(reward_lcb,float);p0=np.asarray(p0,float)
    scenarios=np.asarray(scenarios,float);costs=np.asarray(costs,float)
    g,arms=reward.shape;n=g*arms
    if scenarios.ndim!=2 or scenarios.shape[1]!=g:raise ValueError('shape')
    support=np.ones_like(reward,dtype=bool) if supported is None else np.asarray(supported,bool)
    rows=[];lower=[];upper=[]
    for j in range(g):
        a=np.zeros(n+1);a[j*arms:(j+1)*arms]=1
        rows.append(a);lower.append(1);upper.append(1)
    for p in scenarios:
        rows.append(np.r_[-(p[:,None]*reward).reshape(-1),1.]);lower.append(-np.inf);upper.append(0.)
        rows.append(np.r_[(p[:,None]*np.broadcast_to(costs,(g,arms))).reshape(-1),0.]);lower.append(-np.inf);upper.append(budget)
    result=milp(np.r_[np.zeros(n),-1],integrality=np.r_[np.ones(n),0],
                bounds=Bounds(np.r_[np.zeros(n),-1e5],np.r_[support.reshape(-1).astype(float),1e5]),
                constraints=LinearConstraint(np.array(rows),lower,upper),options={'time_limit':10})
    if not result.success:raise RuntimeError(result.message)
    assignment=tuple(int(x) for x in np.argmax(result.x[:n].reshape(g,arms),axis=1))
    return assignment
