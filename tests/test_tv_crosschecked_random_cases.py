"""Independent exhaustive policy enumeration against robust cutting-plane MILP.

Small-N oracle: enumerate all group actions; each action's exact worst
population shift is solved with a separate LP. Checks optimality, robustness,
upper-bound closure, not only a preselected favorable policy instance.
"""
from __future__ import annotations
import itertools
import numpy as np
from src.tv_robust_allocation import optimize, adversary


def test_randomized_exact_policy_oracle_24_seeds():
    counts={'checked':0,'changed_vs_nominal':0,'cuts_more_than_one':0}
    for seed in range(24):
        rg=np.random.default_rng(seed+20261014)
        groups=3;arms=3
        p=rg.dirichlet([2,3,4])
        rewards=np.c_[np.zeros(groups),rg.uniform(-.5,2.,size=(groups,arms-1))]
        costs=np.array([0.,.4,1.])
        budget=float(rg.uniform(.15,.75));radius=float(rg.uniform(.02,.33))
        solved=optimize(rewards,p,radius,costs,budget,max_iterations=50)
        assert solved.converged, (seed,solved)
        values=[]
        for actions in itertools.product(range(arms),repeat=groups):
            cost=np.array([costs[a] for a in actions])
            wc,_=adversary(p,cost,radius,maximize=True)
            if wc>budget+1e-7:continue
            value=np.array([rewards[j,a] for j,a in enumerate(actions)])
            worst,_=adversary(p,value,radius)
            values.append((worst,actions))
        assert values,seed
        oracle=max(x[0] for x in values)
        assert abs(oracle-solved.true_worst_lcb)<1e-6,(seed,oracle,solved)
        assert solved.worst_budget<=budget+1e-6
        assert abs(solved.upper_bound-solved.lower_bound)<1e-6
        counts['checked']+=1
        counts['cuts_more_than_one']+=(solved.cuts>1)
        nominal=optimize(rewards,p,0.,costs,budget)
        counts['changed_vs_nominal']+=(nominal.actions!=solved.actions)
    assert counts['checked']==24
