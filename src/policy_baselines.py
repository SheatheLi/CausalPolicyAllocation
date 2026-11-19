"""Budget-feasible native greedy policy comparator for a *fixed* TV ambiguity set.

Same training LCB, ESS mask, action costs, population and TV radius as the
cutting-plane MILP. This comparator is a heuristic; not intentionally weakened.
"""
from __future__ import annotations

import numpy as np

from .tv_robust_allocation import adversary


def robust_greedy(reward, population, radius, costs, budget, supported):
    value = np.asarray(reward, float)
    p0 = np.asarray(population,float)
    costs = np.asarray(costs,float)
    support = np.asarray(supported, bool)
    groups, arms = value.shape
    if support.shape != value.shape or costs.shape != (arms,):
        raise ValueError('invalid greedy comparator dimension')

    def score(assign):
        objective,_=adversary(p0,np.array([value[g,a] for g,a in enumerate(assign)]),radius)
        cost,_=adversary(p0,np.array([costs[a] for a in assign]),radius,maximize=True)
        return float(objective),float(cost)

    # Start with the least-cost supported action in each group. This is
    # conservative but not assumed feasible until the exact adversary checks.
    assign=tuple(int(np.argmin(np.where(support[g],costs,np.inf))) for g in range(groups))
    objective,worst_cost=score(assign)
    if worst_cost > budget + 1e-8:
        raise RuntimeError('greedy comparator has no feasible least-cost starting action')
    for _ in range(groups * arms):
        winner=None
        for g in range(groups):
            for a in range(arms):
                if not support[g,a] or assign[g] == a:
                    continue
                trial=list(assign);trial[g]=a;trial=tuple(trial)
                candidate_value,candidate_cost=score(trial)
                if candidate_cost > budget+1e-8 or candidate_value <= objective+1e-10:
                    continue
                if winner is None or candidate_value > winner[0]+1e-10:
                    winner=(candidate_value,candidate_cost,trial)
        if winner is None:
            break
        objective,worst_cost,assign=winner
    return {'actions': list(assign), 'objective':objective, 'worst_case_cost':worst_cost}
