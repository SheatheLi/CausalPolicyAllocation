"""Controlled TV-shift budget correctness demo (not real business outcomes)."""
import json
import numpy as np
from src.tv_robust_allocation import optimize, adversary

def run():
    p=np.array([.55,.3,.15]); payoff=np.array([[.2,1.8],[.2,.7],[.2,.9]])
    costs=np.array([0.,1.]); budget=.65; actual_radius=.26
    nominal=optimize(payoff,p,0,costs,budget)
    robust=optimize(payoff,p,actual_radius,costs,budget)
    def summary(c):
        rewards=np.asarray([payoff[j,a] for j,a in enumerate(c.actions)])
        spend=np.asarray([costs[a] for a in c.actions])
        worst,_=adversary(p,rewards,actual_radius)
        max_spend,_=adversary(p,spend,actual_radius,maximize=True)
        return {'policy':c.actions,'worst_case_reward':worst,'worst_case_spend':max_spend,
                'worst_budget_violation':max(0,max_spend-budget),'cuts':c.cuts}
    return {'population':p.tolist(),'radius':actual_radius,'budget':budget,
            'nominal':summary(nominal),'continuous_tv_robust':summary(robust),
            'limitation':'Fixed synthetic payoffs; no causal identification or statistical coverage claim'}
if __name__=='__main__':print(json.dumps(run(),indent=2))
