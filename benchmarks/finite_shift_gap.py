"""Demonstrate why a finite pairwise grid is not a full TV-ball guarantee."""
import json,numpy as np
from src.tv_robust_allocation import optimize,optimize_finite_scenarios,adversary
from src.robust_allocation import shift_scenarios

def run():
 p=np.array([.05,.05,.9]);rad=.12;cost=np.array([0.,1.]);budget=.95
 rewards=np.array([[0.,.001],[0.,.001],[0.,10.]])
 scenarios=shift_scenarios(p,rad)
 finite=optimize_finite_scenarios(rewards,p,scenarios,cost,budget)
 full=optimize(rewards,p,rad,cost,budget)
 def expose(actions):
  upper,_=adversary(p,np.array([cost[a] for a in actions]),rad,maximize=True)
  worst,_=adversary(p,np.array([rewards[j,a] for j,a in enumerate(actions)]),rad)
  return {'policy':list(actions),'continuous_tv_max_cost':upper,'budget_violation':max(0,upper-budget),'worst_payoff':worst}
 return {'budget':budget,'radius':rad,'scenario_count':len(scenarios),
         'finite_scenario':expose(finite),'tv_cutting_plane':{**expose(full.actions),'cuts':full.cuts,'converged':full.converged},
         'scope':'Constructed three-segment counterexample, not representative real-world rate'}
if __name__=='__main__':print(json.dumps(run(),indent=2))
