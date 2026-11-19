from src.tv_robust_allocation import adversary,optimize
import numpy as np

def test_adversary_exact_and_continuous():
    p=np.array([.6,.3,.1]);v=np.array([.1,.5,1.])
    val,q=adversary(p,v,.2)
    assert abs(q.sum()-1)<1e-9
    assert .5*np.abs(q-p).sum()<=.20000001
    assert val < p@v
    # A reference exhaustive coarse grid is weaker (no exact extreme point).
    coarse=min(w@v for x in range(0,101,5) for y in range(0,101-x,5)
             if (w:=np.array([x,y,100-x-y])/100.) is not None and .5*np.abs(w-p).sum()<=.2+1e-9)
    assert val <= coarse+1e-8

def test_cutting_plane_optimal_vs_exhaustive_policies():
    reward=np.array([[.2,1.8],[.2,.7],[.2,.9]],float)
    p=np.array([.55,.3,.15]);costs=[0.,1.]
    r=optimize(reward,p,.26,costs,.65)
    assert r.converged and r.cuts<=64
    assert r.worst_budget<=.6500001
    import itertools
    values=[]
    for actions in itertools.product(range(2),repeat=3):
        best,_=adversary(p,np.array([reward[j,a] for j,a in enumerate(actions)]),.26)
        cost,_=adversary(p,np.array([costs[a] for a in actions]),.26,True)
        if cost <= .65+1e-7:values.append(best)
    assert abs(max(values)-r.true_worst_lcb)<1e-7
    assert r.upper_bound-r.lower_bound<=1e-7


def test_strict_support_exclusion():
    reward=np.ones((3,2));support=np.array([[1,0],[1,1],[1,1]],bool)
    r=optimize(reward,[.2,.3,.5],.1,[0,1],.8,support)
    assert r.actions[0]==0

def test_tv_solver_wired_to_crossfit_without_oracle_access():
    from src.policy import generate_trial,fit_dr
    from src.tv_robust_allocation import optimize_crossfit_trial
    data=generate_trial(9,800)
    dr=fit_dr(data)
    # Poison simulation-only oracle to ensure this path never reads it.
    data.potential.fill(float('nan'))
    result=optimize_crossfit_trial(data,dr,radius=.10,min_ess=15)
    assert result['certificate'].converged
    assert result['certificate'].worst_budget<=.6500001
    assert len(result['policy'])==4

def test_discrete_stress_set_misses_off_grid_adversary():
    from src.tv_robust_allocation import optimize_finite_scenarios
    from src.robust_allocation import shift_scenarios
    p=np.array([.05,.05,.9]);radius=.12;costs=np.array([0.,1.]);budget=.95
    reward=np.array([[0.,.001],[0.,.001],[0.,10.]])
    sparse=shift_scenarios(p,radius)
    finite=optimize_finite_scenarios(reward,p,sparse,costs,budget)
    exact=optimize(reward,p,radius,costs,budget)
    finite_cost,_=adversary(p,np.array([costs[a] for a in finite]),radius,True)
    assert finite[2]==1
    assert finite_cost>budget
    assert exact.converged and exact.worst_budget<=budget+1e-8
