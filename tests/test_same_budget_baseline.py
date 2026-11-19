"""Greedy baseline is a correct, independently adversarially-validated option."""
import itertools
import numpy as np
from src.policy_baselines import robust_greedy
from src.tv_robust_allocation import adversary, optimize


def test_greedy_feasible_and_same_constraints_as_solver():
    rng=np.random.default_rng(18)
    reward=rng.normal(size=(4,3))
    p=rng.dirichlet([3.,3.,3.,3.])
    support=np.ones((4,3),bool);support[0,2]=False
    costs=[0.,.4,1.];budget=.65;radius=.21
    gre=robust_greedy(reward,p,radius,costs,budget,support)
    chosen=gre['actions']
    assert all(support[g,a] for g,a in enumerate(chosen))
    actual_value,_=adversary(p,[reward[g,a] for g,a in enumerate(chosen)],radius)
    actual_cost,_=adversary(p,[costs[a] for a in chosen],radius,maximize=True)
    assert actual_cost<=budget+1e-8
    assert abs(actual_cost-gre['worst_case_cost'])<1e-8
    assert abs(actual_value-gre['objective'])<1e-8
    exact=optimize(reward,p,radius,costs,budget,support)
    assert exact.true_worst_lcb>=gre['objective']-1e-7


def test_greedy_rejects_infeasible_cheapest_start():
    import pytest
    with pytest.raises(RuntimeError,match='no feasible'):
        robust_greedy([[1.,2.]], [1.], .2, [1.,2.], .5, [[1,1]])
