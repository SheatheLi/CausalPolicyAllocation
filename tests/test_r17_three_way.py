"""Independent finite-search oracle and budget/fencing validation."""
import numpy as np
import pytest

from benchmarks.r17_three_way import exhaustive_oracle, evaluate_assignment
from src.policy_baselines import robust_greedy
from src.tv_robust_allocation import optimize


def test_exact_oracle_matches_milp_with_cost_and_support():
    reward = np.array([[.12,.8,.9],[.11,.6,.7],[.09,.7,.3],[.08,.45,.5]])
    weights = np.array([.35,.20,.25,.20])
    support = np.array([[1,1,0],[1,1,1],[1,0,1],[1,1,1]], dtype=bool)
    costs = (0.,1.,2.)
    radius = .12
    budget = .6
    oracle = exhaustive_oracle(reward,weights,radius,costs,budget,support)
    cert = optimize(reward,weights,radius,costs,budget,support)
    assert cert.converged
    assert cert.true_worst_lcb == pytest.approx(oracle['objective'],abs=1e-7)
    assert list(cert.actions) in oracle['optimal_assignments']
    assert cert.worst_budget <= budget + 1e-8


def test_oracle_rejects_infeasible_support():
    reward = np.ones((2,2))
    population = np.array([.5,.5])
    support = np.array([[False,True],[False,True]])
    with pytest.raises(RuntimeError, match='no feasible policy'):
        exhaustive_oracle(reward,population,0.,(0.,1.),.5,support)


def test_robust_greedy_is_valid_not_weakened_baseline():
    reward = np.array([[.2,.4],[.25,.5],[.15,.7]])
    weights = np.array([.3,.4,.3]); support = np.ones((3,2),bool)
    result = robust_greedy(reward,weights,.1,(0.,1.),.7,support)
    objective,cost = evaluate_assignment(reward,weights,.1,(0.,1.),result['actions'])
    assert objective == pytest.approx(result['objective'])
    assert cost <= .7 + 1e-8
