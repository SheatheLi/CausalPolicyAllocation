import numpy as np
from src.policy import *

def test_randomized_propensities():
    d=generate_trial(n=1500);assert np.allclose(d.propensity.sum(axis=1),1)
    assert np.min(d.t)==0 and np.max(d.t)==2

def test_dr_and_budget_and_support():
    d=generate_trial(n=1800);dr=fit_dr(d);assert dr.shape==(1800,3)
    ans=allocate(d,dr,budget_per_person=.4,min_ess=40)
    assert ans['expected_cost_per_person']<=.400001
    assert len(ans['policy'])==4
    for grp,arm in ans['unsupported_arms']:
        assert ans['policy'][str(grp)]!=arm

def test_estimator_does_not_use_hidden_truth():
    d=generate_trial(n=1200);e=fit_dr(d)
    d.potential[:]=0
    assert np.allclose(e,fit_dr(d))
