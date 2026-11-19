import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.policy import Trial, generate_trial
from src.decision_stress_audit import (assignment_diagnostic, frozen_dr_difference,
    propensity_perturbation_envelope, audit_frozen_policy)

@pytest.fixture
def observed():
    sample=generate_trial(seed=17,n=1400)
    # poison the simulation-only oracle: this diagnostic must never read it
    sample.potential[:]=float('nan')
    predictions=np.tile(np.array([.25,.34,.48]),(len(sample.y),1))
    policy={int(s):int(s%3) for s in np.unique(sample.segment)}
    return sample,predictions,policy

def test_randomized_counts_are_exploratory_not_certificate(observed):
    tr,_,_=observed
    diag=assignment_diagnostic(tr)
    assert diag.degrees_of_freedom==8
    assert len(diag.cells)==12
    assert 0 <= diag.p_value <= 1
    assert 'only' in diag.interpretation

def test_adversarial_assignment_diagnostic_detects_extreme_mismatch(observed):
    tr,_,_=observed
    tr.t[:]=0
    assert assignment_diagnostic(tr).p_value < 1e-20

def test_frozen_policy_no_oracle_access(observed):
    trial,m,policy=observed
    report=audit_frozen_policy(trial,m,policy,min_group_ess=10)
    assert report['status']=='EXPLORATORY_DIAGNOSTIC_ONLY'
    assert np.isfinite(report['perturbation']['nominal_mean_lift'])
    assert len(report['perturbation']['scenarios'])==9

def test_nominal_perturbation_is_nominal_dr(observed):
    tr,m,policy=observed
    baseline=frozen_dr_difference(tr,m,policy).mean()
    report=propensity_perturbation_envelope(tr,m,policy)
    assert abs(report['nominal_mean_lift']-baseline)<1e-12
    assert any(abs(x['mean_lift']-baseline)<1e-12 for x in report['scenarios'])

def test_same_policy_as_baseline_has_zero_advantage(observed):
    tr,m,_=observed
    policy={int(s):0 for s in np.unique(tr.segment)}
    assert np.max(abs(frozen_dr_difference(tr,m,policy)))==0
    report=propensity_perturbation_envelope(tr,m,policy)
    assert report['minimum_scenario_mean_lift']==0

@pytest.mark.parametrize('corruption', ['nonfinite_prediction','missing_segment','wrong_arm','zero_propensity','nan_propensity','invalid_baseline'])
def test_invalid_frozen_policy_rejected(observed, corruption):
    tr,m,policy=observed
    if corruption=='nonfinite_prediction':m[0,0]=np.nan
    elif corruption=='missing_segment':policy.pop(next(iter(policy)))
    elif corruption=='wrong_arm':policy[next(iter(policy))]=99
    elif corruption=='zero_propensity':tr.propensity[0,1]=0
    elif corruption=='nan_propensity':tr.propensity[0,0]=np.nan
    with pytest.raises(ValueError):
        frozen_dr_difference(tr,m,policy,baseline=99 if corruption=='invalid_baseline' else 0)

@pytest.mark.parametrize('shifts', [(),(0.,0.),(float('nan'),),(4.01,)])
def test_invalid_sensitivity_grid_rejected(observed,shifts):
    tr,m,p=observed
    with pytest.raises(ValueError):
        propensity_perturbation_envelope(tr,m,p,relative_log_odds_shifts=shifts)

def test_rare_arms_have_low_group_ess(observed):
    tr,m,policy=observed
    report=audit_frozen_policy(tr,m,policy,min_group_ess=1e8)
    assert not report['all_required_group_support_present']

def test_group_support_same_as_baseline_no_duplicate_arm(observed):
    tr,m,_=observed
    p={int(s):0 for s in np.unique(tr.segment)}
    report=audit_frozen_policy(tr,m,p)
    assert len(report['group_support']) == len(p)

def test_shifts_are_log_odds_alternatives_not_uniform_noise(observed):
    tr,m,p=observed
    env=propensity_perturbation_envelope(tr,m,p,relative_log_odds_shifts=(-.8,0.,.8))
    assert env['minimum_scenario_mean_lift'] <= env['nominal_mean_lift'] <= env['maximum_scenario_mean_lift']
    assert max(c['minimum_propensity'] for c in env['scenarios']) < 1

def test_assignment_inputs_need_consistent_probability_rows(observed):
    tr,_,_=observed
    tr.propensity[0]=[0.4,0.4,0.4]
    with pytest.raises(ValueError):assignment_diagnostic(tr)

def test_group_ids_are_not_oracle_outcomes(observed):
    tr,m,p=observed
    tr.potential=np.full((len(tr.y),3),1e99)
    old=frozen_dr_difference(tr,m,p)
    tr.potential=np.full((len(tr.y),3),-1e99)
    assert np.array_equal(old,frozen_dr_difference(tr,m,p))
