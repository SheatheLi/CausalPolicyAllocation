import numpy as np
from src.policy import generate_trial,fit_dr
from src.robust_allocation import shift_scenarios,fit_robust_policy,synthetic_true_worst_value

def test_shift_probabilities_and_milp_constraints():
    trial=generate_trial(7,800); dr=fit_dr(trial)
    weights=np.array([(trial.segment==g).mean() for g in sorted(set(trial.segment))])
    scenarios=shift_scenarios(weights,.13)
    out=fit_robust_policy(trial,dr,scenarios,budget=.65,robust=True)
    assert len(out['policy'])==4
    assert max(out['scenario_cost'])<=.65000001
    holdout=generate_trial(99,800)
    assert 0<=synthetic_true_worst_value(holdout,out['policy'],scenarios)<=1

def test_insufficient_support_forces_safe_action():
    trial=generate_trial(7,650);dr=fit_dr(trial)
    weights=np.array([(trial.segment==g).mean() for g in sorted(set(trial.segment))])
    out=fit_robust_policy(trial,dr,shift_scenarios(weights),min_ess=1.e10)
    assert set(out['policy'].values())=={0}

def test_distribution_shift_changes_allocation_without_claiming_truth_win():
    trial=generate_trial(4,1200);dr=fit_dr(trial)
    weights=np.array([(trial.segment==g).mean() for g in sorted(set(trial.segment))]); scenarios=shift_scenarios(weights,.12)
    baseline=fit_robust_policy(trial,dr,scenarios,budget=.8,z=0,min_ess=4,robust=False)
    robust=fit_robust_policy(trial,dr,scenarios,budget=.8,z=0,min_ess=4,robust=True)
    assert baseline['policy']!=robust['policy']
    assert min(robust['predicted_scenario_lcb'])>=min(baseline['predicted_scenario_lcb'])-1e-8
