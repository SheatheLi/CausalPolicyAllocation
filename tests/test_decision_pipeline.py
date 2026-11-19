"""Hard fail-closed tests for real-data ingestion and robust decision release."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_synthetic_csv import write_example
from src.decision_pipeline import DataContractError, evaluate_decision, load_logged_trial
from src.tv_robust_allocation import optimize


@pytest.fixture(scope='module')
def logged(tmp_path_factory):
    directory = tmp_path_factory.mktemp('observed_trial')
    write_example(directory, n=1100)
    return directory


def mutate_csv(original, target, fn):
    with original.open(encoding='utf8', newline='') as f:
        rows=list(csv.DictReader(f))
        cols=list(rows[0])
    fn(rows)
    if rows and set(rows[0]) != set(cols):
        cols=list(rows[0])
    with target.open('w', encoding='utf8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return target


def test_real_data_honest_path_does_not_require_oracle(logged):
    result=evaluate_decision(logged/'train.csv',logged/'evaluation.csv',uncertainty_draws=150)
    assert result['status'] in {'PROVISIONAL_RESEARCH_CANDIDATE', 'REJECTED'}
    assert not result['input']['training_sha256']==result['input']['evaluation_sha256']
    assert all(record['ess']>=0 for record in result['evaluation_support'])
    assert result['budget']['worst_case_cost']<=result['budget']['per_person']+1e-7
    assert result['optimizer']['converged']
    assert 'oracle' not in result['estimand'].lower()
    assert np.isnan(load_logged_trial(logged/'train.csv')[0].potential).all()


def test_duplicate_ids_rejected(logged, tmp_path):
    corrupted=mutate_csv(logged/'evaluation.csv',tmp_path/'bad.csv',
                         lambda rows:rows[1].__setitem__('unit_id',rows[0]['unit_id']))
    with pytest.raises(DataContractError, match='repeated'):
        load_logged_trial(corrupted)


def test_train_and_evaluation_leakage_rejected(logged, tmp_path):
    corrupted=mutate_csv(logged/'evaluation.csv',tmp_path/'leak.csv',
                         lambda rows:rows[0].__setitem__('unit_id','train-0'))
    with pytest.raises(DataContractError, match='overlap'):
        evaluate_decision(logged/'train.csv',corrupted,uncertainty_draws=100)


def test_invalid_logging_propensity_fails_without_clipping(logged,tmp_path):
    bad=mutate_csv(logged/'train.csv',tmp_path/'bad.csv',
                   lambda rows:rows[0].__setitem__('p2','0.00001'))
    with pytest.raises(DataContractError,match='positivity'):
        load_logged_trial(bad)


def test_oracle_column_is_rejected(logged,tmp_path):
    p=mutate_csv(logged/'train.csv',tmp_path/'forbidden.csv',
                 lambda rows:[r.__setitem__('potential','0.999') for r in rows])
    with pytest.raises(DataContractError,match='exactly'):
        load_logged_trial(p)


def test_nonfinite_outcome_and_treatment_rejected(logged,tmp_path):
    p=mutate_csv(logged/'train.csv',tmp_path/'bad.csv',
                 lambda rows:rows[0].__setitem__('y','NaN'))
    with pytest.raises(DataContractError):
        load_logged_trial(p)
    p=mutate_csv(logged/'train.csv',tmp_path/'bad.csv',
                 lambda rows:rows[0].__setitem__('t','3'))
    with pytest.raises(DataContractError):
        load_logged_trial(p)


def test_evaluation_support_failure_is_no_go(logged):
    out=evaluate_decision(logged/'train.csv',logged/'evaluation.csv',
                          min_evaluation_ess=1e7,uncertainty_draws=100)
    assert out['status']=='REJECTED'
    assert 'HELDOUT_SUPPORT_BELOW_ESS_THRESHOLD' in out['reasons']


def test_population_shift_outside_tv_is_no_go(logged, tmp_path):
    def skew(rows):
        for r in rows[:850]:
            r['segment']='0'
    p=mutate_csv(logged/'evaluation.csv',tmp_path/'skew.csv',skew)
    out=evaluate_decision(logged/'train.csv',p,radius=.05,uncertainty_draws=100)
    assert out['status']=='REJECTED'
    assert 'HELDOUT_POPULATION_OUTSIDE_DECLARED_TV_SET' in out['reasons']


def test_optimizer_must_not_return_uncertified_unaffordable_policy():
    rng=np.random.default_rng(0)
    reward=rng.normal(size=(4,3))
    weights=rng.dirichlet(np.ones(4))
    with pytest.raises(RuntimeError,match='not certified'):
        optimize(reward,weights,.3,[0,1,2],.9,max_iterations=1)


@pytest.mark.parametrize('population',[
    [-.1,.6,.5], [float('nan'),.5,.5], [.1,.1,.1]
])
def test_optimizer_rejects_invalid_population(population):
    with pytest.raises(ValueError):
        optimize(np.zeros((3,2)),population,.1,[0,1],.5)


def test_cli_does_not_print_success_on_blocked_input(logged,tmp_path):
    report=tmp_path/'must_not_exist.json'
    proc=subprocess.run([sys.executable,'-m','src.decision_pipeline',
                         '--train',str(logged/'train.csv'),
                         '--evaluation',str(logged/'train.csv'),
                         '--output',str(report)],cwd=Path(__file__).resolve().parents[1],
                        capture_output=True,text=True,timeout=45)
    assert proc.returncode==2
    assert json.loads(proc.stdout)['status']=='BLOCKED'
    assert not report.exists()
