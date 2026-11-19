"""Fail-closed, auditable off-policy decision workflow for logged randomized trials.

The real-data path never reads potential outcomes. CSVs represent *observed*
randomized assignments, not claimed external RCT provenance. The caller is
responsible for establishing true randomization, SUTVA and timestamp/feature
provenance. The output is a research decision, never deployment authorization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .policy import Trial, effective_sample_size, fit_dr
from .policy_uncertainty import paired_gate
from .tv_robust_allocation import optimize_crossfit_trial
from .policy_baselines import robust_greedy

FIELDS = ('unit_id', 'segment', 't', 'y', 'x0', 'x1', 'x2', 'x3', 'p0', 'p1', 'p2')
FORBIDDEN = {'potential', 'oracle', 'counterfactual', 'future_outcome', 'y0', 'y1', 'y2'}


class DataContractError(ValueError):
    """Data quality or identification prerequisite is not demonstrably met."""


def load_logged_trial(filename: str | Path, min_propensity: float = 0.01):
    """Require all 3 known logging probabilities and observed binary outcomes.

    Rows and unit_ids may not be repeated. No unobserved counterfactual fields.
    Input cannot establish whether assignment actually followed the logging
    probabilities; that provenance remains an external experimental prerequisite.
    """
    if not np.isfinite(min_propensity) or not 0 < min_propensity <= (1.0 / 3.0):
        raise DataContractError('min_propensity must be finite in (0, 1/3]')
    path = Path(filename)
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise DataContractError('duplicate CSV header')
        if set(columns) != set(FIELDS):
            raise DataContractError(f'CSV must contain exactly {FIELDS}; actual={columns}')
        ids: list[str] = []
        segments: list[int] = []
        actions: list[int] = []
        outcomes: list[int] = []
        features: list[list[float]] = []
        propensities: list[list[float]] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, 2):
            try:
                unit = str(row['unit_id']).strip()
                if not unit or unit in seen:
                    raise DataContractError(f'row {row_number}: empty/repeated unit_id')
                seen.add(unit)
                segment = int(row['segment'])
                action = int(row['t'])
                y = int(row['y'])
                if str(segment) != row['segment'].strip() or str(action) != row['t'].strip() or str(y) != row['y'].strip():
                    raise DataContractError(f'row {row_number}: discrete fields must be canonical integers')
                if segment < 0 or action not in (0, 1, 2) or y not in (0, 1):
                    raise DataContractError(f'row {row_number}: invalid segment/action/outcome')
                x = [float(row[f'x{i}']) for i in range(4)]
                prob = [float(row[f'p{i}']) for i in range(3)]
                if not np.isfinite(x).all() or not np.isfinite(prob).all():
                    raise DataContractError(f'row {row_number}: nonfinite features/probabilities')
                if min(prob) < min_propensity or max(prob) > 1 or abs(sum(prob)-1.) > 1e-8:
                    raise DataContractError(f'row {row_number}: invalid logging probability or positivity')
                ids.append(unit)
                segments.append(segment)
                actions.append(action)
                outcomes.append(y)
                features.append(x)
                propensities.append(prob)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                if isinstance(exc, DataContractError):
                    raise
                raise DataContractError(f'row {row_number}: malformed observed trial record: {exc}') from exc
    if len(ids) < 2:
        raise DataContractError('at least two trial units required')
    n = len(ids)
    return (
        Trial(
            x=np.array(features, dtype=float),
            t=np.array(actions, dtype=np.int64),
            y=np.array(outcomes, dtype=np.int64),
            propensity=np.array(propensities, dtype=float),
            potential=np.full((n, 3), np.nan),  # sentinel: real input NEVER contains truth
            segment=np.array(segments, dtype=np.int64),
        ),
        set(ids),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def evaluate_decision(
    train_path: str | Path,
    evaluation_path: str | Path,
    *,
    budget: float = 0.65,
    radius: float = 0.12,
    costs: tuple[float, float, float] = (0., 1., 2.),
    min_train_ess: float = 20.,
    min_evaluation_ess: float = 20.,
    uncertainty_draws: int = 1200,
    seed: int = 41,
) -> dict[str, Any]:
    """Train policy once, evaluate only afterwards on disjoint held-out units.

    FAIL_CLOSED is returned if evaluation support, shifted population assumption,
    robust feasibility, or the exploratory positive uplift gate does not hold.
    It must never be interpreted as formal policy safety certification.
    """
    if (not np.isfinite(min_train_ess) or not np.isfinite(min_evaluation_ess)
        or min_train_ess <= 0 or min_evaluation_ess <= 0 or uncertainty_draws < 50
        or len(costs)!=3 or not all(np.isfinite(v) and v>=0 for v in costs)
        or not np.isfinite(budget) or not 0 <= budget <= max(costs)
        or not np.isfinite(radius) or not 0 <= radius <= 1):
        raise DataContractError('invalid decision configuration')
    training, training_ids, train_sha = load_logged_trial(train_path)
    evaluation, evaluation_ids, eval_sha = load_logged_trial(evaluation_path)
    overlap = training_ids & evaluation_ids
    if overlap:
        raise DataContractError(f'training/evaluation unit overlap: {len(overlap)}')
    segments = np.unique(training.segment)
    if not np.array_equal(segments, np.unique(evaluation.segment)):
        raise DataContractError('all training segments must exist in held-out evaluation')
    # Need enough observations per group in both datasets for meaningful stderr.
    for name, data in [('train', training), ('evaluation', evaluation)]:
        for group in segments:
            if int((data.segment == group).sum()) < 4:
                raise DataContractError(f'{name} segment {group} lacks 4 records')
        if np.any(np.bincount(data.t, minlength=3) < 12):
            raise DataContractError(f'{name} has insufficient assignments for 3-arm nuisance fitting')

    train_dr = fit_dr(training, seed=seed)
    fit = optimize_crossfit_trial(
        training, train_dr, radius=radius, costs=costs,
        budget=budget, min_ess=min_train_ess,
    )
    cert = fit['certificate']
    if not cert.converged or cert.worst_budget > budget + 1e-7 or cert.upper_bound-cert.lower_bound > 1e-7:
        raise RuntimeError('robust solver returned no certified, budget-feasible policy')
    # This verification is independent of policy selection. The holdout is not
    # used to tune radius, budgets, nuisance parameters or the chosen actions.
    greedy = robust_greedy(
        fit['lcb_matrix'], fit['estimated_prevalence'], radius, costs, budget,
        fit['supported_actions'],
    )
    eval_dr = fit_dr(evaluation, seed=seed + 9)
    actions = np.array([fit['policy'][str(int(g))] for g in evaluation.segment], dtype=int)
    heldout_scores = eval_dr[np.arange(len(actions)), actions]
    baseline_scores = eval_dr[:, 0]
    greedy_policy={str(int(g)): int(a) for g,a in zip(segments,greedy['actions'])}
    greedy_actions=np.array([greedy_policy[str(int(g))] for g in evaluation.segment], dtype=int)
    greedy_scores=eval_dr[np.arange(len(greedy_actions)),greedy_actions]
    gate = paired_gate(heldout_scores, baseline_scores,
                       min_ess=min_evaluation_ess, seed=seed + 17,
                       draws=uncertainty_draws)
    eval_weights = np.array([(evaluation.segment == g).mean() for g in segments])
    train_weights = np.array(fit['estimated_prevalence'])
    shift_distance = float(np.abs(eval_weights-train_weights).sum()/2)
    support_records = []
    support_ok = True
    for group in segments:
        subset = evaluation.segment == group
        chosen = fit['policy'][str(int(group))]
        for arm in sorted({0, chosen}):
            ess = effective_sample_size(evaluation.t[subset], evaluation.propensity[subset], arm)
            ok = bool(ess >= min_evaluation_ess)
            support_ok &= ok
            support_records.append({'segment': int(group),'arm': int(arm),'ess': ess, 'supported':ok})
    reasons=[]
    if shift_distance > radius + 1e-8:
        reasons.append('HELDOUT_POPULATION_OUTSIDE_DECLARED_TV_SET')
    if not support_ok:
        reasons.append('HELDOUT_SUPPORT_BELOW_ESS_THRESHOLD')
    if not gate.accepted:
        reasons.append('HELDOUT_LIFT_NOT_POSITIVE_UNDER_EXPLORATORY_GATE')
    return {
        'status': 'PROVISIONAL_RESEARCH_CANDIDATE' if not reasons else 'REJECTED',
        'reasons': reasons,
        'policy':fit['policy'],
        'input': {'training_sha256':train_sha,'evaluation_sha256':eval_sha,
                  'training_units':len(training.y),'evaluation_units':len(evaluation.y),
                  'disjoint_unit_ids':True},
        'estimand':'held-out mean incremental outcome under frozen segment policy vs arm 0',
        'logging_design':'externally supplied randomized propensity; actual randomization not independently established',
        'budget': {'per_person':budget,'costs':list(costs),
                   'worst_case_cost':cert.worst_budget,'tv_radius':radius,
                   'shift_observed_tv':shift_distance,
                   'within_declared_ambiguity_set':shift_distance<=radius + 1e-8},
        'optimizer':{'converged':cert.converged,'cuts':cert.cuts,
                     'lower_bound':cert.lower_bound,'upper_bound':cert.upper_bound,
                     'objective_gap':cert.upper_bound-cert.lower_bound,
                     'worst_case_lcb':cert.true_worst_lcb,
                     'train_supported_actions':fit['supported_actions'],
                     'lcb_matrix':fit['lcb_matrix']},
        'evaluation_support': support_records,
        'heldout_lift':asdict(gate),
        'same_budget_greedy_comparator': {
            'policy': greedy_policy,
            'training_worst_case_lcb': greedy['objective'],
            'training_worst_case_cost': greedy['worst_case_cost'],
            'heldout_dr_estimated_value':float(np.mean(greedy_scores)),
            'heldout_dr_lift_vs_greedy':float(np.mean(heldout_scores-greedy_scores)),
            'identical_policy_to_optimizer':greedy_policy==fit['policy'],
            'note':'honest heuristic baseline with identical support, budget, and TV ambiguity set'
        },
        'decision_scope':'research triage only; not a frequentist policy-safety/deployment certificate',
        'limitations': [
            'No proof that logged treatment was actually randomized or propensity calibrated',
            'No proof of no interference or correct pre-treatment feature timing',
            'DR robustness needs valid assignment logging and outcome model/propensity conditions',
            'Exploratory bootstrap gate is not a simultaneous selective-inference safety certificate',
            'TV robustness applies to declared segment prevalence shift, not conditional outcome drift',
        ]
    }


def main() -> int:
    p = argparse.ArgumentParser(description='Fail-closed randomized policy evaluation')
    p.add_argument('--train', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--budget', type=float, default=.65)
    p.add_argument('--tv-radius', type=float, default=.12)
    p.add_argument('--min-train-ess', type=float, default=20.)
    p.add_argument('--min-evaluation-ess', type=float, default=20.)
    args = p.parse_args()
    try:
        report = evaluate_decision(
            args.train, args.evaluation, budget=args.budget,
            radius=args.tv_radius, min_train_ess=args.min_train_ess,
            min_evaluation_ess=args.min_evaluation_ess,
        )
    except (DataContractError, RuntimeError, ValueError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc)},ensure_ascii=False))
        return 2
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'], 'reasons':report['reasons'],
                      'report':str(destination)},ensure_ascii=False))
    return 0 if report['status']=='PROVISIONAL_RESEARCH_CANDIDATE' else 3


if __name__ == '__main__':
    raise SystemExit(main())
