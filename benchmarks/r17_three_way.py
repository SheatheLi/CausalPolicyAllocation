"""Round 17 preregistered finite grid: 12 independent seeds × four parameter cells.

All candidates use identical DR training rewards, costs, support and TV uncertainty.
Exhaustive oracle is a separate assignment enumerator, not another MILP.
Synthetic evaluation potential-outcome oracle is benchmark-only and NEVER exposed
while training/choosing policy. No external business validation is claimed.
"""
from __future__ import annotations

import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn

from src.policy import fit_dr, generate_trial, ground_truth_policy_value
from src.policy_baselines import robust_greedy
from src.tv_robust_allocation import (
    adversary,
    optimize_crossfit_trial,
    optimize,
)

SEEDS = (7, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59)
BUDGETS = (.45, .65)
TV_RADII = (.05, .15)
TRAIN_N, EVALUATION_N = 1600, 3000
MIN_ESS = 20
COSTS = (0., 1., 2.)


def evaluate_assignment(reward, population, radius, costs, assignment):
    """Evaluate assignment with independent per-coordinate accumulation.

    This function reuses the project's LP adversary for a common ambiguity
    definition; only policy search is independent of the MILP implementation.
    """
    a = tuple(int(x) for x in assignment)
    gains = np.array([float(reward[i][action]) for i, action in enumerate(a)])
    spending = np.array([float(costs[action]) for action in a])
    robust_reward, _ = adversary(population, gains, radius)
    worst_cost, _ = adversary(population, spending, radius, maximize=True)
    return float(robust_reward), float(worst_cost)


def exhaustive_oracle(reward, population, radius, costs, budget, support):
    """Enumerate every supported assignment; certify finite-grid optimum."""
    a = np.asarray(reward)
    s = np.asarray(support, bool)
    assert a.ndim == 2 and a.shape == s.shape
    options = [[j for j in range(a.shape[1]) if s[i, j]] for i in range(a.shape[0])]
    best_value = float('-inf')
    best_assignments = []
    valid = 0
    total = 0
    for assignment in itertools.product(*options):
        total += 1
        value, spent = evaluate_assignment(a, population, radius, costs, assignment)
        if spent > budget + 1e-8:
            continue
        valid += 1
        if value > best_value + 1e-8:
            best_value = value
            best_assignments = [tuple(assignment)]
        elif abs(value - best_value) <= 1e-8:
            best_assignments.append(tuple(assignment))
    if not best_assignments:
        raise RuntimeError('exhaustive oracle: no feasible policy')
    return {
        'objective': float(best_value), 'optimal_assignments': [list(v) for v in best_assignments],
        'supported_assignments': total, 'feasible_assignments': valid,
    }


def benchmark():
    rows = []
    for seed in SEEDS:
        train = generate_trial(seed=seed, n=TRAIN_N)
        holdout = generate_trial(seed=seed + 10000, n=EVALUATION_N)
        dr = fit_dr(train)
        for budget, radius in itertools.product(BUDGETS, TV_RADII):
            fitted = optimize_crossfit_trial(
                train, dr, budget=budget, radius=radius,
                costs=COSTS, min_ess=MIN_ESS, z=1.,
            )
            reward = fitted['lcb_matrix']
            population = fitted['estimated_prevalence']
            support = fitted['supported_actions']
            # Time only solver operations after shared DR, LCB, ESS and
            # population inputs have been prepared. Avoid unequal preprocessing.
            t0 = time.perf_counter()
            cert = optimize(reward,population,radius,COSTS,budget,support)
            milp_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            greedy = robust_greedy(reward, population, radius, COSTS, budget, support)
            greedy_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            oracle = exhaustive_oracle(reward, population, radius, COSTS, budget, support)
            exhaustive_seconds = time.perf_counter() - t0
            selected = tuple(cert.actions)
            greed = tuple(greedy['actions'])
            milp_value, milp_cost = evaluate_assignment(reward, population, radius, COSTS, selected)
            greedy_value, greedy_cost = evaluate_assignment(reward, population, radius, COSTS, greed)
            # Independent oracle compares objectives; equivalent policies can tie.
            if not cert.converged or abs(milp_value - oracle['objective']) > 1e-6:
                raise AssertionError(f'master not globally optimal for grid cell seed={seed},budget={budget},radius={radius}')
            if abs(cert.true_worst_lcb - milp_value) > 1e-6:
                raise AssertionError('reported MILP objective disagrees with adversarial LP')
            if milp_cost > budget + 1e-7 or greedy_cost > budget + 1e-7:
                raise AssertionError('budget infeasible')
            if greedy_value > milp_value + 1e-7:
                raise AssertionError('greedy beats purported global optimum')
            milp_policy = {str(i): a for i, a in enumerate(selected)}
            greedy_policy = {str(i): a for i, a in enumerate(greed)}
            true_milp = ground_truth_policy_value(holdout, milp_policy)
            true_greedy = ground_truth_policy_value(holdout, greedy_policy)
            rows.append({
                'seed': seed, 'budget': budget, 'tv_radius': radius,
                'milp_assignment': list(selected), 'greedy_assignment': list(greed),
                'milp_training_worst_lcb': milp_value,
                'greedy_training_worst_lcb': greedy_value,
                'milp_worst_cost': milp_cost, 'greedy_worst_cost': greedy_cost,
                'exhaustive_optimum': oracle['objective'],
                'enumerated': oracle['supported_assignments'],
                'feasible': oracle['feasible_assignments'],
                'objective_gap_vs_greedy': milp_value - greedy_value,
                'holdout_true_milp': true_milp,
                'holdout_true_greedy': true_greedy,
                'holdout_truth_difference': true_milp - true_greedy,
                'cuts': cert.cuts,
                'seconds': {'milp_solver_only': milp_seconds,
                            'greedy': greedy_seconds, 'exhaustive': exhaustive_seconds},
            })
    by_cell = []
    for budget, radius in itertools.product(BUDGETS, TV_RADII):
        subset = [r for r in rows if r['budget'] == budget and r['tv_radius'] == radius]
        by_cell.append({
            'budget': budget, 'radius': radius,
            'strict_train_wins': sum(r['objective_gap_vs_greedy'] > 1e-7 for r in subset),
            'mean_holdout_truth_difference': float(np.mean([r['holdout_truth_difference'] for r in subset])),
            'mean_train_objective_gap': float(np.mean([r['objective_gap_vs_greedy'] for r in subset])),
        })
    seed_differences = np.array([
        np.mean([r['holdout_truth_difference'] for r in rows if r['seed']==seed])
        for seed in SEEDS
    ])
    bootstrap_rng = np.random.default_rng(20261016)
    cluster_means = np.array([
        bootstrap_rng.choice(seed_differences, size=len(SEEDS), replace=True).mean()
        for _ in range(5000)
    ])
    cluster_ci = np.quantile(cluster_means, [.025,.975])
    return {
        'schema_version': 1,
        'design': {
            'seeds': list(SEEDS), 'independent_seed_count': len(SEEDS),
            'cells_per_seed': len(BUDGETS) * len(TV_RADII),
            'budget_grid': list(BUDGETS), 'tv_radius_grid': list(TV_RADII),
            'training_n': TRAIN_N, 'independent_evaluation_n': EVALUATION_N,
            'costs': list(COSTS), 'min_ess': MIN_ESS,
            'trained_on': 'observed randomized synthetic data',
            'heldout_truth': 'simulation-only potential outcome oracle inaccessible during policy choice',
            'oracle_limit': 'complete finite-policy enumeration, common LP adversary',
        },
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'summary': {
            'cases': len(rows),
            'global_optimality_verified': sum(abs(r['milp_training_worst_lcb'] - r['exhaustive_optimum']) <= 1e-6 for r in rows),
            'strict_training_wins': sum(r['objective_gap_vs_greedy'] > 1e-7 for r in rows),
            'ties': sum(abs(r['objective_gap_vs_greedy']) <= 1e-7 for r in rows),
            'losses': sum(r['objective_gap_vs_greedy'] < -1e-7 for r in rows),
            'mean_train_objective_gap': float(np.mean([r['objective_gap_vs_greedy'] for r in rows])),
            'mean_heldout_truth_difference': float(np.mean([r['holdout_truth_difference'] for r in rows])),
            'heldout_truth_cluster_bootstrap_95pct': list(map(float,cluster_ci)),
            'cluster_bootstrap_note': '12 independent seeds, 4 coupled grid cells per seed; percentile bootstrap over seed averages, descriptive only',
            'mean_seconds': {k: float(np.mean([r['seconds'][k] for r in rows]))
                             for k in ('milp_solver_only','greedy','exhaustive')},
            'per_cell': by_cell,
        }, 'runs': rows,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='evidence/r17_three_way.json')
    args = parser.parse_args()
    result = benchmark()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
