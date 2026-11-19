"""Independent, narrowly scoped diagnostic of a *frozen* offline policy.

This module intentionally does not select or optimize policies. It checks the
observed assignment distribution and tests how a *fixed* DR calculation changes
under alternative, explicitly hypothetical logging propensities. These are
stress diagnostics, not proofs of randomization, identification, coverage or
safety under unobserved confounding.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping
import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class AssignmentCell:
    segment: int
    arm: int
    units: int
    observed: int
    expected: float
    standardized_residual: float


@dataclass(frozen=True)
class AssignmentDiagnostic:
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    min_expected: float
    sparse_expected_cells: int
    cells: tuple[AssignmentCell, ...]
    interpretation: str


def assignment_diagnostic(trial, min_expected: float = 5.0) -> AssignmentDiagnostic:
    """Compare declared assignment propensities to counts within fixed segments.

    Uses an exploratory Pearson approximation with effective df = segment*(K-1)
    for known rather than estimated propensities. A high p-value is never proof
    that assignments were randomized; small expected cells weaken calibration.
    """
    t = np.asarray(trial.t)
    p = np.asarray(trial.propensity, dtype=float)
    group = np.asarray(trial.segment)
    if (p.ndim != 2 or p.shape[0] != len(t) or len(group) != len(t)
        or len(t) < 2 or p.shape[1] < 2 or np.any(t < 0)
        or np.any(t >= p.shape[1]) or not np.all(np.isfinite(p))
        or np.any(p <= 0) or not np.allclose(p.sum(axis=1), 1.0)
        or not np.isfinite(min_expected) or min_expected <= 0):
        raise ValueError('invalid observed assignment contract')
    chi = 0.0
    cells = []
    min_e = np.inf
    sparse = 0
    for segment in np.unique(group):
        mask = group == segment
        n = int(mask.sum())
        for arm in range(p.shape[1]):
            observed = int(np.count_nonzero(t[mask] == arm))
            expected = float(p[mask, arm].sum())
            min_e = min(min_e, expected)
            sparse += expected < min_expected
            residual = float((observed - expected) / np.sqrt(expected))
            chi += residual**2
            cells.append(AssignmentCell(int(segment), arm, n, observed, expected, residual))
    df = int(len(np.unique(group)) * (p.shape[1] - 1))
    return AssignmentDiagnostic(
        float(chi), df, float(chi2.sf(chi, df)), float(min_e), sparse,
        tuple(cells),
        'Exploratory count calibration only; independent provenance and randomization audit remain required.'
    )


def _validate_frozen_inputs(trial, predictions, policy: Mapping[int | str, int], baseline: int):
    p = np.asarray(trial.propensity, dtype=float)
    m = np.asarray(predictions, dtype=float)
    t = np.asarray(trial.t)
    y = np.asarray(trial.y, dtype=float)
    groups = np.asarray(trial.segment)
    if (m.ndim != 2 or m.shape != p.shape or y.shape != (len(t),)
        or groups.shape != (len(t),) or len(t) < 2
        or np.any(~np.isfinite(m)) or np.any(~np.isfinite(y))
        or np.any(~np.isfinite(p)) or np.any(p <= 0)
        or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1.0)
        or np.any(t < 0) or np.any(t >= m.shape[1])
        or not isinstance(baseline, int) or not 0 <= baseline < m.shape[1]):
        raise ValueError('invalid frozen policy evaluation vectors')
    observed_groups = [int(s) for s in np.unique(groups)]
    normalized = {int(key): int(value) for key, value in policy.items()}
    if (set(normalized) != set(observed_groups)
        or any(a < 0 or a >= m.shape[1] for a in normalized.values())):
        raise ValueError('policy must select exactly one valid action for every observed segment')
    action = np.array([normalized[int(s)] for s in groups], dtype=int)
    return p, m, t, y, groups, action


def frozen_dr_difference(trial, predictions, policy, *, baseline=0, propensity=None):
    """Return per-unit DR policy advantage with no policy selection or truth access."""
    p, m, t, y, _, action = _validate_frozen_inputs(trial, predictions, policy, baseline)
    if propensity is not None:
        p = np.asarray(propensity, dtype=float)
        if (p.shape != m.shape or not np.isfinite(p).all() or np.any(p <= 0)
            or not np.allclose(p.sum(axis=1), 1)):
            raise ValueError('invalid hypothetical propensity matrix')
    ix = np.arange(len(t))
    cand = m[ix, action] + (t == action) * (y - m[ix, action]) / p[ix, action]
    base = m[ix, baseline] + (t == baseline) * (y - m[ix, baseline]) / p[ix, baseline]
    return np.asarray(cand - base, dtype=float)


def propensity_perturbation_envelope(
    trial, predictions, policy,
    *, baseline: int = 0, relative_log_odds_shifts=(-0.4, 0.0, 0.4),
):
    """Explore predeclared logging-odds alternatives; NOT a confounding bound.

    For each arm, multiply its odds versus all other arms by exp(shift).
    Re-normalize the full vector, then recompute a DR advantage with the same
    *frozen* out-of-fold outcome predictions and policy.
    """
    p, m, _, _, _, _ = _validate_frozen_inputs(trial, predictions, policy, baseline)
    shifts = tuple(float(x) for x in relative_log_odds_shifts)
    if not shifts or len(set(shifts)) != len(shifts) or any(not np.isfinite(x) or abs(x) > 4 for x in shifts):
        raise ValueError('provide distinct finite log-odds shifts in [-4,4]')
    cases=[]
    for arm in range(p.shape[1]):
        for delta in shifts:
            q = p.copy()
            q[:, arm] *= np.exp(delta)
            q /= q.sum(axis=1, keepdims=True)
            effect = frozen_dr_difference(trial, m, policy, baseline=baseline, propensity=q)
            cases.append({'arm':arm, 'log_odds_shift':delta, 'mean_lift':float(effect.mean()),
                          'minimum_propensity':float(q.min())})
    original = frozen_dr_difference(trial, m, policy, baseline=baseline)
    values = [v['mean_lift'] for v in cases]
    return {
        'nominal_mean_lift':float(original.mean()),
        'minimum_scenario_mean_lift':float(min(values)),
        'maximum_scenario_mean_lift':float(max(values)),
        'scenarios':cases,
        'scope':'hypothetical logging propensity misspecification only; no identification, policy safety or unobserved-confounding guarantee',
    }


def audit_frozen_policy(trial, predictions, policy, *, baseline=0, min_group_ess=20., shifts=(-0.4, 0., 0.4)):
    """Evidence-rich read-only report for a previously frozen decision."""
    p, m, t, _, groups, action = _validate_frozen_inputs(trial, predictions, policy, baseline)
    if not np.isfinite(min_group_ess) or min_group_ess <= 0:
        raise ValueError('min_group_ess must be positive')
    assignment = assignment_diagnostic(trial)
    support=[]
    supported=True
    for segment in np.unique(groups):
        mask = groups == segment
        for arm in sorted({baseline,int(action[np.flatnonzero(mask)[0]])}):
            w=(t[mask]==arm)/p[mask,arm]
            ess=float(w.sum()**2/np.dot(w,w)) if np.dot(w,w)>0 else 0.
            ok=bool(ess>=min_group_ess)
            supported &= ok
            support.append({'segment':int(segment),'arm':int(arm),'ess':ess,'ok':ok})
    envelope=propensity_perturbation_envelope(trial,m,policy,baseline=baseline,relative_log_odds_shifts=shifts)
    return {
        'status':'EXPLORATORY_DIAGNOSTIC_ONLY',
        'assignment':{**{k:v for k,v in asdict(assignment).items() if k!='cells'},
                      'cells':[asdict(c) for c in assignment.cells]},
        'group_support':support,
        'all_required_group_support_present':bool(supported),
        'perturbation':envelope,
        'external_prerequisites':[
            'Assignment provenance and declared probabilities have not been independently verified',
            'Features must be available before assignment; no interference or hidden confounding assessed',
            'Policy and sensitivity grid must be frozen before evaluation, not tuned on this holdout',
            'Independent production deployment remains outside this statistical audit',
        ],
    }
