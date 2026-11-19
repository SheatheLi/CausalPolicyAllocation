"""Isolated upstream-source regression for sklearn/sklearn#34673.

Runs against a separately copied sklearn 1.8.0 source installation, not a
runtime monkeypatch of the process under test.  The candidate reproduces the
issue reporter's proposed support-clamping algorithm, which is prior art.
"""
from __future__ import annotations
import json, numpy as np, sklearn
from sklearn.ensemble import IsolationForest, BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def checked_fit(name, X, w, seed=0):
    if name == 'isolation':
        return IsolationForest(n_estimators=4, random_state=seed).fit(X, sample_weight=w)
    if name == 'classifier':
        y=(X[:,0]>0.5).astype(int)
        return BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=4,
                                 random_state=seed, bootstrap=False, max_samples=1.0).fit(X,y,sample_weight=w)
    if name == 'regressor':
        y=X[:,0]*4 + X[:,1]
        return BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=4,
                                random_state=seed, bootstrap=False, max_samples=1.0).fit(X,y,sample_weight=w)
    raise ValueError(name)


def run():
    from sklearn.ensemble._bagging import _generate_bagging_indices
    rng=np.random.RandomState(4)
    report={'source_version':sklearn.__version__,'source_path':sklearn.__file__,'cases':[]}
    for name in ('isolation','classifier','regressor'):
        for n, zeros in ((80,0),(80,1),(80,10),(300,60)):
            X=np.random.RandomState(n).uniform(size=(n,4))
            w=np.ones(n);w[:zeros]=0.
            try:
                checked_fit(name,X,w)
                status='success'
            except (ValueError,RuntimeError) as exc:
                status=f'{type(exc).__name__}: {str(exc)[:100]}'
            report['cases'].append({'estimator':name,'n':n,'zeros':zeros,'result':status})
    # The patch must preserve sample IDs, no duplicates, and honor support.
    for bootstrap in (False,True):
        for seed in range(20):
            w=np.array([0,2.,0,3.,1.,0,9.,4.,0.,5.])
            rng=np.random.RandomState(seed)
            try:
                _,indices=_generate_bagging_indices(rng,False,bootstrap,4,10,4,10,w)
            except ValueError as exc:
                if not bootstrap and 'Fewer non-zero entries' in str(exc):
                    report.setdefault('support_failures',0)
                    report['support_failures']+=1
                    continue
                raise
            assert set(indices).isdisjoint({0,2,5,8}),indices
            if not bootstrap:
                assert len(indices)==6 and len(set(indices))==6,indices
            else:
                assert len(indices)==10
    report['support_invariants_passed']=40-report.get('support_failures',0)
    report['fit_successes']=sum(v['result']=='success' for v in report['cases'])
    report['fit_cases']=len(report['cases'])
    return report
if __name__=='__main__':print(json.dumps(run(),indent=2))
