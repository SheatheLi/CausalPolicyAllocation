"""Local isolation study for scikit-learn issue #34673 on installed 1.8.0.

Based on reporter's suggested fix. This is NOT an original upstream fix,
not an accepted PR, and not a fix to CausalAllocation's estimator.
"""
from __future__ import annotations
import inspect
import numpy as np
from sklearn.ensemble import IsolationForest
import sklearn.ensemble._bagging as bag


def attempt_fit()->dict:
    X=np.random.RandomState(0).uniform(size=(80,4))
    w=np.ones(80);w[0]=0.0
    try:
        IsolationForest(random_state=0,n_estimators=8).fit(X,sample_weight=w)
        outcome='success'
    except ValueError as exc:
        outcome=f'{type(exc).__name__}: {exc}'
    return {'outcome':outcome,'n':len(w),'nonzero':int(np.count_nonzero(w))}


def try_local_candidate()->dict:
    original=bag._generate_bagging_indices
    text=inspect.getsource(original)
    anchor='        sample_indices = random_state.choice(\n'
    if text.count(anchor)!=1:raise RuntimeError('installed source does not match guarded candidate patch')
    # Patch the installed function definition IN MEMORY ONLY for an isolation
    # test, preserving all imports, normal dispatch and other sklearn logic.
    patched=text.replace(anchor,
        '        if not bootstrap_samples:\n'
        '            max_samples = min(max_samples, np.count_nonzero(sample_weight))\n'
        +anchor)
    namespace={}
    exec(compile(patched,'scikit-learn-34673-memory-candidate','exec'),bag.__dict__,namespace)
    candidate=namespace[original.__name__]
    bag._generate_bagging_indices=candidate
    try:
        fixed=attempt_fit()
        rng=np.random.RandomState(9);w=np.ones(80);w[:10]=0
        _,ids=candidate(rng,False,False,4,80,4,80,w)
        zero_weight_selected=int(np.sum(ids<10))
        # The bootstrap path should still be able to repeat rows.
        _,bootstrap_ids=candidate(np.random.RandomState(9),False,True,4,80,4,80,w)
        return {'candidate_fit':fixed['outcome'],'unique_nonbootstrap':len(set(ids))==len(ids),
                'nonbootstrap_selected':len(ids),'zero_weight_selected':zero_weight_selected,
                'bootstrap_draws':len(bootstrap_ids),'upstream_patch_authority':'reporter proposed; not original contribution'}
    finally:
        bag._generate_bagging_indices=original

if __name__=='__main__':
    import json
    print(json.dumps({'baseline':attempt_fit(),'candidate':try_local_candidate()},indent=2))
