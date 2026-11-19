"""Paired bootstrap and gate for policy lift; not a guarantee of causal identification.

Requires independent evaluation outcomes or valid DR pseudo-outcomes.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Gate:
    lift:float
    lower:float
    upper:float
    accepted:bool
    effective_n:float
    n:int

def paired_gate(candidate,baseline,weights=None,seed=4,draws=1200,alpha=.05,min_ess=60):
    a=np.asarray(candidate,dtype=float);b=np.asarray(baseline,dtype=float)
    if a.ndim!=1 or b.shape!=a.shape or len(a)<2:raise ValueError('paired vector mismatch')
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):raise ValueError('nonfinite outcomes')
    w=np.ones(len(a)) if weights is None else np.asarray(weights,float)
    if w.shape!=a.shape or np.any(w<0) or not np.all(np.isfinite(w)) or w.sum()<=0:raise ValueError('invalid weight')
    p=w/w.sum();ess=1/np.dot(p,p)
    diff=a-b;mean=float(np.dot(diff,p));rng=np.random.default_rng(seed)
    # Bayesian bootstrap: continuous reweighting avoids pretending IID when sample weights vary.
    sims=np.empty(draws)
    for k in range(draws):
        g=rng.exponential(1,size=len(a))*p
        sims[k]=float(np.dot(diff,g)/g.sum())
    low,hi=np.quantile(sims,[alpha/2,1-alpha/2])
    return Gate(mean,float(low),float(hi),bool(low>0 and ess>=min_ess),float(ess),len(a))
