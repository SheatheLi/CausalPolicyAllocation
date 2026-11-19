"""Targeted runtime tradeoff of upstream issue #34673 source variants."""
from time import perf_counter
import numpy as np,json,sklearn
from sklearn.ensemble import IsolationForest

def bench():
    rows=[]
    for n,zeros in ((80,1),(300,30),(2000,200)):
        X=np.random.RandomState(n+1).uniform(size=(n,4))
        w=np.ones(n);w[:zeros]=0.
        for seed in (0,1,2,3,4):
            f=IsolationForest(n_estimators=12,random_state=seed)
            t0=perf_counter();f.fit(X,sample_weight=w);t1=perf_counter()
            f.score_samples(X[:min(n,30)]);t2=perf_counter()
            rows.append({'n':n,'zeros':zeros,'seed':seed,'fit_ms':(t1-t0)*1000,
                        'score_ms':(t2-t1)*1000,'max_samples':f.max_samples_})
    return {'version':sklearn.__version__,'source':sklearn.__file__,'rows':rows}
if __name__=='__main__':print(json.dumps(bench(),indent=2))
