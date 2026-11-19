"""Ablation: original reporter clamp vs row-exclusion semantics.

A correct zero-row exclusion should reproduce manual row deletion even when
bootstrapping, contamination thresholding, sparse storage and nonuniform
positive weights interact. The independent score oracle is the sklearn fit
on explicitly filtered rows with corresponding positive sample weights.
"""
import numpy as np,json,sklearn
from sklearn.ensemble import IsolationForest
from scipy.sparse import csc_matrix

def result():
    rows=[]
    for n, zeros in ((80,1),(80,10),(300,60)):
        for seed in (0,1,7):
            for bootstrap in (False,True):
                for contamination in ('auto',0.1):
                    for sparse in (False,True):
                        X=np.random.RandomState(9+n).uniform(size=(n,4))
                        w=np.linspace(0.1,2.0,n)
                        w[:zeros]=0.
                        if sparse:X=csc_matrix(X)
                        model=IsolationForest(n_estimators=6,random_state=seed,bootstrap=bootstrap,
                                              contamination=contamination)
                        ref=IsolationForest(n_estimators=6,random_state=seed,bootstrap=bootstrap,
                                            contamination=contamination)
                        model.fit(X,sample_weight=w)
                        ref.fit(X[w>0],sample_weight=w[w>0])
                        obs=X[w>0][:15]
                        delta=float(np.max(np.abs(model.score_samples(obs)-ref.score_samples(obs))))
                        rows.append({'n':n,'zeros':zeros,'seed':seed,'bootstrap':bootstrap,
                                     'contamination':str(contamination),'sparse':sparse,
                                     'max_samples_diff':int(model.max_samples_-ref.max_samples_),
                                     'offset_diff':float(abs(model.offset_-ref.offset_)),
                                     'score_max_abs_error':delta})
    return {'version':sklearn.__version__,'source':sklearn.__file__,'observations':rows,
            'all_equal':all(x['score_max_abs_error']<=1e-12 and x['offset_diff']<=1e-12 and x['max_samples_diff']==0 for x in rows)}
if __name__=='__main__':print(json.dumps(result(),indent=2))
