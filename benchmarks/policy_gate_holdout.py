import json
import numpy as np
from pathlib import Path
from src.policy import generate_trial,fit_dr,allocate
from src.policy_uncertainty import paired_gate
rows=[]
for seed in (3,7,11):
    train=generate_trial(seed=seed,n=1600)
    validation=generate_trial(seed=seed+1000,n=1600)
    policy=allocate(train,fit_dr(train),min_ess=20)['policy']
    validation_dr=fit_dr(validation)
    target=np.array([policy[str(int(g))] for g in validation.segment])
    val=validation_dr[np.arange(len(target)),target]
    baseline=validation_dr[:,0]
    gate=paired_gate(val,baseline,min_ess=60,draws=500)
    rows.append(dict(seed=seed,policy=policy,lift=gate.lift,lower=gate.lower,upper=gate.upper,accepted=gate.accepted,
                     n=gate.n,effective_n=gate.effective_n))
summary=dict(scope='synthetic randomized DGP; selected policy frozen before independent evaluation; confidence intervals rely on iid assumptions',rows=rows)
Path('evidence/独立策略不确定性.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
