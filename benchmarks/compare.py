import json
import numpy as np
from pathlib import Path
from src.policy import *
rows=[]
for seed in (3,5,7,11,13):
    d=generate_trial(seed=seed,n=4000);eval_d=generate_trial(seed=seed+10000,n=4000)
    dr=fit_dr(d);ans=allocate(d,dr,.60,min_ess=30)
    baseline={'0':0,'1':0,'2':0,'3':0}
    rows.append({'seed':seed,'policy':ans['policy'],'true_value':ground_truth_policy_value(eval_d,ans['policy']),
                 'baseline_true_value':ground_truth_policy_value(eval_d,baseline),
                 'expected_cost':ans['expected_cost_per_person'],
                 'unsupported_arms':ans['unsupported_arms'],'ess':ans['ess']})
summary={'data':'seeded randomized synthetic RCT; policy truth is simulator-only',
         'training_n_per_seed':4000,'independent_evaluation_n_per_seed':4000,'seeds':len(rows),'results':rows,
         'mean_value':float(np.mean([r['true_value'] for r in rows])),
         'mean_baseline':float(np.mean([r['baseline_true_value'] for r in rows]))}
Path('evidence/benchmark.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
