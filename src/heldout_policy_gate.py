"""Support-aware doubly-robust *heldout* policy evaluation.

This instrument is not a causal identification proof, and does not fit nuisance
models on evaluation records. It requires external propensity and prediction
inputs, an explicit target policy and minimum per-segment support; refuses to
produce an actionable policy when overlap diagnostics fail. Its intervals are
IID row-bootstrap approximations, not valid for temporal/clustered dependence.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,asdict
import argparse,json,math,pathlib,random

class PolicyGateError(ValueError):pass

@dataclass(frozen=True)
class Observation:
    segment:str
    action:int
    reward:float
    logging_probs:tuple[float,...]
    outcomes:tuple[float,...]
    fold_id:str

@dataclass(frozen=True)
class Decision:
    status:str
    value:float|None
    baseline_value:float|None
    advantage:float|None
    uncertainty_95:tuple[float,float]|None
    diagnostics:dict
    interpretation:str
    def serial(self):return asdict(self)

def _finite(value):return type(value) in (int,float) and math.isfinite(value)

def _observation(row,n_actions):
    if not isinstance(row,dict):raise PolicyGateError('row is not an object')
    probs=row.get('logging_probs');outcomes=row.get('outcomes')
    if not isinstance(probs,list) or not isinstance(outcomes,list) or len(probs)!=n_actions or len(outcomes)!=n_actions:
        raise PolicyGateError('full action propensity and predictions required')
    if not all(_finite(x) and x>=0 for x in probs) or abs(sum(probs)-1)>1e-6:
        raise PolicyGateError('invalid logging probabilities')
    if not all(_finite(x) for x in outcomes):raise PolicyGateError('invalid outcome prediction')
    a=row.get('action'); y=row.get('reward');group=row.get('segment');fold=row.get('fold_id')
    if type(a) is not int or not 0<=a<n_actions or probs[a]<=0:
        raise PolicyGateError('logged action incompatible with probabilities')
    if not _finite(y) or not isinstance(group,str) or not group or not isinstance(fold,str) or not fold:
        raise PolicyGateError('invalid reward, segment or fold provenance')
    return Observation(group,a,float(y),tuple(float(x) for x in probs),tuple(float(x) for x in outcomes),fold)

def _ess(weights):
    sq=sum(x*x for x in weights)
    return sum(weights)**2/sq if sq else 0.

def _dr(obs,action):
    return obs.outcomes[action]+(obs.reward-obs.outcomes[action])/obs.logging_probs[action] if obs.action==action else obs.outcomes[action]

def evaluate(rows:list[dict],target:dict[str,int],baseline:dict[str,int],*,
             n_actions:int,minimum_group_matches:int=3,minimum_ess:float=2.,
             bootstrap_iterations:int=1000,seed:int=19)->Decision:
    if type(n_actions) is not int or n_actions<2:raise PolicyGateError('at least two actions required')
    if not isinstance(rows,list) or not rows:raise PolicyGateError('nonempty holdout rows required')
    if type(minimum_group_matches) is not int or minimum_group_matches<1 or not _finite(minimum_ess) or minimum_ess<=0:
        raise PolicyGateError('invalid support threshold')
    if type(bootstrap_iterations) is not int or bootstrap_iterations<100:
        raise PolicyGateError('at least 100 bootstrap samples')
    data=[_observation(x,n_actions) for x in rows]
    groups=set(x.segment for x in data)
    if not all(isinstance(policy,dict) and set(policy)==groups and all(type(a) is int and 0<=a<n_actions for a in policy.values()) for policy in (target,baseline)):
        raise PolicyGateError('target and baseline must specify valid actions for all observed segments')
    group_records=defaultdict(list)
    for x in data:group_records[x.segment].append(x)
    diagnoses={};blocking=[]
    for group,subset in sorted(group_records.items()):
        diag={'records':len(subset),'folds':sorted(set(x.fold_id for x in subset))}
        # Both policies require support; a model-only baseline cannot establish
        # a reliable DR difference even if the candidate policy is supported.
        for label,policy in (('target',target),('baseline',baseline)):
            action=policy[group]
            matching=[x for x in subset if x.action==action]
            # Very small propensities are numerically unstable and practically
            # represent insufficient overlap; refuse instead of clipping them
            # silently to fabricate a credible policy estimate.
            weights=[1/x.logging_probs[action] for x in matching]
            effective=_ess(weights)
            diag[label]={'action':action,'matched_records':len(matching),
                         'ess':effective,
                         'mean_logging_probability':sum(x.logging_probs[action] for x in subset)/len(subset)}
            if (len(matching)<minimum_group_matches or effective<minimum_ess or
                    any(x.logging_probs[action]<1e-6 for x in matching)):
                blocking.append(group+':'+label)
        diagnoses[group]=diag
    if blocking:
        return Decision('REFUSE',None,None,None,None,
            {'n':len(data),'groups':diagnoses,'blocked_segments':blocking},
            'Observed support is insufficient; no policy value or recommendation issued')
    target_terms=[_dr(x,target[x.segment]) for x in data]
    baseline_terms=[_dr(x,baseline[x.segment]) for x in data]
    diffs=[t-b for t,b in zip(target_terms,baseline_terms)]
    n=len(data);adv=sum(diffs)/n;rng=random.Random(seed)
    samples=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(bootstrap_iterations))
    bounds=(samples[int(.025*(bootstrap_iterations-1))],samples[int(.975*(bootstrap_iterations-1))])
    return Decision('ESTIMATED',sum(target_terms)/n,sum(baseline_terms)/n,adv,bounds,
        {'n':n,'groups':diagnoses,'blocked_segments':[],'assumed_unit':'independent evaluation record'},
        'Conditional DR estimate; depends on identification, propensity validity and out-of-fold nuisance models')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=pathlib.Path)
    args=parser.parse_args()
    config=json.loads(args.experiment.read_text(encoding='utf8'))
    out=evaluate(config['observations'],config['target'],config['baseline'],n_actions=config['n_actions'])
    print(json.dumps(out.serial(),ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
