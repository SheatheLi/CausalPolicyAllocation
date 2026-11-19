import pytest
from src.heldout_policy_gate import evaluate,PolicyGateError

def records(n=30):
    # Exogenous randomized logging: no model fitting inside this module.
    return [dict(segment='s',action=i%2,reward=float((i%2)==1),
        logging_probs=[.5,.5],outcomes=[0.,1.],fold_id=f'holdout-{i%4}') for i in range(n)]

def test_randomized_policy_value_and_equal_policy_difference():
    data=records()
    result=evaluate(data,{'s':1},{'s':0},n_actions=2,bootstrap_iterations=101)
    assert result.status=='ESTIMATED'
    assert abs(result.advantage-1)<1e-12
    same=evaluate(data,{'s':1},{'s':1},n_actions=2,bootstrap_iterations=101)
    assert same.advantage==0

def test_group_with_no_matching_action_is_refused():
    data=records()
    for row in data:row['action']=0;row['logging_probs']=[1.,0.]
    result=evaluate(data,{'s':1},{'s':0},n_actions=2,bootstrap_iterations=101)
    assert result.status=='REFUSE'
    assert result.value is None
    assert result.diagnostics['blocked_segments']==['s:target']

def test_tiny_ess_is_refused():
    data=records(4)
    result=evaluate(data,{'s':1},{'s':0},n_actions=2,minimum_group_matches=3,bootstrap_iterations=101)
    assert result.status=='REFUSE'

@pytest.mark.parametrize('mutate',[
    lambda x:x[0].update(reward=float('nan')),
    lambda x:x[0].update(logging_probs=[1.5,-.5]),
    lambda x:x[0].update(action=True),
    lambda x:x[0].update(fold_id=''),
    lambda x:x[0].update(logging_probs=[.1,.1]),
])
def test_bad_inputs_do_not_generate_policy_claims(mutate):
    data=records();mutate(data)
    with pytest.raises(PolicyGateError):evaluate(data,{'s':1},{'s':0},n_actions=2)

def test_unsupported_baseline_must_also_refuse():
    data=records()
    for row in data:row['action']=1;row['logging_probs']=[0.,1.]
    result=evaluate(data,{'s':1},{'s':0},n_actions=2,bootstrap_iterations=101)
    assert result.status=='REFUSE'
    assert 's:baseline' in result.diagnostics['blocked_segments']
