from real_issues.sklearn_34673 import attempt_fit,try_local_candidate

def test_issue_installed_version():
    before=attempt_fit()
    assert 'Fewer non-zero' in before['outcome']
    after=try_local_candidate()
    assert after['candidate_fit']=='success'
    assert after['zero_weight_selected']==0
    assert after['nonbootstrap_selected']==70
    assert after['unique_nonbootstrap']
    # Candidate mutates only installed module binding for this experiment:
    assert attempt_fit()['outcome']==before['outcome']
