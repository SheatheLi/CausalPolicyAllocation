from scripts.verify_native_sklearn_semantic import verify

def test_native_support_semantics_over_reporter_prior_art():
    x=verify()
    c=x['scoring_comparison']
    assert x['fit_successes']==[9,12,12]
    assert c['reporter_only_max_error']>0
    assert c['reporter_only_max_samples_inconsistent_cases']==c['cases']
    assert c['enhanced_max_error']==0.
    assert c['enhanced_max_samples_inconsistent_cases']==0
