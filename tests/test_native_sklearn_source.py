"""Audit native application of the upstream reporter-derived fix."""
from scripts.verify_native_sklearn import execute

def test_actual_source_before_after(tmp_path):
    result=execute(tmp_path)
    assert result['before']['fit_successes']==9
    assert result['after']['fit_successes']==12
    assert result['before']['support_failures']==20
    assert result['after'].get('support_failures',0)==0
    assert 'Reporter' in result['provenance']
