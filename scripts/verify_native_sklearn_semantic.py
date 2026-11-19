"""Compare original, prior-art clamp, and semantic-consistent native patch.

This is source-isolated (real sklearn Python package copied to temp), not an
in-process monkeypatch. The reported strong win is in the defined zero-weight
'exclude row' semantics, not end-to-end speed or broad library superiority.
"""
from __future__ import annotations
import importlib,os,pathlib,subprocess,sys,tempfile,shutil,json
HERE=pathlib.Path(__file__).resolve().parent
REPORTER=HERE.parent/'patches/sklearn_34673_native_1.8.0.patch'
ENHANCED=HERE.parent/'patches/sklearn_34673_iforest_support_semantics.patch'
REGRESSION=HERE/'native_sklearn_regression.py'
COMPARE=HERE/'check_iforest_semantics.py'

def verify():
 import sklearn
 assert sklearn.__version__=='1.8.0'
 source=pathlib.Path(sklearn.__file__).parent
 with tempfile.TemporaryDirectory(prefix='sklearn_semantics_') as tmp:
  dest=pathlib.Path(tmp)
  shutil.copytree(source,dest/'sklearn',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','tests'))
  libs=source.parent/'scikit_learn.libs'
  if libs.exists():shutil.copytree(libs,dest/'scikit_learn.libs')
  env={**os.environ,'PYTHONPATH':str(dest)}
  def run(path):
   p=subprocess.run([sys.executable,str(path)],cwd=dest,env=env,capture_output=True,text=True,timeout=90)
   if p.returncode:raise RuntimeError(p.stderr[-3000:])
   data=json.loads(p.stdout)
   assert data['source'].startswith(str(dest)) if 'source' in data else data['source_path'].startswith(str(dest))
   return data
  def patch_it(patch):
   dry=subprocess.run(['patch','--dry-run','-p1'],input=patch.read_text(),text=True,cwd=dest,capture_output=True)
   if dry.returncode:raise RuntimeError(dry.stdout+dry.stderr)
   subprocess.run(['patch','-p1'],input=patch.read_text(),text=True,cwd=dest,check=True,capture_output=True)
  original=run(REGRESSION)
  patch_it(REPORTER)
  reporter=run(REGRESSION)
  reporter_sem=run(COMPARE)
  patch_it(ENHANCED)
  enhanced=run(REGRESSION)
  enhanced_sem=run(COMPARE)
  assert original['fit_successes']==9
  assert reporter['fit_successes']==enhanced['fit_successes']==12
  assert not reporter_sem['all_equal']
  assert enhanced_sem['all_equal']
  return {'issue':'scikit-learn/scikit-learn#34673','tested_source':'scikit-learn 1.8.0 isolated copy',
          'origin':'Reporter prior art fix + independently tested IsolationForest semantic extension',
          'scoring_comparison':{
            'reporter_only_max_error':max(x['score_max_abs_error'] for x in reporter_sem['observations']),
            'reporter_only_max_samples_inconsistent_cases':sum(x['max_samples_diff']!=0 for x in reporter_sem['observations']),
            'enhanced_max_error':max(x['score_max_abs_error'] for x in enhanced_sem['observations']),
            'enhanced_max_samples_inconsistent_cases':sum(x['max_samples_diff']!=0 for x in enhanced_sem['observations']),
            'cases':len(reporter_sem['observations'])},
          'fit_successes':[original['fit_successes'],reporter['fit_successes'],enhanced['fit_successes']],
          'scope':'Local unreviewed source-level candidate; original upstream authorship NOT claimed'}

if __name__=='__main__':
 result=verify()
 print(json.dumps(result,indent=2))
