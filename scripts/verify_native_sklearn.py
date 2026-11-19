"""One-command source-isolated upstream reproduction for scikit-learn #34673.

Requires installed scikit-learn 1.8.0 + GNU patch. Doesn't mutate global
site-packages. The functional fix is identical in nature to the ISSUE REPORTER'S
proposal; the script establishes reproducibility, not authorship or acceptance.
"""
from __future__ import annotations
import argparse,importlib.util,json,os,pathlib,shutil,subprocess,sys,tempfile

HERE=pathlib.Path(__file__).resolve().parent
REGRESSION=HERE/'native_sklearn_regression.py'
PATCH=HERE.parent/'patches/sklearn_34673_native_1.8.0.patch'

def execute(dest: pathlib.Path):
    import sklearn
    if sklearn.__version__!='1.8.0':
        raise RuntimeError(f'source pin required: scikit-learn 1.8.0, found {sklearn.__version__}')
    source=pathlib.Path(sklearn.__file__).resolve().parent
    shutil.copytree(source,dest/'sklearn',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','tests'))
    libs=source.parent/'scikit_learn.libs'
    if libs.exists():shutil.copytree(libs,dest/'scikit_learn.libs')
    env={**os.environ,'PYTHONPATH':str(dest)}
    def call():
        p=subprocess.run([sys.executable,str(REGRESSION)],cwd=dest,env=env,
                         capture_output=True,text=True,timeout=90)
        if p.returncode:raise RuntimeError(f'isolated regression exit {p.returncode}\n{p.stderr[-2500:]}')
        output=json.loads(p.stdout)
        assert output['source_path'].startswith(str(dest)),output['source_path']
        return output
    before=call()
    prep=subprocess.run(['patch','--dry-run','-p1'],input=PATCH.read_text(),text=True,
                        cwd=dest,capture_output=True)
    if prep.returncode:raise RuntimeError(prep.stdout+'\n'+prep.stderr)
    apply=subprocess.run(['patch','-p1'],input=PATCH.read_text(),text=True,
                         cwd=dest,capture_output=True,check=True)
    after=call()
    # Known version/source contract: original IsolationForest fails on 3
    # zero-support cases, patched source fixes all 12 tested fit cases.
    assert before['fit_successes']==9,before
    assert before['support_failures']==20,before
    assert after['fit_successes']==12,after
    assert after.get('support_failures',0)==0,after
    return {'issue':'scikit-learn/scikit-learn#34673',
            'provenance':'Reporter proposed support-size clamp (not original)',
            'result':'local native source candidate verified, not upstream merged',
            'patch_dry_run':prep.stdout.strip(),'patch_apply':apply.stdout.strip(),
            'before':before,'after':after,
            'limits':['no full sklearn suite','no upstream review','not robust for zero-total weights']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path)
    args=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix='sklearn34673_') as d:
        result=execute(pathlib.Path(d))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    else:
        print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
