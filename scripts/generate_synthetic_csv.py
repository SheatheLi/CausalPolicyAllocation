"""Generate deterministic synthetic randomized *observed-data* CSV examples.

Potential outcomes are deliberately NOT saved; the engine cannot read them.
"""
from __future__ import annotations
import csv
from pathlib import Path
from src.policy import generate_trial
from src.decision_pipeline import FIELDS


def write_example(directory: Path, n: int = 2400) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    for label, seed in [('train', 13),('evaluation', 941)]:
        t=generate_trial(seed=seed,n=n)
        with (directory/f'{label}.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
            for i in range(n):
                row={'unit_id':f'{label}-{i}','segment':int(t.segment[i]),
                     't':int(t.t[i]),'y':int(t.y[i])}
                row.update({f'x{j}':float(t.x[i,j]) for j in range(4)})
                row.update({f'p{j}':float(t.propensity[i,j]) for j in range(3)})
                w.writerow(row)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dir',default='demo_data');p.add_argument('--n',type=int,default=2400)
    a=p.parse_args();write_example(Path(a.dir),a.n)
    print(f'wrote synthetic OBSERVED train/evaluation CSV to {a.dir}')
