import argparse
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
parser=argparse.ArgumentParser(description='Verify pooled AUC and within/cross-operation pair support.')
parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing authorized locked_test_predictions_seed*.csv files')
parser.add_argument('--out', type=Path, help='Optional output CSV path')
args=parser.parse_args()
base=args.data_dir
rows=[]
for seed in (17,29,43):
 d=pd.read_csv(base/f'locked_test_predictions_seed{seed}.csv')
 for c in ('c1','c2','c3'):
  y=d[f'y_{c}'].to_numpy();p=d[f'p_{c}'].to_numpy();v=d.video.to_numpy();ops=np.unique(v)
  pos=(y==1);neg=~pos;tot=int(pos.sum()*neg.sum());cross=0;within=0
  for op in ops:
   a=pos&(v==op);b=neg&(v==op)
   within+=int(a.sum()*b.sum())
  cross=tot-within
  auc=roc_auc_score(y,p)
  rows.append(dict(seed=seed,criterion=c,pooled_auc=auc,within_pair_fraction=within/tot,cross_pair_fraction=cross/tot,n_operations=len(ops),n_eligible_operations=sum((y[v==o].min()!=y[v==o].max()) for o in ops)))
res=pd.DataFrame(rows)
if args.out:
 args.out.parent.mkdir(parents=True, exist_ok=True)
 res.to_csv(args.out,index=False)
print(res.to_string(index=False))
