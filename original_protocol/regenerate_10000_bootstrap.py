import argparse
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
parser=argparse.ArgumentParser(description='Regenerate the archived 10,000-draw cluster bootstrap.')
parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing authorized locked_test_predictions_seed*.csv files')
parser.add_argument('--out-dir', type=Path, required=True, help='Directory for regenerated summary tables')
args=parser.parse_args()
SEEDS=(17,29,43); CRITERIA=('c1','c2','c3'); B=10000; RNG=20261010
dfs={s:pd.read_csv(args.data_dir/f'locked_test_predictions_seed{s}.csv').sort_values(['video','frame']).reset_index(drop=True) for s in SEEDS}
keys=dfs[17][['video','frame']]
assert all(keys.equals(dfs[s][['video','frame']]) for s in SEEDS)
assert all(np.array_equal(dfs[17][f'y_{c}'],dfs[s][f'y_{c}']) for s in SEEDS for c in CRITERIA)
videos=np.sort(dfs[17].video.unique()); G=len(videos)
assert G==40 and len(keys)==1663
rng=np.random.default_rng(RNG)
counts=np.stack([np.bincount(row,minlength=G) for row in rng.integers(0,G,(B,G))]).astype(float)
out=[]; rank=[]
for c in CRITERIA:
 y=dfs[17][f'y_{c}'].to_numpy()
 pos=np.array([(y[(keys.video==g).to_numpy()]==1).sum() for g in videos],float)
 neg=np.array([(y[(keys.video==g).to_numpy()]==0).sum() for g in videos],float)
 support=(pos*neg)>0
 pooled=[];within=[]
 for s in SEEDS:
  score=dfs[s][f'p_{c}'].to_numpy()
  C=np.zeros((G,G))
  for i,g in enumerate(videos):
   a=score[(keys.video==g).to_numpy() & (y==1)]
   for j,h in enumerate(videos):
    b=score[(keys.video==h).to_numpy() & (y==0)]
    if len(a) and len(b):
     diff=a[:,None]-b[None,:]
     C[i,j]=(diff>0).sum()+.5*(diff==0).sum()
  base=C.sum()/(pos.sum()*neg.sum())
  assert abs(base-roc_auc_score(y,score))<1e-12
  opauc=np.divide(np.diag(C),pos*neg,out=np.zeros(G),where=support)
  numerator=np.einsum('bi,ij,bj->b',counts,C,counts,optimize=True)
  denom=(counts@pos)*(counts@neg)
  pb=numerator/denom
  wb=(counts[:,support]@opauc[support])/counts[:,support].sum(axis=1)
  pooled.append(pb);within.append(wb)
  out.append(dict(criterion=c,seed=s,pooled=base,pooled_lo=np.quantile(pb,.025),pooled_hi=np.quantile(pb,.975),within_equal=opauc[support].mean(),within_lo=np.quantile(wb,.025),within_hi=np.quantile(wb,.975),estimable_ops=int(support.sum())))
 pooled=np.array(pooled);within=np.array(within)
 pdelta=pooled[0]-pooled[1];wdelta=within[0]-within[1]
 rank.append(dict(criterion=c,B=B,prob_different_winner=np.mean(np.argmax(pooled,axis=0)!=np.argmax(within,axis=0)),prob_pooled29_within17=np.mean((np.argmax(pooled,axis=0)==1)&(np.argmax(within,axis=0)==0)),pooled17_minus29=pdelta.mean(),pooled17_minus29_lo=np.quantile(pdelta,.025),pooled17_minus29_hi=np.quantile(pdelta,.975),within17_minus29=wdelta.mean(),within17_minus29_lo=np.quantile(wdelta,.025),within17_minus29_hi=np.quantile(wdelta,.975)))
args.out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(out).to_csv(args.out_dir/'REGENERATED_10000_CLUSTER_BOOTSTRAP.csv',index=False)
pd.DataFrame(rank).to_csv(args.out_dir/'REGENERATED_10000_JOINT_RANKING.csv',index=False)
print('Recreated 10,000 paired cluster-bootstrap draws; exact point AUROC independently verified.')
