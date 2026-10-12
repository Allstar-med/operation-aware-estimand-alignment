"""Operation-aware estimand alignment: descriptive evaluation only.

No threshold or model selection on a locked test set.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def kernel(a,b):
    a=np.asarray(a,float)[:,None];b=np.asarray(b,float)[None,:]
    return ((a>b)+0.5*(a==b)).sum(dtype=float)

def audit(frame, operation='video', label='y', score='score'):
    x=frame[[operation,label,score]].copy()
    if x.isna().any().any():raise ValueError('Missing operation, label or score')
    if not x[label].isin([0,1]).all():raise ValueError('Binary labels required')
    if not np.isfinite(x[score]).all():raise ValueError('Finite scores required')
    pos=x.loc[x[label]==1];neg=x.loc[x[label]==0]
    if not len(pos) or not len(neg):raise ValueError('Both labels required globally')
    p,n=len(pos),len(neg);groups=list(x[operation].unique()); cells=[]; per=[]
    for g in groups:
        z=x[x[operation]==g]; a=z.loc[z[label]==1,score].to_numpy();b=z.loc[z[label]==0,score].to_numpy()
        per.append(dict(operation=g,frames=len(z),positive=len(a),negative=len(b),estimable=bool(len(a) and len(b)),within_auc=kernel(a,b)/(len(a)*len(b)) if len(a) and len(b) else np.nan))
        for h in groups:
            v=x[x[operation]==h];c=v.loc[v[label]==0,score].to_numpy()
            cells.append(dict(pos_operation=g,neg_operation=h,pairs=len(a)*len(c),concordance=kernel(a,c)))
    c=pd.DataFrame(cells);u=pd.DataFrame(per);same=c.pos_operation==c.neg_operation
    total=p*n; pooled=c.concordance.sum()/total
    direct=roc_auc_score(x[label],x[score]); assert abs(pooled-direct)<1e-12
    within_pairs=c.loc[same,'pairs'].sum(); cross_pairs=total-within_pairs
    same_auc=c.loc[same,'concordance'].sum()/within_pairs if within_pairs else np.nan
    cross_auc=c.loc[~same,'concordance'].sum()/cross_pairs if cross_pairs else np.nan
    equal=u.within_auc.mean()
    return dict(frames=len(x),operations=len(groups),positive=p,negative=n,supported_operations=int(u.estimable.sum()),support_fraction=float(u.estimable.mean()),pooled_auc=pooled,equal_operation_auc=equal,within_pair_auc=same_auc,cross_operation_auc=cross_auc,within_pair_fraction=within_pairs/total,cross_pair_fraction=cross_pairs/total,reconstruction_error=abs((c.concordance.sum()/total)-pooled)),u,c

def logit(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def shift_test(frame, operation='video', label='y', score='score', amplitude=2.):
    """Label-free offsets from group mean centered logits; descriptive, not causal."""
    x=frame.copy();z=logit(x[score]);group=pd.Series(z,index=x.index).groupby(x[operation]).transform('mean')
    center=float(z.mean());scale=float(np.std(z-group.to_numpy()))
    if scale<=0:raise ValueError('Zero within-operation logit scale')
    delta=(group-center)/max(float(group.std(ddof=0)),1e-12)
    x[score]=1/(1+np.exp(-(z+amplitude*delta.to_numpy())))
    base,_,_=audit(frame,operation,label,score);after,_,_=audit(x,operation,label,score)
    assert abs(base['equal_operation_auc']-after['equal_operation_auc'])<1e-12
    assert abs(base['within_pair_auc']-after['within_pair_auc'])<1e-12
    return dict(amplitude=amplitude,pooled_before=base['pooled_auc'],pooled_after=after['pooled_auc'],within_before=base['equal_operation_auc'],within_after=after['equal_operation_auc'],invariance_error=abs(base['equal_operation_auc']-after['equal_operation_auc']))

if __name__=='__main__':
 import argparse,json
 q=argparse.ArgumentParser();q.add_argument('csv');q.add_argument('--operation',default='video');q.add_argument('--label',default='y_c1');q.add_argument('--score',default='p_c1');q.add_argument('--out',default='audit.json');a=q.parse_args()
 d=pd.read_csv(a.csv);res,u,c=audit(d,a.operation,a.label,a.score);res['label_free_shift']=shift_test(d,a.operation,a.label,a.score)
 with open(a.out,'w') as f:json.dump(res,f,indent=2)
 u.to_csv(a.out+'.operations.csv',index=False);c.to_csv(a.out+'.cells.csv',index=False)
