import numpy as np,pandas as pd
from oea import audit,shift_test
from sklearn.metrics import roc_auc_score
R=np.random.default_rng(20261011)
for rep in range(150):
 n=int(R.integers(2,12));groups=R.choice([f'op{i}' for i in range(n)],size=120)
 y=R.integers(0,2,120);s=R.choice([0.,.1,.3,.5,.7,.9,1.],size=120)
 x=pd.DataFrame(dict(video=groups,y=y,score=s))
 a,u,c=audit(x)
 assert abs(a['pooled_auc']-roc_auc_score(y,s))<1e-12
 assert c.pairs.sum()==sum(y)*(len(y)-sum(y))
 assert abs(a['within_pair_fraction']+a['cross_pair_fraction']-1)<1e-12
 assert abs(a['pooled_auc']-((a['within_pair_fraction']*a['within_pair_auc'] if np.isfinite(a['within_pair_auc']) else 0)+(a['cross_pair_fraction']*a['cross_operation_auc'] if np.isfinite(a['cross_operation_auc']) else 0)))<1e-12
 if a['supported_operations']:
  assert abs(a['equal_operation_auc']-u.loc[u.estimable,'within_auc'].mean())<1e-12
x=pd.DataFrame({'video':['a']*4+['b']*4,'y':[0,0,1,1]*2,'score':[.1,.2,.7,.8,.15,.25,.65,.75]})
assert shift_test(x,amplitude=2)['invariance_error']<1e-12
for bad in [x.assign(y=1),x.assign(score=np.nan),x.assign(y=2)]:
 try:audit(bad);raise AssertionError('Expected rejection')
 except ValueError:pass
print('PASS 150 randomized exact decomposition cases; ties; support; shift invariance; invalid input rejection')
