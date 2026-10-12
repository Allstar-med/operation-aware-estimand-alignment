import numpy as np,pandas as pd
from oea import audit,shift_test
x=pd.DataFrame({'video':['a']*4+['b']*4,'y':[0,1,0,1]*2,'score':[.1,.8,.4,.7,.2,.6,.3,.9]})
a,u,c=audit(x); assert a['supported_operations']==2
assert abs(a['pooled_auc']-c.concordance.sum()/c.pairs.sum())<1e-12
s=shift_test(x,amplitude=2);assert s['invariance_error']<1e-12
print('PASS: exact decomposition and within-operation invariance')
