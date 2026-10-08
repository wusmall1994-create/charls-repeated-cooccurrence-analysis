from pathlib import Path
import sys,ast,json,warnings

import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf
from scipy.stats import norm
P=Path('work/acer');o=pd.read_pickle(P/'observed.pkl');h=pd.read_pickle(P/'history.pkl')
t=ast.parse(Path('work/decision_validation.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'helpers','exec'))
rows=[]
for orig,dest,label in [(0,[1],'No limitation to IADL'),(0,[2],'No limitation to ADL'),(0,[3],'No limitation to death'),(1,[2],'IADL to ADL'),(1,[0],'IADL recovery'),(1,[3],'IADL to death'),(2,[0,1],'ADL improvement'),(2,[3],'ADL to death')]:
 z=o[o.functional_state.eq(orig)].copy();z['event']=z.destination_state.isin(dest).astype(int);z['iow']=1.;f=fit(z,'event ~ C(state)')
 v=pd.Series(0.,index=f.params.index);v['C(state)[T.3.0]']=1;v['C(state)[T.2.0]']=-1;extract(f,v,label,'Unadjusted',z)
for dest,label in [(2,'History progression'),(0,'History recovery')]:
 z=h.copy();z['event']=z.destination_state.eq(dest).astype(int);z['iow']=1.;f=fit(z,'event ~ C(history)');v=pd.Series(0.,index=f.params.index);v['C(history)[T.Repeated]']=1;v['C(history)[T.First]']=-1;extract(f,v,label,'Unadjusted',z)
(P/'unadjusted_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
