import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,ast,json,warnings

import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf
from scipy.stats import norm
from scipy.linalg import qr
from scipy.special import expit
from patsy import build_design_matrices
P=Path('work/acer');a=pd.read_pickle(P/'all.pkl');long=pd.read_pickle(P/'long.pkl')
t=ast.parse(Path('work/decision_validation.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'helpers','exec'))
base='baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + current_smoker + drank_last_year + binary_covariate_missing + baseline_bmi_imp + bmi_missing + log_hh_income_imp + income_missing + other_chronic_count + chronic_missing'
rows=[];counts=[];diag={};history_saved=None
def weighted(a):
 den=smf.glm('outcome_observed ~ C(state) + C(functional_state) + C(interval) + '+base,a,family=sm.families.Binomial()).fit()
 num=smf.glm('outcome_observed ~ C(functional_state) + C(interval)',a,family=sm.families.Binomial()).fit()
 r=num.predict(a).clip(.02,.995)/den.predict(a).clip(.02,.995);o=a[a.outcome_observed.eq(1)].copy();lo,hi=r[o.index].quantile([.01,.99]);o['iow']=r[o.index].clip(lo,hi)
 return o,den,num,dict(den_min=float(den.predict(a).min()),den_max=float(den.predict(a).max()),p01=float(lo),p99=float(hi),mean=float(o.iow.mean()),max=float(o.iow.max()),effective_intervals=float(o.iow.sum()**2/(o.iow**2).sum()))
models=[(0,[1],'No limitation to IADL'),(0,[2],'No limitation to ADL'),(0,[3],'No limitation to death'),(1,[2],'IADL to ADL'),(1,[0],'IADL recovery'),(1,[3],'IADL to death'),(2,[0,1],'ADL improvement'),(2,[3],'ADL to death')]
saved={}
for name,aa,ll in [('Main',a,long),('Age60',a[a.baseline_age.ge(60)].copy(),long)]:
 o,den,num,d=weighted(aa);diag[name]=d
 if name=='Main':o.to_pickle(P/'observed.pkl');saved.update(o=o,den=den,num=num)
 for orig,dest,label in models:
  z=o[o.functional_state.eq(orig)].copy();z['event']=z.destination_state.isin(dest).astype(int);f=fit(z,'event ~ C(state) + C(interval) + '+base)
  for comp in [1,2,3,'3v2']:
   v=pd.Series(0.,index=f.params.index)
   for q in v.index:
    if q.startswith('C(state)'):
     k=float(q.split('[T.')[1].split(']')[0]);v[q]=(1 if k==3 else -1 if k==2 else 0) if comp=='3v2' else int(k==comp)
   extract(f,v,label+' '+str(comp),name,z)
 wide=ll[ll.wave.isin([1,2,3,4])].pivot(index='ID',columns='wave',values='state');parts=[]
 for w in [2,3,4]:
  z=o[o.wave.eq(w)&o.functional_state.eq(1)&o.state.isin([2,3])].copy();hx=wide.reindex(z.ID)[list(range(1,w+1))];complete=hx.notna().all(axis=1).to_numpy();prior=hx.iloc[:,:-1].eq(3).any(axis=1).to_numpy();z['history']=np.where(z.state.eq(3),np.where(prior,'Repeated','First'),np.where(prior,'Prior_only','Reference'));parts.append(z[complete].copy())
 z=pd.concat(parts);z['history']=pd.Categorical(z.history,categories=['Reference','Prior_only','First','Repeated'])
 if name=='Main':history_saved=z.copy();z.to_pickle(P/'history.pkl')
 counts.extend([dict(scenario=name,group=str(g),intervals=len(q),people=q.ID.nunique(),progression=int(q.destination_state.eq(2).sum()),recovery=int(q.destination_state.eq(0).sum())) for g,q in z.groupby('history',observed=True)])
 for dest,label in [(2,'History progression'),(0,'History recovery')]:
  z['event']=z.destination_state.eq(dest).astype(int);f=fit(z,'event ~ C(history) + C(interval) + '+base)
  for comp in ['Prior_only','First','Repeated','Repeated_vs_First']:
   v=pd.Series(0.,index=f.params.index)
   if comp=='Repeated_vs_First':v['C(history)[T.Repeated]']=1;v['C(history)[T.First]']=-1
   else:v[f'C(history)[T.{comp}]']=1
   extract(f,v,label+' '+comp,name,z)
 print('Finished '+name,flush=True)
pd.DataFrame(rows).to_json(P/'model_results.json',orient='records',indent=2);pd.DataFrame(counts).to_json(P/'history_counts.json',orient='records',indent=2)
# Descriptive table uses one first observed eligible interval per participant.
o=saved['o'];first=o.sort_values('wave').drop_duplicates('ID');desc=[]
for g,z in first.groupby('state'):
 r={'state':int(g),'n':len(z)}
 for v in ['baseline_age','baseline_bmi','cesd10','other_chronic_count']:
  vls=z[v].where(z.chronic_missing.eq(0)) if v=='other_chronic_count' else z[v];r[v]=[float(vls.mean()),float(vls.std()),int(vls.isna().sum())]
 for v in ['female','rural_hukou','current_smoker','drank_last_year']:r[v]=[int(z[v].eq(1).sum()),len(z)]
 for k in [0,1,2]:r['origin_'+str(k)]=int(z.functional_state.eq(k).sum())
 desc.append(r)
person=a.drop_duplicates('ID');diag['sample']={'eligible_people':a.ID.nunique(),'eligible_intervals':len(a),'observed_people':o.ID.nunique(),'observed_intervals':len(o),'deaths':int(o.destination_state.eq(3).sum()),'unknown':int(a.outcome_observed.eq(0).sum()),'bmi_missing_people':int(person.bmi_missing.sum()),'income_missing_people':int(person.income_missing.sum()),'chronic_missing_people':int(person.chronic_missing.sum())}
diag['interval_counts']=a.groupby('interval').agg(eligible=('ID','size'),observed=('outcome_observed','sum')).reset_index().to_dict('records')
diag['transition_counts']=o.groupby(['functional_state','destination_state']).size().rename('n').reset_index().to_dict('records')
(P/'descriptive.json').write_text(json.dumps(desc,indent=2));(P/'diagnostics.json').write_text(json.dumps(diag,indent=2,default=int))
# Logistic standardization supplies bounded risks; RR models above remain modified Poisson.
# Whole-participant bootstrap re-estimates observation models and outcome models.
ids=pd.Index(a.ID.unique());gid=ids.get_indexer(a.ID);ogid=ids.get_indexer(o.ID);a=a.reset_index(drop=True)
def reduce(X):
 _,rr,piv=qr(X,mode='economic',pivoting=True);rank=np.linalg.matrix_rank(rr);cols=np.sort(piv[:rank]);return np.asarray(X)[:,cols],cols
def irls(X,y,w,beta=None):
 b=np.zeros(X.shape[1]) if beta is None else beta.copy()
 for it in range(60):
  mu=expit(X@b);var=np.maximum(mu*(1-mu),1e-8);H=X.T@((w*var)[:,None]*X);score=X.T@(w*(y-mu));step=np.linalg.lstsq(H,score,rcond=1e-10)[0]
  # Step limit protects rare categories in bootstrap fits.
  step=step/max(1.,np.max(np.abs(step))/5);b=b+step
  if np.max(np.abs(step))<1e-7:return b
 if np.max(np.abs(score))>1e-3:raise RuntimeError('IRLS convergence')
 return b
Xd,cd=reduce(saved['den'].model.exog);Xn,cn=reduce(saved['num'].model.exog);y=a.outcome_observed.to_numpy();bd=irls(Xd,y,np.ones(len(a)));bn=irls(Xn,y,np.ones(len(a)));obs=y==1
specs=[]
for label,z,term,lev0,lev1,target in [('Current co-occurrence',o[o.functional_state.eq(1)].copy(),'state',2,3,lambda z:z.state.isin([2,3])),('Repeated versus first',history_saved.copy(),'history','First','Repeated',lambda z:z.history.isin(['First','Repeated']))]:
 z['event']=z.destination_state.eq(2).astype(int)
 form='event ~ C('+term+') + C(interval) + '+base
 f=smf.glm(form,z,family=sm.families.Binomial(),freq_weights=z.iow).fit();X,c=reduce(f.model.exog);tar=z[target(z)].copy();xs=[]
 for level in [lev0,lev1]:
  q=tar.copy();q[term]=level;xs.append(np.asarray(build_design_matrices([f.model.data.design_info],q)[0])[:,c])
 ix=o.index.get_indexer(z.index);assert (ix>=0).all()
 specs.append(dict(label=label,X=X,y=z.event.to_numpy(),ix=ix,g=ids.get_indexer(z.ID),tg=ids.get_indexer(tar.ID),xs=xs,b=irls(X,z.event.to_numpy(),z.iow.to_numpy()),target_n=len(tar)))
def calc(mult,start=False):
 d=bd if start else irls(Xd,y,mult[gid],bd);n=bn if start else irls(Xn,y,mult[gid],bn)
 wr=np.clip(expit(Xn@n),.02,.995)/np.clip(expit(Xd@d),.02,.995);ow=wr[obs];lo,hi=np.quantile(np.repeat(ow,mult[ogid]),[.01,.99]);ow=np.clip(ow,lo,hi);out=[]
 for s in specs:
  b=s['b'] if start else irls(s['X'],s['y'],ow[s['ix']]*mult[s['g']],s['b'])
  risks=[np.average(expit(x@b),weights=mult[s['tg']]) for x in s['xs']];out.append([risks[0],risks[1],risks[1]-risks[0]])
 return out
point=calc(np.ones(len(ids),dtype=int),True);rng=np.random.default_rng(20260912);reps=[];fail=[]
for b in range(500):
 m=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
 try:reps.append(calc(m))
 except Exception as e:fail.append(str(e))
 if (b+1)%25==0:print('Bootstrap '+str(b+1)+' successful '+str(len(reps)),flush=True)
arr=np.array(reps);out=[]
for k,s in enumerate(specs):
 out.append(dict(comparison=s['label'],target_intervals=s['target_n'],reference_risk=point[k][0],exposed_risk=point[k][1],risk_difference=point[k][2],ci=np.quantile(arr[:,k,:],[.025,.975],axis=0).tolist()))
(P/'absolute_risks.json').write_text(json.dumps({'results':out,'successful':len(reps),'attempted':500,'failures':fail},indent=2))
print(json.dumps(out,indent=2),flush=True)
