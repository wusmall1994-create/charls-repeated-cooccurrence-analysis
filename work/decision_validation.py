import os
from pathlib import Path
import sys,json,ast,warnings

import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf
from scipy.stats import norm
D=Path(os.environ['CHARLS_DATA_ROOT']);R=Path.cwd();W=Path('work/decision_validation');W.mkdir(exist_ok=True)
tree=ast.parse((R/'scripts/12_run_functional_multistate_analysis.py').read_text(encoding='utf-8-sig'))
env=dict(np=np,pd=pd,sm=sm,smf=smf,warnings=warnings,norm=norm)
keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['BASE_COVARS','STATE_LABELS','FUNCTION_LABELS'] for t in n.targets)]
exec(compile(ast.Module(body=keep,type_ignores=[]),'helpers','exec'),env)
cols=['ID','ID_w1','inw1','rabyear','rafbdate','r1agey','r1iwy','r1iwm']+[f'inw{w}' for w in [2,3,4]]+[f'r{w}digeste' for w in [1,2,3,4]]
h=pd.read_stata(D/'Harmonized CHARLS/H_CHARLS_D_Data/H_CHARLS_D_Data.dta',columns=cols,convert_categoricals=False);h=h[h.inw1.eq(1)].set_index('ID')
raw11=pd.read_stata(D/'2011/household_and_community_questionnaire_data/health_status_and_functioning.dta',convert_categoricals=False).set_index('ID')
demo=pd.read_stata(D/'2011/household_and_community_questionnaire_data/demographic_background.dta',columns=['ID','ba002_1','ba002_2','ba003','ba004'],convert_categoricals=False).set_index('ID')
by=h.ID_w1.map(demo.ba002_1);bm=h.ID_w1.map(demo.ba002_2);calendar=h.ID_w1.map(demo.ba003)
# Primary verification excludes large birth-year conflicts; alternate age uses raw solar birth dates only.
conflict=by.between(1900,2011)&h.rabyear.notna()&(by-h.rabyear).abs().gt(2)
raw_age=np.floor((h.r1iwy*12+h.r1iwm-(by*12+bm.where(bm.between(1,12),6)))/12).where(calendar.eq(1)&by.between(1900,2011))
baseline_raw=h.ID_w1.map(raw11['da007_10_']).map({1:1.,2:0.})
hist={1:baseline_raw.copy()};frozen={1:baseline_raw.copy()};audit=[]
for w,rel in [(2,'2013/CHARLS2013_Dataset'),(3,'2015/CHARLS2015r'),(4,'2018/CHARLS2018r')]:
 raw=pd.read_stata(D/rel/'Health_Status_and_Functioning.dta',convert_categoricals=False).set_index('ID')
 get=lambda k:raw[k].reindex(h.index)
 pre=hist[w-1].copy() if w==2 else pd.concat([hist[k] for k in range(w-1,0,-1)],axis=1).bfill(axis=1).iloc[:,0]
 pre=pre.where(h[f'inw{w}'].eq(1));direct=get('da007_10_');cur=pd.Series(np.nan,index=h.index)
 if w in [2,3]:
  verify=get('da007_w2_1_10_');new=get('da007_w2_2_10_')
  no=(pre.eq(0)&verify.ne(2)&new.eq(2)) if w==2 else (pre.eq(0)&verify.eq(1))
  cur.loc[no|direct.eq(2)|(pre.eq(1)&verify.eq(2)&new.eq(2))]=0
  yes=(pre.eq(0)&verify.eq(2))|direct.eq(1)|new.eq(1)|(pre.eq(1)&verify.ne(2))
  if w==3:yes=yes|get('zda007_10_').eq(1)
  cur.loc[yes]=1
  # Working history reproduces official retrospective corrections, only for later origins.
  for k in range(1,w):
   old=hist[k].copy();hist[k].loc[pre.eq(0)&verify.eq(2)&old.eq(0)]=1;hist[k].loc[pre.eq(1)&verify.eq(2)&old.eq(1)]=0
 else:
  report=get('da010_w2_2_10_')
  cur.loc[pre.eq(0)|direct.eq(2)|(pre.eq(1)&report.eq(99))]=0
  cur.loc[direct.eq(1)|get('zdisease_10_').eq(1)|(pre.eq(1)&report.between(1,3))]=1
  for k in range(1,w):hist[k].loc[pre.eq(1)&report.eq(99)&hist[k].eq(1)]=0
 cur=cur.where(h[f'inw{w}'].eq(1));hist[w]=cur.copy();frozen[w]=cur.copy()
for w in [1,2,3,4]:
 target=h[f'r{w}digeste'];same=hist[w].eq(target)|(hist[w].isna()&target.isna())
 audit.append({'wave':w,'official_final_mismatch':int((~same).sum()),'frozen_vs_final_changed':int((frozen[w].notna()&target.notna()&frozen[w].ne(target)).sum())})
assert all(r['official_final_mismatch']==0 for r in audit),audit
print('Disease reconstruction independently matches final official values',audit,flush=True)
long=pd.read_csv(R/'outputs/comorbidity_state/comorbidity_state_long.csv',dtype={'ID':str})
allx=pd.read_csv(R/'outputs/comorbidity_state/functional_multistate_intervals_all.csv',dtype={'ID':str})
rawcond=pd.DataFrame({str(k):h.ID_w1.map(raw11[f'da007_{k}_']).map({1:1.,2:0.}) for k in range(1,15) if k!=10})
count=rawcond.sum(axis=1,min_count=13);missing=rawcond.isna().sum(axis=1)
newlong=long.copy()
for w in [1,2,3,4]:
 mask=newlong.wave.eq(w);newlong.loc[mask,'digestive']=newlong.loc[mask,'ID'].map(frozen[w])
newlong['state']=(newlong.digestive+2*newlong.depression).where(newlong.interviewed_alive.eq(1)&newlong.digestive.notna()&newlong.depression.notna())
lookup=newlong.set_index(['ID','wave'])
new=allx.copy();idx=pd.MultiIndex.from_frame(new[['ID','wave']]);new['digestive']=lookup.digestive.reindex(idx).to_numpy();new['state']=lookup.state.reindex(idx).to_numpy()
new=new[new.state.notna()].copy();new['other_chronic_count']=new.ID.map(count);new['chronic_missing']=new.ID.map(missing).gt(0).astype(int)
new['other_chronic_count']=new.other_chronic_count.fillna(count.median())
new['large_age_conflict']=new.ID.map(conflict);new['raw_age']=new.ID.map(raw_age)
rows=[];counts=[]
def extract(f,v,label,scenario,z):
 beta=float(v@f.params);cov=f.audit_cov;se=float(np.sqrt(max(v.to_numpy()@cov@v.to_numpy(),0)))
 rows.append({'scenario':scenario,'comparison':label,'rr':np.exp(beta),'low':np.exp(beta-1.96*se),'high':np.exp(beta+1.96*se),'p':2*norm.sf(abs(beta/se)) if se else None,'intervals':len(z),'people':z.ID.nunique(),'events':int(z.event.sum())})
def fit(z,formula):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');f=smf.glm(formula,z,family=sm.families.Poisson(),freq_weights=z.iow).fit()
 X=f.model.exog;mu=f.fittedvalues.to_numpy();w=z.iow.to_numpy();score=X*(w*(z.event.to_numpy()-mu))[:,None];sums=pd.DataFrame(score).groupby(z.ID.to_numpy()).sum().to_numpy();bread=np.linalg.pinv(X.T@((w*mu)[:,None]*X));n,k=X.shape;g=len(sums);f.audit_cov=bread@sums.T@sums@bread*g/(g-1)*(n-1)/(n-k)
 return f
base=env['BASE_COVARS']+' + chronic_missing'
scenarios=[('Harmonized_age45',allx[allx.baseline_age.ge(45)].assign(chronic_missing=0),long,False),('Frozen_age45',new[new.baseline_age.ge(45)],newlong,False),('Frozen_age45_no_large_conflict',new[new.baseline_age.ge(45)&~new.large_age_conflict],newlong,False),('Frozen_raw_solar_age45',new[new.raw_age.ge(45)].assign(baseline_age=lambda d:d.raw_age),newlong,False),('Frozen_no_income',new[new.baseline_age.ge(45)&~new.large_age_conflict],newlong,True),('Frozen_complete_chronic',new[new.baseline_age.ge(45)&~new.large_age_conflict&new.chronic_missing.eq(0)],newlong,False),('Frozen_severity_adjusted',new[new.baseline_age.ge(45)&~new.large_age_conflict],newlong,False)]
for name,a,l,noincome in scenarios:
 b=base
 if noincome:b=b.replace('log_hh_income_imp + income_missing + ','')
 a=env['prepare'](a.copy()).dropna(subset=['baseline_age'])
 if name=='Frozen_age45_no_large_conflict':a.to_pickle(W/'main_all_intervals.pkl')
 den=smf.glm('outcome_observed ~ C(state) + C(functional_state) + C(interval) + '+b,a,family=sm.families.Binomial()).fit()
 num=smf.glm('outcome_observed ~ C(functional_state) + C(interval)',a,family=sm.families.Binomial()).fit()
 a['wr']=num.predict(a).clip(.02,.995)/den.predict(a).clip(.02,.995);o=a[a.outcome_observed.eq(1)].copy();lo,hi=o.wr.quantile([.01,.99]);o['iow']=o.wr.clip(lo,hi)
 if name=='Frozen_age45_no_large_conflict':o.to_pickle(W/'main_intervals.pkl')
 for orig,dest,label in [(0,[2],'No limitation to ADL'),(1,[2],'IADL to ADL'),(1,[0],'IADL recovery'),(2,[0,1],'ADL improvement')]:
  z=o[o.functional_state.eq(orig)].copy();z['event']=z.destination_state.isin(dest).astype(int)
  if name=='Frozen_severity_adjusted':z=z[z.state.isin([2,3])].copy()
  form='event ~ C(state) + C(interval) + '+b+(' + cesd10' if name=='Frozen_severity_adjusted' else '')
  f=fit(z,form);v=pd.Series(0.,index=f.params.index)
  for t in v.index:
   if t.startswith('C(state)') and t.endswith(('[T.3.0]','[T.3]')):v[t]=1
   if t.startswith('C(state)') and t.endswith(('[T.2.0]','[T.2]')):v[t]=-1
  extract(f,v,label,name,z)
 # Frozen contemporaneous states, never retroactively replace earlier states in exposure history.
 wide=l[l.wave.isin([1,2,3,4])].pivot(index='ID',columns='wave',values='state');parts=[]
 for w in [2,3,4]:
  z=o[o.wave.eq(w)&o.functional_state.eq(1)&o.state.isin([2,3])].copy();hx=wide.reindex(z.ID)[list(range(1,w+1))];complete=hx.notna().all(axis=1).to_numpy();prior=hx.iloc[:,:-1].eq(3).any(axis=1).to_numpy();z['history']=np.where(z.state.eq(3),np.where(prior,'Repeated','First'),np.where(prior,'Prior_only','Reference'));z=z[complete].copy();parts.append(z)
 z=pd.concat(parts);z['history']=pd.Categorical(z.history,categories=['Reference','Prior_only','First','Repeated'])
 counts.extend([dict(scenario=name,group=str(g),intervals=len(q),people=q.ID.nunique(),progression=int(q.destination_state.eq(2).sum()),recovery=int(q.destination_state.eq(0).sum())) for g,q in z.groupby('history',observed=True)])
 for dest,label in [(2,'History progression'),(0,'History recovery')]:
  z['event']=z.destination_state.eq(dest).astype(int);f=fit(z,'event ~ C(history) + C(interval) + '+b+(' + cesd10' if name=='Frozen_severity_adjusted' else ''))
  for contrast in ['First','Repeated','Repeated_vs_First']:
   v=pd.Series(0.,index=f.params.index)
   if contrast=='Repeated_vs_First':v['C(history)[T.Repeated]']=1;v['C(history)[T.First]']=-1
   else:v[f'C(history)[T.{contrast}]']=1
   extract(f,v,label+' '+contrast,name,z)
 print('Finished '+name,flush=True)
pd.DataFrame(rows).to_json(W/'results.json',orient='records',indent=2,force_ascii=False)
pd.DataFrame(counts).to_json(W/'counts.json',orient='records',indent=2,force_ascii=False)
(W/'reconstruction.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False),flush=True)
