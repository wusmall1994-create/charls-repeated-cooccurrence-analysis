"""Supported history-composition and lagged-exposure sensitivity analyses.
Run from package root after the primary pipeline. No individual data redistribution.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,warnings
from pathlib import Path
if os.environ.get('ANALYSIS_DEPS'):sys.path.insert(0,os.environ['ANALYSIS_DEPS'])
import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf
from revision_sensitivity import BASE,weights,model
P=Path('work/acer');O=P/'enhancements'
def make_composition(o,l):
 wide=l[l.wave.isin([1,2,3,4])].pivot(index='ID',columns='wave',values='state');parts=[]
 for w in [2,3,4]:
  z=o[o.wave.eq(w)&o.functional_state.eq(1)&o.depression.eq(1)].copy();hx=wide.reindex(z.ID)[list(range(1,w+1))];keep=hx.notna().all(axis=1).to_numpy();z=z[keep].copy();hx=hx.loc[keep]
  z['cum_co']=hx.eq(3).sum(axis=1).to_numpy();z['cum_dep']=hx.isin([2,3]).sum(axis=1).to_numpy();z['event']=z.destination_state.eq(2).astype(int);parts.append(z)
 d=pd.concat(parts,ignore_index=True);d['stratum']=d.wave.astype(int).astype(str)+'_'+d.cum_dep.astype(str)
 return d

def main():
 a=pd.read_pickle(P/'all.pkl');l=pd.read_pickle(P/'long.pkl');o=pd.read_pickle(P/'observed.pkl');d=make_composition(o,l)
 counts=[];support=set()
 for (w,dep,co),z in d.groupby(['wave','cum_dep','cum_co']):
  row=dict(wave=int(w),depressive_waves=int(dep),cooccurring_waves=int(co),intervals=len(z),people=int(z.ID.nunique()),events=int(z.event.sum()))
  counts.append(row)
  if len(z)>=20 and z.event.sum()>=5 and (1-z.event).sum()>=5:support.add((w,dep,co))
 starts={x for x in support if (x[0],x[1],x[2]+1) in support}
 d['target']=[(w,de,c) in starts for w,de,c in zip(d.wave,d.cum_dep,d.cum_co)]
 formula='event ~ cum_co + C(stratum) + '+BASE
 def fit(z):
  f=smf.glm(formula,z,family=sm.families.Binomial(),freq_weights=z.iow).fit()
  assert f.converged
  t=z[z.target].copy();assert len(t)>0
  p0=f.predict(t).mean();t['cum_co']=t.cum_co+1;p1=f.predict(t).mean()
  return np.array([p0,p1,p1-p0])
 point=fit(d);ids=a.ID.unique();idx={k:np.flatnonzero(a.ID.to_numpy()==k) for k in ids};meta=d[['ID','wave','cum_co','cum_dep','event','stratum','target']]
 rng=np.random.default_rng(261009);samples=[];failures=[]
 for b in range(500):
  draw=rng.choice(ids,len(ids),replace=True);aa=a.iloc[np.concatenate([idx[k] for k in draw])].reset_index(drop=True)
  try:
   oo,_=weights(aa);z=oo.merge(meta,on=['ID','wave'],validate='many_to_one');samples.append(fit(z))
  except Exception as e:failures.append(dict(replicate=b+1,error=str(e)))
  if (b+1)%25==0:print('Composition bootstrap',b+1,'successful',len(samples),flush=True)
 assert len(samples)>=475
 ci=np.quantile(samples,[.025,.975],axis=0)
 result=dict(definition='One additional co-occurring wave replacing a depressive-only wave at fixed total depressive-symptom waves and origin wave; associational, not causal.',formula=formula,joint_cells=counts,target_start_cells=[list(map(int,x)) for x in sorted(starts)],fit_intervals=len(d),fit_people=int(d.ID.nunique()),target_intervals=int(d.target.sum()),target_people=int(d.loc[d.target,'ID'].nunique()),target_events=int(d.loc[d.target,'event'].sum()),target_waves=sorted(d.loc[d.target,'wave'].astype(int).unique().tolist()),estimates={n:dict(estimate=float(point[j]),low=float(ci[0,j]),high=float(ci[1,j])) for j,n in enumerate(['observed_composition_risk','one_more_cooccurrence_risk','risk_difference'])},bootstrap=dict(attempts=500,successful=len(samples),seed=261009,failures=failures,weights_refitted=True,support_fixed=True,target_resampled=True))
 (O/'supported_composition.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='joint_cells'},indent=2),flush=True)
 # Prior-wave exposure precedes the IADL origin; retain origin IADL limitation.
 prev=l[['ID','wave','state']].copy();prev.wave=prev.wave+1;prev=prev.rename(columns={'state':'lag_state'})
 aa=a.merge(prev,on=['ID','wave'],validate='one_to_one');aa=aa[aa.lag_state.notna()].copy()
 den=smf.glm('outcome_observed ~ C(lag_state)+C(state)+C(functional_state)+C(interval)+'+BASE,aa,family=sm.families.Binomial()).fit()
 num=smf.glm('outcome_observed ~ C(functional_state)+C(interval)',aa,family=sm.families.Binomial()).fit()
 assert den.converged and num.converged
 ww=num.predict(aa).clip(.02,.995)/den.predict(aa).clip(.02,.995);obs=aa.outcome_observed.eq(1);lo,hi=ww[obs].quantile([.01,.99]);oo=aa[obs].copy();oo['iow']=ww[obs].clip(lo,hi)
 zz=oo[oo.functional_state.eq(1)].copy();zz['state']=zz.lag_state
 lag=model(zz,'state');lag['definition']='Previous-wave co-occurrence versus previous-wave depressive symptoms only, retaining IADL-only origin function; interval and baseline covariates adjusted, current exposure not adjusted in outcome model.'
 lag['eligible_people']=int(aa.ID.nunique());lag['eligible_intervals']=len(aa);lag['weight_limits']=[float(lo),float(hi)]
 (O/'lagged_exposure.json').write_text(json.dumps(lag,indent=2),encoding='utf-8');print('Lagged',json.dumps(lag),flush=True)
if __name__=='__main__':main()
