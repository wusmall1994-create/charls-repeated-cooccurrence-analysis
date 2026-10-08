from pathlib import Path
import sys,json,warnings
import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf
from scipy.stats import norm,chi2
P=Path('work/acer');O=P/'enhancements';O.mkdir(exist_ok=True)
o=pd.read_pickle(P/'observed.pkl');long=pd.read_pickle(P/'long.pkl')
base=('baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + current_smoker + '
      'drank_last_year + binary_covariate_missing + baseline_bmi_imp + bmi_missing + log_hh_income_imp + '
      'income_missing + other_chronic_count + chronic_missing')
def fit(z,formula):
 with warnings.catch_warnings():warnings.simplefilter('ignore');f=smf.glm(formula,z,family=sm.families.Poisson(),freq_weights=z.iow).fit()
 X=f.model.exog;mu=f.fittedvalues.to_numpy();w=z.iow.to_numpy();score=X*(w*(z.event.to_numpy()-mu))[:,None]
 sums=pd.DataFrame(score).groupby(z.ID.to_numpy()).sum().to_numpy();bread=np.linalg.pinv(X.T@((w*mu)[:,None]*X));n,k=X.shape;g=len(sums)
 f.cv=pd.DataFrame(bread@sums.T@sums@bread*g/(g-1)*(n-1)/(n-k),index=f.params.index,columns=f.params.index);return f
def est(f,name):
 b=f.params[name];se=np.sqrt(f.cv.loc[name,name]);return {'rr':float(np.exp(b)),'low':float(np.exp(b-1.96*se)),'high':float(np.exp(b+1.96*se)),'p':float(2*norm.sf(abs(b/se)))}
wide=long[long.wave.isin([1,2,3,4])].pivot(index='ID',columns='wave',values='state');parts=[]
for wv in [2,3,4]:
 z=o[(o.wave==wv)&(o.functional_state==1)&(o.depression==1)].copy();hx=wide.reindex(z.ID)[list(range(1,wv+1))];keep=hx.notna().all(axis=1).to_numpy();z=z[keep].copy();hx=hx.loc[keep]
 z['cum_co']=hx.eq(3).sum(axis=1).to_numpy();z['cum_dep']=hx.isin([2,3]).sum(axis=1).to_numpy();z['current_co']=z.state.eq(3).astype(int);z['prior_co']=z.cum_co-z.current_co;z['opportunities']=wv;z['cum_prop25']=z.cum_co/wv*4;z['event']=z.destination_state.eq(2).astype(int);z['never_digestive']=~hx.isin([1,3]).any(axis=1).to_numpy();parts.append(z)
d=pd.concat(parts,ignore_index=True)
strat=[]
for interval,z in d.groupby('interval'):
 f=fit(z,'event ~ cum_co + '+base);r=est(f,'cum_co');r.update(interval=interval,intervals=len(z),people=z.ID.nunique(),events=int(z.event.sum()),max_opportunities=int(z.opportunities.iloc[0]));strat.append(r)
f=fit(d,'event ~ cum_co*C(interval) + '+base);names=[x for x in f.params.index if x.startswith('cum_co:C(interval)')]
b=f.params[names].to_numpy();v=f.cv.loc[names,names].to_numpy();wald=float(b@np.linalg.pinv(v)@b);interaction={'wald':wald,'df':len(names),'p':float(chi2.sf(wald,len(names))),'terms':{n:est(f,n) for n in names}}
fp=fit(d,'event ~ cum_prop25 + C(interval) + '+base);proportion=est(fp,'cum_prop25')
fc=fit(d,'event ~ prior_co + current_co + C(interval) + '+base);composition={'per_prior_wave':est(fc,'prior_co'),'current_cooccurrence':est(fc,'current_co')}
fj=fit(d,'event ~ cum_dep + cum_co + C(interval) + '+base);joint={'cumulative_depression_adjusted_for_cooccurrence':est(fj,'cum_dep'),'cumulative_cooccurrence_adjusted_for_depression':est(fj,'cum_co')}
nd=d[d.never_digestive].copy();fn=fit(nd,'event ~ cum_dep + C(interval) + '+base);never=est(fn,'cum_dep');never.update(intervals=len(nd),people=nd.ID.nunique(),events=int(nd.event.sum()),counts={str(k):int(v) for k,v in nd.cum_dep.value_counts().sort_index().items()})
counts=[]
for (inter,c),z in d.groupby(['interval','cum_co']):counts.append({'interval':inter,'cum_co':int(c),'intervals':len(z),'people':z.ID.nunique(),'events':int(z.event.sum())})
out={'wave_stratified_trends':strat,'count_by_interval':counts,'count_by_interval_interaction':interaction,'proportion_sensitivity_per_25pp':proportion,'separate_prior_and_current':composition,'joint_depression_cooccurrence':joint,'never_digestive_depression_trend':never,'definitions':{'cum_co':'number of co-occurring waves through and including origin','cum_dep':'number of waves with elevated depressive symptoms through and including origin','prior_co':'cum_co excluding origin','current_co':'co-occurrence at origin','never_digestive':'no digestive disease report in any wave through origin'}}
(O/'cumulative_bias_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
