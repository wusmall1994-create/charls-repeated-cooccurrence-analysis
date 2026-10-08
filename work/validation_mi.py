from pathlib import Path
import sys,json,ast,warnings

import pandas as pd,numpy as np,statsmodels.api as sm,statsmodels.formula.api as smf
from scipy.stats import norm,t
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
W=Path('work/decision_validation');a=pd.read_pickle(W/'main_all_intervals.pkl')
person=a.sort_values('wave').drop_duplicates('ID').set_index('ID')
variables=['baseline_age','female','education_code','marital_code','rural_hukou','current_smoker','drank_last_year','baseline_bmi','log_hh_income','other_chronic_count']
matrix=person[variables].copy();matrix.loc[person.chronic_missing.eq(1),'other_chronic_count']=np.nan
for wave in [2,3,4]:
 sub=a[a.wave.eq(wave)].set_index('ID')
 for var in ['state','functional_state','destination_state']:
  for level in [0,1,2,3]:matrix[f'{var}_{wave}_{level}']=sub[var].eq(level).astype(float).reindex(matrix.index).fillna(0)
 matrix[f'observed_{wave}']=sub.outcome_observed.reindex(matrix.index).fillna(0)
 matrix[f'eligible_{wave}']=matrix.index.isin(sub.index).astype(float)
# Baseline categorical variables are modeled as auxiliary categories, not randomly imputed here.
for v in ['female','education_code','marital_code','rural_hukou','current_smoker','drank_last_year']:matrix[v]=matrix[v].fillna(matrix[v].mode().iloc[0])
base='baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + current_smoker + drank_last_year + binary_covariate_missing + baseline_bmi + log_hh_income + other_chronic_count'
fits={k:[] for k in ['IADL progression','IADL recovery']};diagnostics=[]
for rep in range(20):
 im=IterativeImputer(max_iter=10,sample_posterior=True,random_state=260911+rep,skip_complete=True)
 filled=pd.DataFrame(im.fit_transform(matrix),index=matrix.index,columns=matrix.columns)
 # Only genuinely missing baseline BMI, income and total disease count are replaced.
 z=a.copy()
 for v in ['baseline_bmi','log_hh_income','other_chronic_count']:
  vals=filled[v]
  if v=='baseline_bmi':vals=vals.clip(10,60)
  if v=='log_hh_income':vals=vals.clip(lower=0)
  if v=='other_chronic_count':vals=vals.clip(0,13).round()
  z[v]=z.ID.map(vals)
 den=smf.glm('outcome_observed ~ C(state) + C(functional_state) + C(interval) + '+base,z,family=sm.families.Binomial()).fit();num=smf.glm('outcome_observed ~ C(functional_state) + C(interval)',z,family=sm.families.Binomial()).fit()
 z['weight']=num.predict(z).clip(.02,.995)/den.predict(z).clip(.02,.995);z=z[z.outcome_observed.eq(1)&z.functional_state.eq(1)].copy()
 # Truncation cutoffs use all observed origins, not only IADL origins.
 obs=a.outcome_observed.eq(1);allpr=num.predict(a.assign(**{v:a.ID.map(filled[v].clip(10,60) if v=='baseline_bmi' else filled[v].clip(lower=0) if v=='log_hh_income' else filled[v].clip(0,13).round()) for v in ['baseline_bmi','log_hh_income','other_chronic_count']})).clip(.02,.995)
 # Use the weights computed before IADL restriction via a separate aligned prediction.
 allz=a.copy()
 for v in ['baseline_bmi','log_hh_income','other_chronic_count']:allz[v]=allz.ID.map(filled[v].clip(10,60) if v=='baseline_bmi' else filled[v].clip(lower=0) if v=='log_hh_income' else filled[v].clip(0,13).round())
 ww=num.predict(allz).clip(.02,.995)/den.predict(allz).clip(.02,.995);lo,hi=ww[obs].quantile([.01,.99]);z['weight']=z.weight.clip(lo,hi)
 for dest,label in [(2,'IADL progression'),(0,'IADL recovery')]:
  z['event']=z.destination_state.eq(dest).astype(int)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');f=smf.glm('event ~ C(state) + C(interval) + '+base,z,family=sm.families.Poisson(),freq_weights=z.weight).fit()
  X=f.model.exog;mu=f.fittedvalues.to_numpy();w=z.weight.to_numpy();s=X*(w*(z.event.to_numpy()-mu))[:,None];ss=pd.DataFrame(s).groupby(z.ID.to_numpy()).sum().to_numpy();b=np.linalg.pinv(X.T@((w*mu)[:,None]*X));n,k=X.shape;g=len(ss);cov=b@ss.T@ss@b*g/(g-1)*(n-1)/(n-k)
  v=np.array([1 if q=='C(state)[T.3.0]' else -1 if q=='C(state)[T.2.0]' else 0 for q in f.params.index]);fits[label].append((float(v@f.params),float(v@cov@v)))
 diagnostics.append({'replicate':rep+1,'mean_imputed_income':float(filled.log_hh_income.mean())});print('MI '+str(rep+1),flush=True)
rows=[]
for label,items in fits.items():
 q=np.array([v[0] for v in items]);u=np.array([v[1] for v in items]);between=q.var(ddof=1);total=u.mean()+1.05*between;df=19*(1+u.mean()/(1.05*between))**2 if between else 1e9;se=np.sqrt(total);crit=t.ppf(.975,df)
 rows.append({'comparison':label,'rr':float(np.exp(q.mean())),'low':float(np.exp(q.mean()-crit*se)),'high':float(np.exp(q.mean()+crit*se)),'p':float(2*t.sf(abs(q.mean()/se),df)),'m':20,'df':float(df),'intervals':len(z),'people':int(z.ID.nunique())})
(W/'mi_results.json').write_text(json.dumps({'results':rows,'diagnostics':diagnostics,'limitations':'Diagnostic MI for continuous baseline covariates and missing disease count with all observed destination-state indicators included; categorical missingness retains main modal/category handling; no imputation of missing exposure or destination, and no substantive-model-compatible guarantee.'},indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
