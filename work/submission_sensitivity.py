"""Additional sensitivity analyses; run from the reproduction package root."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys, json, ast, warnings
from pathlib import Path
if os.environ.get('ANALYSIS_DEPS'): sys.path.insert(0,os.environ['ANALYSIS_DEPS'])
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
import sklearn
from scipy.stats import norm
from scipy.special import expit
from scipy.linalg import qr
from patsy import dmatrix,build_design_matrices
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
P=Path('work/acer'); O=P/'enhancements'; O.mkdir(exist_ok=True)
a=pd.read_pickle(P/'all.pkl').reset_index(drop=True)
hist=pd.read_pickle(P/'history.pkl')
base='baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + current_smoker + drank_last_year + binary_covariate_missing + baseline_bmi_imp + bmi_missing + log_hh_income_imp + income_missing + other_chronic_count + chronic_missing'
def weighted(d):
 den=smf.glm('outcome_observed ~ C(state)+C(functional_state)+C(interval)+'+base,d,family=sm.families.Binomial()).fit()
 num=smf.glm('outcome_observed ~ C(functional_state)+C(interval)',d,family=sm.families.Binomial()).fit()
 r=num.predict(d).clip(.02,.995)/den.predict(d).clip(.02,.995)
 z=d[d.outcome_observed.eq(1)].copy();lo,hi=r.loc[z.index].quantile([.01,.99]);z['iow']=r.loc[z.index].clip(lo,hi)
 return z,den,num
def estimate(z,term,comparison):
 z=z.copy();z['event']=z.destination_state.eq(2).astype(int)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');f=smf.glm('event ~ C('+term+')+C(interval)+'+base,z,family=sm.families.Poisson(),freq_weights=z.iow).fit()
 X=f.model.exog;mu=np.asarray(f.fittedvalues);w=z.iow.to_numpy();scores=X*(w*(z.event.to_numpy()-mu))[:,None]
 sums=pd.DataFrame(scores).groupby(z.ID.to_numpy()).sum().to_numpy();bread=np.linalg.pinv(X.T@((w*mu)[:,None]*X));n,k=X.shape;g=len(sums)
 cov=bread@sums.T@sums@bread*g/(g-1)*(n-1)/(n-k)
 v=np.zeros(k)
 for j,t in enumerate(f.params.index):
  if term=='state':
   if t.startswith('C(state)'):v[j]=1 if t.endswith(('[T.3.0]','[T.3]')) else -1 if t.endswith(('[T.2.0]','[T.2]')) else 0
  else:
   if t=='C(history)[T.Repeated]':v[j]=1
   if t=='C(history)[T.First]':v[j]=-1
 b=float(v@f.params);se=float(np.sqrt(v@cov@v))
 return dict(comparison=comparison,intervals=len(z),people=int(z.ID.nunique()),events=int(z.event.sum()),deaths=int(z.destination_state.eq(3).sum()),rr=float(np.exp(b)),low=float(np.exp(b-1.96*se)),high=float(np.exp(b+1.96*se)))
def hsubset(o):
 z=o.merge(hist[['ID','wave','history']],on=['ID','wave'],how='inner',validate='one_to_one')
 z['history']=pd.Categorical(z.history,categories=['Reference','Prior_only','First','Repeated']);return z
o,den,num=weighted(a)
saved=pd.read_pickle(P/'observed.pkl');assert np.allclose(o.iow.to_numpy(),saved.iow.to_numpy())
rows=[]
for label,z in [('Original count',o),('Survivors only',o[o.destination_state.ne(3)])]:
 for term,zz,comp in [('state',z[z.functional_state.eq(1)],'Current co-occurrence'),('history',hsubset(z),'Repeated versus first')]:
  r=estimate(zz,term,comp);r['scenario']=label;rows.append(r)
baseline=json.loads((P/'model_results.json').read_text())
for r,key in zip(rows[:2],['IADL to ADL 3v2','History progression Repeated_vs_First']):
 old=next(x for x in baseline if x['scenario']=='Main' and x['comparison']==key);assert abs(r['rr']-old['rr'])<1e-7
D=Path(os.environ['CHARLS_DATA_ROOT'])
h=pd.read_stata(D/'Harmonized CHARLS/H_CHARLS_D_Data/H_CHARLS_D_Data.dta',columns=['ID','ID_w1','inw1'],convert_categoricals=False);h=h[h.inw1.eq(1)].set_index('ID')
codes=[k for k in range(1,15) if k not in [10,11,12]]
raw=pd.read_stata(D/'2011/household_and_community_questionnaire_data/health_status_and_functioning.dta',columns=['ID']+[f'da007_{k}_' for k in codes],convert_categoricals=False).set_index('ID')
items=pd.DataFrame({k:h.ID_w1.map(raw[f'da007_{k}_']).map({1:1.,2:0.}) for k in codes})
count=items.sum(axis=1,min_count=11);aa=a.copy();aa['other_chronic_count']=aa.ID.map(count).fillna(count.median());aa['chronic_missing']=aa.ID.map(count).isna().astype(int)
oo,_,_=weighted(aa);assert oo[['ID','wave']].equals(o[['ID','wave']])
for term,z,comp in [('state',oo[oo.functional_state.eq(1)],'Current co-occurrence'),('history',hsubset(oo),'Repeated versus first')]:
 r=estimate(z,term,comp);r['scenario']='11-item count excluding psychiatric and memory-related disease';rows.append(r)
print(json.dumps(rows,indent=2),flush=True)
# IADL-origin multinomial model; empirical symptomatic target matches Table 4.
z=o[o.functional_state.eq(1)].copy();target=z[z.state.isin([2,3])].copy()
X=dmatrix('C(state)+C(interval)+'+base,z,return_type='dataframe');design=X.design_info
scaler=StandardScaler().fit(np.asarray(X));Xs=scaler.transform(np.asarray(X))
xp=[]
for st in [2,3]:
 q=target.copy();q['state']=st;xp.append(scaler.transform(np.asarray(build_design_matrices([design],q)[0])))
def multi(weights,target_weights):
 model=LogisticRegression(C=1e4,solver='lbfgs',max_iter=10000,fit_intercept=True,tol=1e-7)
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning);model.fit(Xs,z.destination_state.astype(int),sample_weight=weights)
 assert np.array_equal(model.classes_,[0,1,2,3])
 return np.array([np.average(model.predict_proba(x),axis=0,weights=target_weights) for x in xp])
point=multi(z.iow.to_numpy(),np.ones(len(target)))
def reduce(X):
 _,rr,piv=qr(X,mode='economic',pivoting=True);cols=np.sort(piv[:np.linalg.matrix_rank(rr)]);return np.asarray(X)[:,cols]
def irls(X,y,w,b):
 b=b.copy()
 for it in range(80):
  mu=expit(X@b);v=np.maximum(mu*(1-mu),1e-8);score=X.T@(w*(y-mu));H=X.T@((w*v)[:,None]*X);step=np.linalg.lstsq(H,score,rcond=1e-10)[0];step/=max(1.,np.max(np.abs(step))/5);b+=step
  if np.max(np.abs(step))<1e-7:return b
 if np.max(np.abs(score))>1e-3:raise RuntimeError('Weight model did not converge')
 return b
Xd=reduce(den.model.exog);Xn=reduce(num.model.exog);y=a.outcome_observed.to_numpy();bd=irls(Xd,y,np.ones(len(a)),np.zeros(Xd.shape[1]));bn=irls(Xn,y,np.ones(len(a)),np.zeros(Xn.shape[1]))
ids=pd.Index(a.ID.unique());ga=ids.get_indexer(a.ID);go=ids.get_indexer(o.ID);gz=ids.get_indexer(z.ID);gt=ids.get_indexer(target.ID)
obs=y==1;zi=o.index.get_indexer(z.index);rng=np.random.default_rng(20261008);reps=[];fail=[]
for b in range(500):
 mult=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
 try:
  bb=irls(Xd,y,mult[ga],bd);nn=irls(Xn,y,mult[ga],bn)
  w=(np.clip(expit(Xn@nn),.02,.995)/np.clip(expit(Xd@bb),.02,.995))[obs]
  lo,hi=np.quantile(np.repeat(w,mult[go]),[.01,.99]);w=np.clip(w,lo,hi)
  reps.append(multi(w[zi]*mult[gz],mult[gt]))
 except Exception as e:fail.append(str(e))
 if (b+1)%25==0:print('Bootstrap',b+1,'successful',len(reps),flush=True)
arr=np.array(reps);dest=[]
assert len(arr)>=475,fail
for j in range(4):
 dest.append(dict(destination=j,reference_risk=float(point[0,j]),co_risk=float(point[1,j]),difference=float(point[1,j]-point[0,j]),reference_ci=np.quantile(arr[:,0,j],[.025,.975]).tolist(),co_ci=np.quantile(arr[:,1,j],[.025,.975]).tolist(),difference_ci=np.quantile(arr[:,1,j]-arr[:,0,j],[.025,.975]).tolist()))
out=dict(sensitivity=rows,multinomial=dest,multinomial_fit_intervals=len(z),multinomial_target_intervals=len(target),target_deaths=int(target.destination_state.eq(3).sum()),attempted=500,successful=len(arr),failures=fail,versions={'Python':sys.version.split()[0],'pandas':pd.__version__,'numpy':np.__version__,'statsmodels':sm.__version__,'scikit-learn':sklearn.__version__},reduced_count_median=float(count.median()),notes='500 whole-participant draws from eligible cohort; observation models and truncation refitted, scaling and missing-covariate substitutions fixed; unweighted empirical symptomatic target.')
(O/'submission_sensitivity.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)

