from pathlib import Path
import json, warnings, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
P=Path('work/acer'); O=P/'enhancements'; O.mkdir(exist_ok=True)
o=pd.read_pickle(P/'observed.pkl'); long=pd.read_pickle(P/'long.pkl')
wide=long[long.wave.isin([1,2,3,4])].pivot(index='ID',columns='wave',values='state')
parts=[]
for wv in [2,3,4]:
 z=o[(o.wave==wv)&(o.depression==1)&(o.functional_state.isin([0,1]))].copy()
 hx=wide.reindex(z.ID)[list(range(1,wv+1))]; keep=hx.notna().all(axis=1).to_numpy();z=z[keep].copy();hx=hx.loc[keep]
 z['cum_co']=hx.eq(3).sum(axis=1).clip(upper=3).to_numpy();parts.append(z)
d=pd.concat(parts,ignore_index=True);d['event']=d.destination_state.eq(2).astype(int)
base_formula='0 + baseline_age_c + female + cesd10 + C(functional_state) + other_chronic_count + C(interval)'
ext_formula=base_formula+' + cum_co'

def design(formula):
 x=dmatrix(formula,d,return_type='dataframe');return x,x.design_info
X0,di0=design(base_formula);X1,di1=design(ext_formula)
y=d.event.to_numpy();groups=d.ID.to_numpy();w=d.iow.to_numpy()
repeats=20;pred0=np.zeros(len(d));pred1=np.zeros(len(d));counts=np.zeros(len(d))
for seed in range(20260920,20260920+repeats):
 cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
 for tr,te in cv.split(X0,y,groups):
  for X,pred in [(X0,pred0),(X1,pred1)]:
   scaler=StandardScaler().fit(np.asarray(X.iloc[tr])); xt=scaler.transform(np.asarray(X.iloc[tr])); xv=scaler.transform(np.asarray(X.iloc[te]))
   model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,fit_intercept=True)
   with warnings.catch_warnings():warnings.simplefilter('ignore');model.fit(xt,y[tr],sample_weight=w[tr])
   pred[te]+=model.predict_proba(xv)[:,1]
  counts[te]+=1
assert np.all(counts==repeats);pred0/=counts;pred1/=counts

def metrics(y,p,w):
 auc=roc_auc_score(y,p,sample_weight=w);brier=np.average((y-p)**2,weights=w)
 lp=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
 fit=sm.GLM(y,sm.add_constant(lp),family=sm.families.Binomial(),freq_weights=w).fit()
 # calibration-in-the-large with slope fixed at 1
 off=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=lp,freq_weights=w).fit()
 return {'auc':float(auc),'brier':float(brier),'calibration_intercept':float(off.params[0]),'calibration_slope':float(fit.params[1])}
m0=metrics(y,pred0,w);m1=metrics(y,pred1,w)
point={'delta_auc':m1['auc']-m0['auc'],'delta_brier':m1['brier']-m0['brier']}
ids=pd.Index(pd.unique(groups)); loc={x:np.flatnonzero(groups==x) for x in ids};rng=np.random.default_rng(20260920);boots=[]
for b in range(1000):
 samp=rng.choice(ids,size=len(ids),replace=True);ii=np.concatenate([loc[x] for x in samp])
 try:
  a0=roc_auc_score(y[ii],pred0[ii],sample_weight=w[ii]);a1=roc_auc_score(y[ii],pred1[ii],sample_weight=w[ii])
  bs0=np.average((y[ii]-pred0[ii])**2,weights=w[ii]);bs1=np.average((y[ii]-pred1[ii])**2,weights=w[ii])
  boots.append((a0,a1,a1-a0,bs0,bs1,bs1-bs0))
 except ValueError:pass
arr=np.asarray(boots); names=['base_auc','extended_auc','delta_auc','base_brier','extended_brier','delta_brier']
ci={n:[float(x) for x in np.quantile(arr[:,j],[.025,.975])] for j,n in enumerate(names)}
# Survivor-only metric sensitivity retains exactly the same prespecified out-of-fold predictions.
surv=d.destination_state.ne(3).to_numpy();ms0=metrics(y[surv],pred0[surv],w[surv]);ms1=metrics(y[surv],pred1[surv],w[surv])
out={'population':{'intervals':len(d),'people':int(d.ID.nunique()),'events':int(y.sum()),'deaths':int(d.destination_state.eq(3).sum()),'cumulative_counts':{str(k):int(v) for k,v in d.cum_co.value_counts().sort_index().items()}},'validation':{'repeated_grouped_folds':repeats,'folds':5,'bootstrap_samples':len(arr),'grouping':'participant'},'base_model':m0,'extended_model':m1,'increment':point,'confidence_intervals':ci,'survivor_sensitivity':{'intervals':int(surv.sum()),'base':ms0,'extended':ms1,'delta_auc':ms1['auc']-ms0['auc'],'delta_brier':ms1['brier']-ms0['brier']},'definition':{'outcome':'ADL limitation at next wave among origins without ADL limitation','death_primary':'non-event competing destination','base_predictors':['baseline age','sex','continuous CES-D 10 score','origin functional state (none or IADL only)','other chronic condition count','survey interval'],'added_predictor':'cumulative co-occurrence waves capped at 3'}}
(O/'prediction_increment.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))



