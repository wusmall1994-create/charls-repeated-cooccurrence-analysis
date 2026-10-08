"""Pool expanded item-level MICE sensitivity analyses for current and history contrasts."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json
from pathlib import Path
if os.environ.get('ANALYSIS_DEPS'):sys.path.insert(0,os.environ['ANALYSIS_DEPS'])
import numpy as np,pandas as pd
from scipy.stats import t
import revision_sensitivity as rs
P=Path('work/acer');T=P/'revision2_private';a=pd.read_pickle(P/'all.pkl');h=pd.read_pickle(P/'history.pkl')
original=pd.read_csv(T/'mi_input.csv',dtype={'ID':str}).set_index('ID')
rs.BASE='baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + current_smoker + drank_last_year + baseline_bmi_imp + log_hh_income_imp + other_chronic_count'
results={'Current co-occurrence':[],'Repeated versus first observed co-occurrence':[]};checks=[]
for rep in range(1,21):
 f=pd.read_csv(T/f'completed_{rep:02d}.csv',dtype={'ID':str}).set_index('ID');assert f.notna().all().all()
 for v in f.columns:
  known=original[v].notna();assert np.allclose(f.loc[known,v],original.loc[known,v]),v
 for v in [c for c in f if c.startswith('condition_')]+['hukou','smoking','drinking']:assert f[v].isin([0,1]).all(),v
 z=a.copy()
 mapping={'female':'female','education_cat':'education','marital_cat':'marital','rural_hukou':'hukou','current_smoker':'smoking','drank_last_year':'drinking','baseline_bmi_imp':'baseline_bmi','log_hh_income_imp':'log_hh_income'}
 for dst,src in mapping.items():
  z[dst]=z.ID.map(f[src])
  if dst.endswith('_cat'):z[dst]=z[dst].astype(float).astype(str)
 z['other_chronic_count']=z.ID.map(f.filter(like='condition_').sum(axis=1))
 o,flow=rs.weights(z)
 for label,term,zz in [('Current co-occurrence','state',o[o.functional_state.eq(1)]),('Repeated versus first observed co-occurrence','history',rs.history(o,h))]:
  r=rs.model(zz,term);r['imputation']=rep;results[label].append(r)
 checks.append(dict(imputation=rep,**flow));print('Pooled analysis fit',rep,flush=True)
pooled=[]
for label,rr in results.items():
 q=np.log([r['rr'] for r in rr]);u=((np.log([r['high'] for r in rr])-np.log([r['low'] for r in rr]))/3.92)**2
 m=len(q);B=q.var(ddof=1);U=u.mean();V=U+(1+1/m)*B;lam=(1+1/m)*B/V;df=(m-1)/lam**2 if lam else 1e12
 se=np.sqrt(V);crit=t.ppf(.975,df)
 pooled.append(dict(comparison=label,rr=float(np.exp(q.mean())),low=float(np.exp(q.mean()-crit*se)),high=float(np.exp(q.mean()+crit*se)),p=float(2*t.sf(abs(q.mean()/se),df)),m=m,within_variance=float(U),between_variance=float(B),total_variance=float(V),df=float(df),missing_variance_fraction=float(lam),mcse_log_rr=float(np.sqrt(B/m)),mcse_to_se=float(np.sqrt(B/m)/se),people=rr[0]['model']['people'],intervals=rr[0]['model']['intervals'],events=rr[0]['model']['events']))
out=dict(results=pooled,per_imputation=results,weight_checks=checks,observed_values_preserved=True,binary_values_valid=True,notes=['Baseline disease count recomputed from individually imputed constituent items.','Missingness indicators omitted from completed-data models.','Observation weights refitted and truncated per imputation.','No missing exposure, functional outcome or history was imputed.','Rubin pooling of log RRs with participant-clustered within-imputation variances; observation weights treated as fixed within each RR fit.','Missing-at-random assumption conditional on imputation predictors remains untestable.'])
(P/'enhancements/expanded_mi_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(pooled,indent=2))
