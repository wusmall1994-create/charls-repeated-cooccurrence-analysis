import os,sys,json
from pathlib import Path
if os.environ.get('ANALYSIS_DEPS'):sys.path.insert(0,os.environ['ANALYSIS_DEPS'])
import numpy as np,pandas as pd
P=Path('work/acer');T=P/'revision2_private';T.mkdir(exist_ok=True)
a=pd.read_pickle(P/'all.pkl');l=pd.read_pickle(P/'long.pkl');h=pd.read_pickle(P/'history.pkl')
D=Path(os.environ['CHARLS_DATA_ROOT'])
cols=['ID','ID_w1','ragender','raeducl','r1mstat','r1rural2','r1smoken','r1drinkl']
r=pd.read_stata(D/'Harmonized CHARLS/H_CHARLS_D_Data/H_CHARLS_D_Data.dta',columns=cols,convert_categoricals=False).set_index('ID')
rc=[f'da007_{k}_' for k in range(1,15) if k!=10]
raw=pd.read_stata(D/'2011/household_and_community_questionnaire_data/health_status_and_functioning.dta',columns=['ID']+rc,convert_categoricals=False).set_index('ID')
p=a.sort_values('wave').drop_duplicates('ID').set_index('ID')
m=p[['baseline_age','baseline_bmi','log_hh_income']].copy()
m['female']=p.index.to_series().map(r.ragender).map({1:0.,2:1.})
m['education']=p.index.to_series().map(r.raeducl)
m['marital']=p.index.to_series().map(r.r1mstat)
for v,col in [('hukou','r1rural2'),('smoking','r1smoken'),('drinking','r1drinkl')]:m[v]=p.index.to_series().map(r[col]).where(lambda x:x.isin([0,1]))
for k in range(1,15):
 if k!=10:m['condition_'+str(k)]=p.index.to_series().map(r.ID_w1).map(raw[f'da007_{k}_']).map({1:1.,2:0.})
basecols=list(m.columns);cc=m.notna().all(axis=1);assert cc.sum()==6478
m['other_chronic_count']=m.filter(like='condition_').sum(axis=1,min_count=13)
rows=[]
def describe(v,label,categorical=False):
 x=m[v] if v in m else p[v]
 levels=sorted(x.dropna().unique()) if categorical else [None]
 for level in levels:
  vals=[]
  for complete in [True,False]:
   q=x[cc.eq(complete)].dropna()
   vals.append(dict(n=len(q),mean=float(q.eq(level).mean() if categorical else q.mean()),sd=float(q.eq(level).std(ddof=1) if categorical else q.std(ddof=1))))
  sd=np.sqrt((vals[0]['sd']**2+vals[1]['sd']**2)/2)
  rows.append(dict(variable=label+(' = '+str(level) if categorical else ''),categorical=categorical,complete=vals[0],incomplete=vals[1],smd=float(abs(vals[0]['mean']-vals[1]['mean'])/sd) if sd>0 else 0))
for v,label,cat in [('baseline_age','Age',False),('female','Female',True),('education','Education',True),('marital','Marital status',True),('hukou','Rural hukou',True),('smoking','Current smoking',True),('drinking','Alcohol use',True),('baseline_bmi','BMI',False),('log_hh_income','Log household income',False),('other_chronic_count','Other-condition count',False),('state','First eligible exposure state',True),('functional_state','First eligible functional state',True)]:describe(v,label,cat)
intervals=[]
for complete in [True,False]:
 z=a[a.ID.map(cc).eq(complete)];o=z[z.outcome_observed.eq(1)];i=o[o.functional_state.eq(1)]
 intervals.append(dict(complete=complete,people=z.ID.nunique(),intervals=len(z),observed=len(o),iadl_intervals=len(i),iadl_events=int(i.destination_state.eq(2).sum())))
(P/'enhancements/missingness_comparison.json').write_text(json.dumps(dict(participants=len(p),complete=int(cc.sum()),incomplete=int((~cc).sum()),variables=rows,intervals=intervals),indent=2),encoding='utf-8')
m=m.drop(columns='other_chronic_count')
# Fully observed auxiliary indicators; structural non-observation distinguished explicitly.
for w in [1,2,3,4]:
 z=l[l.wave.eq(w)].set_index('ID')
 for k in [1,2,3]:m[f'exposure_{w}_{k}']=z.state.eq(k).reindex(m.index).fillna(False).astype(int)
 m[f'exposure_known_{w}']=z.state.notna().reindex(m.index).fillna(False).astype(int)
for w in [2,3,4]:
 z=a[a.wave.eq(w)].set_index('ID');hh=h[h.wave.eq(w)].set_index('ID')
 m[f'eligible_{w}']=m.index.isin(z.index).astype(int)
 m[f'observed_{w}']=z.outcome_observed.reindex(m.index).fillna(0)
 for k in [1,2]:m[f'origin_{w}_{k}']=z.functional_state.eq(k).reindex(m.index).fillna(False).astype(int)
 for k in [1,2,3]:m[f'destination_{w}_{k}']=z.destination_state.eq(k).reindex(m.index).fillna(False).astype(int)
 for k in ['Prior_only','First','Repeated']:m[f'history_{w}_{k}']=hh.history.eq(k).reindex(m.index).fillna(False).astype(int)
m.reset_index().to_csv(T/'mi_input.csv',index=False)
(T/'mi_columns.json').write_text(json.dumps(basecols),encoding='utf-8')
print('Missingness comparison saved. MI matrix',m.shape,'missing',m[basecols].isna().sum().to_dict(),flush=True)
