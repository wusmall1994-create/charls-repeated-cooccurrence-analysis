from pathlib import Path
import json
exec(Path('work/decision_validation.py').read_text(encoding='utf-8').split('rows=[];counts=[]')[0])
P=Path('work/acer');P.mkdir(exist_ok=True)
f=pd.read_csv(R/'outputs/comorbidity_state/functional_multistate_long.csv',dtype={'ID':str})
f=f.drop(columns=['state','state_label','digestive','depression','cesd10','digestive_dispute'],errors='ignore').merge(newlong[['ID','wave','state','digestive','depression','cesd10']],on=['ID','wave'],how='left')
hv=pd.read_stata(D/'Harmonized CHARLS/H_CHARLS_D_Data/H_CHARLS_D_Data.dta',columns=['ID','r3iwstat','r4iwstat'],convert_categoricals=False).set_index('ID')
sv=pd.read_stata(D/'2020/CHARLS2020r/Sample_Infor.dta',columns=['ID','died'],convert_categoricals=False).set_index('ID')
parts=[]
for w in [2,3,4]:
 s=f[f.wave.eq(w)].copy();e=f[f.wave.eq(w+1)].set_index('ID');v=s.ID.map(hv[f'r{w+1}iwstat']) if w<4 else s.ID.map(sv.died)
 dead=v.eq(5) if w<4 else v.eq(1);alive=v.isin([1,4]) if w<4 else v.eq(0)
 dest=s.ID.map(e.functional_state).where(alive);dest.loc[dead]=3
 s['destination_state']=dest;s['outcome_observed']=dest.notna().astype(int);s['known_dead']=dead.astype(int)
 s['interval']={2:'2013-2015',3:'2015-2018',4:'2018-2020'}[w]
 parts.append(s)
x=pd.concat(parts,ignore_index=True)
stages=[]
def record(label,z):stages.append(dict(step=label,people=int(z.ID.nunique()),intervals=len(z)))
record('All baseline cohort origin rows',x)
x=x[x.alive_interview.eq(1)&x.functional_state.notna()&x.state.notna()].copy();record('Living origin with complete functional and exposure states',x)
x=x[x.baseline_age.ge(45)].copy();record('Baseline age at least 45',x)
x['large_age_conflict']=x.ID.map(conflict);x=x[~x.large_age_conflict].copy();record('No birth year discrepancy exceeding two years',x)
x['other_chronic_count']=x.ID.map(count).fillna(count.median());x['chronic_missing']=x.ID.map(missing).gt(0).astype(int)
x['raw_age']=x.ID.map(raw_age)
old=pd.read_pickle(W/'main_all_intervals.pkl')
oldkeys=set(zip(old.ID,old.wave));newkeys=set(zip(x.ID,x.wave))
qa={'flow':stages,'added_to_previous_scaffold':len(newkeys-oldkeys),'removed_from_previous_scaffold':len(oldkeys-newkeys)}
for c in ['state','functional_state','destination_state','other_chronic_count']:
 u=x.set_index(['ID','wave'])[c];v=old.set_index(['ID','wave'])[c];common=u.index.intersection(v.index);qa[c+'_mismatch']=int((~(u.loc[common].eq(v.loc[common])|(u.loc[common].isna()&v.loc[common].isna()))).sum())
x=env['prepare'](x)
x.to_pickle(P/'all.pkl');newlong.to_pickle(P/'long.pkl')
(P/'cohort_audit.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa,indent=2))
