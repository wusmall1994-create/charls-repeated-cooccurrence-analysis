from pathlib import Path
import sys, json, warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from patsy import dmatrix, build_design_matrices
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Reuse the independently verified raw-data reconstruction before its model loop.
src = Path('work/decision_validation.py').read_text(encoding='utf-8')
exec(src.split('rows=[];counts=[]')[0])
P=Path('work/acer'); O=P/'enhancements'; O.mkdir(exist_ok=True)
a=pd.read_pickle(P/'all.pkl'); o=pd.read_pickle(P/'observed.pkl'); long=pd.read_pickle(P/'long.pkl')
base=('baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + '
      'current_smoker + drank_last_year + binary_covariate_missing + baseline_bmi_imp + '
      'bmi_missing + log_hh_income_imp + income_missing + other_chronic_count + chronic_missing')

def fit_cluster(z, formula, weight='iow'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f=smf.glm(formula,z,family=sm.families.Poisson(),freq_weights=z[weight]).fit()
    X=f.model.exog; mu=f.fittedvalues.to_numpy(); w=z[weight].to_numpy()
    score=X*(w*(z.event.to_numpy()-mu))[:,None]
    sums=pd.DataFrame(score).groupby(z.ID.to_numpy()).sum().to_numpy()
    bread=np.linalg.pinv(X.T@((w*mu)[:,None]*X)); n,k=X.shape; g=len(sums)
    f.audit_cov=pd.DataFrame(bread@sums.T@sums@bread*g/(g-1)*(n-1)/(n-k),index=f.params.index,columns=f.params.index)
    return f

def contrast(f,v):
    b=float(v@f.params); se=float(np.sqrt(max(v.to_numpy()@f.audit_cov.to_numpy()@v.to_numpy(),0)))
    return {'rr':float(np.exp(b)),'low':float(np.exp(b-1.96*se)),'high':float(np.exp(b+1.96*se)),
            'p':float(2*norm.sf(abs(b/se)))}

# Cumulative co-occurrence waves, using current and prior observations only.
wide=long[long.wave.isin([1,2,3,4])].pivot(index='ID',columns='wave',values='state'); parts=[]
for wv in [2,3,4]:
    z=o[o.wave.eq(wv)&o.functional_state.eq(1)&o.depression.eq(1)].copy()
    hx=wide.reindex(z.ID)[list(range(1,wv+1))]; complete=hx.notna().all(axis=1).to_numpy()
    z=z[complete].copy(); hx=hx.loc[complete]
    z['cum_co']=hx.eq(3).sum(axis=1).to_numpy()
    z['cum_cat']=pd.Categorical(np.where(z.cum_co>=3,'3+',z.cum_co.astype(int).astype(str)),categories=['0','1','2','3+'])
    parts.append(z)
dose=pd.concat(parts,ignore_index=True); dose_counts=[]; dose_results=[]
for g,z in dose.groupby('cum_cat',observed=False):
    dose_counts.append({'group':str(g),'intervals':len(z),'people':z.ID.nunique(),
                        'progression':int(z.destination_state.eq(2).sum()),
                        'recovery':int(z.destination_state.eq(0).sum()),'death':int(z.destination_state.eq(3).sum())})
for dest,label in [(2,'progression'),(0,'recovery')]:
    dose['event']=dose.destination_state.eq(dest).astype(int)
    f=fit_cluster(dose,'event ~ C(cum_cat) + C(interval) + '+base)
    for lev in ['1','2','3+']:
        v=pd.Series(0.,index=f.params.index); v[f'C(cum_cat)[T.{lev}]']=1
        r=contrast(f,v); r.update(outcome=label,comparison=f'{lev} vs 0'); dose_results.append(r)
    ft=fit_cluster(dose,'event ~ cum_co + C(interval) + '+base)
    v=pd.Series(0.,index=ft.params.index); v['cum_co']=1
    r=contrast(ft,v); r.update(outcome=label,comparison='per additional wave'); dose_results.append(r)

def reconstruct(code,targetstub):
    histx={1:h.ID_w1.map(raw11[f'da007_{code}_']).map({1:1.,2:0.})}; froz={1:histx[1].copy()}
    for wv,rel in [(2,'2013/CHARLS2013_Dataset'),(3,'2015/CHARLS2015r'),(4,'2018/CHARLS2018r')]:
        raw=pd.read_stata(D/rel/'Health_Status_and_Functioning.dta',convert_categoricals=False).set_index('ID')
        get=lambda k:raw[k].reindex(h.index)
        pre=histx[wv-1].copy() if wv==2 else pd.concat([histx[k] for k in range(wv-1,0,-1)],axis=1).bfill(axis=1).iloc[:,0]
        pre=pre.where(h[f'inw{wv}'].eq(1)); direct=get(f'da007_{code}_'); cur=pd.Series(np.nan,index=h.index)
        if wv in [2,3]:
            verify=get(f'da007_w2_1_{code}_'); newv=get(f'da007_w2_2_{code}_')
            no=(pre.eq(0)&verify.ne(2)&newv.eq(2)) if wv==2 else (pre.eq(0)&verify.eq(1))
            cur.loc[no|direct.eq(2)|(pre.eq(1)&verify.eq(2)&newv.eq(2))]=0
            yes=(pre.eq(0)&verify.eq(2))|direct.eq(1)|newv.eq(1)|(pre.eq(1)&verify.ne(2))
            if wv==3: yes=yes|get(f'zda007_{code}_').eq(1)
            cur.loc[yes]=1
            for k in range(1,wv):
                old=histx[k].copy(); histx[k].loc[pre.eq(0)&verify.eq(2)&old.eq(0)]=1
                histx[k].loc[pre.eq(1)&verify.eq(2)&old.eq(1)]=0
        else:
            report=get(f'da010_w2_2_{code}_')
            cur.loc[pre.eq(0)|direct.eq(2)|(pre.eq(1)&report.eq(99))]=0
            cur.loc[direct.eq(1)|get(f'zdisease_{code}_').eq(1)|(pre.eq(1)&report.between(1,3))]=1
            for k in range(1,wv): histx[k].loc[pre.eq(1)&report.eq(99)&histx[k].eq(1)]=0
        cur=cur.where(h[f'inw{wv}'].eq(1)); histx[wv]=cur.copy(); froz[wv]=cur.copy()
    final=pd.read_stata(D/'Harmonized CHARLS/H_CHARLS_D_Data/H_CHARLS_D_Data.dta',
                        columns=['ID']+[f'r{w}{targetstub}' for w in range(1,5)],convert_categoricals=False).set_index('ID')
    for wv in range(1,5):
        tar=final[f'r{wv}{targetstub}'].reindex(h.index)
        same=histx[wv].eq(tar)|(histx[wv].isna()&tar.isna())
        assert same.all(),(targetstub,wv,int((~same).sum()))
    return froz

snaps={'digestive':frozen,'heart':reconstruct(7,'hearte'),'arthritis':reconstruct(13,'arthre')}
baseline14=pd.DataFrame({k:h.ID_w1.map(raw11[f'da007_{k}_']).map({1:1.,2:0.}) for k in range(1,15)})
control=[]
for disease,code in [('digestive',10),('heart',7),('arthritis',13)]:
    z=a.copy(); idx=pd.MultiIndex.from_frame(z[['ID','wave']])
    lookup=pd.concat({wv:s for wv,s in snaps[disease].items()},names=['wave','ID']).reorder_levels(['ID','wave'])
    z['index_disease']=lookup.reindex(idx).to_numpy(); z=z[z.index_disease.notna()].copy()
    z['index_state']=z.index_disease+2*z.depression
    cnt=baseline14.drop(columns=code).sum(axis=1,min_count=13); miss=baseline14.drop(columns=code).isna().sum(axis=1)
    z['other_chronic_count']=z.ID.map(cnt); z['chronic_missing']=z.ID.map(miss).gt(0).astype(int)
    z['other_chronic_count']=z.other_chronic_count.fillna(cnt.median())
    den=smf.glm('outcome_observed ~ C(index_state)+C(functional_state)+C(interval)+'+base,z,family=sm.families.Binomial()).fit()
    num=smf.glm('outcome_observed ~ C(functional_state)+C(interval)',z,family=sm.families.Binomial()).fit()
    z['wr']=num.predict(z).clip(.02,.995)/den.predict(z).clip(.02,.995)
    zz=z[z.outcome_observed.eq(1)&z.functional_state.eq(1)&z.index_state.isin([2,3])].copy()
    lo,hi=zz.wr.quantile([.01,.99]); zz['iw']=zz.wr.clip(lo,hi); zz['event']=zz.destination_state.eq(2).astype(int)
    f=fit_cluster(zz,'event ~ index_disease + C(interval) + '+base,'iw')
    v=pd.Series(0.,index=f.params.index); v['index_disease']=1
    r=contrast(f,v); r.update(disease=disease,intervals=len(zz),people=zz.ID.nunique(),events=int(zz.event.sum()),
                              co_intervals=int(zz.index_disease.sum()),co_events=int(zz.loc[zz.index_disease.eq(1),'event'].sum()))
    control.append(r)

# Sex interaction for IADL-to-ADL progression.
z=o[o.functional_state.eq(1)&o.state.isin([2,3])].copy(); z['co']=z.state.eq(3).astype(int); z['event']=z.destination_state.eq(2).astype(int)
base_no_sex=base.replace('female + ','')
f=fit_cluster(z,'event ~ co*female + C(interval) + '+base_no_sex)
sex=[]
for female,label in [(0,'men'),(1,'women')]:
    v=pd.Series(0.,index=f.params.index); v['co']=1
    if female:v['co:female']=1
    r=contrast(f,v); r.update(group=label); sex.append(r)
b=f.params['co:female']; se=np.sqrt(f.audit_cov.loc['co:female','co:female'])
interaction={'ratio_of_rr':float(np.exp(b)),'low':float(np.exp(b-1.96*se)),'high':float(np.exp(b+1.96*se)),
             'p':float(2*norm.sf(abs(b/se)))}

# Joint multinomial destination model for no-limitation origins.
z=o[o.functional_state.eq(0)&o.state.isin([2,3])].copy(); z['co']=z.state.eq(3).astype(int)
formula='co + C(interval) + '+base
X=dmatrix(formula,z,return_type='dataframe'); design=X.design_info
scaler=StandardScaler().fit(np.asarray(X))
Xs=scaler.transform(np.asarray(X))
model=LogisticRegression(C=1e4,solver='lbfgs',max_iter=500,fit_intercept=True)
with warnings.catch_warnings():
    warnings.simplefilter('ignore'); model.fit(Xs,z.destination_state.astype(int),sample_weight=z.iow)
target=z.copy(); pred=[]
for val in [0,1]:
    q=target.copy(); q['co']=val; xq=scaler.transform(np.asarray(build_design_matrices([design],q)[0]))
    pred.append(np.average(model.predict_proba(xq),axis=0,weights=q.iow))
point=np.array(pred); ids=pd.Index(z.ID.unique()); g=ids.get_indexer(z.ID); rng=np.random.default_rng(20260919); reps=[]
for b in range(500):
    mult=rng.multinomial(len(ids),np.full(len(ids),1/len(ids))); wgt=z.iow.to_numpy()*mult[g]
    if wgt.sum()==0: continue
    try:
        mm=LogisticRegression(C=1e4,solver='lbfgs',max_iter=400,fit_intercept=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); mm.fit(Xs,z.destination_state.astype(int),sample_weight=wgt)
        pp=[]
        for val in [0,1]:
            q=target.copy();q['co']=val;xq=scaler.transform(np.asarray(build_design_matrices([design],q)[0]))
            pp.append(np.average(mm.predict_proba(xq),axis=0,weights=wgt))
        reps.append(pp)
    except Exception: pass
reps=np.array(reps); multi=[]
for j,cls in enumerate(model.classes_):
    dd=reps[:,1,j]-reps[:,0,j]; ci=np.quantile(dd,[.025,.975])
    multi.append({'destination':int(cls),'reference_risk':float(point[0,j]),'co_risk':float(point[1,j]),
                  'difference':float(point[1,j]-point[0,j]),'difference_ci':[float(ci[0]),float(ci[1])]})

def evalue(rr):
    x=rr if rr>=1 else 1/rr
    return float(x+np.sqrt(x*(x-1)))
models=json.loads((P/'model_results.json').read_text())
def pick(x):return next(r for r in models if r['scenario']=='Main' and r['comparison']==x)
evalues=[]
for label,key in [('Current co-occurrence','IADL to ADL 3v2'),('Repeated versus first','History progression Repeated_vs_First')]:
    r=pick(key); evalues.append({'comparison':label,'rr':r['rr'],'ci':[r['low'],r['high']],
      'evalue_estimate':evalue(r['rr']),'evalue_ci_nearest_null':evalue(r['low']) if r['low']>1 else 1.0})

out={'dose_counts':dose_counts,'dose_results':dose_results,'disease_controls':control,
     'sex_stratified':sex,'sex_interaction':interaction,'multinomial_destination_probabilities':multi,
     'multinomial_bootstrap_success':len(reps),'evalues':evalues}
(O/'enhancement_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)





