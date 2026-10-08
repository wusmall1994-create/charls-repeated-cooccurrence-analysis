"""Additional complete-case, baseline-ADL and effect-modification analyses.

Run from the reproducibility-package root after run_all.py preparatory steps.
Inputs: work/acer/{all,observed,history}.pkl and authorized Harmonized CHARLS D.
Output: work/acer/enhancements/revision_sensitivity.json (aggregate only).
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, warnings
from pathlib import Path
if os.environ.get('ANALYSIS_DEPS'):
    sys.path.insert(0, os.environ['ANALYSIS_DEPS'])
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from patsy import build_design_matrices

BASE = ('baseline_age_c + female + C(education_cat) + C(marital_cat) + '
        'rural_hukou + current_smoker + drank_last_year + binary_covariate_missing + '
        'baseline_bmi_imp + bmi_missing + log_hh_income_imp + income_missing + '
        'other_chronic_count + chronic_missing')

def weights(a):
    den = smf.glm('outcome_observed ~ C(state)+C(functional_state)+C(interval)+'+BASE,
                  a, family=sm.families.Binomial()).fit()
    num = smf.glm('outcome_observed ~ C(functional_state)+C(interval)',
                  a, family=sm.families.Binomial()).fit()
    assert den.converged and num.converged
    w = num.predict(a).clip(.02,.995) / den.predict(a).clip(.02,.995)
    o = a[a.outcome_observed.eq(1)].copy()
    lo,hi = w.loc[o.index].quantile([.01,.99])
    o['iow'] = w.loc[o.index].clip(lo,hi)
    return o, dict(eligible_people=int(a.ID.nunique()), eligible_intervals=len(a),
                   observed_people=int(o.ID.nunique()), observed_intervals=len(o),
                   weight_low=float(lo), weight_high=float(hi))

def history(o, h):
    z = o.merge(h[['ID','wave','history']],on=['ID','wave'],validate='one_to_one')
    z['history'] = pd.Categorical(z.history,categories=['Reference','Prior_only','First','Repeated'])
    assert z.functional_state.eq(1).all() and z.state.isin([2,3]).all()
    return z

def model(z, term, modifier=None):
    z = z.copy()
    z['event'] = z.destination_state.eq(2).astype(int)
    exposure = 'C('+term+')'
    formula = 'event ~ '+exposure+('*'+modifier if modifier else '')+'+C(interval)+'+BASE
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*cov_type.*')
        f = smf.glm(formula,z,family=sm.families.Poisson(),freq_weights=z.iow).fit()
    assert f.converged and len(f.fittedvalues)==len(z)
    X=f.model.exog;mu=np.asarray(f.fittedvalues);w=z.iow.to_numpy()
    scores=X*(w*(z.event.to_numpy()-mu))[:,None]
    sums=pd.DataFrame(scores).groupby(z.ID.to_numpy()).sum().to_numpy()
    bread=np.linalg.pinv(X.T@((w*mu)[:,None]*X))
    n,k=X.shape;g=len(sums)
    cov=bread@sums.T@sums@bread*g/(g-1)*(n-1)/(n-k)
    def vector(group=None):
        pair=pd.concat([z.iloc[[0]],z.iloc[[0]]],ignore_index=True)
        pair[term]=[2.,3.] if term=='state' else ['First','Repeated']
        if modifier: pair[modifier]=group
        xx=np.asarray(build_design_matrices([f.model.data.design_info],pair)[0])
        v=xx[1]-xx[0]
        assert np.max(np.abs(v-v@np.linalg.pinv(X)@X))<1e-7, 'Non-estimable contrast'
        return v
    def contrast(v):
        b=float(v@f.params);variance=float(v@cov@v)
        assert variance>0
        se=np.sqrt(variance)
        return dict(rr=float(np.exp(b)),low=float(np.exp(b-1.96*se)),
                    high=float(np.exp(b+1.96*se)),p=float(2*norm.sf(abs(b/se))))
    def counts(zz):
        levels=[2,3] if term=='state' else ['First','Repeated']
        return dict(people=int(zz.ID.nunique()),intervals=len(zz),
                    events=int(zz.destination_state.eq(2).sum()),
                    cells=[dict(level=str(l),intervals=int(zz[term].eq(l).sum()),
                                people=int(zz.loc[zz[term].eq(l),'ID'].nunique()),
                                events=int((zz[term].eq(l)&zz.destination_state.eq(2)).sum())) for l in levels])
    result=dict(term=term,formula=formula,model=counts(z),converged=bool(f.converged))
    if modifier:
        result['modifier']=modifier
        result['strata']=[dict(group=i,**counts(z[z[modifier].eq(i)]),**contrast(vector(i))) for i in [0,1]]
        result['interaction']=contrast(vector(1)-vector(0))
    else: result.update(contrast(vector()))
    return result

def main():
    p=Path('work/acer');out=p/'enhancements';out.mkdir(exist_ok=True)
    a=pd.read_pickle(p/'all.pkl');h=pd.read_pickle(p/'history.pkl')
    o,main_flow=weights(a)
    saved=pd.read_pickle(p/'observed.pkl')
    assert o[['ID','wave']].equals(saved[['ID','wave']])
    assert np.allclose(o.iow,saved.iow,rtol=1e-9,atol=1e-9)
    baseline=[model(o[o.functional_state.eq(1)],'state'),model(history(o,h),'history')]
    old=json.loads((p/'model_results.json').read_text())
    for result,label in zip(baseline,['IADL to ADL 3v2','History progression Repeated_vs_First']):
        expected=next(v for v in old if v['scenario']=='Main' and v['comparison']==label)
        assert abs(result['rr']-expected['rr'])<1e-7
    d=Path(os.environ['CHARLS_DATA_ROOT'])
    cols=['ID','ragender','raeducl','r1mstat','r1rural2','r1smoken','r1drinkl','r1adlab_c','r1adlabm_c','inw1']
    raw=pd.read_stata(d/'Harmonized CHARLS/H_CHARLS_D_Data/H_CHARLS_D_Data.dta',columns=cols,convert_categoricals=False).set_index('ID')
    # Original pre-substitution covariate validity, not model-ready filled values.
    valid=(raw.ragender.isin([1,2]) & raw.raeducl.notna() & raw.r1mstat.notna()
           & raw.r1rural2.isin([0,1]) & raw.r1smoken.isin([0,1]) & raw.r1drinkl.isin([0,1]))
    cc=(a.ID.map(valid).eq(True)&a.baseline_age.notna()&a.baseline_bmi.notna()
        &a.log_hh_income.notna()&a.chronic_missing.eq(0))
    noadl=raw.inw1.eq(1)&raw.r1adlabm_c.eq(0)&raw.r1adlab_c.eq(0)
    known_adl=raw.inw1.eq(1)&raw.r1adlabm_c.eq(0)&raw.r1adlab_c.notna()
    rows=[];flows={}
    for label,mask in [('Complete baseline covariates',cc),('No ADL difficulty in 2011',a.ID.map(noadl).eq(True))]:
        oo,flow=weights(a[mask].copy());flows[label]=flow
        for term,z in [('state',oo[oo.functional_state.eq(1)]),('history',history(oo,h))]:
            row=model(z,term);row['scenario']=label;rows.append(row)
    eligible=a.drop_duplicates('ID')
    flows['baseline_adl_classification']=dict(
        no_difficulty=int(eligible.ID.map(noadl).sum()),
        difficulty=int((eligible.ID.map(known_adl)&~eligible.ID.map(noadl)).sum()),
        indeterminate=int((~eligible.ID.map(known_adl)).sum()))
    interactions=[]
    for mod in ['age60','rural_hukou']:
        aa=a.copy()
        if mod=='rural_hukou': aa=aa[aa.ID.map(raw.r1rural2).isin([0,1])].copy()
        aa['age60']=aa.baseline_age.ge(60).astype(int)
        oo,flow=weights(aa);flows[mod]=flow
        for term,z in [('state',oo[oo.functional_state.eq(1)]),('history',history(oo,h))]:
            interactions.append(model(z,term,mod))
    missing_checks={
        'Baseline age':eligible.baseline_age.notna(),
        'Sex':eligible.ID.map(raw.ragender).isin([1,2]),
        'Education':eligible.ID.map(raw.raeducl).notna(),
        'Marital status':eligible.ID.map(raw.r1mstat).notna(),
        'Rural hukou':eligible.ID.map(raw.r1rural2).isin([0,1]),
        'Smoking':eligible.ID.map(raw.r1smoken).isin([0,1]),
        'Alcohol use':eligible.ID.map(raw.r1drinkl).isin([0,1]),
        'BMI':eligible.baseline_bmi.notna(),
        'Log household income':eligible.log_hh_income.notna(),
        '13-condition count':eligible.chronic_missing.eq(0)}
    missingness=[dict(variable=k,eligible_people=len(eligible),missing_people=int((~v).sum()),
                      missing_percent=float((~v).mean()*100)) for k,v in missing_checks.items()]
    result=dict(main=baseline,flow=flows,sensitivity=rows,interactions=interactions,missingness=missingness,
                versions=dict(Python=sys.version.split()[0],pandas=pd.__version__,numpy=np.__version__,statsmodels=sm.__version__),
                notes=['All additional analyses were specified for this revision, not preregistered.',
                       'Weights refitted within each eligible restricted cohort; interaction models share adjustment coefficients across strata.',
                       'Hukou is baseline registration status, not current residence. Unknown original hukou excluded.',
                       'Wald CIs and interaction tests use participant-clustered variance; observation weights treated as fixed.',
                       'No adjustment for multiplicity. Missing baseline ADL excluded from the 2011 restriction.',
                       'Age groups: baseline 45-59 and >=60 years. Death is an observed non-event.'])
    (out/'revision_sensitivity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
