# CHARLS digestive disease, depressive symptoms and functional transitions

Statistical analysis code for repeated co-occurrence of digestive disease and depressive symptoms and subsequent functional transitions in CHARLS. This repository contains source code, software requirements and operating instructions only. It contains no manuscript, participant records, analysis outputs or author contact information.

## Requirements and data access

Use Python 3.12 with the versions in `requirements.txt`. Obtain the CHARLS releases independently under their registration and data-use conditions at https://charls.charlsdata.com/. This repository does not grant access to or permission to redistribute CHARLS data.

Set `CHARLS_DATA_ROOT` to the directory containing these relative paths (filenames are case-sensitive on Linux):

```text
Harmonized CHARLS/H_CHARLS_D_Data/H_CHARLS_D_Data.dta
2011/household_and_community_questionnaire_data/health_status_and_functioning.dta
2011/household_and_community_questionnaire_data/demographic_background.dta
2013/CHARLS2013_Dataset/Health_Status_and_Functioning.dta
2015/CHARLS2015r/Health_Status_and_Functioning.dta
2018/CHARLS2018r/Health_Status_and_Functioning.dta
2020/CHARLS2020r/Health_Status_and_Functioning.dta
2020/CHARLS2020r/Sample_Infor.dta
```

Install the dependencies into a compatible Python environment with `python -m pip install -r requirements.txt` if needed. Then, from this repository directory:

PowerShell:
```powershell
$env:CHARLS_DATA_ROOT = 'PATH_TO_AUTHORIZED_CHARLS_DATA'
python run_all.py
```

POSIX shell:
```sh
export CHARLS_DATA_ROOT=/path/to/authorized/charls/data
python run_all.py
```

`ANALYSIS_DEPS` is optional and can point to an existing compatible dependency directory. The runner propagates it to child processes. Full execution includes multiple imputation, cross-validation and bootstrap analyses and can take substantial time. Individual scripts depend on earlier outputs; run the pipeline in order.

## Analysis sequence

| Script | Purpose |
| --- | --- |
| `scripts/09_build_comorbidity_state_data.py` | Build exposure states and baseline covariates |
| `scripts/11_build_functional_multistate_data.py` | Construct functional states and consecutive survey intervals |
| `scripts/12_run_functional_multistate_analysis.py` | Shared covariate definitions and preparation helpers |
| `work/decision_validation.py` | Reconstruct time-specific disease snapshots and age sensitivity analyses |
| `work/acer_prepare.py` | Assemble the final eligible intervals and death destinations |
| `work/acer_analysis.py` | Main weighted modified Poisson models, exposure histories and standardized absolute risks |
| `work/acer_unadjusted.py` | Unadjusted comparisons |
| `work/validation_mi.py` | Diagnostic multiple-imputation sensitivity analysis |
| `work/enhancement_analysis.py` | Cumulative exposure, disease controls, sex analyses, no-limitation joint destinations and E-values |
| `work/prediction_increment.py` | Exploratory internal cross-validation and incremental discrimination |
| `work/cumulative_bias_checks.py` | Accumulation opportunities and depressive-symptom persistence |
| `work/submission_sensitivity.py` | Death exclusion, 11-condition adjustment and IADL-origin joint destinations |
| `work/revision_sensitivity.py` | Complete-case and 2011 ADL restrictions, age/hukou interactions and baseline missingness |

## Outputs and reproducibility

Scripts create local intermediate records under `outputs/comorbidity_state`, `work/decision_validation` and `work/acer`. These include participant-level CSV and pickle files: do not upload or redistribute them. All generated output directories and non-source files are ignored by Git. Aggregate model summaries are produced locally as JSON; they are not distributed in this repository.

The model definitions, contrasts and random seeds are retained from the verified analysis scripts. The new IADL-origin joint model uses 500 participant bootstrap draws, refitting observation and outcome models and weight truncation while holding covariate scaling and missing-covariate substitutions fixed. Death-exclusion models estimate survivor-conditional associations. The 11-condition sensitivity removes psychiatric and memory-related conditions from the adjustment count and refits observation weights on the same eligible intervals.

The outcome is next-wave functional status, not time to first disability. Intervals within individuals are clustered. The analyses were not preregistered; prediction and disease-control analyses are exploratory. These observational estimates do not establish causal effects or digestive specificity.

## Additional sensitivity and subgroup analyses

`work/revision_sensitivity.py` uses the locked `work/acer/all.pkl`, `observed.pkl`, `history.pkl` and `model_results.json` produced by the preceding steps, plus baseline variables from the authorized Harmonized CHARLS D file. It first verifies the unchanged main current and repeated-versus-first RRs and observation weights. It then produces `work/acer/enhancements/revision_sensitivity.json` with aggregate estimates, confidence intervals, interaction tests, group-specific event counts, restriction flows and missingness counts.

Complete cases require all baseline adjustment variables before substitution. The baseline functional restriction requires complete 2011 ADL items and no reported difficulty; unknown status is excluded separately. Both restrictions refit observation weights and outcome models. Age interactions compare baseline ages 45–59 and ≥60 years while retaining continuous-age adjustment. Hukou interactions use original rural/urban registration, exclude unknown hukou and do not measure residence. Models share other adjustment coefficients across strata and use participant-clustered variance. Interaction tests compare direct contrasts, not separate significance levels. These analyses were specified after the primary analysis, are not preregistered and do not correct for multiplicity. No participant records or aggregate results are uploaded to this source-only repository.
