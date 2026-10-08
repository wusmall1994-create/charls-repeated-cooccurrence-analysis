import numpy as np
import pandas as pd
STATE_LABELS = {0: 'Neither', 1: 'Digestive only', 2: 'Depression only', 3: 'Co-occurring'}
FUNCTION_LABELS = {0: 'No limitation', 1: 'IADL limitation only', 2: 'ADL limitation', 3: 'Death'}
BASE_COVARS = 'baseline_age_c + female + C(education_cat) + C(marital_cat) + rural_hukou + current_smoker + drank_last_year + binary_covariate_missing + baseline_bmi_imp + bmi_missing + log_hh_income_imp + income_missing + other_chronic_count'

def prepare(df):
    x = df.copy()
    x['baseline_age_c'] = x['baseline_age'] - 60
    x['education_cat'] = x['education_code'].fillna(-1).astype(str)
    x['marital_cat'] = x['marital_code'].fillna(-1).astype(str)
    x['binary_covariate_missing'] = x[['rural_hukou', 'current_smoker', 'drank_last_year']].isna().any(axis=1).astype(int)
    for v in ['female', 'rural_hukou', 'current_smoker', 'drank_last_year']:
        x[v] = x[v].fillna(x[v].mode(dropna=True).iloc[0])
    x['bmi_missing'] = x['baseline_bmi'].isna().astype(int)
    x['baseline_bmi_imp'] = x['baseline_bmi'].fillna(x['baseline_bmi'].median())
    x['income_missing'] = x['log_hh_income'].isna().astype(int)
    x['log_hh_income_imp'] = x['log_hh_income'].fillna(x['log_hh_income'].median())
    x['state12'] = x['digestive'] + 2 * x['cesd10'].ge(12).astype(int)
    return x