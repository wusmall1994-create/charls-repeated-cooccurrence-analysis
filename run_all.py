import os,sys,subprocess
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
if not os.environ.get('CHARLS_DATA_ROOT'):raise SystemExit('Set CHARLS_DATA_ROOT to the authorized raw-data directory')
if os.environ.get('ANALYSIS_DEPS'):
 os.environ['PYTHONPATH']=os.environ['ANALYSIS_DEPS']+os.pathsep+os.environ.get('PYTHONPATH','')
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
for p in ['scripts/09_build_comorbidity_state_data.py','scripts/11_build_functional_multistate_data.py','work/decision_validation.py','work/acer_prepare.py','work/acer_analysis.py','work/acer_unadjusted.py','work/validation_mi.py','work/enhancement_analysis.py','work/prediction_increment.py','work/cumulative_bias_checks.py','work/submission_sensitivity.py']:
 print('Running '+p,flush=True)
 subprocess.run([sys.executable,p],check=True)
