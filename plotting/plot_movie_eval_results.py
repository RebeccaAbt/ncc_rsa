#%%

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from rsatoolbox.inference import eval_fixed

sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_MEG_config_instance


#%%

config_class_name = 'MEGconfig_C'

cfg = load_MEG_config_instance(config_class_name) # uses '*' if no subjectID supplied
models = cfg.get_model_RDM()

inDir = '/home/reabt/experiments/ncc/MEG/data/movie_RDMs/C/'
inFiles = os.listdir(inDir)

#%%
results_senso = []
results_supra = []


for file in inFiles:
    if "first" not in file:
        print(file)

        subj_data = joblib.load(os.path.join(inDir, file))
        eval_results = eval_fixed(models, subj_data, theta=None, method='corr_cov')

        results_senso.append(eval_results.evaluations[0,0])
        results_supra.append(eval_results.evaluations[0,1])

# %%

# Convert lists of (200,) arrays → 2D arrays (n_subjects, n_timepoints)
senso_arr = np.vstack(results_senso)   # shape (n_subj, 200)
supra_arr = np.vstack(results_supra)

# Compute mean and SEM (standard error of the mean) across subjects
senso_mean = senso_arr.mean(axis=0)
senso_sem  = senso_arr.std(axis=0, ddof=1) / np.sqrt(senso_arr.shape[0])

supra_mean = supra_arr.mean(axis=0)
supra_sem  = supra_arr.std(axis=0, ddof=1) / np.sqrt(supra_arr.shape[0])

# X-axis (time points)
time = np.arange(senso_arr.shape[1])

# Plot with shaded error bars
plt.figure(figsize=(10,5))

plt.plot(time, senso_mean, label='Sensory', color='tab:blue')
plt.fill_between(time, senso_mean - senso_sem, senso_mean + senso_sem, 
                 color='tab:blue', alpha=0.3)

plt.plot(time, supra_mean, label='Suprasensory', color='tab:orange')
plt.fill_between(time, supra_mean - supra_sem, supra_mean + supra_sem, 
                 color='tab:orange', alpha=0.3)

plt.xlabel("Time points")
plt.ylabel("Evaluation")
plt.title("Mean evaluations over time with SEM shaded")
plt.legend()
plt.tight_layout()
plt.show()

# %%
