'''
This is not really a 'run' since it does not run a job on the cluster, 
but instead just computes stuff locally (because it doesn't need a lot of ressources). 
But it is an important processing step, since we do the comparison of the MEG movie-RDMs 
with the model RDMs here.

We will use the output from here to plot the temportal course of the correlation with 
the models for each subjects, so we can determine inter-subjects variability. 
This info will then be used to decide, whether we average MEG RDMs over subjects or not 
before doing the cluster permutation test with the commonality coefficients of the models

We also compute the mean of the subjects' RDM movies and save them.

At the end, we do some plotting.

Script for more plotting: /home/reabt/experiments/ncc/MRI/code/tools/plot_MEG_eval_peaks.py
'''
#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from utils.subj import get_MEG_subjects
from utils.load_cfg import load_config_instance
from utils.plots import *
from utils.rsa import *
from utils.subj import *

import joblib

import rsatoolbox as rsa
from copy import deepcopy
from glob import glob

#%%
all_subj = get_MEG_subjects()

cfg = load_config_instance('MEGconfig_E', all_subj[0]) # Because this config uses crossnobis distance & the correct RSA measure 'corr_cov'

# create output directories if they don't exist already
os.makedirs(os.path.split(cfg.get_outFile_names()['movie'])[0], exist_ok=True)
os.makedirs(os.path.split(cfg.get_outFile_names()['movie_eval'])[0], exist_ok=True)

models = cfg.get_model_RDM()

for subj in all_subj:
    print(f'\nProcessing subject {subj}')
    cfg.subjectID = subj

    movieFile = cfg.get_outFile_names()['movie']
    resultsFile = cfg.get_outFile_names()['movie_eval']

    if os.path.exists(movieFile):

        print(f'     - input file: {os.path.split(movieFile)[1]}')

        rdm_movie = joblib.load(movieFile)
        results = rsa.inference.eval_fixed(models, rdm_movie, method=cfg.RSAmethod)#method='corr_cov')
        # print(f'{results.evaluations.shape=} = (n_bootstrap_samples, n_models, n_timepoints * n_subjects)')

        print(f'     - output file: {os.path.split(resultsFile)[1]}')

        joblib.dump(results, resultsFile)

    else:
        print(f"\nskipping subjects {subj} because we don't have the necessary input data\n")


#%% Plotting the results after averaging RDMs across subjects on top of single-subjects results:


all_RDMs = []
for subj in all_subj:

    cfg.subjectID = subj

    movieFile = cfg.get_outFile_names()['movie']
    resultsFile = cfg.get_outFile_names()['movie_eval']

    if os.path.exists(movieFile):
        rdm_movie = joblib.load(movieFile)
        all_RDMs.append(rdm_movie.get_vectors())

mean_rdms = np.mean(np.array(all_RDMs), axis = 0)

mean_movie = deepcopy(rdm_movie)
mean_movie.dissimilarities = mean_rdms # put the data back in necessary RDMs structure

joblib.dump(mean_movie, cfg.get_outFile_names()['movie_mean'])

# do model evaluation with averaged RDM movie
mean_results = rsa.inference.eval_fixed(models, rdm_movie, method=cfg.RSAmethod)#method='corr_cov')


# loading normal results files
results_files = sorted(glob(f'{MEG_DATA_DIR}/movie_eval/model_eval_results*'))

sensory_data, suprasensory_data = [], []

for file in results_files:
    results = joblib.load(file)
    sensory_data.append(results.evaluations[0][0])
    suprasensory_data.append(results.evaluations[0][1])


sensory_max = [(np.max(data).item(), np.argmax(data).item()) for data in sensory_data]
suprasensory_max = [(np.max(data).item(), np.argmax(data).item()) for data in suprasensory_data]

x = np.arange(200)

# [1] Sensory ---------------------------------------------------
#  a) Plot sincle subjects results
for n, data in enumerate(sensory_data):
    plt.plot(x, 
             data, 
             label=f"Subject {n}", 
             alpha=0.3, 
             linewidth=1)
#  b) plot results avter averaging RDMs before
plt.plot(x, 
         mean_results.evaluations[0][0],
         alpha = 1,
         c = 'black',
         linewidth = 3)

# c) show peaks of single subjects
[plt.scatter(m[1], 
        m[0], 
        alpha=1,
        s = 5,
        c = 'red') for n, m in enumerate(sensory_max)]
plt.title('Sensory')
plt.show()

# [2] Supraensory -----------------------------------------------
#  a) Plot sincle subjects results
for n, data in enumerate(suprasensory_data):
    plt.plot(x, 
             data, 
             label=f"Subject {n}", 
             alpha=0.3, 
             linewidth=1)
    
plt.plot(x, 
         mean_results.evaluations[0][1],
         alpha = 1,
         c = 'black',
         linewidth = 3)    
    
[plt.scatter(m[1], 
        m[0], 
        alpha=1,
        s = 5,
        c = 'red') for n, m in enumerate(suprasensory_max)]
plt.title('Suprasensory')
plt.show()

#%%




#%%

import os 

#%%
base_dir = os.getcwd() 

print("Current Working Directory:", base_dir)

print(os.path.dirname(base_dir))
# %%
