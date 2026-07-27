
#%%


import os
import sys
import joblib
import mne
from pymatreader import read_mat

from plus_slurm import Job

sys.path.append('/home/reabt/experiments/ncc/MRI/code/utils/')
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.raw import Raw
from utils.epochs import get_epochs_R
from utils.clean_epochs import *
from utils.subj import *


all_subjects = get_MEG_raw_subjects()


#%%
all_subjects = [all_subjects[i] for i in [1, 13, 24, 27]]

#%%
potato_setting = []
# LWF
potato_setting.append({
	'estimator' : "lwf", # 'lwf' or 'oas'
	'use_field': True,
	'picks': "grad", # egal, wenn use_field = True, da grads+mags
	'n_potatoes': 2, # only if use_field
	'z_threshold': 3})

potato_setting.append({
	'estimator' : "lwf", # 'lwf' or 'oas'
	'use_field': False,
	'picks': "mag",
	'n_potatoes': 2, # only if use_field
	'z_threshold': 3})

potato_setting.append({
	'estimator' : "lwf", # 'lwf' or 'oas'
	'use_field': False,
	'picks': "grad",
	'n_potatoes': 2, # only if use_field
	'z_threshold': 3})

potato_setting.append({
	'estimator' : "lwf", # 'lwf' or 'oas'
	'use_field': False,
	'picks': "mag",
	'n_potatoes': 2, # only if use_field
	'z_threshold': 2})

# OAS
potato_setting.append({
	'estimator' : "oas", # 'lwf' or 'oas'
	'use_field': True,
	'picks': "grad", # egal, wenn use_field = True, da grads+mags
	'n_potatoes': 2, # only if use_field
	'z_threshold': 3})

potato_setting.append({
	'estimator' : "oas", # 'lwf' or 'oas'
	'use_field': False,
	'picks': "mag",
	'n_potatoes': 2, # only if use_field
	'z_threshold': 3})

potato_setting.append({
	'estimator' : "oas", # 'lwf' or 'oas'
	'use_field': False,
	'picks': "grad",
	'n_potatoes': 2, # only if use_field
	'z_threshold': 3})

potato_setting.append({
	'estimator' : "oas", # 'lwf' or 'oas'
	'use_field': False,
	'picks': "mag",
	'n_potatoes': 2, # only if use_field
	'z_threshold': 2})

potatos = dict()
for subj in all_subjects[:5]:

	epochs = mne.read_epochs(f'/home/reabt/Desktop/ncc/MEG/data/epochs_icaTest/{subj}/{subj}_maxfilter_True__ica_True__0.5-99Hz__fs_1000__[-1.5_1.5]s_meg-epo.fif')


	potato_info = []
	bad_idx = []

	for opts in potato_setting:
		print(f"Running potato with settings: {opts}")
		_, info = reject_bad_epochs_with_potato(
								epochs,
								**potato_setting,
								p_threshold=0.001,
								return_details=True,
								reject_epochs = False
								)
		potato_info.append(info)
		bad_idx.append(info['bad_idx'])

	potato_outputs  = dict(settings = potato_setting, info = potato_info, bad_idx = bad_idx)

	potatos[subj] = potato_outputs


joblib.dump(potatos, 'tmp_potatos_output.pkl')
