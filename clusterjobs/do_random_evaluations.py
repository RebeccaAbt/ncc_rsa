#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import joblib
import mne
import nibabel as nib
import numpy as np
from nilearn import datasets, image, plotting

from utils.load_cfg import load_fusion_config_instance, load_MEG_config_instance, load_MRI_config_instance
from utils.subj import *
from utils.fusion_stat import *
from utils.rsa import *
from utils.model import get_random_model_like
from utils.rsa import data3d_to_masked1d
from utils.rsa_eval import find_common_centers
from plus_slurm import Job
mne.viz.set_browser_backend('matplotlib')


#%%

def _get_random_evaluation(subjectID, config_class_name, model, random_models, mask_1d, show_progress=False, n_jobs = -1):
	cfg = load_MRI_config_instance(config_class_name, subjectID)
	cfg.modelType = model
	print(f'start processing subject {subjectID}', flush=True)
	_, _, _, RDM_brain_list = get_RSA_for_models(cfg, random_models, show_progress=show_progress, n_jobs=n_jobs)
	print(f'finished processing subject {subjectID}', flush=True)
	return data3d_to_masked1d(RDM_brain_list, mask_1d)


class RandomEval_fMRI(Job):
	def run(self,
			all_subjects,
			config_class_name = 'MRIconfig_E2',  # not relevant, if e.g. "E1" or "E2" because searchlight radius not relevant here",
			allModels = ALL_MODELS,
			n_random = 100,
			method = 'label', # 'label' or 'random'
			n_jobs_eval = 4,
			n_jobs_subj = 12,
			show_progress = False
			):
		
		for model in allModels:
			# model = allModels[1]

			# normal model evaluations
			RDM_brain_list = []
			centers_masks = []
			for subjectID in all_subjects:
				print(subjectID, flush=True)
				cfg = load_MRI_config_instance(config_class_name, subjectID)
				cfg.modelType = model  
				outFiles = cfg.get_outFile_names()

				if os.path.isfile(outFiles['RDM_brain']):
					print(f'\t file for {subjectID} exists! File: {outFiles['RDM_brain']}', flush=True)
					centers_masks.append(cfg.get_centers_mask())
					RDM_brain = joblib.load(outFiles['RDM_brain'])
					mask =nib.load(cfg.get_mask_file())
					RDM_brain_list.append(RDM_brain)
				else:
					print(f'\t file  for {subjectID} does not exists! File: {outFiles['RDM_brain']}', flush=True)

			common_centers_mask = find_common_centers(centers_masks, mask, show_plot=False)
			mask_1d = common_centers_mask.flatten()

			X = data3d_to_masked1d(RDM_brain_list, mask_1d)

			# random model evaluations
			random_models = get_random_model_like(n=n_random, like_model=model, method=method)

			X_random = joblib.Parallel(n_jobs=n_jobs_subj)(
				joblib.delayed(_get_random_evaluation)(
					subjectID, config_class_name, model, random_models, mask_1d, show_progress=show_progress, n_jobs = n_jobs_eval
				)
				for subjectID in all_subjects
			)

			outFile = os.path.join(cfg.outDir_inference, f'{n_random}_models_{model}_method_{method}.pkl')
			joblib.dump(
				{'X': X,
				'X_random': X_random}, 
				outFile)
