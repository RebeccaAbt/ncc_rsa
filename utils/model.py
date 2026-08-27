#%%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.plots import *
import joblib
import pandas as pd
import numpy as np
import rsatoolbox as rsa
import matplotlib.pyplot as plt

#%%

def _get_rdm_descriptor():
	descr_modality_vec = np.concatenate([['aud_']*8, ['tac_']*8, ['vis_']*8])
	descr_perc_vec = np.tile(np.concatenate([['hit']*4, ['miss']*4]), 3)
	descr_stim_vec = np.tile([('_1'), ('_2'), ('_3'), ('_4')], 6)
	RDM_descriptor = [f"{m}{p}{s}" for m, p, s in zip(descr_modality_vec, descr_perc_vec, descr_stim_vec)]
	return RDM_descriptor


def _get_concept_df():
	# Modalities: Auditory, Somatosensory, Visual
	modality_vec = np.concatenate([np.zeros((1,8)), np.ones((1,8)), np.ones((1,8))*2]).flatten()
	perc_vec = np.tile(np.concatenate([np.zeros((1,4)), np.ones((1,4))]).flatten(), 3)
	stim_vec = np.tile(np.array(np.array(np.arange(1,5))).flatten(), 6)
	data = {'Modality': modality_vec,
			'Perceived': perc_vec,
			'Stimulus': stim_vec}

	concept_df = pd.DataFrame(data)
	return concept_df

def _random_model(data, k=1, RDM_descriptor=_get_rdm_descriptor()):
	# data must be a lower-triangle vector, e.g. from a rsa model :model.rdm_obj.get_vectors()
	# k: number attached to model name
	n_conditions = int((1 + np.sqrt(1 + 8 * data.size)) / 2)
	if n_conditions * (n_conditions - 1) // 2 != data.size:
		raise ValueError('modeldata must contain exactly the lower triangle data.')

	rng = np.random.default_rng()
	rng.shuffle(data)
	lower_idx = np.tril_indices(n_conditions, k=-1)
	randomMat = np.zeros((n_conditions, n_conditions), dtype=data.dtype)
	randomMat[lower_idx] = data
	randomMat += randomMat.T
	np.fill_diagonal(randomMat, 0)
	randomModel = rsa.model.ModelFixed(f'random_model_{k}', randomMat)
	randomModel.rdm_obj.pattern_descriptors['condition'] = RDM_descriptor
	return randomModel

def get_spec_modelRDM(model_name = ALL_MODELS):
	if isinstance(model_name,list):
		return [get_spec_modelRDM(m) for m in model_name]
	else:
		concept_df = _get_concept_df()
		RDM_descriptor = _get_rdm_descriptor()

		rdm_mat = np.zeros((len(concept_df), len(concept_df)))

		if model_name == 'sensory':
			for i in range(0,len(concept_df)):
				for k in range(0,len(concept_df)):
					if concept_df["Modality"].iloc[i] == concept_df["Modality"].iloc[k]:
						rdm_mat[i, k] = np.abs(concept_df["Stimulus"].iloc[i] - concept_df["Stimulus"].iloc[k]) / 4
					else:
						rdm_mat[i, k] = 1

		elif model_name == 'suprasensory':
			for i in range(0,len(concept_df)):
				for k in range(0,len(concept_df)):
					if concept_df["Perceived"].iloc[i] == concept_df["Perceived"].iloc[k]:
						rdm_mat[i, k] = 0
					else:
						rdm_mat[i, k] = 1

		elif model_name == 'sensory2':
			for i in range(0,len(concept_df)):
				for k in range(0,len(concept_df)):
					if concept_df["Modality"].iloc[i] == concept_df["Modality"].iloc[k]:
						rdm_mat[i, k] = 0
					else:
						rdm_mat[i, k] = 1

		else:
			raise ValueError(f"Invalid RDM name: {model_name}")

		model = rsa.model.ModelFixed(model_name, rdm_mat)
		model.rdm_obj.pattern_descriptors['condition'] = RDM_descriptor

		return model


def get_random_model_like(n=1, like_model='sensory'):
	# 'like_model' can be
	#		- a string of a model type
	# 		- an RSA model object
	if isinstance(like_model, str): # get model if input was a string
		like_model = get_spec_modelRDM(like_model)

	if isinstance(like_model, rsa.model.model.ModelFixed): # if input is (now) a model, get lower tri data
		like_model = like_model.rdm_obj.get_vectors()

	data = np.asarray(like_model).ravel()

	return [_random_model(data,k) for k in range(0,n)]




#%%


# randomModels = get_random_model_like(4)