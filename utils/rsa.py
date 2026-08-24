import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import sys
from copy import deepcopy

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
from contextlib import redirect_stdout
import rsatoolbox as rsa
from rsatoolbox.data import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.rdm import calc_rdm, RDMs
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import (
	get_volume_searchlight,
	evaluate_models_searchlight
)

from mne import read_epochs

from utils.load_cfg import *
from utils.subj import *
from utils.plots import *


def data4modelRDM():
	# create the matrices the model_RDMs will be built from. 
	# Will be used to remove certain conditions before building the modelRDMs, 
	# when there was missing data (full condition missing) in a subject

	# conditions
	modality_vec = np.concatenate([np.zeros((1,8)), np.ones((1,8)), np.ones((1,8))*2]).flatten()
	perc_vec = np.tile(np.concatenate([np.zeros((1,4)), np.ones((1,4))]).flatten(), 3)
	stim_vec = np.tile(np.array(np.array(np.arange(1,5))).flatten(), 6)

	# RDM descriptors:
	descr_modality_vec = np.concatenate([['aud_']*8, ['tac_']*8, ['vis_']*8])
	descr_perc_vec = np.tile(np.concatenate([['hit']*4, ['miss']*4]), 3)
	descr_stim_vec = np.tile([('_1'), ('_2'), ('_3'), ('_4') ], 6)
	RDM_descriptor = descr_modality_vec + descr_perc_vec + descr_stim_vec
	RDM_descriptor = [str(s) for s in RDM_descriptor]

	data = {'Modality': modality_vec,
			'Perceived': perc_vec,
			'Stimulus': stim_vec}
	concept_df = pd.DataFrame(data)
	ConsciousMat = np.zeros((len(concept_df), len(concept_df)))
	SensoryMat = np.zeros((len(concept_df), len(concept_df)))
	for ii in range(0,len(concept_df)):
		for kk in range(0,len(concept_df)):
			
			#make the supramodal model
			if concept_df["Perceived"].iloc[ii] == concept_df["Perceived"].iloc[kk]:
				ConsciousMat[ii, kk] = 0
			else:
				ConsciousMat[ii, kk] = 1
			
			#make the sensory model    
			if concept_df["Modality"].iloc[ii] == concept_df["Modality"].iloc[kk]:
				SensoryMat[ii, kk] = np.abs(concept_df["Stimulus"].iloc[ii] - concept_df["Stimulus"].iloc[kk]) / 4
			else:
				SensoryMat[ii, kk] = 1

	return SensoryMat, ConsciousMat, RDM_descriptor


def check_condition_missing_in_all_runs(info):
	info_full = joblib.load(os.path.join(RESOURCE_DIR, 'info.pkl'))
	
	conditions_subj = np.unique(np.asarray(info['condition']))
	conditions_full = np.unique(np.asarray(info_full['condition']))

	missing_in_all_runs = np.setdiff1d(conditions_full, conditions_subj)

	return missing_in_all_runs


def check_conditions_missing(info, info_full):
	'''
	check whether either
		a) one ore more conditions are missing in some of the runs 
		   --> We will replace the missing data with the mean of the respective 
			   condition from the other runs. We will use the "full_info" (that includes all conditions ins all runs) 

		b) one condition is missing in all runs 
		   --> compute the RDMs for the subjects using the subject-specific info (where one condition is completely missing).
			   We will later construct individual model-RDMs that don't include the missing condition for this subject  
	'''
	conditions_missing = False
	which_conditions_missing = None
	# missing_in_all_runs = np.setdiff1d(np.unique(np.asarray(info['condition'])), np.unique(np.asarray(info_full['condition']))).size > 0 # check if there is a condition that is missing in all runs --> no imputation needed, but different model-RDM later
	missing_in_all_runs = check_condition_missing_in_all_runs(info).size > 0
	print(f'missing in all runs: {missing_in_all_runs}')
	if not missing_in_all_runs:
		
		if len(info_full) > len(info): # imputation --> replace missing value with mean of condition in other runs
			conditions_missing = True 
			which_conditions_missing = np.setdiff1d(info_full['identifier'].values, info['identifier'].values) # missing conditions; key = 'identifiers' do identify condition in different runs
			
			print(f'\nwe have {which_conditions_missing.shape[0]} missing conditions --> using imputation\n')
	else:
		print(f'\n-----------------------------------------------------------------\n'\
				'at least one condition is missing in all runs \n --> using subject-specific instead of full conditions-info'\
			   '\n-----------------------------------------------------------------\n')

	return conditions_missing, which_conditions_missing


def deal_with_missing_conditions(info, info_full):
	conditions_missing, _ = check_conditions_missing(info, info_full)

	if conditions_missing: # condisions were missing , but not in all runs --> imputation is used to replace the missing condition onme or more runs, so we need to use the full info instead, since we manupulated the data to be complete again
		conditions = list(dict.fromkeys(info_full['condition'])) # use full list of conditions because we replaced missing values when RDMs were computed
	else: # no conditions were missing, or a condition was missing in all runs, so we didn't use imputation, so we can just use the info that is already there
		conditions = list(dict.fromkeys(info['condition']))
	return conditions


def reformat_info(info, info_full, events, conditions_missing):
	info['events'] = info[events]
	info_double = [] 
	info_double = pd.DataFrame(info_double)

	if conditions_missing:
		info_full['events'] = info_full[events]
		info_double['events'] = info_full['events'].astype('double')
		info_double['run_number'] = info_full['run_number'].astype('double')

	else:
		info_double['events'] = info['events'].astype('double')
		info_double['run_number'] = info['run_number'].astype('double')
	return info_double


def adjust_descriptors(descriptors):
	modality_map = {
			'auditory': 'aud',
			'somato': 'tac',
			'visual': 'vis'
		}

		# transform MEG descriptors into same format as fMRI descriptors
	renamed_events = []
	for e in descriptors:
		parts = e.split('/')  # e.g., ['auditory', 'hit', '1']

		if 'NT' in parts: #<------------------------ this is new... coded this blindly for testing. Should adjust the descriptors, if they contain "NT" prefix
			idx_fix = 1
		else:
			idx_fix = 0
		modality = modality_map[parts[0+idx_fix]]
		state = parts[1+idx_fix]
		number = parts[2+idx_fix]
		normalized = f"{modality}_{state}_{number}"
		renamed_events.append(normalized)
	return renamed_events


def _count_trials_per_condition_from_file(MEG_file):
	'''
	MEG_file: can pe 'pkl', '.dat' or '.fif'. Must contain epoched data.

	'''
	if MEG_file.endswith('.pkl') or MEG_file.endswith('.dat'):
		epochs = (joblib.load(MEG_file)['epochs_meg'])
	elif MEG_file.endswith('.fif'):
		epochs = read_epochs(MEG_file)['NT']

	event_id = epochs.event_id
	unique_events, counts = np.unique(epochs.events[:,-1], return_counts=True)
	event_counts = dict([[u.item(), c.item()] for u, c in zip(unique_events, counts)])
	event_matched = {k: event_counts[v] for k, v in event_id.items()}
	return event_counts, event_matched


def which_conditions_enough_MEG_trials(cfg):

	subjectID = cfg.subjectID
	min_trls  = cfg.min_MEG_trials

	# MEG_cfg = MEG_cfg

	if hasattr(cfg, "MEG_config"): # input cfg is a fusion config
		event_count_dir = os.path.join(cfg.MEG_cfg.dataDir, cfg.MEG_cfg.dataFolder)
		MEG_file = cfg.MEG_cfg.MEG_inFile[0]
	else: # input cfg is a MEG config
		event_count_dir = os.path.join(cfg.dataDir, cfg.dataFolder)
		MEG_file = cfg.MEG_inFile[0]

	thisFile = os.path.join(event_count_dir, f'{subjectID}_event_counts.pkl')
	if os.path.exists(thisFile):
		counts = joblib.load(thisFile)['event_matched']
	else:
		_, counts = _count_trials_per_condition_from_file(MEG_file)

	[counts.pop(k) for k in [key for key, value in counts.items() if value < min_trls]]

	if len(counts) < 24:
		print(f'subjectID {subjectID} has only {len(counts)} conditions with enough trials\n', flush=True)

	return adjust_descriptors(counts.keys())


def adjust_model_rdm(model, descriptors_data):
	model_subset = model.rdm_obj.subset_pattern('condition', descriptors_data)
	model_name = model.name
	del model
	model = rsa.model.ModelFixed(model_name, model_subset)
	return model


def reorder_rdms(SL_rdms, models):
	descriptors_data = SL_rdms.pattern_descriptors['condition']
	if isinstance(models, list): # list with multiple models
		descriptors_model = models[0].rdm_obj.pattern_descriptors['condition']
	else: # single
		descriptors_model = models.rdm_obj.pattern_descriptors['condition']

	if len(descriptors_data) == len(descriptors_model):
		RDM_conditions_order = np.searchsorted(descriptors_model, descriptors_data,'left')
		SL_rdms.reorder(RDM_conditions_order)
	elif len(descriptors_data) < len(descriptors_model): # adjust model-RDM
		print(f'\n-----------------------------------------------------------------\n'\
				'at least one condition is missing in all runs \n --> constructing individual model-RDM'\
				'\n-----------------------------------------------------------------\n', flush=True)
		if isinstance(models, list):
			models = [adjust_model_rdm(model, descriptors_data) for model in models]
			descriptors_model = models[0].rdm_obj.pattern_descriptors['condition']
		else:
			models = adjust_model_rdm(models, descriptors_data)
			descriptors_model = models.rdm_obj.pattern_descriptors['condition']
		RDM_conditions_order = np.searchsorted(descriptors_model, descriptors_data, 'left') 
		SL_rdms.reorder(RDM_conditions_order)
	else:
		print("There is a problem! There are more conditions in the data than in the model")
	return SL_rdms, models


def reorder_and_subset_all_data(cfg, SL_rdms, rdm_movie, models, save_info=True):
	'''
	Reorder and subset all data (MEG, fMRI, model) to only include conditions that are both in MEG and fMRI data and have enough trials in MEG data. 

	Steps:
		1) find conditions, where we dont't have enough trials in the MEG data (e.g. less than 10) and drop them from the MRI data by subsetting the RDMs.
		2) only use conditions that are both in MRI and (selected) MEG data
		3) subset the MRI data to match the MEG data
		4) adjust model, if conditions are missing in fMRI data (could be because of subsetting or because they were missing from the beginning)
		5) adjust MEG data to match the fMRI data, if condition was missing in fMRI data (or not enough trials in MEG data)

	'''
	output_info = []
	 
	# ~~~ Reordering step 1: find conditions, where we dont't have enough trials in the MEG data (e.g. less than 10) and drop them from the MRI data by subsetting the RDMs.
	selected_MEG_conditions = which_conditions_enough_MEG_trials(cfg)
	
	# ~~~ Reordering step 2: only use conditions that are both in (selected) MEG and MRI 
	common_valid_conditions = list(set(selected_MEG_conditions) & set(SL_rdms.pattern_descriptors['condition']))

	output_info.append('\n----------------------------------------------------------------------------------------\n'
					f'\tnumber of conditions with enough MEG trials: {len(selected_MEG_conditions)}'
					f'\tnumber of conditions in MRI rdms:  {len(SL_rdms.pattern_descriptors["condition"])}\n'
					f'\tnumber of common_valid_conditions: {len(common_valid_conditions)}')
	if len(selected_MEG_conditions)<24:
		output_info.append(f'\n\t\t missing MEG condition(s):\t{list(set(models[0].rdm_obj.pattern_descriptors['condition'])-set(selected_MEG_conditions))}')
	if len(SL_rdms.pattern_descriptors["condition"])<24:
		output_info.append(f'\n\t\t missing MRI condition(s):\t{list(set(models[0].rdm_obj.pattern_descriptors['condition'])-set(SL_rdms.pattern_descriptors["condition"]))}')
	if len(common_valid_conditions)<24:
		output_info.append(f'\n\t\t missing condition(s) total:\t{list(set(models[0].rdm_obj.pattern_descriptors['condition'])-set(common_valid_conditions))}') 

	# ~~~ Reordering step 3: subset the MRI data to match the MEG data
	SL_rdms = SL_rdms.subset_pattern('condition', common_valid_conditions)
	output_info.append(f'\n\tnumber of conditions in MRI rdms after subsetting:  {len(SL_rdms.pattern_descriptors["condition"])}')

	# ~~~ Reordering step 4: adjust model, if conditions were missing in fMRI data (or not enough trials in MEG data)
	SL_rdms, models = reorder_rdms(SL_rdms, models) 

	output_info.append(f'\tnumber of conditions in MRI rdms after reordering:  {len(SL_rdms.pattern_descriptors["condition"])}\n'
					   f'\tnumber of conditions in model rdms after reordering:  {len(models[0].rdm_obj.pattern_descriptors['condition'])} and {len(models[1].rdm_obj.pattern_descriptors['condition'])}')

	# ~~~ Reordering step 5: adjust MEG data if condition was missing in fMRI data
	subset_descriptors = models[0].rdm_obj.pattern_descriptors['condition']
	rdm_movie = rdm_movie.subset_pattern('condition', subset_descriptors)
	output_info.append(f'\tnumber of conditions in MEG rdms after subsetting:  {len(rdm_movie.pattern_descriptors["condition"])}\n'
					   '----------------------------------------------------------------------------------------\n')
	
	print("".join(output_info), flush=True)
	if save_info:
		print('Saving info...', flush=True)
		filepath = os.path.join(cfg.fusionDir, cfg.__class__.__name__, f'valid_conditions_{cfg.subjectID}.txt')
		print(f'filepath: {filepath}', flush=True)
		with open(filepath, 'w') as f:
			with redirect_stdout(f):
				print("".join(output_info), flush=True)
	return SL_rdms, rdm_movie, models


def get_searchlight_RDMs_crossnobis(spm,
									centers, 
									neighbors, 
									mask,
									reg_mask, 
									info, 
									events='condition_number',
									method='crossnobis', 
									verbose=True):
	
	info_full = joblib.load(os.path.join(RESOURCE_DIR, 'info.pkl'))

	conditions_missing, which_conditions_missing = check_conditions_missing(info, info_full)

	info_double = reformat_info(info, info_full, events, conditions_missing)

	# original mask
	mask_bool = mask.get_fdata() > 0
	n_voxels_total = np.prod(mask_bool.shape)
	mask_bool_1D = mask_bool.flatten()
	mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)
	
	#initalize variables
	n_centers = centers.shape[0]

	def get_dataset(center, nb):

		SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
		SL_mask_1D[nb] = True # !!!!!!! important!! must be  [nb], not SL_mask_idx
		SL_mask_3D = SL_mask_1D.reshape(mask.shape)
		SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)
		
		# print('     - Loading SL_betas...', flush=True)
		SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
		SL_beta = np.nan_to_num(SL_beta)
		SL_ResMS = np.nan_to_num(SL_ResMS)
		SL_beta = SL_beta[reg_mask.to_numpy(), :]

		if conditions_missing: # if we have missing conditions
			for id in which_conditions_missing:     
				missing_identifier_idx = np.where(info_full['identifier'].values == id)[0].item()           # 1) index of identifier in full set of conditions
				missing_condition = which_conditions_missing[0][:-5]                                        # 2) find indices/labels of condition in other runs of current data
				missing_condition_idx = info.index[info['condition'] == missing_condition].tolist() 
				missing_condition_values = np.mean(SL_beta[missing_condition_idx], axis = 0)                 # 3) compute sum of betas at these indices

				SL_beta = np.insert(SL_beta, missing_identifier_idx, missing_condition_values, axis = 0)    # 4) insert condition at right index to mathc full_info order            

		# print(f'       - Loading Residuals...', flush=True)
		SL_residuals, _, _ = spm.get_residuals(SL_mask_img)
		SL_residuals = np.nan_to_num(SL_residuals)

		# print(f'         - Computing Precision Matrix...', flush=True)

		SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')
		# SL_Prec = np.nan_to_num(SL_Prec)

		measurements = SL_beta / np.sqrt(SL_ResMS)
		measurements = np.nan_to_num(measurements)
		ds = Dataset(measurements = measurements, 
					descriptors={'center': center},
					obs_descriptors=dict(info_double),
					channel_descriptors={'voxels': nb})
		
		return ds, SL_Prec

	if n_centers > 1000:
		print('processing centers in chunks...', flush=True)
		# we can't run all centers at once, that will take too much memory
		# so lets to some chunking
		chunked_center = np.split(np.arange(n_centers),
									np.linspace(0, n_centers, 
												101, dtype=int)[1:-1]) 
		# loop over chunks
		n_conds = len(np.unique(info_double['events']))
		n_conds = len(np.unique(info_double['events']))
		RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
		
		for chunk in tqdm(chunked_center, desc='Calculating RDMs...', flush=True):
			center_data, center_noise = [], []
			for c in chunk:

				center = centers[c]
				nb = neighbors[c]

				print(f'current center Nr:{c} = center {center}', flush=True)

				ds, SL_Prec = get_dataset(center, nb)
				
				center_data.append(ds)
				center_noise.append(SL_Prec)

			print('calculating RDMs for current chunk...', flush=True)

			RDM_corr = calc_rdm(center_data, 
						method=method,
						descriptor='events', 
						noise=center_noise,
						cv_descriptor='run_number')
			
			'''
			inside 'calc_rdm', the function 'build_rdm' (from rsatoolbox.utils) is called at some point which  would nortmally append the noise as
			a descriptor. But since not all Searchlights have exactly the same number of voxels and hence the noise cov matrices have different
			dimesions, adding the noise as descriptor causes an error, since descriptors need to have the same dimensions.
			I therefore commented out the lines 

			if noise is not None:
				rdm.noise = noise[i]
				rdm.noise_descriptor = 'noise'    

			inside 'build_rdm' as a patch, so we can run this code anyways.
			
			'''

			RDM[chunk, :] = RDM_corr.dissimilarities
		
	else:
		print('processing all centers at once...', flush=True)
		center_data, center_noise = [], []
		for c in range(n_centers):
			center = centers[c]
			nb = neighbors[c]

			print(f'current center Nr:{c} = center {center}', flush=True)

			ds, SL_Prec = get_dataset(center, nb)
			
			center_data.append(ds)
			center_noise.append(SL_Prec)
		
		RDM = calc_rdm(center_data, 
			method=method,
			descriptor='events', 
			noise=center_noise,
			cv_descriptor='run_number').dissimilarities

	SL_rdms = RDMs(RDM,
				rdm_descriptors={'voxel_index': centers},
				dissimilarity_measure=method)
	
	return SL_rdms


def get_searchlight_RDMs_crossnobis_parallel(spm,
									centers, 
									neighbors, 
									mask,
									reg_mask, 
									info, 
									events='condition_number',
									method='crossnobis', 
									verbose=True):
	
	'''
	Docstring for get_searchlight_RDMs_crossnobis_parallel. Since "(n_jobs=-1)" the number of parallel computations
	depends on the number of cores available. It also differes from the function from above in the way that chunking/parallelisation 
	starts if n_cencter >500 instead of >1000.
	
	Does the same as get_searchlight_RDMs_crossnobis but uses prallel computing. 

	:param spm: Description
	:param centers: Description
	:param neighbors: Description
	:param mask: Description
	:param reg_mask: Description
	:param info: Description
	:param events: Description
	:param method: Description
	:param verbose: Description
	'''
	
	info_full = joblib.load(os.path.join(RESOURCE_DIR, 'info.pkl'))

	conditions_missing, which_conditions_missing = check_conditions_missing(info, info_full)

	info_double = reformat_info(info, info_full, events, conditions_missing)

	# original mask
	mask_bool = mask.get_fdata() > 0
	n_voxels_total = np.prod(mask_bool.shape)
	mask_bool_1D = mask_bool.flatten()
	mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)
	
	#initalize variables
	n_centers = centers.shape[0]

	def get_dataset(center, nb):

		SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
		SL_mask_1D[nb] = True # !!!!!!! important!! must be  [nb], not SL_mask_idx
		SL_mask_3D = SL_mask_1D.reshape(mask.shape)
		SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)
		
		# print('     - Loading SL_betas...', flush=True)
		SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
		SL_beta = np.nan_to_num(SL_beta)
		SL_ResMS = np.nan_to_num(SL_ResMS)
		SL_beta = SL_beta[reg_mask.to_numpy(), :]

		if conditions_missing: # if we have missing conditions
			for id in which_conditions_missing:     
				missing_identifier_idx = np.where(info_full['identifier'].values == id)[0].item()           # 1) index of identifier in full set of conditions
				missing_condition = which_conditions_missing[0][:-5]                                        # 2) find indices/labels of condition in other runs of current data
				missing_condition_idx = info.index[info['condition'] == missing_condition].tolist() 
				missing_condition_values = np.mean(SL_beta[missing_condition_idx], axis = 0)                 # 3) compute sum of betas at these indices =================> I HAD MISTAKE HERE; BECAUSE I USED "SUM" INSTEAD OF "MEAN" HERE BEFORE!!!

				SL_beta = np.insert(SL_beta, missing_identifier_idx, missing_condition_values, axis = 0)    # 4) insert condition at right index to mathc full_info order            

		# print(f'       - Loading Residuals...', flush=True)
		SL_residuals, _, _ = spm.get_residuals(SL_mask_img)
		SL_residuals = np.nan_to_num(SL_residuals)

		# print(f'         - Computing Precision Matrix...', flush=True)

		SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')
		# SL_Prec = np.nan_to_num(SL_Prec)

		measurements = SL_beta / np.sqrt(SL_ResMS)
		measurements = np.nan_to_num(measurements)
		ds = Dataset(measurements = measurements, 
					descriptors={'center': center},
					obs_descriptors=dict(info_double),
					channel_descriptors={'voxels': nb})
		
		return ds, SL_Prec

	def process_center(c):
		center = centers[c]
		nb = neighbors[c]

		'''
		for debugging and visualizing the centers use:
		
			from utils.plots import plot_centers
			plot_centers(neighbors[c], cfg.subjectID)
		'''
		print(f'current center Nr:{c} = center {center}', flush=True)
		ds, SL_Prec = get_dataset(center, nb)
		return ds, SL_Prec
	
	if n_centers > 500:
		print('processing centers in chunks...', flush=True)
		# we can't run all centers at once, that will take too much memory
		# so lets to some chunking
		chunked_center = np.split(np.arange(n_centers),
									np.linspace(0, n_centers, 
												101, dtype=int)[1:-1]) 
		# loop over chunks
		n_conds = len(np.unique(info_double['events']))
		RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))

		for chunk in tqdm(chunked_center, desc='Calculating RDMs...'):
			# Parallelize over centers in the chunk
			results = Parallel(n_jobs=-1)(
				delayed(process_center)(c) for c in chunk
			)
			center_data, center_noise = zip(*results)

			print('calculating RDMs for current chunk...', flush=True)

			RDM_corr = calc_rdm(center_data, 
						method=method,
						descriptor='events', 
						noise=center_noise,
						cv_descriptor='run_number')

			'''
			inside 'calc_rdm', the function 'build_rdm' (from rsatoolbox.utils) is called at some point which  would nortmally append the noise as
			a descriptor. But since not all Searchlights have exactly the same number of voxels and hence the noise cov matrices have different
			dimesions, adding the noise as descriptor causes an error, since descriptors need to have the same dimensions.
			I therefore commented out the lines 

			if noise is not None:
				rdm.noise = noise[i]
				rdm.noise_descriptor = 'noise'    

			inside 'build_rdm' as a patch, so we can run this code anyways.
			
			'''

			RDM[chunk, :] = RDM_corr.dissimilarities
		
	else:
		print('processing all centers at once...', flush=True)
		center_data, center_noise = [], []

		# original sequential method:
		# for c in range(n_centers):
		# 	center = centers[c]
		# 	nb = neighbors[c]
		# 	print(f'current center Nr:{c} = center {center}', flush=True)
		# 	ds, SL_Prec = get_dataset(center, nb)
		# 	center_data.append(ds)
		# 	center_noise.append(SL_Prec)

		# parallel version:
		args = [(centers[c], neighbors[c], c) for c in range(n_centers)]
		def _wrap(arg):
			center, nb, idx = arg
			print(f'current center Nr:{idx} = center {center}', flush=True)
			return get_dataset(center, nb)

		results = joblib.Parallel(n_jobs=-1)(joblib.delayed(_wrap)(a) for a in args)

		for ds, SL_Prec in results:
			center_data.append(ds)
			center_noise.append(SL_Prec)

		RDM = calc_rdm(center_data, 
			method=method,
			descriptor='events', 
			noise=center_noise,
			cv_descriptor='run_number').dissimilarities
		
		'''
		inside 'calc_rdm', the function 'build_rdm' (from rsatoolbox.utils) is called at some point which  would nortmally append the noise as
		a descriptor. But since not all Searchlights have exactly the same number of voxels and hence the noise cov matrices have different
		dimesions, adding the noise as descriptor causes an error, since descriptors need to have the same dimensions.
		I therefore commented out the lines 

		if noise is not None:
			rdm.noise = noise[i]
			rdm.noise_descriptor = 'noise'    

		inside 'build_rdm' as a patch, so we can run this code anyways.
		
		'''

	SL_rdms = RDMs(RDM,
				rdm_descriptors={'voxel_index': centers},
				dissimilarity_measure=method)
	
	return SL_rdms


def get_RDM_brain(mask, SL_rdms, eval_score):
	x, y, z = mask.shape
	RDM_brain = np.zeros([x*y*z])
	RDM_brain[SL_rdms.rdm_descriptors['voxel_index']] = eval_score
	RDM_brain = RDM_brain.reshape(mask.shape)
	return RDM_brain


def get_mean_movie_new(cfg_in):
	'''
	Compute the mean of the RDM-movies of sujects, that are currently included 
	in the fusion analysis (=subjects where MEG + fMRI data exists).
	Outputs a variable of the class "RDMs"
	'''

	all_subj = get_fusion_subjects()
	cfg = load_MEG_config_instance(cfg_in.MEG_config, all_subj[0]) 

	all_RDMs = []
	for subj in all_subj:

		cfg.subjectID = subj

		movieFile = cfg.get_outFile_names()['movie']

		if os.path.exists(movieFile):
			rdm_movie = joblib.load(movieFile)

			# selected_MEG_conditions = which_conditions_enough_MEG_trials(subj, min_trls=cfg_in.min_MEG_trials)
			# rdm_movie = rdm_movie.subset_pattern('condition', selected_MEG_conditions)

			all_RDMs.append(rdm_movie.get_vectors())

	mean_rdms = np.mean(np.array(all_RDMs), axis = 0)
	mean_movie = deepcopy(rdm_movie)
	mean_movie.dissimilarities = mean_rdms # put the data back in necessary RDMs structure
	return mean_movie


def tmp_add_condition_rdm_descriptor(cfg, modelNames = ALL_MODELS):    
	print("running the 'tmp_add_condition_rdm_descriptor' function", flush=True)
	if cfg.maskNr == 0:
		SL_rdms     = joblib.load(cfg.get_outFile_names()['SL_rdms'])
	else:
		SL_rdms     = joblib.load(cfg.get_outFile_names()['SL_rdms_partial'])

	info        = joblib.load(cfg.get_outFile_names()['info'])
	info_full   = joblib.load(os.path.join(RESOURCE_DIR, 'info.pkl'))
	models      = cfg.get_model_RDM()
	mask        = nib.load(cfg.get_mask_file())
	outFiles	= cfg.get_outFile_names()

	conditions = deal_with_missing_conditions(info, info_full)

	SL_rdms.pattern_descriptors['condition'] = conditions

	SL_rdms, models = reorder_rdms(SL_rdms, models)
		# 6) compare Searchlight RDMs with model RDM

	print(f"conditions: {conditions}")
	print("Saving SL_rdms")

	if cfg.maskNr == 0:
		joblib.dump(SL_rdms, outFiles['SL_rdms'])
	else:
		joblib.dump(SL_rdms, outFiles['SL_rdms_partial'])


def save_RSA_outputs(cfg, modelNames = ALL_MODELS):    
	'''
	Saves the output (e.g. model evaluations. Also adds the "conditions" descriptor as RDM descriptor and saves the SL_Rdms again. 
	This is necessarty, because we need the descriptor for the fusion)
	'''
	if cfg.maskNr == 0:
		SL_rdms     = joblib.load(cfg.get_outFile_names()['SL_rdms'])
	else:
		SL_rdms     = joblib.load(cfg.get_outFile_names()['SL_rdms_partial'])

	info        = joblib.load(cfg.get_outFile_names()['info'])
	info_full   = joblib.load(os.path.join(RESOURCE_DIR, 'info.pkl'))
	models      = cfg.get_model_RDM()
	mask        = nib.load(cfg.get_mask_file())
	outFiles	= cfg.get_outFile_names()

	conditions = deal_with_missing_conditions(info, info_full)

	SL_rdms.pattern_descriptors['condition'] = conditions

	SL_rdms, models = reorder_rdms(SL_rdms, models)
	

	# 6) compare Searchlight RDMs with model RDM

	print('    - Evaluating RDMs...')

	eval_results = evaluate_models_searchlight(SL_rdms, models, eval_fixed, method = cfg.RSAmethod)

	# 8) extract and plot data separately for sensory & suprasensory model
	if cfg.maskNr == 0:
		joblib.dump(SL_rdms, outFiles['SL_rdms'])
		joblib.dump(eval_results, outFiles['eval_results'])

		for model in modelNames:
			# ------------------------ v 
			cfg.modelType = model
			cfg.configure_paths()
			outFiles = cfg.get_outFile_names()
			# ------------------------ ^ important! for right prefix of files (sensory/suprasensory)

			eval_score = list(np.concatenate([e.evaluations[0][cfg.modelIdx] for e in eval_results]))
			RDM_brain = get_RDM_brain(mask, SL_rdms, eval_score) # to display eval scores of model comparison in brain-shape

			joblib.dump(eval_score, outFiles['eval_score'])
			joblib.dump(RDM_brain, outFiles['RDM_brain'])

			eval_score_histogram(cfg, 
								eval_score)
			
			plot_brain_map(cfg, 
						mask, 
						RDM_brain, 
						eval_score)
			
			plot_max_modelFit_rdm(cfg, 
								SL_rdms, 
								eval_score)
	else:
		print(f"-------------------------------------------------------------- \n Since this is only the partial mask, we skip computing the eval_score and RDM_brain\n-------------------------------------------------------------- \n ")
		joblib.dump(SL_rdms, outFiles['SL_rdms_partial'])
		joblib.dump(eval_results, outFiles['eval_results_partial'])


def get_RSA_outputs(cfg, modelName = 'sensory'):    

	cfg.modelType = modelName
	cfg.configure_paths()
	outFiles = cfg.get_outFile_names()

	if cfg.maskNr == 0:
		SL_rdms     = joblib.load(cfg.get_outFile_names()['SL_rdms'])
	else:
		SL_rdms     = joblib.load(cfg.get_outFile_names()['SL_rdms_partial'])
	info        = joblib.load(cfg.get_outFile_names()['info'])
	info_full   = joblib.load(os.path.join(RESOURCE_DIR, 'info.pkl'))
	models      = cfg.get_model_RDM()
	mask        = nib.load(cfg.get_mask_file())

	conditions = deal_with_missing_conditions(info, info_full)

	SL_rdms.pattern_descriptors['condition'] = conditions

	print('    - Reordering RDMs and models...')
	SL_rdms, models = reorder_rdms(SL_rdms, models)

	print('    - Evaluating RDMs...')

	eval_results = evaluate_models_searchlight(SL_rdms, models, eval_fixed, method = cfg.RSAmethod, n_jobs=-1)

	eval_score = [float(e.evaluations[0][0][0]) for e in eval_results]
	RDM_brain = get_RDM_brain(mask, SL_rdms, eval_score)

	return SL_rdms, models, eval_results, eval_results, eval_score, RDM_brain




def get_searchlight_RDMs_parallel(data_2d, centers, neighbors, events,
						 method='correlation', verbose=True):
	"""Iterates over all the searchlight centers and calculates the RDM

	Args:

		data_2d (2D numpy array): brain data,
		shape n_observations x n_channels (i.e. voxels/vertices)

		centers (1D numpy array): center indices for all searchlights as provided
		by rsatoolbox.util.searchlight.get_volume_searchlight

		neighbors (list): list of lists with neighbor voxel indices for all searchlights
		as provided by rsatoolbox.util.searchlight.get_volume_searchlight

		events (1D numpy array): 1D array of length n_observations

		method (str, optional): distance metric,
		see rsatoolbox.rdm.calc for options. Defaults to 'correlation'.

		verbose (bool, optional): Defaults to True.

	Returns:
		RDM [rsatoolbox.rdm.RDMs]: RDMs object with the RDM for each searchlight
							  the RDM.rdm_descriptors['voxel_index']
							  describes the center voxel index each RDM is associated with
	"""

	data_2d, centers = np.array(data_2d), np.array(centers)
	n_centers = centers.shape[0]

	# For memory reasons, we chunk the data if we have more than 1000 RDMs
	if n_centers > 1000:
		# we can't run all centers at once, that will take too much memory
		# so lets to some chunking
		chunked_center = np.split(np.arange(n_centers),
								  np.linspace(0, n_centers,
											  101, dtype=int)[1:-1])

		# loop over chunks
		n_conds = len(np.unique(events))
		RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
		# for chunks in tqdm(chunked_center, desc='Calculating RDMs...'):
		#     center_data = []
		#     for c in chunks:
		#         # grab this center and neighbors
		#         center = centers[c]
		#         center_neighbors = neighbors[c]
		#         # create a database object with this data
		#         ds = Dataset(data_2d[:, center_neighbors],
		#                      descriptors={'center': center},
		#                      obs_descriptors={'events': events},
		#                      channel_descriptors={'voxels': center_neighbors})
		#         center_data.append(ds)

		#     RDM_corr = calc_rdm(center_data, method=method,
		#                         descriptor='events')
		#     RDM[chunks, :] = RDM_corr.dissimilarities


		def process_chunk(chunks, data_2d, centers, neighbors, events, method):
			center_data = []
			for c in chunks:
				center = centers[c]
				center_neighbors = neighbors[c]
				ds = Dataset(
					data_2d[:, center_neighbors],
					descriptors={'center': center},
					obs_descriptors={'events': events},
					channel_descriptors={'voxels': center_neighbors})

				center_data.append(ds)

			rdm_corr = calc_rdm(
				center_data,
				method=method,
				descriptor='events')

			return chunks, rdm_corr.dissimilarities


		results = Parallel(n_jobs=-1,backend='loky')(delayed(process_chunk)(
														chunks=chunks,
														data_2d=data_2d,
														centers=centers,
														neighbors=neighbors,
														events=events,
														method=method)
			for chunks in tqdm(chunked_center,desc='Submitting RDM chunks...'))

		for chunks, dissimilarities in results:
			RDM[chunks, :] = dissimilarities

	else:
		center_data = []
		for c in range(n_centers):
			# grab this center and neighbors
			center = centers[c]
			nb = neighbors[c]
			# create a database object with this data
			ds = Dataset(data_2d[:, nb],
						 descriptors={'center': c},
						 obs_descriptors={'events': events},
						 channel_descriptors={'voxels': nb})
			center_data.append(ds)
		# calculate RDMs for each database object
		RDM = calc_rdm(center_data, method=method,
					   descriptor='events').dissimilarities

	SL_rdms = RDMs(RDM,
				   rdm_descriptors={'voxel_index': centers},
				   dissimilarity_measure=method)

	return SL_rdms