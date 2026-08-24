

#%% 

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from glob import glob
import joblib
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils.load_cfg import load_fusion_config_instance
from utils.fusion_stat import *
from utils.subj import *

#%%
def find_common_centers(masks, maskFile='/home/reabt/experiments/ncc/MRI/data/sync/19910703eigl/NCC/firstLevel_sensory_M1C/mask.nii', show_plot=True):
	'''
	masks: list of length n_subjects containing 3d boolean masks.
	returns a mask that is True only for voxels that are True in ALL input masks.
	'''
	mask =  np.stack(masks,axis=0)
	print(mask.shape)
	mask = np.all(mask, axis=0) # shape should be (s1, s2, s3) again, but only True where all subjects had True

	# mask =  np.stack([np.all(np.isfinite(t), axis=-1) for t in data], axis=0)

	if show_plot:
		plot_img = new_img_like(maskFile, mask)
		fig = plt.figure(figsize=(12, 3))
		display = plotting.plot_stat_map(
					plot_img, 
					display_mode='z', 
					draw_cross=False, 
					figure=fig,
					# cmap='viridis',
					black_bg=False, 
					annotate=False)
		plt.show()
	return mask.astype(bool)

def get_masked_data(data, mask_1D):
	'''
	Docstring for get_masked_data
	
	:param data: list of length n_subj with arrays of shape (s1 x s2 x s3)
	:param mask_1D: 1d format of original 3d spatial mask

	NOTE: adapted this quickly from the 4d fusion stuff... 
	we also could have just used the 3d mask for masking here, 
	or added some logik to detect the input dimensions to make it more flexible...
	'''
	s_3D = data[0].shape
	# print(f'\t\t\t  - shape of 3D data (s1 x s2 x s3): {s_3D}')
	maskedData = []

	for d in data:
		subj_maskedData = d.flatten()
		subj_maskedData = subj_maskedData[mask_1D] # single subject
		maskedData.append(subj_maskedData) # all subjects
	return np.stack([d for d in maskedData]) 

def significant_clusters_to_3d(clusters, cluster_p_values, no_nan_mask, alpha=0.05, number_per_cluster = True):

	n_vox = int(no_nan_mask.sum())
	cluster_labels_1d = np.zeros(n_vox, dtype=int)

	clu_nr = 0
	for clu, p in zip(clusters, cluster_p_values):
		if p < alpha:
			clu_nr += 1
			idx = clu[0] if isinstance(clu, tuple) else clu
			if number_per_cluster:
				cluster_labels_1d[idx] = clu_nr
			else:
				cluster_labels_1d[idx] = 1

	cluster_img_3d = np.zeros(no_nan_mask.shape, dtype=int)
	cluster_img_3d[no_nan_mask] = cluster_labels_1d
	return cluster_img_3d

def plot_panel(ax, values, title, use_mean=True):
	if use_mean:
		values = np.mean(np.stack(values), axis=0)

	vals = np.ravel(values)

	weights = np.ones(len(vals)) / len(vals)

	lo, hi = np.nanmin(vals), np.nanmax(vals)

	ax.hist(
		vals,
		bins=200,              # fewer bins -> more visible bars
		range=(lo, hi),       # include the full range, including tails
		weights=weights,
		edgecolor="k",
		alpha=0.7
	)

	ax.set_xlim(lo, hi)
	# ax.set_ylim(0, 0.02)      # or inspect the actual max and set a fixed range
	ax.yaxis.set_major_formatter(PercentFormatter(1))
	ax.axvline(x=0, color='g', linestyle='--', linewidth=1)
	ax.set_title(title)


def find_common_centers(masks, maskFile='/home/reabt/experiments/ncc/MRI/data/sync/19910703eigl/NCC/firstLevel_sensory_M1C/mask.nii', show_plot=True):
	'''
	masks: list of length n_subjects containing 3d boolean masks.
	returns a mask that is True only for voxels that are True in ALL input masks.
	'''
	mask =  np.stack(masks,axis=0)
	print(mask.shape)
	mask = np.all(mask, axis=0) # shape should be (s1, s2, s3) again, but only True where all subjects had True

	# mask =  np.stack([np.all(np.isfinite(t), axis=-1) for t in data], axis=0)

	if show_plot:
		plot_img = new_img_like(maskFile, mask)
		fig = plt.figure(figsize=(12, 3))
		display = plotting.plot_stat_map(
					plot_img, 
					display_mode='z', 
					draw_cross=False, 
					figure=fig,
					# cmap='viridis',
					black_bg=False, 
					annotate=False)
		plt.show()
	return mask.astype(bool)
#%% ======================================================================== plot fusion commonalir values


for config_class_name in [
			"FusionConfig_E2",
			"FusionConfig_E5",
			"FusionConfig_C2",
			"FusionConfig_C5"
	]:
	
	cluster_def_method = 'p' # cluster_def_method = 't'
	cp_variables = ['X_pre', 'X_post'] #['X_pre', 'X_post', 'X_diff']

	cfg = load_fusion_config_instance(config_class_name) # since no subject ID is specified, '*' will be used --> creates pattern with subject-wildcard instead of subject filename 

	preStim_time = np.array(range(0, 100)) # should be adapted to dynamically find the right time points... but for now this selects times <=0
	postStim_time = np.array(range(100, 200))

	all_data_pre_sensory, all_data_post_sensory, all_data_pre_suprasensory, all_data_post_suprasensory, maskData_all, centers_masks = [], [], [], [], [], []

	print(f'\n[A] loading the data', flush=True)

	fusionFiles = sorted(glob(cfg.get_outFile_names()['fusion_pkl']))

	print(f'fusionFiles: {fusionFiles}', flush=True)

	cfg.subjectID = '19910823ssld' # dummy subject
	cfg.configure_paths()

	def process_subject(file):
		"""Process a single subject's data and return results for all modalities."""
		cfg.subjectID = check_subj_id(file, cfg)
		cfg.configure_paths()

		# print(f"Processing subject {cfg.subjectID}...", flush=True)
		fusion_data = joblib.load(file)

		mask = nib.load(cfg.get_mask_file())
		mask_data = mask.get_fdata()
		mask_size = mask.shape

		indices = np.unravel_index(fusion_data['voxel_index_py'], mask_size)
		voxel_coords = np.stack(indices, axis=1)

		result = {
			# "maskData": mask_data,
			"pre_sensory": partial_subj_data(mask_size, voxel_coords, fusion_data, preStim_time, modality="sensory", do_print=False),
			"post_sensory": partial_subj_data(mask_size, voxel_coords, fusion_data, postStim_time, modality="sensory", do_print=False),
			"pre_suprasensory": partial_subj_data(mask_size, voxel_coords, fusion_data, preStim_time, modality="suprasensory", do_print=False),
			"post_suprasensory": partial_subj_data(mask_size, voxel_coords, fusion_data, postStim_time, modality="suprasensory", do_print=False),
		}
		return result

	# Run in parallel
	results = Parallel(n_jobs=-1, backend='loky')(delayed(process_subject)(f) for f in fusionFiles)

	# Collect results
	for res in results:
		# maskData_all.append(res["maskData"])
		# centers_masks.append(res["centers_mask"])
		all_data_pre_sensory.append(res["pre_sensory"])
		all_data_post_sensory.append(res["post_sensory"])
		all_data_pre_suprasensory.append(res["pre_suprasensory"])
		all_data_post_suprasensory.append(res["post_suprasensory"])

	all_subj_mask = np.all(np.isfinite(all_data_post_sensory), axis=0)
	all_data_pre_sensory = [np.where(all_subj_mask, data, np.nan) for data in all_data_pre_sensory]
	all_data_post_sensory = [np.where(all_subj_mask, data, np.nan) for data in all_data_post_sensory]
	all_data_pre_suprasensory = [np.where(all_subj_mask, data, np.nan) for data in all_data_pre_suprasensory]
	all_data_post_suprasensory = [np.where(all_subj_mask, data, np.nan) for data in all_data_post_suprasensory]

	fig, axs = plt.subplots(2, 2, sharey=True, figsize=(10, 7))
	fig.suptitle(f"{config_class_name} all subjects")
	plot_panel(axs[0, 0], all_data_pre_sensory, "pre_sensory", use_mean=False)
	plot_panel(axs[1, 0], all_data_post_sensory, "post_sensory", use_mean=False)
	plot_panel(axs[0, 1], all_data_pre_suprasensory, "pre_suprasensory", use_mean=False)
	plot_panel(axs[1, 1], all_data_post_suprasensory, "post_suprasensory", use_mean=False)
	plt.show()

	fig, axs = plt.subplots(2, 2, sharey=True, figsize=(10, 7))
	fig.suptitle(f"{config_class_name} mean across subjects")
	plot_panel(axs[0, 0], all_data_pre_sensory, "pre_sensory", use_mean=True)
	plot_panel(axs[1, 0], all_data_post_sensory, "post_sensory", use_mean=True)
	plot_panel(axs[0, 1], all_data_pre_suprasensory, "pre_suprasensory", use_mean=True)
	plot_panel(axs[1, 1], all_data_post_suprasensory, "post_suprasensory", use_mean=True)
	plt.show()


#%% ======================================================================== plot MRI RSA values

all_subjects =get_MRI_subjects()

for config_class_name in [
	'MRIconfig_C2', 
	'MRIconfig_C5', 
	'MRIconfig_E2', 
	'MRIconfig_E5'
	]:

	allModels = ALL_MODELS
	data_full = dict()
	for i, model in enumerate(allModels):
		# model = allModels[1]

		RDM_brain_list = []
		centers_masks = []
		for subjectID in all_subjects:
			# print(subjectID)
			cfg = load_MRI_config_instance(config_class_name, subjectID)
			cfg.modelType = model  
			outFiles = cfg.get_outFile_names()
			
			RDM_brain = joblib.load(outFiles['RDM_brain'])
			mask =nib.load(cfg.get_mask_file())
			centers_masks.append(cfg.get_centers_mask())
			RDM_brain_list.append(RDM_brain)

		common_centers_mask = find_common_centers(centers_masks, mask, show_plot=False)
		data_full[model] = [np.where(common_centers_mask, data, np.nan) for data in RDM_brain_list]

	fig, axs = plt.subplots(2, 2, sharey=True, figsize=(10, 7))
	fig.suptitle(config_class_name)
	
	for i, model in enumerate(allModels):
		plot_panel(axs[0, i], data_full[model] , f"{model} all subjects", use_mean=False)
		plot_panel(axs[1, i], data_full[model] , f"{model} subjects mean", use_mean=True)
	plt.show()



