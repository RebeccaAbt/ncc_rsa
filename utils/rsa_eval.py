import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
from nilearn.image import new_img_like
from mne.stats import ttest_1samp_no_p

from utils.load_cfg import *
from utils.subj import *
from utils.plots import *


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

def get_all_centers_masks(all_subjects, config_class_name):

	def _get_centers_masks(subjectID, config_class_name):
		cfg = load_MRI_config_instance(config_class_name, subjectID)
		return cfg.get_centers_mask()

	centers_masks = joblib.Parallel(n_jobs=-1)(
					joblib.delayed(_get_centers_masks)(subjectID, config_class_name)
					for subjectID in all_subjects
				)
	cfg = load_MRI_config_instance(config_class_name, all_subjects[0])
	dummyMask =nib.load(cfg.get_mask_file())

	return centers_masks, dummyMask


def get_common_centers_masks(config_class_name):
	all_subjects = get_MRI_subjects(config_class_name)
	centers_masks, mask = get_all_centers_masks(all_subjects, config_class_name)

	assert np.all([len(m)>0 for m in centers_masks]), 'some centers masks appear to have a length < 1 which means that something went wrong'
	common_centers_mask = find_common_centers(centers_masks, mask, show_plot=False)
	return common_centers_mask, mask


def get_model_t_map(X, i_rand = None):
	if isinstance(X[0], np.ndarray) and X[0].ndim == 1:
		print(f"'X' seems to contain the data of {len(X)} subjects, 1 model and {len(X[0])} observations")
		inData = np.stack(X)
	else:
		if i_rand == 0:
			print(f"'X' seems to contain the data of {len(X)} subjects, {len(X[0])} models and {len(X[0][0])} observations. \n\t(showing this output only for the first iteration)")
		inData = np.stack([x[i_rand] for x in X])
	assert inData.shape[0] < inData.shape[1], f'shape of input data is {inData.shape}. The first dimensions should be smaller than the second dimension since the data should be  "subj x voxel"'
	return ttest_1samp_no_p(inData)

def get_random_model_t_values(X_random, all_subjects):
	assert len(X_random) == len(all_subjects), 'expected len(X_random) to be len(all_subjects). Either the dimensions/structure of X_random is wrong or it doesn\'t match the subjects array' 
	n_rand = range(len(X_random[0]))
	print(f"Calculating t-maps for {n_rand} random models with {len(X_random[0][0])} observations")

	random_t_maps = []
	for i_rand in tqdm(n_rand, desc='computing t-maps for random models...'):
		random_t_maps.append(get_model_t_map(X_random, i_rand))
	return random_t_maps



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


def get_masked_data(data, mask, output_dim='1d', do_stack=True):
	'''
	Mask 3D data using a 3D or flattened mask.
	
	:param data: list of length n_subj with arrays of shape (s1 x s2 x s3)
	:param mask: 3D or flattened spatial mask
	:param output_dim: ``'1d'`` for masked voxels or ``'3d'`` for masked maps

	NOTE: adapted this quickly from the 4d fusion stuff... 
	we also could have just used the 3d mask for masking here.
	'''

	if output_dim not in ('1d', '3d'):
		raise ValueError("output_dim must be either '1d' or '3d'")

	mask = np.asarray(mask) # safety, because the output of mask.get_fdata() is a memmap

	if mask.ndim == 3:
		mask_3d = mask
		mask_1d = mask.flatten.astype(bool)
	else:
		mask_1d = mask.astype(bool)

	data_items = [data] if isinstance(data, np.ndarray) and data.ndim == 3 else data

	if mask.ndim == 1 and output_dim == '3d':
		assert data_items[0].ndim == 3, 'cannot create 3d output if input data and mask are both 1d'
		s_3D = data[0].shape
		mask_3d = mask_1d.reshape(s_3D)

	data_items
	masked_data = []

	for d in data_items:
		d = np.asarray(d)
		# if d.ndim != 3 or d.size != mask_1d.size:
		# 	raise ValueError('each data item must be a 3D array matching the mask size')
		if output_dim == '1d':
			masked = d.flatten()[mask_1d]
		if output_dim == '3d':
			if d.ndim == 1:# or d.shape != mask.shape:
				d = d.reshape(s_3D)
				# raise ValueError('3D output requires a 3D mask matching the data shape')
			masked_3d = np.zeros(d.shape, dtype=d.dtype)
			masked_3d[mask_3d.astype(bool)] = masked
			masked = masked_3d
		masked_data.append(masked)

	if len(masked_data) > 1: 
		if do_stack:
			masked_data = np.stack(masked_data)
	else:
		masked_data = masked_data[0]

	return masked_data