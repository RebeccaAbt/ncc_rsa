

#%% 

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from copy import deepcopy
from glob import glob
import joblib
import nibabel as nib
import numpy as np
from pqdm.processes import pqdm
from joblib import Parallel, delayed
import shutil
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.image import new_img_like 	 	


from utils.load_cfg import load_fusion_config_instance
from utils.fusion_stat import *

from plus_slurm import Job

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

#%%


thresholds = [90, 95, 99, 99.9, None]

for config_class_name in [
			# "FusionConfig_E2",
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

		print(f"\nProcessing subject {cfg.subjectID}...\n", flush=True)
		fusion_data = joblib.load(file)

		mask = nib.load(cfg.get_mask_file())
		mask_data = mask.get_fdata()
		mask_size = mask.shape

		indices = np.unravel_index(fusion_data['voxel_index_py'], mask_size)
		voxel_coords = np.stack(indices, axis=1)

		result = {
			"maskData": mask_data,
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
		maskData_all.append(res["maskData"])
		# centers_masks.append(res["centers_mask"])
		all_data_pre_sensory.append(res["pre_sensory"])
		all_data_post_sensory.append(res["post_sensory"])
		all_data_pre_suprasensory.append(res["pre_suprasensory"])
		all_data_post_suprasensory.append(res["post_suprasensory"])

	# common_centers_mask = find_common_centers(centers_masks, nib.load(cfg.get_mask_file()), show_plot=False)

	data_pre_dict = {'sensory': all_data_pre_sensory, 'suprasensory': all_data_pre_suprasensory}
	data_post_dict = {'sensory': all_data_post_sensory, 'suprasensory': all_data_post_suprasensory}

	print(f'[B] Looping over modalities')

	modalities = ALL_MODELS

	for modality  in modalities:

		cfg.modelType = modality

		print(f'\n[{modality}]\n')
		print(f'\t [1] preparing the data', flush=True)

		_, data_pre_noNan = stack_data(data_pre_dict, modality)

		all_data_pre = data_pre_dict[modality]
		all_data_post = data_post_dict[modality]
		
		n_subj = len(all_data_pre)

		mask = find_voxels_noNan_allSubj(all_data_pre, all_data_post, cfg.get_mask_file())

		mask_1d = mask.flatten() 
		s_4d = all_data_post[0].shape

		print(f'\n\t\t\t shape of 4D data (s1 x s2 x s3 x time): {s_4d}')
		print(f'\t\t\t number of valid voxels in mask: {np.sum(mask_1d)}')
		print(f"\t\t\t number of subjects (length of 'all_data_pre'): {len(all_data_pre)}\n")

		X_pre   = get_masked_data(all_data_pre, mask_1d) # we mask the data here, because the adjacency has onlly the same length as the number of True values inside the mask 
		X_post  = get_masked_data(all_data_post, mask_1d)
		X_diff  = X_post - X_pre

		
		X_all = {'X_pre': X_pre, 'X_post': X_post, 'X_diff': X_diff}

	# Average the 4D post-stimulus data across subjects
	mean_post_dict = {
		model: np.nanmean(np.stack(data_post_dict[model], axis=0), axis=0)
		for model in ALL_MODELS
	}

	# Expected shape of each entry:
	# x × y × z × time
	for model, data in mean_post_dict.items():
		print(f'{model}: mean 4D data shape = {data.shape}')


	def create_frame(args):
		(	mask_img,
			data_t,
			transparency_mask,
			time_point,
			model,
			vmin,
			vmax) = args

		fig = plt.figure(figsize=(6, 6))

		data_img = new_img_like(mask_img, data_t, copy_header=True)

		if np.any(transparency_mask):
			transparency = new_img_like(mask_img, transparency_mask, copy_header=True)
		else:  # because we get an error, if the transparency mask is empty
			# print('unique in data with no transparency: ', np.unique(data_t), flush=True)
			transparency = 0

		plotting.plot_glass_brain(
			data_img,
			transparency = transparency,
			display_mode='ortho',
			cut_coords=(0, 0, 0),
			draw_cross=False,
			black_bg=False,
			figure=fig,
			cmap='viridis',
			threshold=None,
			vmin=vmin,
			vmax=vmax,
			annotate=False,
			colorbar=True,
			plot_abs=False,
			title=f'{model}: {time_point} ms',
		)

		fig.canvas.draw()
		width, height = fig.canvas.get_width_height()
		buffer = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
		frame  = buffer[:, :, [1, 2, 3]]  # ARGB -> RGB
		plt.close(fig)

		return frame


	output_folder = 'fusion_outputs'
	os.makedirs(output_folder, exist_ok=True)

	for percentile in thresholds:
		fps = 5
		time_array = np.arange(0, 1000, 10)

		mask_img = nib.load(cfg.get_mask_file())

		for model in ALL_MODELS:

			print(f'\nCreating movie for {model}...', flush=True)

			mean_4d = deepcopy(mean_post_dict[model])

			if mean_4d.shape[-1] != len(time_array):
				raise ValueError(
					f'{model}: found {mean_4d.shape[-1]} time points, '
					f'but time_array contains {len(time_array)} values.'
				)
			'''
			The weird transparency mask / thresholding locig is needed, because
			even though plot_class_brain has  a thresholding option, the resulting plots alays have a weird background 
			and I think this is the only way to remove the background.
			So we do the thresholding before the plot and then use a transparency mask to mask 
			values we don't want to see (below Thres or NaNs) and set vmin to the lowest value in the thresholded data

			'''
			if percentile:
				# valid_values = mean_4d[np.isfinite(mean_4d)]
				valid_values = mean_4d[(np.isfinite(mean_4d)) & (mean_4d > 0)]
				threshold = np.percentile(valid_values, percentile)
				thresholded_mean_4d = np.where(mean_4d >= threshold, mean_4d, np.nan)
				transparency_mask 	= np.where((mean_4d < threshold) | ~np.isfinite(mean_4d), 0, 1)
				vmin = threshold
						
			else: # no threshold
				threshold=None
				thresholded_mean_4d = mean_4d
				transparency_mask = np.where(~np.isfinite(mean_4d), 0, 1)
				vmin = np.nanmin(mean_4d)
			
			vmax = np.nanmax(thresholded_mean_4d)

			# thresholded_mean_4d = np.where(np.isnan(thresholded_mean_4d), vmin - 1, thresholded_mean_4d)

			args = [
				(	mask_img,
					thresholded_mean_4d[..., t],
					# mean_4d[..., t],
					transparency_mask[..., t].astype(bool),
					time_array[t],
					model,
					vmin,
					vmax) for t in range(thresholded_mean_4d.shape[-1]
				 	# vmax) for t in range(mean_4d.shape[-1]
					  )]

			frames = pqdm( args, create_frame, n_jobs=5, desc=f'Creating {model} frames')

			movie_file = os.path.join(
				'/home/scc_e_393956/ncc/rsa/fusion_outputs',
				f'meanCommonalities_{config_class_name.split('_')[1]}_{model}_thres{percentile}.mp4',
			)

			with imageio.get_writer(
				movie_file,
				fps=fps,
				codec='libx264',
				format='FFMPEG',
				pixelformat='yuv420p',
			) as writer:

				for frame in frames:
					writer.append_data(frame)

			print(f'Saved movie:\n{movie_file}', flush=True)
	# %%
