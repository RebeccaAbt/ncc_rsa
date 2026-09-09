# -----------------------------------------------------------------------
# helper Functions to accumulate data processed in separate martial masks 
# -----------------------------------------------------------------------
import os
import sys
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.load_cfg import load_MRI_config_instance
from utils.rsa import adust_edge_searchlights

import nibabel as nib
import numpy as np
import os
import warnings
import rsatoolbox as rsa
from rsatoolbox.util.searchlight import get_volume_searchlight
import joblib

def _check_if_duplicate_searchlights_are_equal(d_v, SL_sizes, SL_rdms_full):
	size_map = defaultdict(list)
	for i, sz in enumerate(SL_sizes):
		size_map[int(sz)].append(i)

	maxkey = np.max(list(size_map.keys())) # get key that represents biggers SL size
	maxvals = size_map[maxkey] # indices in d_v of the searchlights with the maximum size

	if len(maxvals) > 1: # if multiple SL at the same voxel have the biggest size: Check if they have the same dissimilarities
		
		SL_idxs = [d_v[x] for x in maxvals] # index SL in d_v with the maximum size

		assert all(np.isclose(
							SL_rdms_full.dissimilarities[SL_idxs[0]], 
							SL_rdms_full.dissimilarities[SL_idx], 
							atol=1e-12).all() 
							for SL_idx in SL_idxs[1:]), \
							'Searchlight dissimilarieties differ, even though they have the same size and should be at the same voxel position'
		
def find_empty_masks(subjectID='19910823ssld', config_class_name='MRIconfig_C2'):
	'''
	This function also returns the searchlight centers and neighbors for all non-empty masks (correted for edge searchlights!!) 
	which can be used later to compile the RDMs.
	'''

	# 1) load config
	cfg = load_MRI_config_instance(config_class_name, subjectID)

	out_file = os.path.join(cfg.outDir, "empty_masks.txt")

	partialMasks = list(map(int, np.arange(1, 61)))
	all_centers = []
	all_neighbors = []

	def process_mask(maskNr):
		cfg_mask = load_MRI_config_instance(
			config_class_name,
			subjectID,
			maskNr
		)

		mask = nib.load(cfg_mask.get_mask_file())
		mask_data = mask.get_fdata()
		mask_bool = mask_data > 0
		
		try:
			centers, neighbors = get_volume_searchlight(
				mask_bool,
				radius=cfg_mask.SLradius,
				threshold=cfg_mask.SLthr
			)
			if cfg.SLthr < 1:
				neighbors = adust_edge_searchlights(mask, neighbors)

		except ValueError as e:
			if "multi_index must be a sequence of length 3" in str(e):
				print(
					f"Empty mask found: subject={subjectID}, maskNr={maskNr}",
					flush=True
				)
				return maskNr, None, None
			else:
				raise

		return None, centers, neighbors

	results = joblib.Parallel(n_jobs=-1)(
		joblib.delayed(process_mask)(maskNr)
		for maskNr in partialMasks
	)

	for maskNr, centers, neighbors in results:
		if maskNr is not None:
			continue
		if centers is None and neighbors is None:
			continue
		all_centers.append(centers)
		all_neighbors.append(neighbors)

	empty_masks = sorted([
		maskNr for maskNr, _, _ in results if maskNr is not None
	])

	with open(out_file, "w") as f:
		for maskNr in empty_masks:
			f.write(f"{maskNr}\n")

	print(f"\nSaved {len(empty_masks)} empty masks to:\n{out_file}", flush=True)
	print(f"Collected searchlights from {len(all_centers)} non-empty partial masks.", flush=True)
	return all_centers, all_neighbors



def compile_SL_rdms_files_old(SL_rdms_files):
	
	sl_rdms_list = [joblib.load(f) for f in SL_rdms_files]
	if not sl_rdms_list:
		raise ValueError('No SL_rdms files were provided for compilation.')

	SL_rdms_full = rsa.rdm.rdms.RDMs.copy(sl_rdms_list[0])

	for sl_rdms in sl_rdms_list[1:]:
		SL_rdms_full.append(sl_rdms)

	# ------------------------------------------------------------
	# Sanity check: overlapping partial masks should not change the
	# same voxel's value. This can happen at mask borders where the
	# same voxel is covered by more than one partial mask.
	# ------------------------------------------------------------
	voxel_indices = np.asarray(SL_rdms_full.rdm_descriptors['voxel_index'])
	voxel_map = defaultdict(list)
	for i, voxel in enumerate(voxel_indices):
		voxel_map[int(voxel)].append(i)

	duplicates = {voxel: idxs for voxel, idxs in voxel_map.items() if len(idxs) > 1}

	if duplicates:
		print(f"    -> number of duplicate indices: {len(duplicates)}")
		all_equal = True
		for voxel, idxs in duplicates.items():
			values = SL_rdms_full.dissimilarities[idxs]
			if not np.allclose(values, values[0], rtol=1e-8, atol=1e-12, equal_nan=True):
				all_equal = False
				print(f"Voxel index {voxel} has different RDM values at indices {idxs}: values={values}")
		print("    -> All duplicates match." if all_equal else "    -> Some duplicates differ!")
	else:
		print("    -> No duplicate voxel indices across masks.")

	# Get unique voxel indices and the first index at which each unique value occurs
	unique_voxel_indices, unique_indices = np.unique(voxel_indices, return_index=True)
	unique_indices_sorted = np.sort(unique_indices)

	# Apply to dissimilarities and all rdm_descriptors
	SL_rdms_full.dissimilarities = SL_rdms_full.dissimilarities[unique_indices_sorted]
	for key in SL_rdms_full.rdm_descriptors:
		SL_rdms_full.rdm_descriptors[key] = np.array(SL_rdms_full.rdm_descriptors[key])[unique_indices_sorted]

	# Update count
	SL_rdms_full.n_rdm = len(unique_indices_sorted)

	return SL_rdms_full

def compile_SL_rdms_files(SL_rdms_files, all_centers, all_neighbors):
	SL_rdms_list = [joblib.load(f) for f in SL_rdms_files]
	SL_rdms_full = rsa.rdm.rdms.RDMs.copy(SL_rdms_list[0])
	centers_full = np.concatenate(all_centers)
	neighbors_full = [nb for sublist in all_neighbors for nb in sublist]

	for sl_rdms, nb in zip(SL_rdms_list[1:], all_neighbors[1:]):
		SL_rdms_full.append(sl_rdms)

	# Get unique voxel indices and the first index at which each unique value occurs
	voxel_indices = np.array(SL_rdms_full.rdm_descriptors['voxel_index'])

	assert np.all(voxel_indices == centers_full), 'Mismatch between voxel indices and centers_full'

	voxel_map = defaultdict(list)
	for i, voxel in enumerate(voxel_indices):
		voxel_map[int(voxel)].append(i)

	duplicates = {voxel: idxs for voxel, idxs in voxel_map.items() if len(idxs) > 1}
	uniques = 	 {voxel: idxs for voxel, idxs in voxel_map.items() if len(idxs) == 1}
	keep_unique_centers = [idxs[0] for idxs in uniques.values()]
	keep_dupe_centers = []  # Keep the index of the largest searchlight. We need this, since the same voxel can be part of multiple searchlights. And at the margins where masks overlap, some searchlights are smaller than others.

	for _, d_v in duplicates.items():
		SL_sizes = [len(neighbors_full[i]) for i in d_v]

		if len(set(SL_sizes)) < len(SL_sizes):
			_check_if_duplicate_searchlights_are_equal(d_v, SL_sizes, SL_rdms_full)

		keep_dupe_centers.append(d_v[np.argmax(SL_sizes)]) 

	center_indices = keep_unique_centers + keep_dupe_centers

	# Apply to dissimilarities and all rdm_descriptors
	SL_rdms_full.dissimilarities = SL_rdms_full.dissimilarities[center_indices]
	for key in SL_rdms_full.rdm_descriptors:
		SL_rdms_full.rdm_descriptors[key] = np.array(SL_rdms_full.rdm_descriptors[key])[center_indices]

	# Update count
	SL_rdms_full.n_rdm = len(center_indices)
	return SL_rdms_full





def get_compiled_centers(cfg):

	"""
	Get accumulated centers for a given mask number.
	"""
	# cfg --> is an instance of one of the subclasses in /configs/config.py that contain setting configurations
	if cfg.maskNr != 0:
		warnings.warn('! ! ! ! Not the full brain mask file is provided in cfg! Review your inputs!', UserWarning)
	fullBrain_mask = nib.load(cfg.get_mask_file())
	partialMasks = list(map(int, np.concatenate([np.arange(1, 61)])))

	acc_centers = np.zeros(fullBrain_mask.shape)

	for maskNr in partialMasks:
		mask_file = mask_file = os.path.join(cfg.masksDir, f'SL_marg{cfg.maskMargin}_mask_part_{maskNr}.nii') 
		mask = nib.load(mask_file)
		mask_data = np.array(mask.dataobj)
		mask_bool = mask_data > 0

		try:
			centers, _ = get_volume_searchlight(mask_bool, radius=2, threshold=1)
			overlay_data = np.zeros(fullBrain_mask.shape)
			overlay_data[np.unravel_index(centers, mask.shape)] = 1  # Mark centers

			acc_centers += overlay_data

		except Exception as e:
			print(f"Failed to get searchlight centers for mask {maskNr}: {e}")
			continue
	
	overlay_img = nib.Nifti1Image(acc_centers, affine = fullBrain_mask.affine)
	return acc_centers, overlay_img

