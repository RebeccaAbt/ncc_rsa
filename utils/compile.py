# -----------------------------------------------------------------------
# helper Functions to accumulate data processed in separate martial masks 
# -----------------------------------------------------------------------
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.load_cfg import load_MRI_config_instance

import nibabel as nib
import numpy as np
import os
import warnings
import rsatoolbox as rsa
from rsatoolbox.util.searchlight import get_volume_searchlight
import joblib


def find_empty_masks(
		subjectID='19910823ssld',
		config_class_name='MRIconfig_C2'):

	# 1) load config
	cfg = load_MRI_config_instance(config_class_name, subjectID)

	out_file = os.path.join(cfg.outDir, "empty_masks.txt")

	# if os.path.exists(out_file):
	# 	print(f"Empty masks file already exists for subject {subjectID}. Skipping.")
	# 	return

	partialMasks = list(map(int, np.arange(1, 61)))

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

		except ValueError as e:
			if "multi_index must be a sequence of length 3" in str(e):
				print(
					f"Empty mask found: subject={subjectID}, maskNr={maskNr}",
					flush=True
				)
				return maskNr
			else:
				raise

		return None

	empty_masks = joblib.Parallel(n_jobs=-1)(
		joblib.delayed(process_mask)(maskNr)
		for maskNr in partialMasks
	)

	empty_masks = sorted([m for m in empty_masks if m is not None])

	out_file = os.path.join(cfg.outDir, "empty_masks.txt")

	with open(out_file, "w") as f:
		for maskNr in empty_masks:
			f.write(f"{maskNr}\n")

	print(f"\nSaved {len(empty_masks)} empty masks to:\n{out_file}", flush=True)



def compile_SL_rdms_files(SL_rdms_files):
    
    sl_rdms_list = [joblib.load(f) for f in SL_rdms_files]
    SL_rdms_full = rsa.rdm.rdms.RDMs.copy(sl_rdms_list[0])

    for sl_rdms in sl_rdms_list[1:]:
        SL_rdms_full.append(sl_rdms)

    # Get unique voxel indices and the first index at which each unique value occurs
    voxel_indices = np.array(SL_rdms_full.rdm_descriptors['voxel_index'])
    unique_voxel_indices, unique_indices = np.unique(voxel_indices, return_index=True)
    unique_indices_sorted = np.sort(unique_indices)

    # Apply to dissimilarities and all rdm_descriptors
    SL_rdms_full.dissimilarities = SL_rdms_full.dissimilarities[unique_indices_sorted]
    for key in SL_rdms_full.rdm_descriptors:
        SL_rdms_full.rdm_descriptors[key] = np.array(SL_rdms_full.rdm_descriptors[key])[unique_indices_sorted]

    # Update count
    SL_rdms_full.n_rdm = len(unique_indices_sorted)

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