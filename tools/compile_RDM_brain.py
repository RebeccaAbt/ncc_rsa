# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import re
from glob import glob

import joblib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


from utils.files import fix_spm_rawdata_paths
from utils.plots import plot_overlay_on_mask, plot_centers, plot_rdm, plot_RDMbrain, plot_rsa_hist
from utils.rsa import get_searchlight_RDMs_crossnobis

# Optional: reload utils.plots if developing interactively
# import importlib
# import utils.plots
# importlib.reload(utils.plots)
#%% inputs:

subjID = '19910823ssld'
spmDir = get_spm_dir(subjID)
inDir = f'{MRI_DATA_DIR}/rsa_old/{subjID}/'

brainMaskFile = os.path.join(spmDir, 'mask.nii')
allCentersMaskFile = os.path.join(inDir, 'thr_1_allCenters.nii')
rdm_brain_files = glob(os.path.join(inDir, 'test_partial*_M4B_model2_crossnobis_spearman_2_RDM_brain.pkl'))
#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List all files in inDir that contain the pattern "_RDM_brain.pkl"


print("Found RDM brain files:")
for f in rdm_brain_files:
    print(f)
print(len(rdm_brain_files))

mask_numbers = []
pattern = re.compile(r'test_partial_(\d{1,2})_M4B_model2_crossnobis')

# Extract mask numbers for each file
for file in rdm_brain_files:
    match = pattern.search(os.path.basename(file))
    if match:
        mask_numbers.append(int(match.group(1)))
    else:
        mask_numbers.append(None)  # In case pattern doesn't match

# Pair mask numbers with files and sort by mask number
paired = sorted(zip(mask_numbers, rdm_brain_files))
# Remove any pairs where mask number is None
paired = [(num, file) for num, file in paired if num is not None]

# Unzip sorted pairs
mask_numbers_sorted, rdm_brain_files_sorted = zip(*paired)

print("Sorted mask numbers:", mask_numbers_sorted)
print("Sorted files:")
for f in rdm_brain_files_sorted:
    print(f)

# Load RDM_brain variables into a list of dicts with mask number
rdm_brain_data = []
for num, file in zip(mask_numbers_sorted, rdm_brain_files_sorted):
    RDM_brain = joblib.load(file)
    rdm_brain_data.append({'mask_number': num, 'RDM_brain': RDM_brain})


#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Find all coordinates where two or more RDM_brain arrays have nonzero values ---
# Assume all RDM_brain arrays have the same shape
rdm_arrays = [d['RDM_brain'] for d in rdm_brain_data]
mask_numbers = [d['mask_number'] for d in rdm_brain_data]
shape = rdm_arrays[0].shape

# Stack all arrays to shape (n_masks, x, y, z)
stacked = np.stack(rdm_arrays, axis=0)

# Find coordinates where at least two arrays have nonzero values
nonzero_counts = np.count_nonzero(stacked, axis=0)
coords_multi_nonzero = np.where(nonzero_counts >= 2)

print(f"Found {coords_multi_nonzero[0].shape[0]} coordinates with nonzero values in two or more RDM_brain arrays.")


# For each such coordinate, collect the values and mask numbers where value is nonzero
multi_nonzero_values = []
all_equal = True
for idx in zip(*coords_multi_nonzero):
    values = []
    masks = []
    for i, arr in enumerate(rdm_arrays):
        val = arr[idx]
        if val != 0:
            values.append(val)
            masks.append(mask_numbers[i])
    if len(values) >= 2:
        multi_nonzero_values.append({
            'indices': idx,
            'mask_numbers': masks,
            'values': values
        })
        # Check if all values are the same
        if not all(v == values[0] for v in values):
            all_equal = False
            print(f"Different values at {idx}: mask_numbers={masks}, values={values}")
print('-----------------------------------------------------------------------')
if all_equal:
    print(f"All of the {coords_multi_nonzero[0].shape[0]} multi-nonzero coordinates have the same value across masks.")
else:
    print("! ! ! ! Some coordinates have different values across masks. See above for details.")

print('-----------------------------------------------------------------------')
print(f"Found {len(multi_nonzero_values)} coordinates with nonzero values in two or more RDM_brain arrays.")
# Example: print the first few
for entry in multi_nonzero_values[:5]:
    print(f"indices: {entry['indices']}, mask_numbers: {entry['mask_numbers']}, values: {entry['values']}")

#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Combine all RDM_brain arrays so that for each coordinate, only one nonzero value is kept ---
# If multiple arrays have nonzero at the same coordinate, keep the first nonzero value (by mask order)

# Initialize combined array with zeros
RDM_brain_combined = np.zeros(shape, dtype=stacked.dtype)

# For each mask, fill in values only where combined is still zero
for arr in stacked:
    mask = (RDM_brain_combined == 0) & (arr != 0)
    RDM_brain_combined[mask] = arr[mask]

print("Combined RDM_brain shape:", RDM_brain_combined.shape)
print("Nonzero count in combined:", np.count_nonzero(RDM_brain_combined))

#%%
from utils.compile import *
cfg = 'MRIconfig_C5'
acc_centers, acc_centers_img = get_compiled_centers(cfg)


#%%
allCentersMask_bool = nib.load(allCentersMaskFile).get_fdata() > 0

eval_score = RDM_brain_combined[allCentersMask_bool]

plot_RDMbrain(RDM_brain_combined, eval_score)
plot_rsa_hist(eval_score)

# %%
