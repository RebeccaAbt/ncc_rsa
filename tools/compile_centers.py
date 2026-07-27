# %%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from functools import partial
from utils.files import fix_spm_rawdata_paths
from utils.plots import plot_overlay_on_mask, plot_centers
from utils.rsa import get_searchlight_RDMs_crossnobis

from glob import glob
from pprint import pprint

import dill
import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from tqdm import tqdm

from nilearn import plotting
from nilearn.image import new_img_like

import rsatoolbox as rsa
from rsatoolbox.io.spm import SpmGlm
from rsatoolbox.data import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.data.ops import merge_datasets
from rsatoolbox.rdm import calc_rdm, calc_rdm_unbalanced, RDMs
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.descriptor_utils import dict_to_list
from rsatoolbox.util.searchlight import (
    get_volume_searchlight,
    get_searchlight_RDMs,
    evaluate_models_searchlight
)
import matplotlib.colors as mcolors
from plus_slurm import Job


#%%
import utils.plots
import importlib
importlib.reload(utils.plots)
import utils.plots
from utils.plots import plot_overlay_on_mask, plot_centers
#%%


# Define parameters
subjectID = '19910823ssld'
firstLevel_model = 'M4B'

model_RM = 'model2'
RDMmethod = 'crossnobis'
RSAmethod = 'spearman'

dataDir = MRI_DATA_DIR
outDir = f'{dataDir}/rsa_old/{subjectID}/'
spmDir = get_spm_dir(subjectID)
masksDir = f'{dataDir}/masks/{subjectID}/'

fullBrain_maskFile = os.path.join(spmDir, 'mask.nii')
fullBrain_mask = nib.load(fullBrain_maskFile)

allCenters_niiFile = os.path.join(outDir, 'thr_1_allCenters.nii')

#%%
partialMasks = list(map(int, np.concatenate([np.arange(1, 16), np.arange(17, 20), np.arange(21, 56), np.arange(57, 61)])))


acc_centers = np.zeros(fullBrain_mask.shape)

for maskNr in partialMasks:
    print(maskNr)
    mask_file = os.path.join(masksDir, f'SL_marg2_mask_part_{maskNr}.nii') 

    mask = nib.load(mask_file)
    mask_data = np.array(mask.dataobj)

    mask_bool = mask_data > 0

    try:
        centers, neighbors = get_volume_searchlight(mask_bool, radius=2, threshold=1)
        overlay_data = np.zeros(fullBrain_mask.shape)
        overlay_data[np.unravel_index(centers, mask.shape)] = 1  # Mark centers

        acc_centers += overlay_data

        # overlay_img = nib.Nifti1Image(acc_centers, affine = fullBrain_mask.affine)

        # # Define a custom colormap
        # cmap = mcolors.ListedColormap(['black', 'green', 'yellow', 'orange' 'red', 'magenta', 'purple', 'cyan', 'blue'])  # Black for background, red for centers, blue for neighbors
        # bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Define boundaries for the colormap
        # norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # # Plot the overlay on top of the mask
        # plotting.plot_stat_map(
        #     overlay_img,
        #     bg_img=fullBrain_mask,  # Use the original image as the background
        #     title="Centers Visualization",
        #     threshold=0.1,  # Lower threshold to ensure visibility
        #     display_mode="ortho",  # Orthogonal view
        #     colorbar=True,
        #     cbar_tick_format = '%%i',
        #     cmap=cmap,
        #     vmax=8,  # Maximum value for the colorbar
        #     alpha = 0.9
        # )
        # plotting.show()

    except Exception as e:
        
        print(f"Failed to get searchlight centers for mask {maskNr}: {e}")
        continue

#%%

overlay_img = nib.Nifti1Image(acc_centers, affine = fullBrain_mask.affine)
nib.save(overlay_img, allCenters_niiFile)
#%%

# Define a custom colormap
cmap = mcolors.ListedColormap(['black', 'green', 'yellow', 'orange' 'red', 'magenta', 'purple', 'cyan', 'blue'])  # Black for background, red for centers, blue for neighbors
bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Define boundaries for the colormap
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot the overlay on top of the mask
plotting.plot_stat_map(
    overlay_img,
    bg_img=fullBrain_mask,  # Use the original image as the background
    title="Centers Visualization",
    threshold=0.1,  # Lower threshold to ensure visibility
    display_mode="ortho",  # Orthogonal view
    colorbar=True,
    cbar_tick_format = '%%i',
    cmap=cmap,
    vmax=8,  # Maximum value for the colorbar
    alpha = 0.9
)
plotting.show()



# Searchlight
# print('    - Getting searchlight centers and neighbors...But using "threshold=1" this time')

# centers, neighbors = get_volume_searchlight(mask_bool, radius=2, threshold=1)

# print('    - saving centers...')

# plot_centers(centers)
# plt.savefig(centerPlot_title, dpi=300)
# plt.close()
# %%
filePrefix = 'thr_0_5_'

partialMasks = list(map(int, np.concatenate([np.arange(1, 16), np.arange(17, 20), np.arange(21, 56), np.arange(57, 61)])))

acc_centers = np.zeros(fullBrain_mask.shape)
centers_dict = {}

for maskNr in partialMasks:
    print(maskNr)
    mask_file = os.path.join(masksDir, f'SL_marg2_mask_part_{maskNr}.nii') 

    mask = nib.load(mask_file)
    mask_data = np.array(mask.dataobj)

    mask_bool = mask_data > 0

    try:
        centers, neighbors = get_volume_searchlight(mask_bool, radius=2, threshold=0.5)
        centers_data = np.zeros(fullBrain_mask.shape)
        centers_data[np.unravel_index(centers, mask.shape)] = 1  # Mark centers
        overlay_img = nib.Nifti1Image(centers_data, affine = fullBrain_mask.affine)
        nib.save(overlay_img, os.path.join(outDir, f'{filePrefix}centers_{maskNr}.nii'))
        # acc_centers += overlay_data

        # Store centers and neighbors in the dictionary
        centers_dict[maskNr] = {
            'maskNr': maskNr,
            'centers': centers,
            'neighbors': neighbors
        }

    # After the loop, save centers_dict as a .mat file for MATLAB
    except Exception as e:
        print(f"Failed to get searchlight centers for mask {maskNr}: {e}")
        continue


# import scipy.io
# # Convert integer keys to strings for savemat compatibility
# # Prepare a structure array where each element is a dict with fields maskNr, centers, neighbors
# centers_struct_array = []
# for k, v in centers_dict.items():
#     # print(f'k = {k}')
#     # print(f'v = {v}')
#     centers_struct_array.append({
#         'maskNr': v['maskNr'],
#         'centers': v['centers'],
#         'neighbors': v['neighbors']
#     })
# scipy.io.savemat(
#     os.path.join(outDir, f'{filePrefix}allPartMasks_centers_neighbors.mat'),
#     {'centers_info': centers_struct_array}
# )

#%%

# Gather all centers from centers_dict into a single array
allCenters = []
for v in centers_dict.values():
    allCenters.extend(list(map(int, v['centers'])))

allCenters = np.array(allCenters)

# Count occurrences of each value
vals, idx, counts = np.unique(allCenters, return_inverse=True, return_counts=True)

# Find elements that appear only once
unique_once = vals[counts == 1]

# Find elements that appear more than once
notUnique = np.setdiff1d(allCenters, unique_once)
notunique_single = np.unique(notUnique)  # For plotting: numbers that appeared >1x
# %%


thisFile = f'{MRI_RSA_DIR}/19910823ssld/old_M4_model2_euclidean_spearman_RDM_brain.pkl'
joblib.load(thisFile)
