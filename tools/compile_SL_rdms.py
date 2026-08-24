# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from functools import partial
import re

from utils.files import fix_spm_rawdata_paths
from utils.plots import plot_overlay_on_mask, plot_centers, plot_rdm, plot_RDMbrain
from utils.rsa import get_searchlight_RDMs_crossnobis

from glob import glob
import joblib

import rsatoolbox as rsa

from plus_slurm import Job

#%%
subjID = EXAMPLE_SUBJ_1
firstLevel = 'firstLevel_supraSens_M4B'
spmDir = f'{MRI_DATA_DIR}/sync/{subjID}/NCC/{firstLevel}'
inDir = f'{MRI_DATA_DIR}rsa/{subjID}/'

brainMaskFile = os.path.join(spmDir, 'mask.nii')
allCentersMaskFile = os.path.join(inDir, 'thr_1_allCenters.nii')

#%% Load data

# List all SL_rdms files in inDir
sl_rdms_files = glob(os.path.join(inDir, 'test_partial*_M4B_model2_crossnobis_spearman_2_SL_rdms.pkl'))
print("Found SL_rdms files:")
for f in sl_rdms_files:
    print(f)
print(f"Total: {len(sl_rdms_files)}")

# Load all SL_rdms objects
sl_rdms_list = [joblib.load(f) for f in sl_rdms_files]

#%% Merge RDM datasets

SL_rdms_full = rsa.rdm.rdms.RDMs.copy(sl_rdms_list[0])
for sl_rdms in sl_rdms_list[1:]:
    SL_rdms_full.append(sl_rdms)


#%% Plot single RDMs

rsa.vis.show_rdm(SL_rdms_full[5000], pattern_descriptor='index')

#%% some sanity checks

idx_correction = sl_rdms_list[0].dissimilarities.shape[0] + sl_rdms_list[1].dissimilarities.shape[0]

idx = 12

print(sl_rdms_list[2].dissimilarities[idx])
print(SL_rdms_full.dissimilarities[idx_correction+idx])

print(f'voxel_index (original): {sl_rdms_list[2].rdm_descriptors['voxel_index'][idx]}')
print(f'voxel_index (appended): {SL_rdms_full.rdm_descriptors['voxel_index'][idx_correction+idx]}')

print(f'index (original): {sl_rdms_list[1].rdm_descriptors['index'][idx]}')
print(f'index (appended): {SL_rdms_full.rdm_descriptors['index'][idx_correction+idx]}')
