# %%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from copy import deepcopy
from utils.load_cfg import load_config_instance
from utils.files import fix_spm_rawdata_paths
import joblib

import numpy as np
import pandas as pd
import nibabel as nib
from copy import deepcopy

from rsatoolbox.io.spm import SpmGlm
from rsatoolbox.data import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.data.ops import merge_datasets
from rsatoolbox.rdm import calc_rdm, RDMs
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import (
get_volume_searchlight,
evaluate_models_searchlight
)

from plus_slurm import Job
from configs.config import BaseConfig as cfg
import importlib
import configs.config as config
importlib.reload(config)


#%%


subjectID = '19951227eipo'
maskNr = 1
config_class_name = 'SensoryConfig_C2'
evaluate_model_singleSubj = True

print('----------------------------------------------------')
print('running...')
print(f'     - subject:       {subjectID}')
print(f'     - maskNr:        {maskNr}')
print(f'     - configuration: {config_class_name}')
print('----------------------------------------------------')

# 1) load instance of selected class object --> this loads all the settings for the computations

cfg = load_config_instance(config_class_name, subjectID, maskNr) 
cfg.print_summary()
cfg.save_summary()

outFiles = cfg.get_outFile_names()

model = cfg.get_model_RDM()

print(f'Running crossnobis Searchlight part {maskNr}', flush=True)
os.makedirs(cfg.outDir, exist_ok=True)

# 2) defining and loading some more variables
spm = SpmGlm(cfg.spmDir)
spm.get_info_from_spm_mat()

print('    - Renaming rawdata file paths in SPM.mat...', flush=True)
fix_spm_rawdata_paths(spm, cfg.dataDir + 'sync')        


# 3) loading the betas and info

print('    - Loading betas and info...', flush=True)
_, _, info = spm.get_betas(cfg.maskFile)

info = pd.DataFrame(info)
reg_mask = info['reg_name'].str.contains('_hit') | info['reg_name'].str.contains('_miss')
info = info[reg_mask].reset_index(drop=True)

info['modality'] = info['reg_name'].str.extract(r'^(aud|tac|vis)')
info['awareness'] = info['reg_name'].str.extract(r'_(hit|miss)')

if cfg.nCond == 6:
    info['condition'] = info['modality'] + '_' + info['awareness']

elif cfg.nCond == 24:
    info['stimulus'] = info['reg_name'].str.extract(r'_(1|2|3|4)')
    info['condition'] = info['modality'] + '_' + info['awareness'] + '_' + info['stimulus']
    info['identifier'] = info['condition'] + '_run' + info['run_number'].astype('str')

# Add new key with numeric condition codes
condition_numbers, _ = pd.factorize(info['condition'])
condition_numbers += 1

info['condition_number'] = condition_numbers


mask = nib.load(cfg.maskFile)
mask_data = mask.get_fdata()
mask_bool = mask_data > 0

# 4) finding Searchlights
print(f'    - Getting searchlight centers and neighbors of mask nr. {cfg.maskNr} with radius {cfg.SLradius} voxels and threshold {cfg.SLthr}')

centers, neighbors = get_volume_searchlight(mask_bool, radius = cfg.SLradius, threshold = cfg.SLthr)

#%% check where size of searchlight is different
all_SLsizes = [nb.shape[0] for nb in neighbors]
unique_SLsizes = np.unique(all_SLsizes)
SL_sizediff_idx = np.where(np.array(all_SLsizes) ==unique_SLsizes[0]) # indices where shape is different

#%% 

centers = np.array(centers)

events='condition_number'
method='crossnobis'
verbose=True

# first find missing conditions --> compare to full set of conditions

# ---------------------------------------------------------------------- v
info_full = joblib.load(f'{CODE_DIR}/resources/info.pkl')

do_imputation = False
if len(info_full) > len(info): do_imputation = True  # imputation --> replace missing value with mean

if do_imputation:
    missing_identifiers = np.setdiff1d(info_full['identifier'].values, info['identifier'].values) # missing conditions; key = 'identifiers' do identify condition in different runs


info['events'] = info[events]
info_full['events'] = info_full[events]
info_double = [] # use info that contains all the conditions even if conditions are missing --> because we will replace missing conditions later on
info_double = pd.DataFrame(info_double)
info_double['events'] = info_full['events'].astype('double')
info_double['run_number'] = info_full['run_number'].astype('double')
# ---------------------------------------------------------------------- ^

# original mask
mask_bool = mask.get_fdata() > 0
n_voxels_total = np.prod(mask_bool.shape)
mask_bool_1D = mask_bool.flatten()
mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)

#initalize variables
n_centers = centers.shape[0]

chunked_center = np.split(np.arange(n_centers),
                            np.linspace(0, n_centers, 
                                        101, dtype=int)[1:-1]) 
# loop over chunks
n_conds = len(np.unique(info_double['events']))
RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))

chunk = chunked_center[0]

chunk_voxels  = np.unique(np.concatenate([neighbors[i] for i in chunk]))
#%%
center_data, center_noise = [], []
# for c in chunk:
for c in chunk:
    center = centers[c]
    nb = neighbors[c]

    print(f'\ncurrent center Nr:{c} = center {center}')
    print(f'size of Searchlight: {nb.shape[0]}')


    SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
    SL_mask_1D[nb] = True # !!!!!!! important!! must be  [nb], not SL_mask_idx
    SL_mask_3D = SL_mask_1D.reshape(mask.shape)
    SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)

    print('     - Loading SL_betas...')
    SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
    # print(f'       size of ResMS: {SL_ResMS.size}')
    SL_beta = np.nan_to_num(SL_beta)
    SL_ResMS = np.nan_to_num(SL_ResMS)
    SL_beta = SL_beta[reg_mask.to_numpy(), :]

    if do_imputation: # if we have missing conditions
        for id in missing_identifiers:     
            missing_identifier_idx = np.where(info_full['identifier'].values == id)[0].item()           # 1) index of identifier in full set of conditions
            missing_condition = missing_identifiers[0][:-5]                                             # 2) find indices/labels of condition in other runs of current data
            missing_condition_idx = info.index[info['condition'] == missing_condition].tolist() 
            missing_condition_values = np.sum(SL_beta[missing_condition_idx], axis = 0)                 # 3) compute sum of betas at these indices

            SL_beta = np.insert(SL_beta, missing_identifier_idx, missing_condition_values, axis = 0)    # 4) insert condition at right index to mathc full_info order

    print(f'       - Loading Residuals...')
    SL_residuals, _, _ = spm.get_residuals(SL_mask_img)
    SL_residuals = np.nan_to_num(SL_residuals)

    print(f'         - Computing Precision Matrix...')

    SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')
    measurements = SL_beta / np.sqrt(SL_ResMS)
    measurements = np.nan_to_num(measurements)

    ds = Dataset(measurements = measurements, 
                descriptors={'center': center},
                obs_descriptors=dict(info_double),
                channel_descriptors={'voxels': nb})

    center_data.append(ds)
    center_noise.append(SL_Prec)

#%%
print('calculating RDMs for current chunk...')

RDM_corr = calc_rdm(center_data, 
            method=method,
            descriptor='events', 
            noise=center_noise,
            cv_descriptor='run_number')

RDM[chunk, :] = RDM_corr.dissimilarities


# %%
