# %%

import os
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_config_instance
from utils.files import fix_spm_rawdata_paths
from utils.plots import plot_centers

import numpy as np
import pandas as pd
import nibabel as nib

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


subjectID = '19970302urmr'
maskNr = 21
config_class_name = 'SensoryConfig_C4'
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

condition_numbers, _ = pd.factorize(info['condition'])
condition_numbers += 1
# Add new key with numeric condition codes
info['condition_number'] = condition_numbers


mask = nib.load(cfg.maskFile)
mask_data = mask.get_fdata()
mask_bool = mask_data > 0

# 4) finding Searchlights
print(f'    - Getting searchlight centers and neighbors of mask nr. {cfg.maskNr} with radius {cfg.SLradius} voxels and threshold {cfg.SLthr}')
#%%
centers, neighbors = get_volume_searchlight(mask_bool, radius = cfg.SLradius, threshold = cfg.SLthr)

#%% check where size of searchlight is different
all_SLsizes = [nb.shape[0] for nb in neighbors]
unique_SLsizes = np.unique(all_SLsizes)
SL_sizediff_idx = np.where(np.array(all_SLsizes) ==unique_SLsizes[0]) # indices where shape is different
# #%%
# chunk = center_vox  

# chunk_voxels  = np.unique(np.concatenate([neighbors[i] for i in chunk]))
# plot_centers(chunk_voxels, subjectID)


# subjectID1 = '19970302urmr'
# maskFile1 = nib.load(f'/home/reabt/experiments/ncc/MRI/data/sync/{subjectID1}/NCC/firstLevel_sensory_M1B/mask.nii').get_fdata()
# partialMaskFile1 = nib.load(f'/home/reabt/experiments/ncc/MRI/data/masks/{subjectID1}/24cond_SL_marg2_mask_part_21.nii').get_fdata()
# mask_voxels1 = np.flatnonzero(maskFile1)
# partialMask_voxels1 = np.flatnonzero(partialMaskFile1)
# plot_centers(partialMask_voxels1, subjectID1)

# subjectID2 = '19910823ssld'
# maskFile2 = nib.load(f'/home/reabt/experiments/ncc/MRI/data/sync/{subjectID2}/NCC/firstLevel_sensory_M1B/mask.nii').get_fdata()
# partialMaskFile2 = nib.load(f'/home/reabt/experiments/ncc/MRI/data/masks/{subjectID2}/24cond_SL_marg2_mask_part_21.nii').get_fdata()
# mask_voxels2 = np.flatnonzero(maskFile2)
# partialMask_voxels2 = np.flatnonzero(partialMaskFile2)
# plot_centers(partialMask_voxels2, subjectID2)

# plot_centers(neighbors[0][7], subjectID1)


# mask = nib.load(f'/home/reabt/experiments/ncc/MRI/data/sync/{subjectID}/NCC/firstLevel_sensory_M1B/mask.nii')

#%%
# def plot_centers(centers, subjectID='19910823ssld'):
#     from nilearn import plotting
#     import nibabel as nib
#     import matplotlib.colors as mcolors

#     mask = nib.load(f'/home/reabt/experiments/ncc/MRI/data/sync/{subjectID}/NCC/firstLevel_sensory_M1B/mask.nii')

#     overlay_data = np.zeros(mask.shape)
#     overlay_data[np.unravel_index(centers, mask.shape)] = 1  # Mark centers
#     overlay_img = nib.Nifti1Image(overlay_data, affine = mask.affine)

#     # Define a custom colormap
#     cmap = mcolors.ListedColormap(['black', 'red'])  # Black for background, red for centers, blue for neighbors
#     bounds = [0, 0.5, 1.5]  # Define boundaries for the colormap
#     norm = mcolors.BoundaryNorm(bounds, cmap.N)

#     # Plot the overlay on top of the mask
#     plotting.plot_stat_map(
#         overlay_img,
#         # bg_img=mask,  # Use the original image as the background
#         # transparency = 0.5,
#         # transparency=None,
#         title="Centers Visualization",
#         threshold=0.1,  # Lower threshold to ensure visibility
#         display_mode="ortho",  # Orthogonal view
#         colorbar=True,
#         cmap=cmap,
#         vmax=1,  # Maximum value for the colorbar
#         alpha = 0.9,
#         black_bg="False"
#     )
#     plotting.show()

# plot_centers(mask_voxels, '19970302urmr')

#%%


# print('    - Computing searchlight RDMs...')
# #%% --------------------------------------------------------------- option A) compute RDMs using function
# # 5) computing RDMs for every searchlight
# SL_rdms = get_searchlight_RDMs_crossnobis(
#     spm,
#     centers, 
#     neighbors, 
#     mask,
#     reg_mask, 
#     info, 
#     'condition', 
#     'crossnobis')

# conditions = list(dict.fromkeys(info['condition']))
# SL_rdms.pattern_descriptors['condition'] = conditions    


# #%% --------------------------------------------------------------- option B) compute RDMs using extracted function logic
# from tqdm import tqdm

# events='condition'
# method='crossnobis'
# verbose=True
    
# info['events'] = info[events]

# # original mask
# mask_bool = mask.get_fdata() > 0
# n_voxels_total = np.prod(mask_bool.shape)
# mask_bool_1D = mask_bool.flatten()
# mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)

# #initalize variables
# n_centers = centers.shape[0]

# if n_centers > 1000:
#     print('processing centers in chunks...')
#     # we can't run all centers at once, that will take too much memory
#     # so lets to some chunking
#     chunked_center = np.split(np.arange(n_centers),
#                                 np.linspace(0, n_centers, 
#                                             101, dtype=int)[1:-1]) 
#     # loop over chunks
#     n_conds = len(np.unique(info['events']))
#     RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    
#     for chunk in tqdm(chunked_center, desc='Calculating RDMs...'):
#         print(f'chunk: {chunk}')
#         center_data, center_noise = [], []
#         for c in chunk:

#             center = centers[c]
#             nb = neighbors[c]

#             print(f'current center Nr:{c} = center {center}')

#             SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
#             SL_mask_1D[nb] = True # !!!!!!! important!! must be  [nb], not SL_mask_idx
#             SL_mask_3D = SL_mask_1D.reshape(mask.shape)
#             SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)
            
#             print('     - Loading SL_betas...')
#             SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
#             # print(f'       size of ResMS: {SL_ResMS.size}')
#             SL_beta = np.nan_to_num(SL_beta)
#             SL_ResMS = np.nan_to_num(SL_ResMS)
#             SL_beta = SL_beta[reg_mask.to_numpy(), :]

#             print(f'       - Loading Residuals...')
#             SL_residuals, _, _ = spm.get_residuals(SL_mask_img)
#             SL_residuals = np.nan_to_num(SL_residuals)


#             print(f'         - Computing Precision Matrix...')

#             SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')
#             measurements = SL_beta / np.sqrt(SL_ResMS)
#             measurements = np.nan_to_num(measurements)
#             ds = Dataset(measurements = measurements, 
#                         descriptors={'center': center},
#                         obs_descriptors=dict(info),
#                         channel_descriptors={'voxels': nb})
            
#             center_data.append(ds)
#             center_noise.append(SL_Prec)

#         print('calculating RDMs for current chunk...')


#         # print("Type of center_data:", type(center_data))
#         # print("Type of method:", type(method))
#         print("Type of center_noise:", type(center_noise))
#         print("Type of info['events']:", type(info['events']) if 'events' in info else 'N/A')
#         print("Type of cv_descriptor (hardcoded as 'run_number'):", type('run_number'))
#         RDM_corr = calc_rdm(center_data, 
#                     method=method,
#                     descriptor='events', 
#                     noise=center_noise,
#                     cv_descriptor='run_number')

#         RDM[chunk, :] = RDM_corr.dissimilarities
    
# else:
#     print('processing all centers at once...')
#     center_data, center_noise = [], []
#     for c in range(n_centers):
#         center = centers[c]
#         nb = neighbors[c]

#         print(f'current center Nr:{c} = center {center}')

#         SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
#         SL_mask_1D[nb] = True # !!!!!!! important!! must be  [nb], not SL_mask_idx
#         SL_mask_3D = SL_mask_1D.reshape(mask.shape)
#         SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)
        
#         print('     - Loading SL_betas...')
#         SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
#         # print(f'       size of ResMS: {SL_ResMS.size}')
#         SL_beta = np.nan_to_num(SL_beta)
#         SL_ResMS = np.nan_to_num(SL_ResMS)
#         SL_beta = SL_beta[reg_mask.to_numpy(), :]

#         print(f'       - Loading Residuals...')
#         SL_residuals, _, _ = spm.get_residuals(SL_mask_img)
#         SL_residuals = np.nan_to_num(SL_residuals)

#         print(f'         - Computing Precision Matrix...')

#         SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')
#         # SL_Prec = np.nan_to_num(SL_Prec)

#         measurements = SL_beta / np.sqrt(SL_ResMS)
#         measurements = np.nan_to_num(measurements)
#         ds = Dataset(measurements = measurements, 
#                     descriptors={'center': center},
#                     obs_descriptors=dict(info),
#                     channel_descriptors={'voxels': nb})
        
#         center_data.append(ds)
#         center_noise.append(SL_Prec)


#     RDM = calc_rdm(center_data, 
#         method=method,
#         descriptor='events', 
#         noise=center_noise,
#         cv_descriptor='run_number').dissimilarities

# SL_rdms = RDMs(RDM,
#             rdm_descriptors={'voxel_index': centers},
#             dissimilarity_measure=method)


#%% --------------------------------------------------------------- option C) compute single RDM for specific center

centers = np.array(centers)

events='condition_number'
method='crossnobis'
verbose=True

info['events'] = info[events]

info_new = []
info_new = pd.DataFrame(info_new)
info_new['events'] = info['events'].astype('double')
info_new['run_number'] = info['run_number'].astype('double')

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
n_conds = len(np.unique(info['events']))
RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))


chunk = chunked_center[0]

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

    print(f'       - Loading Residuals...')
    SL_residuals, _, _ = spm.get_residuals(SL_mask_img)
    SL_residuals = np.nan_to_num(SL_residuals)


    print(f'         - Computing Precision Matrix...')

    SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')
    measurements = SL_beta / np.sqrt(SL_ResMS)
    measurements = np.nan_to_num(measurements)


    ds = Dataset(measurements = measurements, 
                descriptors={'center': center},
                obs_descriptors=dict(info_new),
                channel_descriptors={'voxels': nb})

    center_data.append(ds)
    center_noise.append(SL_Prec)

    print('calculating RDMs for current chunk...')


    # print("Type of center_data:", type(center_data))
    # print("Type of method:", type(method))
    print("Type of center_noise:", type(center_noise))
    print("Type of info['events']:", type(info_new['events']) if 'events' in info else 'N/A')
    print("Type of cv_descriptor (hardcoded as 'run_number'):", type('run_number'))
#%%
RDM_corr = calc_rdm(center_data, 
            method=method,
            descriptor='events', 
            noise=center_noise,
            cv_descriptor='run_number')
#%%
RDM[chunk, :] = RDM_corr.dissimilarities


#%%

if evaluate_model_singleSubj:

# 6) compare Searchlight RDMs with model RDM
    print('    - Evaluating RDMs...')

    eval_results = evaluate_models_searchlight(SL_rdms, model, eval_fixed, method = cfg.RSAmethod)
    eval_score = [float(e.evaluations.item()) for e in eval_results]

    x, y, z = mask.shape
    RDM_brain = np.zeros([x*y*z])
    RDM_brain[SL_rdms.rdm_descriptors['voxel_index']] = eval_score
    RDM_brain = RDM_brain.reshape(mask.shape)



