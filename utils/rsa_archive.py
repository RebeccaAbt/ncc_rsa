import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from pqdm.processes import pqdm
from rsatoolbox.data import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.rdm import calc_rdm, RDMs
from joblib import Parallel, delayed
from rsatoolbox.data import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.rdm import calc_rdm
from pqdm.processes import pqdm

def get_searchlight_RDMs_crossnobis_chunksOnly(spm,
                                    centers, 
                                    neighbors, 
                                    mask,
                                    reg_mask, 
                                    info, 
                                    events='condition', 
                                    method='crossnobis', 
                                    verbose=True):
    
    info['events'] = info[events]

    # original mask
    mask_bool = mask.get_fdata() > 0
    n_voxels_total = np.prod(mask_bool.shape)
    mask_bool_1D = mask_bool.flatten()
    mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)
    
    #initalize variables
    n_centers = centers.shape[0]
    n_conds = len(np.unique(info['events']))
    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    chunked_center = np.split(np.arange(n_centers),
                                np.linspace(0, n_centers, 101, dtype=int)[1:-1]) 

    for chunk in tqdm(chunked_center, desc='Calculating RDMs...'):
        center_data, center_noise = [], []
        for c in chunk:

            center = centers[c]
            nb = neighbors[c]

            print(f'current center: {center}')

            SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
            SL_mask_1D[nb] = True # !!!!!!! important!! must be  [nb], not SL_mask_idx
            SL_mask_3D = SL_mask_1D.reshape(mask.shape)
            SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)
            
            print('     - Loading SL_betas...')
            SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
            print(f'size of ResMS: {SL_ResMS.size}')
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
                        obs_descriptors=dict(info),
                        channel_descriptors={'voxels': nb})
            
            center_data.append(ds)
            center_noise.append(SL_Prec)

        print('calculating RDMs for current chunk...')


        # print("Type of center_data:", type(center_data))
        # print("Type of method:", type(method))
        print("Type of center_noise:", type(center_noise))
        print("Type of info['events']:", type(info['events']) if 'events' in info else 'N/A')
        print("Type of cv_descriptor (hardcoded as 'run_number'):", type('run_number'))
        RDM_corr = calc_rdm(center_data, 
                    method=method,
                    descriptor='events', 
                    noise=center_noise,
                    cv_descriptor='run_number')

        RDM[chunk, :] = RDM_corr.dissimilarities
    
    SL_rdms = RDMs(RDM,
                rdm_descriptors={'voxel_index': centers},
                dissimilarity_measure=method)
    
    return SL_rdms


def get_searchlight_RDMs_crossnobis_test(spm,
                                    centers, 
                                    neighbors, 
                                    mask,
                                    reg_mask, 
                                    info, 
                                    events='condition', 
                                    method='crossnobis', 
                                    verbose=True):
    
    info['events'] = info[events]

    # original mask
    mask_bool = mask.get_fdata() > 0
    n_voxels_total = np.prod(mask_bool.shape)
    mask_bool_1D = mask_bool.flatten()
    mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)
    
    #initalize variables
    n_centers = centers.shape[0]
    n_conds = len(np.unique(events))
    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    chunked_center = np.split(np.arange(n_centers),
                                np.linspace(0, n_centers, 101, dtype=int)[1:-1]) 


    center_data, center_noise = [], []

    center = centers[2000]
    nb = neighbors[2000]

    print(f'current center: {center}')

    SL_idx_1D = mask_bool_1D_idx[nb]
    SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
    SL_mask_1D[SL_idx_1D] = True
    SL_mask_3D = SL_mask_1D.reshape(mask.shape)
    SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)
    
    print('     - Loading SL_betas...')
    SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
    
    SL_beta = SL_beta[reg_mask.to_numpy(), :]

    print(f'       - Loading Residuals...')
    SL_residuals, _, _ = spm.get_residuals(SL_mask_img)

    print(f'         - Computing Precision Matrix...')

    SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')

    ds = Dataset(measurements = SL_beta / np.sqrt(SL_ResMS), 
                descriptors={'center': center},
                obs_descriptors=dict(info),
                channel_descriptors={'voxels': nb})
    
    center_data.append(ds)
    center_noise.append(SL_Prec)

    print('calculating RDMs for current chunk...')
    RDM_corr = calc_rdm(ds, 
                        method=method,
                        descriptor='events', 
                        noise=SL_Prec,
                        cv_descriptor='run_number')

    RDM = RDM_corr.dissimilarities

    SL_rdms = RDMs(RDM,
                rdm_descriptors={'voxel_index': centers},
                dissimilarity_measure=method)
    
    return SL_rdms


def get_searchlight_RDMs_crossnobis(spm,
                                    centers, 
                                    neighbors, 
                                    mask,
                                    reg_mask, 
                                    info, 
                                    events='condition', 
                                    method='crossnobis', 
                                    verbose=True):
    
    info['events'] = info[events]

    # original mask
    mask_bool = mask.get_fdata() > 0
    n_voxels_total = np.prod(mask_bool.shape)
    mask_bool_1D = mask_bool.flatten()
    mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)
    
    #initalize variables
    n_centers = centers.shape[0]
    n_conds = len(np.unique(events))
    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    chunked_center = np.split(np.arange(n_centers),
                                np.linspace(0, n_centers, 101, dtype=int)[1:-1]) 

    for chunk in tqdm(chunked_center, desc='Calculating RDMs...'):
        center_data, center_noise = [], []
        for c in chunk:

            center = centers[c]
            nb = neighbors[c]

            print(f'current center: {center}')
        
            SL_idx_1D = mask_bool_1D_idx[nb]
            SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
            SL_mask_1D[SL_idx_1D] = True
            SL_mask_3D = SL_mask_1D.reshape(mask.shape)
            SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)
            
            print('     - Loading SL_betas...')
            SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
            
            SL_beta = SL_beta[reg_mask.to_numpy(), :]

            print(f'       - Loading Residuals...')
            SL_residuals, _, _ = spm.get_residuals(SL_mask_img)

            print(f'         - Computing Precision Matrix...')

            SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')

            ds = Dataset(measurements = SL_beta / np.sqrt(SL_ResMS), 
                        descriptors={'center': center},
                        obs_descriptors=dict(info),
                        channel_descriptors={'voxels': nb})
            
            center_data.append(ds)
            center_noise.append(SL_Prec)

        print('calculating RDMs for current chunk...')
        RDM_corr = calc_rdm(center_data, 
                            method=method,
                            descriptor='events', 
                            noise=center_noise,
                            cv_descriptor='run_number')

        RDM[chunk, :] = RDM_corr.dissimilarities
    
    SL_rdms = RDMs(RDM,
                rdm_descriptors={'voxel_index': centers},
                dissimilarity_measure=method)
    
    return SL_rdms

def process_chunk(chunk, 
                  centers, 
                  neighbors, 
                  spm, 
                  mask,
                  reg_mask,
                  mask_bool_1D_idx,
                  n_voxels_total,
                  info,
                  method):

    print(f'info: [{info}]')
    print(f'data type of info: {type(info)}')
    center_data = []
    center_noise = []

    for c in chunk:
        center = centers[c]
        nb = neighbors[c]

        SL_idx_1D = mask_bool_1D_idx[nb]
        SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
        SL_mask_1D[SL_idx_1D] = True
        SL_mask_3D = SL_mask_1D.reshape(mask.shape)
        SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)

        SL_beta, SL_ResMS, SL_info = spm.get_betas(SL_mask_img)
        # SL_info = pd.DataFrame(SL_info)
        # reg_mask = SL_info['reg_name'].str.contains('_hit') | SL_info['reg_name'].str.contains('_miss')
        # SL_info = SL_info[reg_mask].reset_index(drop=True)
        SL_beta = SL_beta[reg_mask.to_numpy(), :]

        SL_residuals, _, _ = spm.get_residuals(SL_mask_img)
        SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')

        ds = Dataset(
            measurements=SL_beta / np.sqrt(SL_ResMS),
            descriptors={'center': center},
            obs_descriptors=dict(info),
            channel_descriptors={'voxels': nb}
        )

        center_data.append(ds)
        center_noise.append(SL_Prec)

    rdm_chunk = calc_rdm(center_data,
                         method=method,
                         descriptor='events',
                         noise=center_noise,
                         cv_descriptor='run_number')
    return rdm_chunk.dissimilarities


def parallel_searchlight_RDMs_crossnobis(spm,
                                    centers, 
                                    neighbors, 
                                    mask,
                                    reg_mask, 
                                    info, 
                                    events='condition', 
                                    method='crossnobis', 
                                    verbose=True):
    
    info['events'] = info[events]
    print(f'info first: {info}')
    print(f'data type of info: {type(info)}')

    # original mask
    mask_bool = mask.get_fdata() > 0
    n_voxels_total = np.prod(mask_bool.shape)
    mask_bool_1D = mask_bool.flatten()
    mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)
    
    #initalize variables
    n_centers = centers.shape[0]
    n_conds = len(np.unique(events))
    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    chunked_center = np.split(np.arange(n_centers),
                                np.linspace(0, n_centers, 101, dtype=int)[1:-1]) 

    # Process chunks in parallel
    rdm_results = Parallel(n_jobs=10)(
        delayed(process_chunk)(
            chunk, 
            centers, 
            neighbors, 
            spm, 
            mask, 
            reg_mask, 
            mask_bool_1D_idx, 
            n_voxels_total, 
            info, 
            method
        ) for chunk in chunked_center
    )

    # Merge chunk results
    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    for chunk, rdm_chunk in zip(chunked_center, rdm_results):
        RDM[chunk, :] = rdm_chunk

    SL_rdms = RDMs(RDM,
                rdm_descriptors={'voxel_index': centers},
                dissimilarity_measure=method)
        
    return SL_rdms
    


def pqdm_searchlight_RDMs_crossnobis(spm,
                                    centers, 
                                    neighbors, 
                                    mask,
                                    reg_mask, 
                                    info, 
                                    events='condition', 
                                    method='crossnobis', 
                                    verbose=True):
    
    info['events'] = info[events]

    # original mask
    mask_bool = mask.get_fdata() > 0
    n_voxels_total = np.prod(mask_bool.shape)
    mask_bool_1D = mask_bool.flatten()
    mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)
    
    #initalize variables
    n_centers = centers.shape[0]
    n_conds = len(np.unique(events))
    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    chunked_center = np.split(np.arange(n_centers),
                                np.linspace(0, n_centers, 101, dtype=int)[1:-1]) 

    for chunk in pqdm(chunked_center, desc='Calculating RDMs...'):
        center_data, center_noise = [], []
        for c in chunk:

            center = centers[c]
            nb = neighbors[c]

            print(f'current center: {center}')
        
            SL_idx_1D = mask_bool_1D_idx[nb]
            SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
            SL_mask_1D[SL_idx_1D] = True
            SL_mask_3D = SL_mask_1D.reshape(mask.shape)
            SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)
            
            print('     - Loading SL_betas...')
            SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
            
            SL_beta = SL_beta[reg_mask.to_numpy(), :]

            print(f'       - Loading Residuals...')
            SL_residuals, _, _ = spm.get_residuals(SL_mask_img)

            print(f'         - Computing Precision Matrix...')

            SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')

            ds = Dataset(measurements = SL_beta / np.sqrt(SL_ResMS), 
                        descriptors={'center': center},
                        obs_descriptors=dict(info),
                        channel_descriptors={'voxels': nb})
            
            center_data.append(ds)
            center_noise.append(SL_Prec)

        print('calculating RDMs for current chunk...')
        RDM_corr = calc_rdm(center_data, 
                            method=method,
                            descriptor='events', 
                            noise=center_noise,
                            cv_descriptor='run_number')

        RDM[chunk, :] = RDM_corr.dissimilarities
    
    SL_rdms = RDMs(RDM,
                rdm_descriptors={'voxel_index': centers},
                dissimilarity_measure=method)
    
    return SL_rdms


def process_chunk_pqdm(chunk, centers, neighbors, spm, mask, mask_bool_1D_idx, n_voxels_total, reg_mask, info, method):
    from rsatoolbox.data import Dataset
    from rsatoolbox.data.noise import prec_from_residuals
    from rsatoolbox.rdm import calc_rdm

    center_data, center_noise = [], []
    for c in chunk:
        center = centers[c]
        nb = neighbors[c]

        SL_idx_1D = mask_bool_1D_idx[nb]
        SL_mask_1D = np.zeros(n_voxels_total, dtype=bool)
        SL_mask_1D[SL_idx_1D] = True
        SL_mask_3D = SL_mask_1D.reshape(mask.shape)
        SL_mask_img = nib.Nifti1Image(SL_mask_3D.astype(np.int16), mask.affine)

        SL_beta, SL_ResMS, _ = spm.get_betas(SL_mask_img)
        SL_beta = SL_beta[reg_mask.to_numpy(), :]

        SL_residuals, _, _ = spm.get_residuals(SL_mask_img)
        SL_Prec = prec_from_residuals(SL_residuals, dof=spm.eff_df, method='shrinkage_diag')

        ds = Dataset(
            measurements=SL_beta / np.sqrt(SL_ResMS),
            descriptors={'center': center},
            obs_descriptors=info.to_dict(orient='list'),
            channel_descriptors={'voxels': nb}
        )

        center_data.append(ds)
        center_noise.append(SL_Prec)

    RDM_chunk = calc_rdm(
        center_data,
        method=method,
        descriptor='events',
        noise=center_noise,
        cv_descriptor='run_number'
    )

    return RDM_chunk.dissimilarities