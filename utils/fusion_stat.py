#%% 
import os
import re
from copy import deepcopy

import mne
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
from tqdm import tqdm
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
import scipy


#%%
def check_subj_id(file, cfg):
    match = re.search(r'\d{8}[A-Za-z]{4}', os.path.split(file)[1]) # Extract subject ID: 8 digits followed by 4 letters
    if match:
        return match.group(0)
    else:
        raise ValueError(f"Could not extract subject ID from filename: {file}")
        

def partial_subj_data(maskSize, voxel_coords, fusion_data, times, modality = 'sensory', do_print=True):

    input_data = [fusion_data['data'][t][modality] for t in times]
    subj_data = np.full([maskSize[0], maskSize[1], maskSize[2], len(input_data)], np.nan)
    
    if do_print:
        print(f'\t [3] Unpacking {modality} data to full 4D size + fill missings with nans', flush=True)
    for n_data, data in enumerate(input_data):
        for n_r, r in enumerate(data):
            subj_data[(*voxel_coords[n_r], n_data)] = r    # the "*" indicates that the array is unpacked first!
                        
    return subj_data


def _remove_data_containing_nans(data_stacked):

    indices_full = np.arange(data_stacked.shape[1])
    indices_noNan = []
    data_noNan = []

    for i in indices_full:
        if not np.any(np.isnan(data_stacked[:,i])):

            indices_noNan.append(indices_full[i])
            data_noNan.append(data_stacked[:,i])

    return indices_noNan, data_noNan 


def stack_data(data_dict, modality=None):

    data_stacked = np.stack([d.reshape(-1) for d in data_dict[modality]])
    idx_noNan, data_noNan = _remove_data_containing_nans(data_stacked)

    return data_stacked, data_noNan


def get_cluster_def_thres(data_pre, method, tail, pval = 0.001, t_percent = 99.5):
    '''
    method: 't' or 'p'
        t : one-sample t-test of pre-stim data, 99.5 precentile
        p: based on df / arbitrary p
    '''

    X_thres = np.transpose(np.array(data_pre)) # get nSubjects*observations shape
    if method == 't':
        print(f'\t\t - Computing cluster definition threshold on pre-stimulus data + get adjacency', flush=True)
        print(f'\t\t\t - Shape of input variable "X": {X_thres.shape} (n_subj x valid_voxels_all_timepoints)', flush=True)
        t_results = mne.stats.ttest_1samp_no_p(X_thres)
        cluster_def_thres = np.percentile(t_results, t_percent)
    elif method == 'p':
        print(f'\t\t - Computing cluster definition threshold based on df and arbitrary  p', flush=True)
        n_subj = X_thres.shape[0]
        df = n_subj - 1  # degrees of freedom for the test
        if tail == 1: # 1-tailed
            cluster_def_thres = scipy.stats.t.ppf(1 - pval, df)  # 1-tailed
        elif tail == 0: # 2-tailed
            cluster_def_thres = scipy.stats.t.ppf(1 - pval / 2, df)

    return cluster_def_thres


def find_voxels_noNan_allSubj(data_pre, data_post, maskFile='/home/reabt/experiments/ncc/MRI/data/sync/19910703eigl/NCC/firstLevel_sensory_M1C/mask.nii'):
    '''
    for every subject, check for every voxel, if it is finite (= not nan) at timepoint (=last dimension) (= in pre- and post-stim data). <
    mask is true for a voxel if it is a valid voxel for all subjects
    
    :param data_pre: Description
    :param data_post: Description
    '''
    print(f'\t\t - Masking data so we only use voxels with valid data from all subjects')

    mask =  np.all(np.stack([np.all(np.isfinite(t), axis=-1) for t in data_post]), axis=0) & \
            np.all(np.stack([np.all(np.isfinite(t), axis=-1) for t in data_pre]), axis=0)

    # sanity check: plot the mask to make sure this makes sense

    plot_img = new_img_like(maskFile, mask)
    fig = plt.figure(figsize=(12, 3))
    display = plotting.plot_stat_map(
                plot_img, 
                # colorbar=True, 
                # threshold=threshold,
                display_mode='z', 
                draw_cross=False, 
                figure=fig,
                # title=f'suprasensory', 
                cmap='viridis',
                black_bg=False, 
                annotate=False)
    plt.show()
    return np.bool(mask)


def get_spatial_adjacency_3d(mask_3d):
    '''
    Docstring for get_spatial_adjacency_3d
    
    :param mask_3d: Description
    

    My approach for adjacency matrix: 3D adjacency instead of 4d adjacency: Cluster-permutation dopcumentation says that we can also compute a adjacency 
    matrix from only spatial 3d data instead of spatiotemporal 4d data. We can then set the "max_step" argument 
    to something that accounts for the temporal adjacency. 

    For the 3D adjacency matrix we can use the function:

    sklearn.feature_extraction.image.grid_to_graph(
                    n_x, 
                    n_y, 
                    n_z=1, 
                    *, 
                    mask=None, 
                    return_as=<class 'scipy.sparse._coo.coo_matrix'>, 
                    dtype=<class 'int'>)

    Because it seems like we can use a mask here to mask the nan values from the data!!!
    '''
    print(f'\t\t - Getting adjacency')

    s_3D = mask_3d.shape
    return grid_to_graph(   n_x = s_3D[0], 
                            n_y = s_3D[1], 
                            n_z= s_3D[2],
                            mask=mask_3d   )

def get_masked_data(data, mask_1D):
    '''
    Docstring for get_masked_data
    
    :param data: list of length n_subj with arrays of shape (s1 x s2 x s3 x time)
    :param mask_1D: 1d format of original 3d spatial mask
    '''
    s_4D = data[0].shape
    print(f'\t\t\t  - shape of 4D data (s1 x s2 x s3 x time): {s_4D}')
    maskedData = []

    for d in data:
        subj_maskedData = np.reshape(d.transpose(3, 0, 1, 2), (s_4D[3], s_4D[0]*s_4D[1]*s_4D[2]))
        subj_maskedData = subj_maskedData[:, mask_1D] # single subject
        maskedData.append(subj_maskedData) # all subjects
    return np.stack([d for d in maskedData]) 


def plot_fusion_all_subjects(data_pre, data_post, modality):

    mean_post = np.mean(data_post, axis=-1)#.flatten() # mean over subjects and voxels
    mean_pre = np.mean(data_pre, axis=-1)#.flatten()

    mycolors = ['blue', 'cyan', 'navy', 'green', 'turquoise', 'teal', 'slateblue', 'darkgreen', 'rebeccapurple', 'indigo', 'forestgreen', 'seagreen', 'yellowgreen']*2

    plt.figure(figsize=(12, 6))

    for i in range(mean_pre.shape[0]):

        plt.plot(range(mean_post.shape[1]), 
                mean_post[i,:], 
                color=mycolors[i], 
                alpha=1, 
                #  s=3, 
                label=f'post subj {i}')
        plt.plot(range(mean_pre.shape[1]), 
                mean_pre[i,:], 
                color='red', 
                alpha=1, 
                #  s=3, 
                #  label='pre'
                )

    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title(f'{modality} commonality (mean over subjects & voxels)  over time')
    plt.show()


def cluster_permutation(X, threshold, adjacency, n_permutations=5000, tail=1, t_power = 1, n_jobs=None, seed=None, buffer_size=None, verbose=None):
    
    print(f'\t [2] Running the cluster permutation test')

    T_obs, clusters, cluster_pv, H0 = \
        mne.stats.spatio_temporal_cluster_1samp_test(
                X = X, 
                threshold = threshold,
                n_permutations=n_permutations, 
                tail=tail, 
                adjacency=adjacency, 
                n_jobs=n_jobs, 
                seed=seed, 
                max_step=1, 
                step_down_p=0, 
                t_power=t_power, # t_power=1: by cluster SUM --> t_power=0 for cluster SIZE
                out_type='indices', 
                check_disjoint=False, 
                buffer_size=buffer_size, 
                verbose=verbose)
    return T_obs, clusters, cluster_pv, H0

def get_good_clusters_4d(clu, mask, thres = 0.05):
    '''
    Docstring for get_good_clusters_4d
    
    :param clu: output of my cluster_permutation function as tuple
    :param thres: threshold we want tu use
    :param mask_1d: valid voxels (constructed in "find_voxels_noNan_allSubj") as a flat boolean mask
    '''
    print(f'\t [3] Accumulate good clusters to 4D shape')

    T_obs, clusters, cluster_pv, H0 = clu
    mask_1d = mask.flatten()
    good_clusters=[clusters[i] for i in np.where(cluster_pv<=thres)[0]]
    shape = T_obs.shape  # should be (n_times, n_voxels)
    cluster_array_dummy = np.zeros(shape, dtype=bool) # Reconstruct the (time, voxel) array of zeros
    n_times = cluster_array_dummy.shape[0]   
    all_clusters_4d = np.zeros((n_times, *mask.shape))

    for c in good_clusters:

        times, voxels = c      # clusters[cluster_idx] is a tuple: (time_indices, voxel_indices) 

        cluster_array = deepcopy(cluster_array_dummy) # Reconstruct the (time, voxel) array of zeros
        cluster_array[times, voxels] = True # shape (n_times, n_voxels)
 
        full_cluster_4d = np.zeros((n_times, mask_1d.size))
        full_cluster_4d[:, mask_1d] = cluster_array # Fill only masked voxels
        full_cluster_4d = full_cluster_4d.reshape((n_times, *mask.shape))
        all_clusters_4d += full_cluster_4d

    return all_clusters_4d