'''

Problem with the first test-script:
We did the cluster-permutation t-test against zero (because we only defined one condition: the post stimulus data). 
But for cluster permutation, we always need multiple conditions which we can exchange.
Since we compare two conditions, we substract one from the other. sign flipping in this scenario means that we can exchange the two conditions (i.e. randomize the labels).

So things we need to correct now:


1) [ x ] make sure both conditions have equal nuber of time points 
         (also check, that timepoint 0 belongs to post-stim)

2) [ o ] insert second (prestim condition) into cluster-permutation function

3) [ x ] improve how timepoints are selected 
        --> instead of hardcoding indices, find a way to soft-code it based on 
            real-time values the timepoints refer to 
            [ o ] Inwhich file/variable is the time point information stored?

4) [ o ] add t_lim to filename in config class so that it is clear which timepoints 
         were used for the analysis

5) [ o ] Use clustrer-sum instead of cluster size --> hack the source code if necessary


'''


#%% 

import sys
from glob import glob
import joblib
import nibabel as nib
import numpy as np
from pqdm.processes import pqdm

sys.path.append('/home/reabt/experiments/ncc/MRI/code/')
from utils.load_cfg import load_fusion_config_instance
from utils.fusion_stat import *

from plus_slurm import Job


#%%

class Fusion_CP(Job):
    def run(self,
            config_class_name = 'FusionConfig_C5',
            thres = 0.1     # for defnition of good clusters after cluster permutation
            ):

        cfg = load_fusion_config_instance(config_class_name) # since no subject ID is specified, '*' will be used --> creates pattern with subject-wildcard instead of subject filename 

        preStim_time = np.array(range(0, 100)) # should be adapted to dynamically find the right time points... but for now this selects times <=0
        postStim_time = np.array(range(100, 200))

        all_data_pre_sensory, all_data_post_sensory, all_data_pre_suprasensory, all_data_post_suprasensory, maskData_all = [], [], [], [], []

        print(f'\n[A] loading the data')

        fusionFiles = sorted(glob(cfg.get_outFile_names()['fusion_mean_pkl']))

        for file in fusionFiles:
            
            cfg.subjectID = check_subj_id(file, cfg)
            cfg.configure_paths() # update paths for subject ID instead of '*' pattern

            print(f'\nProcessing subject {cfg.subjectID}...\n', flush=True)
            print(f'\t [1] loading fusion data', flush=True)

            fusion_data = joblib.load(file)

            maskData_all.append(nib.load(cfg.get_mask_file()).get_fdata())
            maskSize = nib.load(cfg.get_mask_file()).shape

            indices = np.unravel_index(fusion_data['voxel_index_py'], maskSize)
            voxel_coords = np.stack(indices, axis=1) 

            print(f'\t [2] defining pre- and post-stim data', flush=True)

            all_data_pre_suprasensory.append(partial_subj_data(maskSize, voxel_coords, fusion_data, preStim_time, modality = 'suprasensory'))
            all_data_post_suprasensory.append(partial_subj_data(maskSize, voxel_coords, fusion_data, postStim_time, modality = 'suprasensory'))

            all_data_pre_sensory.append(partial_subj_data(maskSize, voxel_coords, fusion_data, preStim_time, modality = 'sensory'))
            all_data_post_sensory.append(partial_subj_data(maskSize, voxel_coords, fusion_data, postStim_time, modality = 'sensory'))

        data_pre_dict = {'sensory': all_data_pre_sensory, 'suprasensory': all_data_pre_suprasensory}
        data_post_dict = {'sensory': all_data_pre_suprasensory, 'suprasensory': all_data_post_suprasensory}

        print(f'[B] Looping over modalities')
        
        modalities = ALL_MODELS

        maskFile = cfg.get_mask_file()

        for modality  in modalities:

            print(f'\n[{modality}]\n')
            print(f'\t [1] preparing the data', flush=True)

            # data_pre, data_pre_noNan = stack_data(t = 'pre', modality=modality)
            # data_post, data_post_noNan = stack_data(t = 'post', modality=modality)

            data_pre, data_pre_noNan = stack_data(data_pre_dict, modality)
            data_post, data_post_noNan = stack_data(data_post_dict, modality)

            all_data_pre = data_pre_dict[modality]
            all_data_post = data_post_dict[modality]

            cluster_def_thres   = get_cluster_def_thres(data_pre_noNan)
            mask                = find_voxels_noNan_allSubj(all_data_pre, all_data_post, maskFile)
            spatial_adjacency   = get_spatial_adjacency_3d(mask)

            mask_1d = mask.flatten()
            s_4d = all_data_post[0].shape

            print(f'\n\t\t\t shape of 4D data (s1 x s2 x s3 x time): {s_4d}')
            print(f'\t\t\t number of valid voxels in mask: {np.sum(mask_1d)}')
            print(f"\t\t\t number of subjects (length of 'all_data_pre'): {len(all_data_pre)}\n")

            X_pre   = get_masked_data(all_data_pre, mask_1d)
            X_post  = get_masked_data(all_data_post, mask_1d)
            X_diff  = X_post - X_pre

            plot_fusion_all_subjects(X_pre, X_post, modality)

            T_obs, clusters, cluster_pv, H0 = clu = \
                cluster_permutation(X_diff, cluster_def_thres, spatial_adjacency) # shape of T_obs: (n_times, n_voxels)

            thres = 0.1

            good_clusters = get_good_clusters_4d(clu, mask, thres)

            n_subj = len(all_data_pre)
            thres_str = str(thres).replace('.','_')

            print(f'\t [4] Saving variables...', flush=True)

            joblib.dump(clu, f'cluster_outputs/clu_results_{n_subj}_subj_{modality}_{config_class_name}.pkl')
            joblib.dump(good_clusters, f'cluster_outputs/clusters_4d_thres_{thres_str}_{modality}_{config_class_name}.pkl')

