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
from joblib import Parallel, delayed

sys.path.append('/home/reabt/experiments/ncc/MRI/code/')
from utils.load_cfg import load_fusion_config_instance
from utils.fusion_stat import *

from plus_slurm import Job


#%%

config_class_name = 'FusionConfig_C5'
thres = 0.1,    # for defnition of good clusters after cluster permutation
meanMEG = False

cfg = load_fusion_config_instance(config_class_name) # since no subject ID is specified, '*' will be used --> creates pattern with subject-wildcard instead of subject filename 

preStim_time = np.array(range(0, 100)) # should be adapted to dynamically find the right time points... but for now this selects times <=0
postStim_time = np.array(range(100, 200))

all_data_pre_sensory, all_data_post_sensory, all_data_pre_suprasensory, all_data_post_suprasensory, maskData_all = [], [], [], [], []

print(f'\n[A] loading the data', flush=True)

if meanMEG:
    fusionFiles = sorted(glob(cfg.get_outFile_names()['fusion_mean_pkl']))
else:
    fusionFiles = sorted(glob(cfg.get_outFile_names()['fusion_pkl']))

print(f'fusionFiles: {fusionFiles}', flush=True)
cfg.subjectID = '19910823ssld' # dummy subject
cfg.configure_paths()

def process_subject(file):
    """Process a single subject's data and return results for all modalities."""
    cfg.subjectID = check_subj_id(file, cfg)
    cfg.configure_paths()

    print(f"\nProcessing subject {cfg.subjectID}...\n", flush=True)
    fusion_data = joblib.load(file)

    mask = nib.load(cfg.get_mask_file())
    mask_data = mask.get_fdata()
    mask_size = mask.shape

    indices = np.unravel_index(fusion_data['voxel_index_py'], mask_size)
    voxel_coords = np.stack(indices, axis=1)

    result = {
        "maskData": mask_data,
        "pre_sensory": partial_subj_data(mask_size, voxel_coords, fusion_data, preStim_time, modality="sensory", do_print=False),
        "post_sensory": partial_subj_data(mask_size, voxel_coords, fusion_data, postStim_time, modality="sensory", do_print=False),
        "pre_suprasensory": partial_subj_data(mask_size, voxel_coords, fusion_data, preStim_time, modality="suprasensory", do_print=False),
        "post_suprasensory": partial_subj_data(mask_size, voxel_coords, fusion_data, postStim_time, modality="suprasensory", do_print=False),
    }
    return result

#%%
# Run in parallel
results = Parallel(n_jobs=-1, backend='loky')(delayed(process_subject)(f) for f in fusionFiles)
#%%
# Collect results
for res in results:
    maskData_all.append(res["maskData"])
    all_data_pre_sensory.append(res["pre_sensory"])
    all_data_post_sensory.append(res["post_sensory"])
    all_data_pre_suprasensory.append(res["pre_suprasensory"])
    all_data_post_suprasensory.append(res["post_suprasensory"])

data_pre_dict = {'sensory': all_data_pre_sensory, 'suprasensory': all_data_pre_suprasensory}
data_post_dict = {'sensory': all_data_post_sensory, 'suprasensory': all_data_post_suprasensory}

print(f'[B] Looping over modalities')

modalities = ALL_MODELS

for modality  in modalities:

    cfg.modelType = modality

    print(f'\n[{modality}]\n')
    print(f'\t [1] preparing the data', flush=True)

    # data_pre, data_pre_noNan = stack_data(t = 'pre', modality=modality)
    # data_post, data_post_noNan = stack_data(t = 'post', modality=modality)
    print(f'data_pre_dict: {data_pre_dict}', flush=True)
    _, data_pre_noNan = stack_data(data_pre_dict, modality)
    # data_post, data_post_noNan = stack_data(data_post_dict, modality)

    all_data_pre = data_pre_dict[modality]
    all_data_post = data_post_dict[modality]

    cluster_def_thres   = get_cluster_def_thres(data_pre_noNan)
    mask                = find_voxels_noNan_allSubj(all_data_pre, all_data_post, cfg.get_mask_file())
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
        cluster_permutation(X_diff, cluster_def_thres, spatial_adjacency, n_jobs=8) # shape of T_obs: (n_times, n_voxels)

    good_clusters = get_good_clusters_4d(clu, mask, thres)

    n_subj = len(all_data_pre)
    thres_str = str(thres).replace('.','_')

    print(f'\t [4] Saving variables...', flush=True)

    file_note_clu = f'{n_subj}_subj_{modality}' # anottate number of subjects & thresold used for defining "good" clusters
    file_note_good_clu = f'{n_subj}_subj_thres_{thres_str}_{modality}'

    if meanMEG == False:
        fileName_clu = cfg.get_outFile_names()['cp'].replace('cp', f'cp_{file_note_clu}')
        fileName_good_clu = cfg.get_outFile_names()['good_clusters'].replace('clusters', f'clusters_{file_note_good_clu}')
    else:
        fileName_clu = cfg.get_outFile_names()['cp_mean'].replace('cp', f'cp_{file_note_clu}')
        fileName_good_clu = cfg.get_outFile_names()['good_clusters_mean'].replace('clusters', f'clusters_{file_note_good_clu}')
    print(f'\nsaving CP results as {fileName_clu}')
    print(f'\nsaving good clusters as {fileName_good_clu}')
    joblib.dump(clu, fileName_clu)
    joblib.dump(good_clusters, fileName_good_clu)
