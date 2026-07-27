#%%

import joblib
import rsatoolbox as rsa
from tqdm import tqdm
from pqdm.processes import pqdm
import numpy as np
import nibabel as nib
from plus_slurm import Job
from scipy.io import savemat
from pingouin import partial_corr
import pandas as pd
import os

import rsatoolbox as rsa
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_config_instance
from utils.fusion import fuse_timepoint, get_mean_movie
from utils.rsa import reorder_rdms

#%% 

# Wrap the function to unpack args
def fuse_wrapper(arg):
    return fuse_timepoint(*arg)

#%%
class Fusion(Job):
    def run(self,
            subjectID = '19910823ssld',
            config_class_name = 'SetupConfig_E1',
            nJobs = 10,
            meanMEG = False,
            cv_descriptor = None
            ):

        print(f'\n-------------------------------------\n'\
              'number of Jobs running parallel: {nJobs}\n\n'\
                'using the mean MEG RDM-movie: {meanMEG}\n'\
                    '-------------------------------------\n')

        print('[1] load config & data\n')

        cfg = load_config_instance(config_class_name, subjectID) 
        cfg.MEG_cv_descriptor = cv_descriptor
        cfg.print_summary()
        cfg.configure_paths()

        outFiles = cfg.get_outFile_names()

        if meanMEG:
            print('loading mean RDM-movie file...')
            # rdm_movie = joblib.load(outFiles['movie_mean']) # this would load the mean RDMs of all MEG subjects (but most of them don't have matching fMRI data for the fusion)
            rdm_movie = get_mean_movie() # computes the mean MEG movie-RDMs of the subjects currently included in the fusion

        else:
            print(f'Loading RDM-movie file {os.path.split(outFiles['movie'])[1]}...')
            rdm_movie = joblib.load(outFiles['movie'])

        SL_rdms = joblib.load(outFiles['SL_rdms_fullBrain'])

        models = cfg.get_model_RDM()
        mask = nib.load(cfg.get_mask_file()) # we need this only for fransforming c-indiuces to fortran indices because linear indexing in matlab works differently 
        s = mask.shape

        print('[2] Reorder data\n')

        SL_rdms, models = reorder_rdms(SL_rdms, models) # adjust model, if conditions were missing in fMRI data

        subset_descriptors = models[0].rdm_obj.pattern_descriptors['condition']
        rdm_movie = rdm_movie.subset_pattern('condition', subset_descriptors) # adjust MEG data if condition was missing in fMRI data

        print('[3] Calculate commonality\n')

        sensory_vec = models[0].rdm_obj.get_vectors()[0]
        suprasensory_vec = models[1].rdm_obj.get_vectors()[0]

        # Prepare arguments
        args = [(t, SL_rdms, sensory_vec, suprasensory_vec, s) for t in rdm_movie]

        # Run in parallel:
        # Do the fusion by computing the commonality coefficients of both models
        fusion_data = pqdm(args, fuse_wrapper, n_jobs=nJobs)  # adjust n_jobs as needed

        # Add voxel info
        voxel_index_py = [SL.rdm_descriptors['voxel_index'][0].item() for SL in SL_rdms]
        voxel_index_mat = np.ravel_multi_index(np.unravel_index(voxel_index_py, s), dims=s, order='F') + 1

        fusion_data = {'voxel_index_py': voxel_index_py,
                        'voxel_index_mat': voxel_index_mat,
                        'data': fusion_data,
                        }

        print('[4] Reformat the output (using "recarray" and export it to Matlab\n')

        n_voxel = len(SL_rdms)
        n_times = len(rdm_movie)

        # defined data type for our recarray
        data_dtype = [
            ('time', 'f4'),             # scalar float
            ('sensory', 'f4', (n_voxel,)),
            ('suprasensory', 'f4', (n_voxel,))
        ]

        # Create an empty recarray with n_times entries + fill it with data
        data_mat = np.recarray(n_times, dtype=data_dtype)
        for i in range(len(fusion_data['data'])): # creates right format for matlab structure array
            data_mat[i].time = fusion_data['data'][i]['time']
            data_mat[i].sensory = [f.item() for f in fusion_data['data'][i]['sensory']]
            data_mat[i].suprasensory = [f.item() for f in fusion_data['data'][i]['suprasensory']]

        # defined data type for our nested (final) recarray
        fusion_dtype = [
            ('voxel_index_py', 'i4', (n_voxel,)), 
            ('voxel_index_mat', 'i4', (n_voxel,)),          
            ('data', data_mat.dtype, (n_times,))  # nested: include dtype from data_mat         
        ]

        fusion_mat = np.recarray(1, dtype=fusion_dtype)
        fusion_mat.voxel_index_py = fusion_data['voxel_index_py']
        fusion_mat.voxel_index_mat = fusion_data['voxel_index_mat']
        fusion_mat.data = data_mat

        if meanMEG:
            joblib.dump(fusion_data, outFiles['fusion_mean_pkl'])   
            savemat(outFiles['fusion_mean_mat'], {'fusion_data': fusion_mat})
            print(f'python outfile saved as {outFiles['fusion_mean_pkl']}')
            print(f'matlab outfile saved as {outFiles['fusion_mean_mat']}')
        else:
            joblib.dump(fusion_data, outFiles['fusion_pkl'])   
            savemat(outFiles['fusion_mat'], {'fusion_data': fusion_mat})
            print(f'python outfile saved as {outFiles['fusion_pkl']}')
            print(f'matlab outfile saved as {outFiles['fusion_mat']}')

        