
'''
Docstring for clusterjobs.do_fusion_np_parallel_outer_2


This is almost the same as "do_fusion_np_parallel_outer", but "do_fusion_np_parallel_outer" used "pqdm". "do_fusion_np_parallel_outer_2" uses jpblib Parallel. I want to compare the performance
'''


#%%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import joblib
from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from plus_slurm import Job
from scipy.io import savemat
from tqdm import tqdm
import pqdm

from utils.fusion import fuse_timepoint_np, fuse_timepoint_np_parallel
from utils.load_cfg import *
from utils.rsa import reorder_rdms, reorder_and_subset_all_data
from pqdm.processes import pqdm
import pickle

# Wrap the function to unpack args
def fuse_wrapper(arg):
    return fuse_timepoint_np(*arg)


#%%
class Fusion_np(Job):
    def run(self,
            subjectID = '19910823ssld',
            config_class_name = 'FusionConfig_C2',
            meanMEG = False
            ):

        print(f'\n-------------------------------------\n'\
                f'using the mean MEG RDM-movie: {meanMEG}\n'\
                    '-------------------------------------\n')

        print('[1] load config & data\n')

        cfg = load_fusion_config_instance(config_class_name, subjectID) 
        cfg.print_summary()
        cfg.configure_paths()

        outFiles = cfg.get_outFile_names()
        print(f'outFiles:{outFiles}', flush=True)
        print(f'meanMEG: {meanMEG}', flush=True)
        print(f'file already exists: {os.path.exists(outFiles['fusion_pkl'])}', flush=True)
        
        if not meanMEG and not os.path.exists(outFiles['fusion_pkl']): # If we dont't use the mean file, only run if output file does not exist yet
            print(f'Loading RDM-movie file {os.path.split(cfg.MEG_input)[1]}...\n')
            rdm_movie = joblib.load(cfg.MEG_input) # load MEG RDM-movie of current subject
        
        elif not meanMEG and os.path.exists(outFiles['fusion_pkl']): # stop the execution here, if the output file already exists
            msg = (
                '---------------------------------------------------------------------------------------\n'
                f'Fusion file already exists: {outFiles['fusion_pkl']}. Skipping computation.'
                '---------------------------------------------------------------------------------------\n')
            print(msg)
            return
        
        elif meanMEG: # if we use the mean MEG RDM-movie, always run, because running it with new subjects included means different mean MEG data
            print('loading mean RDM-movie file...')
            # rdm_movie = joblib.load(outFiles['movie_mean']) # this would load the mean RDMs of all MEG subjects (but most of them don't have matching fMRI data for the fusion)
            rdm_movie = cfg.get_mean_movie() # computes the mean MEG movie-RDMs of the subjects currently included in the fusion

        else:
            print("\n Something weird happened... \n")
    
        SL_rdms = joblib.load(cfg.MRI_input)

        models = cfg.get_model_RDM()
        mask = nib.load(cfg.get_mask_file()) # we need this only for fransforming c-indiuces to fortran indices because linear indexing in matlab works differently 
        s = mask.shape

        print('[2] Reorder data\n')
        # -------------- v this was before removing conditions with too little MEG trials
        # SL_rdms, models = reorder_rdms(SL_rdms, models) # adjust model, if conditions were missing in fMRI data

        # subset_descriptors = models[0].rdm_obj.pattern_descriptors['condition']
        # rdm_movie = rdm_movie.subset_pattern('condition', subset_descriptors) # adjust MEG data if condition was missing in fMRI data
        # -------------- ^

        SL_rdms, rdm_movie, models = reorder_and_subset_all_data(cfg, SL_rdms, rdm_movie, models)

        print('[3] Calculate commonality\n')

        sensory_vec = models[0].rdm_obj.get_vectors()[0]
        suprasensory_vec = models[1].rdm_obj.get_vectors()[0]

        # Run in parallel:
        # Do the fusion by computing the commonality coefficients of both models
        # fusion_data = pqdm(args, fuse_wrapper, n_jobs=nJobs)  # adjust n_jobs as needed
        fusion_data = Parallel(n_jobs=-1)(delayed(fuse_timepoint_np)(t, SL_rdms, sensory_vec, suprasensory_vec) for t in tqdm(rdm_movie, desc="Fusing fMRI/MEG data for time point..."))

        # Add voxel info
        voxel_index_py = [SL.rdm_descriptors['voxel_index'][0].item() for SL in SL_rdms]
        voxel_index_mat = np.ravel_multi_index(np.unravel_index(voxel_index_py, s), dims=s, order='F') + 1

        fusion_data = {'voxel_index_py': voxel_index_py,
                        'voxel_index_mat': voxel_index_mat,
                        'data': fusion_data,
                        }

        print('\n[4] Reformat the output (using "recarray" and export it to Matlab)\n', flush=True)

        n_voxel = len(SL_rdms)
        n_times = len(rdm_movie)
        print(f"n_voxel: {n_voxel}")
        print(f"n_times: {n_times}")

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
            data_mat[i].sensory = [f for f in fusion_data['data'][i]['sensory']]
            data_mat[i].suprasensory = [f for f in fusion_data['data'][i]['suprasensory']]

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
        print('saving the data...', flush=True)
        if meanMEG:
            # joblib.dump(fusion_data, 'cluster_outputs/fusion_data_mean_test.pkl')   
            joblib.dump(fusion_data, outFiles['fusion_mean_pkl'])   
            savemat(outFiles['fusion_mean_mat'], {'fusion_data': fusion_mat})
            print(f'python outfile saved as {outFiles['fusion_mean_pkl']}')
            print(f'matlab outfile saved as {outFiles['fusion_mean_mat']}')
        else:
            # joblib.dump(fusion_data, 'cluster_outputs/fusion_data_test.pkl')   
            joblib.dump(fusion_data, outFiles['fusion_pkl'])   
            savemat(outFiles['fusion_mat'], {'fusion_data': fusion_mat})
            print(f'python outfile saved as {outFiles['fusion_pkl']}')
            print(f'matlab outfile saved as {outFiles['fusion_mat']}')
