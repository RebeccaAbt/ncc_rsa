#%%

import joblib
import rsatoolbox as rsa
from tqdm import tqdm
import numpy as np
import nibabel as nib
from plus_slurm import Job
from scipy.io import savemat

import rsatoolbox as rsa
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_config_instance
from utils.fusion import reorder_rdms, commonality_fusion, adjust_descriptors

#%% 

class Fusion(Job):
    def run(self,
            subjectID = '19910823ssld',
            config_class_name = 'SetupConfig_E1',  
            maskNr = 0
            ):

        print('[1] load config & data\n')

        cfg = load_config_instance(config_class_name, subjectID, maskNr) 
        cfg.print_summary()
        cfg.configure_paths()
        outFiles = cfg.get_outFile_names()
        MEG_file = cfg.MEG_inFile[0]
        fileName_movie = outFiles['movie']

        rdm_movie = joblib.load(fileName_movie)
        SL_rdms = joblib.load(outFiles['SL_rdms_fullBrain']) 
        models = cfg.get_model_RDM()
        mask = nib.load(cfg.maskFile) # we need this only for fransforming c-indiuces to fortran indices because linear indexing in matlab works differently 
        s = mask.shape

        print('[2] Reorder data\n')

        SL_rdms, models = reorder_rdms(SL_rdms, models) # adjust model, if conditions were missing in fMRI data

        subset_descriptors = models[0].rdm_obj.pattern_descriptors['condition']
        rdm_movie = rdm_movie.subset_pattern('condition', subset_descriptors) # adjust MEG data if condition was missing in fMRI data

        print('[3] Calculate commonality\n')

        sensory_vec = models[0].rdm_obj.get_vectors()[0]
        suprasensory_vec = models[1].rdm_obj.get_vectors()[0]

        fusion_data = []
        i = 0

        for t in tqdm(rdm_movie, desc='Fusing fMRI and MEG data...'):
            i += 1

            print(f'Processing timepoint {i} of {len(rdm_movie)}')

            MEG_vec = t.get_vectors()[0]
            timepoint = t.rdm_descriptors['index'][0]
            # data = []
            voxel_index_py = []
            sensory = []
            suprasensory = []
            for SL in SL_rdms: # loop over MRI searchlights
            # for SL in tqdm(SL_rdms, desc='Calculating commonality...'): # loop over MRI searchlights

                fMRI_vec = SL.get_vectors()[0]
                unique_sensory, unique_suprasensory = commonality_fusion(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec)
                
                voxel_index_py.append(SL.rdm_descriptors['voxel_index'][0])
                sensory.append(unique_sensory)
                suprasensory.append(unique_suprasensory)
                
            voxel_index_mat = np.ravel_multi_index(np.unravel_index(voxel_index_py, s), dims = s, order = 'F')+1

            fusion_data.append({'time': timepoint,
                        'voxel_index_py': voxel_index_py,
                        'voxel_index_mat': voxel_index_mat,
                        'sensory': sensory, 
                        'suprasensory': suprasensory                 
                        })

        joblib.dump(fusion_data, outFiles['fusion_pkl'])
        print(f'python outfile saved as {outFiles['fusion_pkl']}')

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
        for i in range(len(fusion_data)): # creates right format for matlab structure array
            data_mat[i].time = fusion_data[i]['time']
            data_mat[i].sensory = fusion_data[i]['sensory']
            data_mat[i].suprasensory = fusion_data[i]['suprasensory']

        # defined data type for our nested (final) recarray
        fusion_dtype = [
            ('voxel_index_py', 'i4', (n_voxel,)), 
            ('voxel_index_mat', 'i4', (n_voxel,)),          
            ('data', data_mat.dtype, (3,))  # nested: include dtype from data_mat         
        ]

        fusion_mat = np.recarray(1, dtype=fusion_dtype)
        fusion_mat.voxel_index_py = fusion_data[0]['voxel_index_py']
        fusion_mat.voxel_index_mat = fusion_data[0]['voxel_index_mat']
        fusion_mat.data = data_mat

        savemat(outFiles['fusion_mat'], {'fusion_data': fusion_mat})

        print(f'matlab outfile saved as {outFiles['fusion_mat']}')