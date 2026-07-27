# %%
import os
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_MRI_config_instance
from utils.files import fix_spm_rawdata_paths
from utils.rsa import *
from utils.plots import *

import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

from nilearn import plotting
from nilearn.image import new_img_like

import rsatoolbox as rsa
from rsatoolbox.io.spm import SpmGlm
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import (
    get_volume_searchlight,
    get_searchlight_RDMs,
    evaluate_models_searchlight
)

from plus_slurm import Job
from configs.config import BaseConfig as cfg
import importlib
import configs.config as config
importlib.reload(config)
from configs.config import BaseConfig as cfg

from io import BytesIO
from PIL import Image


# %%
# trying Euclidean distance again
class SL_euclidean(Job):
    def run(self,
            subjectID = '19910823ssld',
            config_class_name = 'SetupConfig_E1',  
            maskNr = 0
            ):

        print('----------------------------------------------------')
        print('running...')
        print(f'     - subject:       {subjectID}')
        print(f'     - maskNr:        {maskNr}')
        print(f'     - configuration: {config_class_name}')
        print('----------------------------------------------------')

        # 1) load instance of selected class object --> this loads all the settings for the computations

        cfg = load_MRI_config_instance(config_class_name, subjectID, maskNr) 
        cfg.print_summary()
        cfg.save_summary()

        outFiles = cfg.get_outFile_names()
    
        models = cfg.get_model_RDM()

        print('Running whole Brain Searchlight Analysis with Euclidean Distance')
        os.makedirs(cfg.outDir, exist_ok=True)

        # 2) defining and loading some more variables

        spm = SpmGlm(cfg.spmDir)
        spm.get_info_from_spm_mat()

        print('    - Renaming rawdata file paths in SPM.mat...', flush=True)
        fix_spm_rawdata_paths(spm, cfg.dataDir + 'sync')    

        # 3) loading the betas and info

        print('    - Loading betas and info...', flush=True)
        betas, _, info = spm.get_betas(cfg.get_mask_file())
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

        betas = betas[reg_mask.to_numpy(), :]

        print('    - Loading full brain Mask...', flush=True)
        mask = nib.load(cfg.get_mask_file())
        mask_bool = mask.get_fdata() > 0
        n_voxels_total = np.prod(mask_bool.shape)
        mask_bool_1D = mask_bool.flatten()
        mask_bool_1D_idx = np.where(mask_bool_1D)[0] # shape=(95797,)

        mask_data = mask.get_fdata()

        # 4) finding Searchlights
        print('    - Finding centers and neighbors...', flush=True)
        centers, neighbors = get_volume_searchlight(mask_data, radius = cfg.SLradius, threshold=cfg.SLthr)

        data2d = np.zeros([betas.shape[0],n_voxels_total])
        data2d[:,mask_bool_1D_idx] = betas
        data2d = np.nan_to_num(data2d)

        # 4) compute searchlight RDMs
        print('    - Computing Searchlight RDMs...', flush=True)
        SL_rdms = get_searchlight_RDMs(data2d, centers, neighbors, info['condition'], method=cfg.RDMmethod)

        conditions = list(dict.fromkeys(info['condition']))
                
        SL_rdms.pattern_descriptors['condition'] = conditions  
        SL_rdms, models = reorder_rdms(SL_rdms, models) # reorder RDMs and check if we need so create a separate model

        # 6) compare Searchlight RDMs to model RDM

        print('    - Evaluating RDMs...')
        eval_results = evaluate_models_searchlight(SL_rdms, models, eval_fixed, method = cfg.RSAmethod)
        
        # 7) save general variables

        joblib.dump(info, outFiles['info'])
        joblib.dump(centers, outFiles['centers'])
        joblib.dump(neighbors, outFiles['neighbors'])
        joblib.dump(SL_rdms, outFiles['SL_rdms_fullBrain'])
        joblib.dump(eval_results, outFiles['eval_results_fullBrain'])
                                
        # 8) extract and plot data separately for sensory & suprasensory model

        allModels = ALL_MODELS
        for model in allModels:

            # ------------------------ v 
            cfg.modelType = model
            cfg.configure_paths()
            outFiles = cfg.get_outFile_names()
            # ------------------------ ^ important! for right prefix of files (suprasuprasensory/suprasensory)

            eval_score = [float(e.evaluations[0][cfg.modelIdx]) for e in eval_results]

            RDM_brain = get_RDM_brain(mask, SL_rdms, eval_score)

            joblib.dump(eval_score, outFiles['eval_score_fullBrain'])
            joblib.dump(RDM_brain, outFiles['RDM_brain_fullBrain'])

            eval_score_histogram(cfg, 
                                eval_score)
            plot_brain_map(cfg, 
                        mask, 
                        RDM_brain, 
                        eval_score)
            plot_max_modelFit_rdm(cfg, 
                                SL_rdms, 
                                eval_score)
                