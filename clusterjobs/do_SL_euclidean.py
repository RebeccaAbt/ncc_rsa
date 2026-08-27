# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.load_cfg import load_MRI_config_instance
from utils.files import fix_spm_rawdata_paths
from utils.rsa import *
from utils.plots import *

import joblib
import numpy as np
import pandas as pd
import nibabel as nib

from rsatoolbox.io.spm import SpmGlm
from rsatoolbox.util.searchlight import (
    get_volume_searchlight,
    get_searchlight_RDMs
)

from plus_slurm import Job
from utils.provenance import configure_subject_logging, record_artifact

# %%

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
        models = cfg.get_model_RDM()

        outFiles = cfg.get_outFile_names()
        logger, _ = configure_subject_logging(outFiles['SL_rdms'], subjectID)
        logger.info('Starting fMRI Euclidean searchlight')
        logger.info('Configuration class: %s', config_class_name)
        logger.info('Configuration values: %s', vars(cfg))
        # if os.path.isfile(outFiles['SL_rdms']):
        #     print(f'    - Output files already exist. Skipping computation for maskNr {maskNr}. \n But we temporally load the ')
        #     tmp_add_condition_rdm_descriptor(cfg) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! THIS NEEDS TO BE REMOVED!!
        #     return


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
            info['identifier'] = info['condition'] + '_run' + info['run_number'].astype('str')

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
        # SL_rdms = get_searchlight_RDMs(data2d, centers, neighbors, info['condition'], method=cfg.RDMmethod)
        SL_rdms = get_searchlight_RDMs_parallel(data2d, centers, neighbors, info['condition'], method=cfg.RDMmethod)

        info_full = joblib.load(f'{CODE_DIR}/resources/info.pkl')

        conditions_missing, _ = check_conditions_missing(info, info_full)

        if conditions_missing:
            conditions = list(dict.fromkeys(info_full['condition'])) # use full list of conditions because we replaced missing values when RDMs were computed
        else:
            conditions = list(dict.fromkeys(info['condition']))

        SL_rdms.pattern_descriptors['condition'] = conditions    
        SL_rdms, models = reorder_rdms(SL_rdms, models)

        joblib.dump(info, outFiles['info'])
        joblib.dump(SL_rdms, outFiles['SL_rdms'])
        record_artifact(
            output_path=outFiles['SL_rdms'],
            operation_name='SL_euclidean.run',
            parameters={
                'config_class_name': config_class_name,
                'config': vars(cfg),
                'subjectID': subjectID,
                'maskNr': maskNr,
                'n_searchlights': int(len(centers)),
                'n_conditions': int(len(conditions)),
            },
            input_paths=[cfg.get_mask_file(), os.path.join(CODE_DIR, 'resources', 'info.pkl')],
        )
        logger.info('Wrote fMRI Euclidean searchlights and provenance')

        save_RSA_outputs(cfg)
