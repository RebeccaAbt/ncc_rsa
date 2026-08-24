# %%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.load_cfg import load_MRI_config_instance
from utils.files import fix_spm_rawdata_paths
from utils.plots import *
from utils.rsa import *

import joblib
import pandas as pd
import nibabel as nib


from rsatoolbox.io.spm import SpmGlm
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import (
    get_volume_searchlight,
    evaluate_models_searchlight
)

from plus_slurm import Job


#%%

class SL_crossnobis_full(Job):
    def run(self,
            subjectID = '19910823ssld',
            maskNr = 0, 
            config_class_name = 'MRIconfig_C2'):
        
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
        
        print(f'Running crossnobis Searchlight part {maskNr}', flush=True)
        # os.makedirs(cfg.outDir, exist_ok=True) # --> moved this down a bit so we only create the directory after we made sure that the input data for this subject it available so we don't create empty subject folders that interfere with the missing subject search function

        # Stop execution if output files already exist
        if os.path.isfile(outFiles['SL_rdms']):
            print(f'    - Output files already exist. Skipping computation for maskNr {maskNr}.')
            return
        
    # 2) defining and loading some more variables

        spm = SpmGlm(cfg.spmDir)
        spm.get_info_from_spm_mat()

        print('    - Renaming rawdata file paths in SPM.mat...', flush=True)
        fix_spm_rawdata_paths(spm, cfg.dataDir + 'sync')   

    # 3) loading the betas and info

        print('    - Loading betas and info...', flush=True)
        _, _, info = spm.get_betas(cfg.get_mask_file())
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

        condition_numbers, _ = pd.factorize(info['condition'])
        condition_numbers += 1
        # Add new key with numeric condition codes
        info['condition_number'] = condition_numbers

        print('\n-----------------------------------------------------------------\n'\
            f'Mask file: {cfg.get_mask_file()}'\
            '\n-----------------------------------------------------------------\n')

        condition_numbers, _ = pd.factorize(info['condition'])
        condition_numbers += 1
        # Add new key with numeric condition codes
        info['condition_number'] = condition_numbers

        mask = nib.load(cfg.get_mask_file())
        mask_data = mask.get_fdata()
        mask_bool = mask_data > 0
 
    # 4) finding Searchlights
        print(f'    - Getting searchlight centers and neighbors of mask nr. {cfg.maskNr} with radius {cfg.SLradius} voxels and threshold {cfg.SLthr}')

        centers, neighbors = get_volume_searchlight(mask_bool, radius = cfg.SLradius, threshold = cfg.SLthr)

        print('    - Computing searchlight RDMs...')


        os.makedirs(cfg.outDir, exist_ok=True) # do it here instead of above so we don't create a new directory in case the script is missing data --> this way we ensure the functionality of my get_subjects functions

    # 5) computing RDMs for every searchlight
        SL_rdms = get_searchlight_RDMs_crossnobis_parallel(
            spm,
            centers, 
            neighbors, 
            mask,
            reg_mask, 
            info, 
            'condition_number', 
            'crossnobis')


        info_full = joblib.load(f'{CODE_DIR}/resources/info.pkl')

        conditions_missing, _ = check_conditions_missing(info, info_full)

        if conditions_missing:
            conditions = list(dict.fromkeys(info_full['condition'])) # use full list of conditions because we replaced missing values when RDMs were computed
        else:
            conditions = list(dict.fromkeys(info['condition']))

        SL_rdms.pattern_descriptors['condition'] = conditions    
        SL_rdms, models = reorder_rdms(SL_rdms, models)

        # 6) compare Searchlight RDMs with model RDM

        print('    - Evaluating RDMs...')
        eval_results = evaluate_models_searchlight(SL_rdms, models, eval_fixed, method = cfg.RSAmethod)
        
        # 7) save general variables

        # joblib.dump(info, outFiles['info'])
        # joblib.dump(centers, outFiles['centers'])
        # joblib.dump(neighbors, outFiles['neighbors'])
        # joblib.dump(SL_rdms, outFiles['SL_rdms'])
        # joblib.dump(eval_results, outFiles['eval_results'])

        joblib.dump(info, f'job_outputs/{subjectID}_info.pkl')
        joblib.dump(centers, f'job_outputs/{subjectID}_centers.pkl')
        joblib.dump(neighbors, f'job_outputs/{subjectID}_neighbors.pkl')
        joblib.dump(SL_rdms, f'job_outputs/{subjectID}_SL_rdms.pkl')
        joblib.dump(eval_results, f'job_outputs/{subjectID}_eval_results.pkl')
                                
        # 8) extract and plot data separately for sensory & suprasensory model

        allModels = ALL_MODELS
        for model in allModels:

            # ------------------------ v 
            cfg.modelType = model
            cfg.configure_paths()
            outFiles = cfg.get_outFile_names()
            # ------------------------ ^ important! for right prefix of files (suprasuprasensory/suprasensory)

            eval_score = [float(e.evaluations[0][cfg.modelIdx]) for e in eval_results]

            # to display eval scores of model comparison in brain-shape
            RDM_brain = get_RDM_brain(mask, SL_rdms, eval_score)

            joblib.dump(eval_score, f'job_outputs/{subjectID}_{model}_eval_score.pkl')
            joblib.dump(RDM_brain, f'job_outputs/{subjectID}_{model}_RDM_brain.pkl')

            # eval_score_histogram(cfg, 
            #                     eval_score)

        

