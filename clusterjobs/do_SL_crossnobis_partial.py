#%%

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
from datetime import datetime
import fcntl
import traceback
from pathlib import Path


from rsatoolbox.io.spm import SpmGlm
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import (
    get_volume_searchlight,
    evaluate_models_searchlight
)

from plus_slurm import Job
from utils.provenance import configure_subject_logging, record_artifact


#%%
# def init_empty_mask_log(subject_folder):
#     log_file = Path(subject_folder) / "empty_masks.tsv"
#     log_file.parent.mkdir(parents=True, exist_ok=True)

#     with open(log_file, "a+") as f:
#         fcntl.flock(f, fcntl.LOCK_EX)

#         f.seek(0)
#         if f.read().strip() == "":
#             f.write("timestamp\tsubjectID\tmaskNr\tconfig_class_name\treason\n")
#             f.flush()

#         fcntl.flock(f, fcntl.LOCK_UN)

#     return log_file


# def log_empty_mask(log_file, subjectID, maskNr, config_class_name, reason):
#     log_file = Path(log_file)

#     line_id = f"{subjectID}\t{maskNr}\t{config_class_name}"

#     with open(log_file, "a+") as f:
#         fcntl.flock(f, fcntl.LOCK_EX)

#         f.seek(0)
#         existing_text = f.read()

#         # Avoid duplicate entries if the same job is restarted
#         if line_id not in existing_text:
#             timestamp = datetime.now().isoformat(timespec="seconds")
#             reason = str(reason).replace("\n", " ").replace("\t", " ")
#             f.write(f"{timestamp}\t{subjectID}\t{maskNr}\t{config_class_name}\t{reason}\n")
#             f.flush()

#         fcntl.flock(f, fcntl.LOCK_UN)


class SL_crossnobis_partial(Job):
    def run(self,
            subjectID = '19910823ssld',
            maskNr = 1, 
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
        models = cfg.get_model_RDM()

        # log_file = init_empty_mask_log(cfg.outDir)

        outFiles = cfg.get_outFile_names()
        logger, _ = configure_subject_logging(outFiles['SL_rdms_partial'], subjectID)
        logger.info('Starting fMRI crossnobis partial searchlight')
        logger.info('Configuration class: %s', config_class_name)
        logger.info('Configuration values: %s', vars(cfg))
        
        print(f'Running crossnobis Searchlight part {maskNr}', flush=True)
        
        #  # Stop execution if output files already exist
        # if os.path.isfile(outFiles['SL_rdms_partial']):
        #     print(f'    - Output files already exist. Skipping computation for maskNr {maskNr}.')
        #     return
        
    # 2) defining and loading some more variables

        spm = SpmGlm(cfg.spmDir)
        spm.get_info_from_spm_mat()

        print('    - Renaming rawdata file paths in SPM.mat...', flush=True)
        
        # fix_spm_rawdata_paths(spm, cfg.dataDir + 'sync')   # <-- path locally, without mounting
        fix_spm_rawdata_paths(spm, MRI_RAW_DIR)    # <-- path if data is mounted

    # 3) loading the betas and info

        print('    - Loading betas and info...', flush=True)
        _, _, info = spm.get_betas(cfg.get_mask_file())
        info = pd.DataFrame(info)
        reg_mask = info['reg_name'].str.contains('_hit') | info['reg_name'].str.contains('_miss')
        info = info[reg_mask].reset_index(drop=True)
        print(f'regressors: {info['reg_name'].tolist()}', flush=True)

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

        print(f' --- mask file: {cfg.get_mask_file()} --- ', flush=True)
        mask = nib.load(cfg.get_mask_file())
        mask_data = mask.get_fdata()
        mask_bool = mask_data > 0
 
    # 4) finding Searchlights
        print(f'    - Getting searchlight centers and neighbors of mask nr. {cfg.maskNr} with radius {cfg.SLradius} voxels and threshold {cfg.SLthr}', flush=True)
        
        
        try:
            centers, neighbors = get_volume_searchlight(mask_bool, radius = cfg.SLradius, threshold = cfg.SLthr)
        except ValueError as e:
            if "multi_index must be a sequence of length 3" in str(e):
                # log_empty_mask(
                #     log_file=log_file,
                #     subjectID=subjectID,
                #     maskNr=maskNr,
                #     config_class_name=config_class_name,
                #     reason="no valid searchlights found"
                # )
                print(f"Skipping empty mask: subject={subjectID}, maskNr={maskNr}")
                return
            else:
                raise

    # ~~~~~~~~~~~~~~~~~~ this is new: trying to allow searchlights that are smaller at the edge of the brain
        if cfg.SLthr < 1:
            mask_bool_1D_idx = np.where(mask_bool.flatten())[0] 
            for i, n in enumerate(neighbors):
                n = n[np.isin(n, mask_bool_1D_idx)]
                neighbors[i] = n
            edge_searchlights_adjusted = True
        else: edge_searchlights_adjusted = False
    # ~~~~~~~~~~~~~~~~~~

        print('    - Computing searchlight RDMs...', flush=True)

        os.makedirs(cfg.outDir, exist_ok=True) # do it here instead of above so we don't create a new directory in case the script is missing data --> this way we ensure the functionality of my get_subjects functions

        # print(f"Shape of Searchlights: \n {[n.shape for n in neighbors]}")
    # 5) computing RDMs for every searchlight
        SL_rdms = get_searchlight_RDMs_crossnobis_parallel(
            spm,
            centers, 
            neighbors, 
            mask,
            reg_mask, 
            info, 
            events='condition_number',
            replace_missing = cfg.replace_missing,
            method=cfg.RDMmethod)
        
        print(f'Info: \n{info}', flush=True)

        info_full = joblib.load(f'{CODE_DIR}/resources/info.pkl')

        conditions_missing, _ = check_conditions_missing(info, info_full)

        if conditions_missing:
            conditions = list(dict.fromkeys(info_full['condition'])) # use full list of conditions because we replaced missing values when RDMs were computed
        else:
            conditions = list(dict.fromkeys(info['condition']))

        SL_rdms.pattern_descriptors['condition'] = conditions    
        SL_rdms, models = reorder_rdms(SL_rdms, models)

        joblib.dump(SL_rdms, outFiles['SL_rdms_partial'])
        joblib.dump(info, outFiles['info'])
        record_artifact(
            output_path=outFiles['SL_rdms_partial'],
            operation_name='SL_crossnobis_partial.run',
            parameters={
                'config_class_name': config_class_name,
                'config': vars(cfg),
                'subjectID': subjectID,
                'maskNr': maskNr,
                'n_searchlights': int(len(centers)),
                'n_conditions': int(len(conditions)),
                'edge_searchlights_adjusted': edge_searchlights_adjusted,
                'edge_searchlights_description': (
                    'Brain-edge searchlights were reduced to contain only in-brain voxels.'
                    if edge_searchlights_adjusted else None
                ),
            },
            input_paths=[cfg.get_mask_file(), os.path.join(CODE_DIR, 'resources', 'info.pkl')],
        )
        logger.info('Partial mask %s searchlights: %s', maskNr, len(centers))
