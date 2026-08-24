
import os
import sys

CODE_DIR	= os.path.dirname(os.path.dirname(__file__))
FABI_DIR	= '/home/scc_e_393956/ncc/Fabi'

#mounts
# SINUHE_DATA = '/home/scc_e_393956/mnt/data/sinuhe/data_raw/'
# SINUHE_STUDY = 'aw_ncc'
SINUHE_DATA = '/home/scc_e_393956/Desktop/reabt/ncc/MEG'
SINUHE_STUDY = 'raw'

#python
PYTHON_BIN	= '/home/scc_e_393956/ncc/rsa/.pixi/envs/default/bin/python'
JOBS_DIR = f'{CODE_DIR}/jobs'

#directories
MEG_DATA_DIR			= '/home/scc_e_393956/Desktop/reabt/ncc/MEG'
MEG_EPOCHS_DIR 			= '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2'
MEG_CLEAN_EPOCHS_DIR 	= '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/manual_finish'
MEG_ICA_DIR 			= '/home/scc_e_393956/Desktop/reabt/ncc/MEG/ica2'

EMPTY_ROOM_DATA_DIR		= '/home/scc_e_393956/Desktop/reabt/ncc/empty_room'
ICA_DIR 				= '/home/scc_e_393956/Desktop/reabt/ncc/MEG/ica_qc2'
ICA_BLOCK_DIR 			= '/home/scc_e_393956/Desktop/reabt/ncc/MEG/ica_qc_blockwise2'

MRI_DATA_DIR			= '/home/scc_e_393956/Desktop/reabt/ncc/MRI'
MRI_RAW_FOLDER 			= 'sync'
MRI_RAW_DIR 			= f'{MRI_DATA_DIR}/{MRI_RAW_FOLDER}'
MRI_MASKS_DIR 			= f'{MRI_DATA_DIR}/masks'
MRI_1ST_LEVEL_FOLDER 	= 'NCC/firstLevel_sensory_M1C'
MRI_PREPROC_FOLDER		= 'NCC/prepro_V1B'
MRI_RSA_DIR 			= '/home/scc_e_393956/Desktop/reabt/ncc/MRI/rsa2'

RESOURCE_DIR = f'{CODE_DIR}/resources/'
MODELS_DIR 		= RESOURCE_DIR
FUSION_DATA_DIR	= '/home/scc_e_393956/Desktop/reabt/ncc/fusion'

#subjects
EXAMPLE_SUBJ_1 	= '19910823ssld'
EXAMPLE_SUBJ_2 	= '19840930bigs'

#models
ALL_MODELS 		= ['sensory', 'suprasensory']

#functions
def get_spm_dir(subj):
	return f'{MRI_RAW_DIR}/{subj}/{MRI_1ST_LEVEL_FOLDER}'

def get_mask_file(subj):
	return f'{MRI_RAW_DIR}/{subj}/{MRI_1ST_LEVEL_FOLDER}/mask.nii'