'''
First step of the whole pipeline - Andy-Style:
roughly fuilter data [0.5 - 99Hz], do ICA, save epochs.
'''

#%% 
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.subj import *
from utils.subj import _remove_bad_subj
from utils.submit_jobs import auto_args, job_setup
# from clusterjobs.do_save_epochs import SaveEpochs_andiStyle
from clusterjobs.do_save_epochs import SaveEpochs
from plus_slurm import JobCluster

#%%


all_subjects = _remove_bad_subj(os.listdir('/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean_mean_head_pos/manual_finish'))


#%%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
run_part = 2 # only first or second part of ICA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
overwrite = False # needs to be Truew if we do the blockwise stuff, becaue there we cannot properly load the data with the existing pipeline

use_mean_headpos = False
preproc_settings = {'maxfilter': True,
					'downsample_f': None,
					'notch': True,
					'l_pass': 99,
					'h_pass': 0.5},

ica_settings = {'ica_method': "picard",
		'fit_params': None,
		'ica_threshold': 0.3,
		'n_components': 50, 
		'eog': True,
		'ecg': True,		
		'train_thresh': 2,
		'train_freq': 16.7
		}

epochs_settings={
		 'tmin': -4,
		 'tmax': 4,
		 'baseline': None,
		 'preload': True,
		 'h_freq': 20,
		 'detrend': None,
		 'fs': 100} # 0=constant, 1=linear, None=no detrending

# all_subjects = get_MEG_subjects()

# block_subjects = [
# 	'19910703eigl',
# 	'19910823ssld', 
# 	'19960630cahi', 
# 	'19970302urmr',
# 	'19971028mrhs',
# 	'19990810mrkh',
# 	'20000107ptfu',
# 	'20031022ekse',
# 	'20070324hlti',
# 	'19960531hibu',
# 	'20000118sbnb',
# 	'20040819knee',
# 	'20050204vrao',
# 	'20040627vrrj',
# 	'20050610atbu',
# 	'20040630gbaf'
# ]

# normal_subjects = [subj for subj in all_subjects if subj not in block_subjects]
# all_subjects = get_MEG_subjects()
#%%

job_kwargs = job_setup(ram='128',
                       cpus=16,
                       time=2*60,
                    #    qos='high_prio',
                       name = 'epochs_new.sh',
				   jobs_dir='epochs_new'
                       )

job_cluster = JobCluster(**job_kwargs)

# ------------------------------------ NORMAL
# job_cluster.add_job(SaveEpochs,
#                     subjectID = auto_args('19910703eigl'),
# 					blockwise_ica = False, # <----------------------------------!!! experimental
# 					run_part = run_part, 
# 					overwrite = overwrite,
# 					ica_out_root = ICA_DIR, # !!! careful which folder!!!
# 					ica_settings = ica_settings,
# 					epochs_settings = epochs_settings,
# 					use_mean_headpos = use_mean_headpos
# )

# # # ------------------------------------ BLOCK-WISE
job_cluster.add_job(SaveEpochs,
                    subjectID = auto_args(all_subjects),
					blockwise_ica = True,
					run_part = run_part, 
					overwrite = overwrite,
					ica_out_root = f"{ICA_BLOCK_DIR}", # !!! careful which folder!!!
					ica_settings = ica_settings,
					epochs_settings = epochs_settings,
					use_mean_headpos = use_mean_headpos
)

job_cluster.submit(do_submit=True)
