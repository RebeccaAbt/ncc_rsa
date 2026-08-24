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
from utils.submit_jobs import auto_args, job_setup
# from clusterjobs.do_save_epochs import SaveEpochs_andiStyle
from clusterjobs.do_save_epochs import SaveEpochs
from plus_slurm import JobCluster

#%%

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
run_part = 1 # only first or second part of ICA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
overwrite = True

use_mean_headpos = False

ica_settings = {'ica_method': "picard",
		'fit_params': None,
		'ica_threshold': 0.3,
		'n_components': 50, 
		'train_thresh': 2,
		'train_freq': 16.7
		}

epochs_settings={
		 'tmin': -1.5,
		 'tmax': 1.5,
		 'baseline': None,
		 'preload': True,
		 'l_freq': 0.5,
		 'h_freq': None,
		 'detrend': 1} # 0=constant, 1=linear, None=no detrending

all_subjects = get_MEG_subjects()

block_subjects = [
	'19910703eigl',
	'19910823ssld', 
	'19960630cahi', 
	'19970302urmr',
	'19971028mrhs',
	'19990810mrkh',
	'20000107ptfu',
	'20031022ekse',
	'20070324hlti',
	'19960531hibu',
	'20000118sbnb',
	'20040819knee',
	'20050204vrao',
	'20040627vrrj',
	'20050610atbu',
	'20040630gbaf'
]

normal_subjects = [subj for subj in all_subjects if subj not in block_subjects]

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
					blockwise_ica = True, # <----------------------------------!!! experimental
					run_part = run_part, 
					overwrite = overwrite,
					ica_out_root = f"{ICA_BLOCK_DIR}", # !!! careful which folder!!!
					ica_settings = ica_settings,
					epochs_settings = epochs_settings,
					use_mean_headpos = use_mean_headpos
)

job_cluster.submit(do_submit=True)



#%%
# run_part = 2 # only second part of ICA
# overwrite = False

# use_mean_headpos = False

# ica_settings = {'ica_method': "picard",
# 		'fit_params': None,
# 		'ica_threshold': 0.35,
# 		'n_components': None
# 		}

# epochs_settings={
# 		 'tmin': -1.5,
# 		 'tmax': 1.5,
# 		 'baseline': None,
# 		 'preload': True,
# 		 'l_freq': 0.5,
# 		 'h_freq': None,
# 		 'detrend': 1} # 0=constant, 1=linear, None=no detrending

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

# #%%

# job_kwargs = job_setup(ram='64',
#                        cpus=8,
#                        time=3*60,
#                     #    qos='high_prio',
#                        name = 'epochs_new.sh',
# 				   jobs_dir='epochs_new'
#                        )

# job_cluster = JobCluster(**job_kwargs)

# # ------------------------------------ NORMAL
# job_cluster.add_job(SaveEpochs,
#                     subjectID = auto_args(normal_subjects),
# 					blockwise_ica = False, # <----------------------------------!!! experimental
# 					run_part = run_part, 
# 					overwrite = overwrite,
# 					ica_out_root = ICA_DIR, # !!! careful which folder!!!
# 					ica_settings = ica_settings,
# 					epochs_settings = epochs_settings,
# 					use_mean_headpos = use_mean_headpos
# )

# # # ------------------------------------ BLOCK-WISE
# job_cluster.add_job(SaveEpochs,
#                     subjectID = auto_args(block_subjects),
# 					blockwise_ica = True, # <----------------------------------!!! experimental
# 					run_part = run_part, 
# 					overwrite = overwrite,
# 					ica_out_root = f"{ICA_BLOCK_DIR}", # !!! careful which folder!!!
# 					ica_settings = ica_settings,
# 					epochs_settings = epochs_settings,
# 					use_mean_headpos = use_mean_headpos
# )


# job_cluster.submit(do_submit=True)
