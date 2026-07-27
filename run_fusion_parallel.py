#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants


from clusterjobs.do_fusion_np_parallel import Fusion_np

from plus_slurm import JobCluster
from utils.subj import *
from utils.submit_jobs import job_setup, auto_args
#%%
'''
Fusion_pd: original function I wrote: uses pandas df --> slow
Fusion_np: newer version (with chatGPT code): uses np --> should be way faster
Fusion_pqdm: uses np, but lopes over timepoints, not voxel
'''
# go to the "/configs/config_fusion.py file" --> use one of the classes there or create a new subclass that contains all the settings for the analysis!

all_configs =[
		"FusionConfig_E2",
		"FusionConfig_E5",
		"FusionConfig_C2",
		"FusionConfig_C5",
]

thisConfig = all_configs[3]
meanMEG = False

all_subjects    = get_new_fusion_subjects(thisConfig,meanMEG) # only works with 1 value for meanMEG

# all_subjects = all_subjects[0]
#%%
'''
Info about parallelisation of old script ('do_fusion_parallel_pqdm'):
 # 20 parallel jobs / 8 cores used ~16GB. So 32 GB for 20 jobs should be enough
# 10 jobs for 4 cores seems to work well (uses 3.7 cores). If more jobs are run in parallel --> use more cores,
# 20 jobs on 8 cores took ~2:30 hours

'''

job_kwargs = job_setup(ram  = '64',       # best: use 64G and 48 CPU and 1-2 h
					   cpus = 48,        
					   time = 2*60,       
					#    qos  = 'high_prio',
					   name = f'fusion_{thisConfig.split('_')[1]}.sh',
					   jobs_dir = 'fusion'
					   )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(
	Fusion_np,
	subjectID=auto_args(all_subjects),
	config_class_name = auto_args(thisConfig),
	meanMEG = auto_args(meanMEG)
 )

job_cluster.submit(do_submit=True)
