





'''
=================================== S T O P ! ! ! ==================================

Ich habe zwischenzeitlich eine andere Art gewählt, den cluster-definitiomn threshold zu berechnen!
auch das Suffix für den OutFile name sind anders!!!
--> muss wieder geändert werden!!!

=====================================================================================
'''

#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from clusterjobs.do_fusion_stats import Fusion_CP

from plus_slurm import JobCluster
from utils.submit_jobs import job_setup, auto_args
#%%
# all_configs =  ["FusionConfig_E2", "FusionConfig_E5", "FusionConfig_C2", "FusionConfig_C5"]
all_configs =  ["FusionConfig_C5"]

for thisConfig in all_configs:

#    thres = [0.01, 0.05, 0.1]
	thres = [0.05] # needs to be a  list
   # meanMEG = [True,False]
	meanMEG = False
   # meanMEG = True


	print(f'current config: {thisConfig}')

	job_kwargs = job_setup(ram  = '120G',
						  cpus = 15,
						  time = 2*60,
						#   qos  = 'high_prio',
						  name = f'cp_{thisConfig.split('_')[1]}.sh',
						  jobs_dir = 'fusion_stat'
						  )

	job_cluster = JobCluster(**job_kwargs)

	job_cluster.add_job(
	   Fusion_CP,
	   config_class_name = auto_args(thisConfig),
	   thresholds = thres, # don't use PermuteArgument oder auto_args here, because we iterate over list within the Job!
	   meanMEG = auto_args(meanMEG)
	)
	job_cluster.submit(do_submit=True)
