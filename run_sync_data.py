'''
First step of the whole pipeline - Andy-Style:
roughly fuilter data [0.5 - 99Hz], do ICA, save epochs.
'''

#%% 

from utils.submit_jobs import auto_args, job_setup
# from clusterjobs.do_save_epochs import SaveEpochs_andiStyle
from rclone.sync_data	import SyncMEG, SyncMRI, SyncEmptyRoom
from plus_slurm import JobCluster

#%%


job_kwargs = job_setup(ram='128',
                       cpus=10,
                       time=10*60,
                       name = 'sync_data.sh',
				        jobs_dir='sync_data'
                       )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(SyncEmptyRoom)


job_cluster.submit(do_submit=True)
