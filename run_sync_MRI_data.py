'''
First step of the whole pipeline - Andy-Style:
roughly fuilter data [0.5 - 99Hz], do ICA, save epochs.
'''

#%% 

from utils.submit_jobs import auto_args, job_setup
# from clusterjobs.do_save_epochs import SaveEpochs_andiStyle
from clusterjobs.sync_MRI_data	import DataSync
from plus_slurm import JobCluster

#%%


job_kwargs = job_setup(ram='64',
                       cpus=8,
                       time=20*60,
                       name = 'sync_data.sh',
				        jobs_dir='sync_data'
                       )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(DataSync)


job_cluster.submit(do_submit=True)
