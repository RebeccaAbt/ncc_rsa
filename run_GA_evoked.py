# %% imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.submit_jobs import auto_args, job_setup
from clusterjobs.tmp_GA_evoked import EvokedGA
from plus_slurm import JobCluster


job_kwargs = job_setup(ram='256',
                       cpus=10,
                       time=3*60,
                       name = 'evoked.sh',
					      jobs_dir='evoked'
                       )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(EvokedGA, 
					inDir = MEG_CLEAN_EPOCHS_DIR, 
			    #   suffix = 'clean-epo', 
				suffix = 'maxfilter_True__ica_True__0.5-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1_meg-epo_clean', #'maxfilter_ica_0.1-99Hz__fs_1000__[-1.5_1.5]s_detrend_1_meg_clean-epo', run_Ga
			      h_freq = 20,
			      baseline=(-0.2, 0))
job_cluster.submit(do_submit=True)