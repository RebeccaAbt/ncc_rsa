# %% imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from utils.submit_jobs import auto_args, job_setup
from clusterjobs.tmp_GA_evoked import EvokedGA
from plus_slurm import JobCluster


job_kwargs = job_setup(ram='300G',
                       cpus=10,
                       time=4*60,
                       qos='high_prio',
                       name = 'evoked.sh',
					      jobs_dir='evoked'
                       )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(EvokedGA, 
					inDir = MEG_CLEAN_EPOCHS_DIR, 
			    #   suffix = 'clean-epo', 
				suffix = 'maxfilter_ica_1-99Hz__fs_1000__[-1.5_1.5]s_detrend_1_meg_clean-epo',
			      h_freq = 20,
			      baseline=(-0.8, 0))
job_cluster.submit(do_submit=True)