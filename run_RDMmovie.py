#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from clusterjobs.do_RDMmovie import RDMmovie
from plus_slurm import JobCluster
from utils.submit_jobs import auto_args, job_setup
from utils.subj import *

#%%

meg_config = ["MEGconfig_E", "MEGconfig_C"]

for thisConfig in meg_config:

    all_subjects = get_MEG_subjects(thisConfig)
    job_kwargs = job_setup(ram='64G',
                           cpus=10,
                           time=1*60,
                        #    qos='high_prio',
                           name = f'movie_{thisConfig.split('_')[1]}.sh',
    					   jobs_dir = 'movie'
                           )

    job_cluster = JobCluster(**job_kwargs)

    job_cluster.add_job(
        RDMmovie,
        subjectID=auto_args(all_subjects),
        config_class_name = auto_args(thisConfig)
    )

    job_cluster.submit(do_submit=True)