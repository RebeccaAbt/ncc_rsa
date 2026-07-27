import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants


from plus_slurm import JobCluster, PermuteArgument
from clusterjobs.do_SL_euclidean import SL_euclidean
from utils.subj import *
from utils.submit_jobs import auto_args, job_setup

for thisConfig in ["MRIconfig_E2", "MRIconfig_E5"]: # go to the "/configs/config.py file" --> use one of the classes there or create a new subclass that contains all the settings for the analysis! 

    all_subjects = get_MRI_subjects(thisConfig)


    job_kwargs = job_setup(ram='64G',
                           cpus=10,
                           time=30,
                        #    qos='high_prio',
                           name = f'SL_{thisConfig.split('_')[1]}.sh',
                           jobs_dir = 'rsa_E'
                           )

    job_cluster = JobCluster(**job_kwargs)

    job_cluster.add_job(
        SL_euclidean,
        subjectID=auto_args(all_subjects),
        config_class_name = auto_args(thisConfig)
    )

    job_cluster.submit(do_submit=True)