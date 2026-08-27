'''
For new subjects, make sure you first run
    1) code/rclone/copy_MRI_to_SCC.py   to cget the data from the mri server
    2) code/tools/make_partialMasks.py  to make the partial brain masks we need to shorten computation time for crossnobis

'''

#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import numpy as np
from clusterjobs.do_SL_crossnobis_partial import SL_crossnobis_partial
from plus_slurm import JobCluster
from utils.subj import *
from utils.submit_jobs import auto_args, job_setup
#%%

partialMasks = list(map(int, np.concatenate([np.arange(1, 61)])))  # This will include all masks from 1 to 60])))

for thisConfig in ['MRIconfig_C5full_nan']:#, ['MRIconfig_C5full', 'MRIconfig_C5_nan', 'MRIconfig_C5full_nan']:
    all_subjects = get_MRI_subjects(thisConfig)
    # all_subjects = '19880331igse'

    '''
    Resource info: 
    Single subject, mask 42 (340 serachlights --> automatically no parallel processing):
        - average of 2.25 CPUs
        - maximum of 1.62GB of RAM

    Because of that, memory usage info for parallel jobs (e.g. avg. 0.87 CPUs/ 1.41GB of RAM) doesn't really make sense in comparison 
        --> should probably better be ignored
    '''
    job_kwargs = job_setup(ram='64G',
                           cpus=8,
                           time=3*60, # 1h is enough for SL_rasius=2, but for SL-_radius=5, we need more time! --> 2.5 - 3 h
                        #    qos='high_prio',
                           name = f'MRI_{thisConfig.partition('_')[2]}.sh',
    					   jobs_dir = f'rsa_mri'
                           )

    job_cluster = JobCluster(**job_kwargs)

    job_cluster.add_job(
        SL_crossnobis_partial,
        subjectID=auto_args(all_subjects),
        maskNr=auto_args(partialMasks),
        config_class_name = auto_args(thisConfig)
     )

    job_cluster.submit(do_submit=True)

#%%
