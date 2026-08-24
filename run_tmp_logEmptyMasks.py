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

from clusterjobs.tmp_find_empty_masks import FindEmptyMasks
from plus_slurm import JobCluster
from utils.subj import *
from utils.submit_jobs import auto_args, job_setup
#%%



thisConfig = ['MRIconfig_C2', 'MRIconfig_C5']

all_subjects = get_MRI_subjects(thisConfig)
# all_subjects = ['19840930bigs']

#%%
'''
Resource info: 
Single subject, mask 42 (340 serachlights --> automatically no parallel processing):
    - average of 2.25 CPUs
    - maximum of 1.62GB of RAM

Because of that, memory usage info for parallel jobs (e.g. avg. 0.87 CPUs/ 1.41GB of RAM) doesn't really make sense in comparison 
    --> should probably better be ignored
'''
job_kwargs = job_setup(ram='64G',
                       cpus=10,
                       time=60,
                       qos='high_prio',
                       name = f'logs_{thisConfig[0].split('_')[1]}.sh',
					   jobs_dir = f'empty_masks'
                       )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(
    FindEmptyMasks,
    subjectID=auto_args(all_subjects),
    config_class_name = auto_args(thisConfig)
 )

job_cluster.submit(do_submit=True)

# %%
