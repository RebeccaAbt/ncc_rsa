import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import sys

from clusterjobs.do_SL_crossnobis_fullBrain import SL_crossnobis_full
from plus_slurm import JobCluster, PermuteArgument
from utils.subj import *
from utils.submit_jobs import auto_args, job_setup
#%%


partialMasks = 0
thisConfig = 'MRIconfig_C5'
all_subjects = '19970302urmr'

job_kwargs = job_setup(ram='4GB',
                       cpus=1,
                       time=4*60,
                       qos='high_prio',
                       name = 'fullBrain.sh'
                       )

job_cluster = JobCluster(**job_kwargs)

#%%
job_cluster.add_job(
    SL_crossnobis_full,
    subjectID=auto_args(all_subjects),
    maskNr=auto_args(1),
    config_class_name = auto_args(thisConfig)
 )

job_cluster.submit(do_submit=True)