'''
compile the results from the crossnobis RSA done on the partial brain masks so we have results 
for the full brain
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from plus_slurm import JobCluster
from clusterjobs.do_SL_crossnobis_compileResults import SL_crossnobis_compileResults

from utils.subj import *
from utils.submit_jobs import auto_args, job_setup

#%%

# thisConfig = ['MRIconfig_C2'] #, 'MRIconfig_C5'] # go to the "/configs/config.py file" --> use one of the classes there or create a new subclass that contains all the settings for the analysis! 
thisConfig = 'MRIconfig_C2'

all_subjects = get_MRI_subjects()


job_kwargs = job_setup(ram='16G',
                       cpus=1,
                       time=1*60,
                    #    qos='high_prio',
                       name = f"compile_{thisConfig.split('_')[1]}.sh",
					   jobs_dir = 'compile'
					                  )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(
    SL_crossnobis_compileResults,
    subjectID=auto_args(all_subjects),
    config_class_name = auto_args(thisConfig)
)

job_cluster.submit(do_submit=True)
