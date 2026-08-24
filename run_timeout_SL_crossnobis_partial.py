import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants


from clusterjobs.do_SL_crossnobis_partial import SL_crossnobis_partial

from plus_slurm import JobCluster, PermuteArgument
from utils.submit_jobs import job_setup
#%%

job_kwargs = job_setup(ram='8G',
                       cpus=16,
                       time=10*60,
                       # qos='high_prio'
                       )

job_cluster = JobCluster(**job_kwargs)

# now add all the timeout configs:

thisConfig = 'MRIconfig_C5'
subjectID = '19921205crfi'
partialMasks = [27, 29, 33, 34]

job_cluster.add_job(
	SL_crossnobis_partial,
	subjectID=subjectID,
	maskNr=PermuteArgument(partialMasks),
	config_class_name = thisConfig
)

thisConfig = 'MRIconfig_C5'
subjectID = '19961123crsh'
partialMasks = [27, 28, 29, 33, 34]

job_cluster.add_job(
	SL_crossnobis_partial,
	subjectID=subjectID,
	maskNr=PermuteArgument(partialMasks),
	config_class_name = thisConfig
)

job_cluster.submit(do_submit=True)
