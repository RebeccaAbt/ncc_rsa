import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')
from clusterjobs.do_SL_crossnobis_24_partial import SL_crossnobis_partial

from plus_slurm import JobCluster, PermuteArgument
import numpy as np
from utils.subj import get_all_subjects
#%%

thisConfig = 'SensoryConfig_C2' # go to the "/configs/config.py file" --> use one of the classes there or create a new subclass that contains all the settings for the analysis! 

partialMasks = list(map(int, np.concatenate([np.arange(1, 61)])))  # This will include all masks from 1 to 60])))
# partialMasks = list(map(int, np.concatenate([np.arange(6, 61)])))  # This will include all masks from 1 to 60])))
# all_subjects = ["19910823ssld", "19951227eipo", "19970302urmr", "20020705ttbr", "19991211mrbn"]
all_subjects = get_all_subjects()

job_cluster = JobCluster(
    required_ram='8G',
    request_cpus=2,
    request_time=2*60,
    qos='high_prio',
    python_bin='/home/reabt/experiments/ncc/MRI/code/.pixi/envs/default/bin/python'
)

# job_cluster.add_job(
#     SL_crossnobis_partial,
#     subjectID='19910823ssld',
#     maskNr=1,
#     config_class_name = PermuteArgument(thisConfig)
# )

# job_cluster.add_job(
#     SL_crossnobis_partial,
#     subjectID='19991211mrbn',
#     maskNr=1,
#     config_class_name = thisConfig
# )

job_cluster.add_job(
    SL_crossnobis_partial,
    subjectID=PermuteArgument(all_subjects),
    maskNr=PermuteArgument(partialMasks),
    config_class_name = thisConfig
)

job_cluster.submit(do_submit=True)
