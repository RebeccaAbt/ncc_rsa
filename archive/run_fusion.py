#%%
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

# from plus_slurm import JobCluster, PermuteArgument
from archive.do_fusion import Fusion
from plus_slurm import JobCluster, PermuteArgument
from utils.subj import get_fusion_subjects
#%%
thisConfig = ["SetupConfig_E1", "SetupConfig_E2", "SetupConfig_C1", "SetupConfig_C2"] # go to the "/configs/config.py file" --> use one of the classes there or create a new subclass that contains all the settings for the analysis! 

all_subjects = get_fusion_subjects()

job_cluster = JobCluster(
    required_ram='64G',
    request_cpus=1,
    request_time=20*60,
    # qos='high_prio',
    python_bin='/home/reabt/experiments/ncc/MRI/code/.pixi/envs/default/bin/python'
)

job_cluster.add_job(
    Fusion,
    subjectID=PermuteArgument(all_subjects),
    config_class_name = PermuteArgument(thisConfig)

    # subjectID='19910823ssld',
    # config_class_name = 'SetupConfig_C1'
)

job_cluster.submit(do_submit=True)