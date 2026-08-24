#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from clusterjobs.do_movieplot2 import Movie

from plus_slurm import JobCluster
from utils.submit_jobs import job_setup, auto_args
#%%
       
thisConfig = ["FusionConfig_C2", "FusionConfig_C5"]
meanMEG = [True,False]
thres = [0.1, 0.05] 
# thres = 0.05

n_subj = 7

job_kwargs = job_setup(ram  = '64G',       
                       cpus = 10,        
                       time = 1*60,       
                       qos  = 'high_prio',
                       name = 'movie.sh'
                       )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(
    Movie,
    config_class_name = auto_args(thisConfig),
    meanMEG = auto_args(meanMEG),
    thres = auto_args(thres),
    n_subj = auto_args(n_subj)
)

job_cluster.submit(do_submit=True)
