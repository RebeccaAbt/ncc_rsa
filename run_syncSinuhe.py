#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from rclone.do_sync_sinuhe import CloneSinuhe
from plus_slurm import JobCluster
from utils.submit_jobs import auto_args, job_setup
from utils.subj import *

#%%

job_kwargs = job_setup(ram='128G',
                       cpus=10,
                       time=1*60,
                    #    qos='high_prio',
                       name = f'sync_data.sh',
					   jobs_dir = 'sync_data'
                       )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(
    CloneSinuhe
)

job_cluster.submit(do_submit=True)