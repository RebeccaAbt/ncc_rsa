# %% imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from utils.submit_jobs import auto_args, job_setup
from clusterjobs.tmp_do_drop_same_epochs import DropSameEpochs
from plus_slurm import JobCluster

cleanDir = MEG_CLEAN_EPOCHS_DIR

all_subjects = [
	name for name in os.listdir(cleanDir)
	if os.path.isdir(os.path.join(cleanDir, name))
]

job_kwargs = job_setup(ram='32G',
					   cpus=4,
					   time=3*60,
					   qos='high_prio',
					   name = 'drop_epos.sh',
						  jobs_dir='drop_epos'
					   )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(DropSameEpochs,
					subjectID = auto_args(all_subjects),
					cleanDir = cleanDir)

job_cluster.submit(do_submit=True)