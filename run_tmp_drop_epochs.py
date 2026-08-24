# %% imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.submit_jobs import auto_args, job_setup
from clusterjobs.tmp_do_drop_same_epochs import DropEpochsFromEpo, DropEpochsFromTxt
from plus_slurm import JobCluster

cleanDir = MEG_CLEAN_EPOCHS_DIR

all_subjects = [
	name for name in os.listdir(cleanDir)
	if os.path.isdir(os.path.join(cleanDir, name))
]

job_kwargs = job_setup(ram='32G',
					   cpus=4,
					   time=1*60,
					   name = 'drop_epos.sh',
						  jobs_dir='drop_epos'
					   )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(DropEpochsFromTxt,
					subjectID = auto_args(all_subjects),
					cleanDir = cleanDir,
					epoSuffix = 'maxfilter_True__ica_True__0.5-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1_meg-epo')

job_cluster.submit(do_submit=True)