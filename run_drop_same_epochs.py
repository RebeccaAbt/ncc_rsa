# %% imports
import os
import sys

# from utils.subj import get_MEG_subjects
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.submit_jobs import auto_args, job_setup
from clusterjobs.do_drop_same_epochs import DropEpochsFromEpo, DropEpochsFromTxt
from plus_slurm import JobCluster

cleanDir = MEG_CLEAN_EPOCHS_DIR

subj_list_dir =  '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean_mean_head_pos/manual_finish'
all_subjects = [
	name for name in os.listdir(subj_list_dir)
	if os.path.isdir(os.path.join(subj_list_dir, name))
]

all_subjects = all_subjects[0:5]
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
					epoSuffix = 'maxfilter_True__ica_True__0.5-40Hz__fs_1000__[-1.5_1.5]s_detrend_None', # 'maxfilter_True__ica_True__0.1-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1_bio-epo.fif''#
					epoSuffix2 = '_meg-epo'

					# oldSuffix = 'clean-epo',
					# newSuffix = 'maxfilter_True__ica_True__0.1-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1',
					# newSuffix2 = '_meg-epo'
					)

job_cluster.submit(do_submit=True)