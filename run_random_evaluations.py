import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants


from plus_slurm import JobCluster, PermuteArgument
from clusterjobs.do_random_evaluations import RandomEval_fMRI
from utils.subj import *
from utils.submit_jobs import auto_args, job_setup

show_progress = False
n_random = 1000
method = 'label'

# cpu_spec = [   # [eval, subj]
    # [4, 24],
    # [8, 12], 
    # [2, 24],  # 2*24 CPU * 8 GB for n_rand = 10 --> 35 min (also 100 --> ~6 h?)
    # [4, 12],
    # [8, 6],
    # [16, 3],
# ]

spec = [5, 24] # n CPUs [eval, subj]

# thisConfig = "MRIconfig_E5"
for thisConfig in ["MRIconfig_C2full_nan"]:
    all_subjects = get_MRI_subjects(thisConfig)

# for spec in cpu_spec:

    n_jobs_eval = spec[0]
    n_jobs_subj = spec[1]
    n_cpus      = n_jobs_eval*n_jobs_subj
    ram_per_cpu = 4
    ram_total   = n_cpus*ram_per_cpu

    print(f"CPUs: {n_jobs_eval} x {n_jobs_subj} --> {n_cpus} \n RAM: {ram_per_cpu} per CPU --> {ram_total}")

    job_kwargs = job_setup(ram=f'{ram_total}',
                           cpus=n_cpus,
                           time=15*60,
                        #    qos='high_prio',
                        #    name = f'bench200_{n_jobs_eval}x{n_jobs_subj}x{ram_per_cpu}GB.sh',
                        name = f'rand{n_random}_{thisConfig.partition('_')[2]}',
                           jobs_dir = 'randomModels'
                           )

    job_cluster = JobCluster(**job_kwargs)

    job_cluster.add_job(
        RandomEval_fMRI,
        all_subjects,
        config_class_name = thisConfig,  # not relevant, if e.g. "E1" or "E2" because searchlight radius not relevant here",
        allModels = ALL_MODELS,
        n_random = n_random,
        method = method,
        n_jobs_eval = n_jobs_eval,
        n_jobs_subj = n_jobs_subj,
        show_progress = show_progress
    )

    job_cluster.submit(do_submit=True)


