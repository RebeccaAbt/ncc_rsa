import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import re
import argparse
from plus_slurm import PermuteArgument

'''
To submit a job with a dependency, call your script from the command line after activating the environment and add the job ID of the job you want to depend on as an argument, e.g.:

$> pixi shell
$> python run_RDMmovie.py --dependency 1396365

'''


def dependency_input():
    parser = argparse.ArgumentParser(description="Run Job with custom SLURM dependency.")
    parser.add_argument('--dependency', type=str, default=None,
                        help="SLURM dependency argument (e.g., 'afterany:764988')")
    args = parser.parse_args()

    print(f"dependency: {args.dependency}")
    
    return args.dependency


def job_setup(  ram='4G',
                cpus=1,
                time=10,
                # qos='high_prio',
                qos = None,
                python_bin = PYTHON_BIN,
                # exclude_nodes='node09.scc-pilot.plus.ac.at,node10.scc-pilot.plus.ac.at',
                exclude_nodes=None,
                name='submit.sh',
                jobs_dir = 'jobs'):
    
    ram = ram if ram.endswith("G") else (re.sub(r"\D", "", ram) or "0") + "G"

    jobs_dir = jobs_dir if jobs_dir.startswith("jobs/") else "jobs/" + jobs_dir


    job_cluster_kwargs = dict(required_ram=ram,
                              request_cpus=cpus,
                              request_time=time,
                              qos=qos,
                              python_bin=python_bin,
                              exclude_nodes=exclude_nodes,
                              submit_file_name = name,
                              jobs_dir = jobs_dir
                              )
    dep = dependency_input()

    if dep != None:
        job_cluster_kwargs['extra_slurm_args'] = [f"--dependency={dep}"]
    return job_cluster_kwargs


def auto_args(arg):
    '''
    automatically wrap multiple arguments into PermuteArgument
    '''
    if isinstance(arg, list):
        if len(arg) > 1:
            return PermuteArgument(arg)
        else:
            return arg[0]
    
    else:
        return arg