#%%
import subprocess
import re
import sys
import traceback

log_file = "submit_jobs.log"

def log(msg):
    with open(log_file, "a") as f:
        print(msg, file=f)
        f.flush()

def submit(script, dependency=None):
    log(f"Script: {script}")
    cmd = ["python", script]
    log(f"cmd1: {cmd}")
    if dependency is not None:
        cmd.append("--dependency")
        cmd.append(f"afterany:{dependency}")
        log(f"cmd2: {cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log(result.stdout)
    if result.stderr:
        log("STDERR:\n" + result.stderr)
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        raise RuntimeError(f"Could not find job ID in output of {script}")
    return m.group(1)

try:
    jid1 = '765052' # partial SL_analysis --> currently running

    jid2 = submit("run_SL_crossnobis_compile.py", dependency=jid1)
    log("Second job submitted with ID: " + str(jid2))

    jid3 = submit("run_fusion_parallel.py", dependency=jid2)
except Exception as e:
    with open(log_file, "a") as f:
        f.write("Exception occurred:\n")
        traceback.print_exc(file=f)
