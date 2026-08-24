#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import re
import subprocess
from collections import defaultdict
#%%
jobsDir = os.path.join(JOBS_DIR, 'rsa_mri')
jobID = '784818'


def find_job_folder(jobID, jobsDir):
    for folder in os.listdir(jobsDir):
        logFile =os.path.join(jobsDir, folder, 'log', 'out_1.log')
        if os.path.isfile(logFile):
            with open(logFile) as f :
                lines = f.readlines()
            for line in lines:
                # match = re.search(r"Job ID:\s*(\d{6})", line)
                match = re.search(r"Job ID:\s*(\d+)(?=,)", line)
                if match:
                    if match.group(1) == jobID:
                        print(f"Job folder: {folder}\n")
                        return folder
                    else:
                        break
                        
jobFolder = find_job_folder(jobID, jobsDir)

jobPath = os.path.join(jobsDir, jobFolder, 'log')

outFile_failed = os.path.join(f'{CODE_DIR}/job_outputs', f'job_{jobID}.txt')
outFile_timeout = os.path.join(f'{CODE_DIR}/job_outputs', f'job_TO_{jobID}.txt')

cmd_failed = f"sacct -j {jobID} --format=JobID,JobName,State,ExitCode -n | grep -E 'FAILED' > {outFile_failed}"
cmd_timeout = f"sacct -j {jobID} --format=JobID,JobName,State,ExitCode -n | grep -E 'TIMEOUT' > {outFile_timeout}"

subprocess.run(cmd_failed, shell=True, check=False)
subprocess.run(cmd_timeout, shell=True, check=False)

# [1] failed jobs ---------------------------------------------

# Load file
with open(outFile_failed) as f:
    lines = f.readlines()

# Extract subjob numbers (e.g. 16, 20, 27, ...)
subjob_ids = sorted(set(
    int(match.group(1))
    for line in lines
    if (match := re.search(rf"{jobID}_(\d+)\b", line))
))

pattern = "Found 0 searchlights"
failed_due_to_searchlights = []

for job_id in subjob_ids:
    log_file = os.path.join(jobPath, f"out_{job_id}.log")
    if not os.path.isfile(log_file):
        print(f"[Missing] {log_file}")
        continue
    with open(log_file) as f:
        content = f.read()
        if pattern in content:
            failed_due_to_searchlights.append(job_id)

print("Jobs that failed due to 0 Searchlights:\n", failed_due_to_searchlights)

otherErrors = list(set(subjob_ids) - set(failed_due_to_searchlights))

print("\nOther errors (not '0 Searchlights'):")

for job_id in otherErrors:
    log_file = os.path.join(jobPath, f"out_{job_id}.log")
    if not os.path.isfile(log_file):
        print(f"[Missing] {log_file}")
        continue
    with open(log_file) as f:
        lines = f.readlines()
        if lines:
            print(f"[Job {job_id}] {lines[-1].strip()}")
        else:
            print(f"[Job {job_id}] <empty file>")


# [2] Timeout jobs ---------------------------------------------

with open(outFile_timeout) as f:
    lines = f.readlines()

# Extract subjob numbers
subjob_ids = sorted(set(
    int(match.group(1))
    for line in lines
    if (match := re.search(rf"{jobID}_(\d+)\b", line))
))

print(f'\nJobs that failed due to Timeout:\n {subjob_ids}\n')

if subjob_ids: 

    job_errors = []

    for job_id in subjob_ids:
        log_file = os.path.join(jobPath, f"out_{job_id}.log")
        if not os.path.isfile(log_file):
            print(f"[Missing] {log_file}")
            continue

        with open(log_file) as f:
            lines = f.readlines()

        # Default values
        subject = maskNr = config = "N/A"
        # last_line = lines[-1].strip() if lines else "<empty file>"

        for line in lines:
            if "subject:" in line:
                subject = line.split("subject:")[1].strip()
            elif "maskNr:" in line:
                maskNr = line.split("maskNr:")[1].strip()
            elif "configuration:" in line:
                config = line.split("configuration:")[1].strip()

        job_errors.append({
            "job_id": job_id,
            "subject": subject,
            "maskNr": maskNr,
            "config": config,
        })

    # Sort first by config, then by subject
    job_errors_sorted = sorted(job_errors, key=lambda x: (x["config"], x["subject"]))


    # Group: config -> subject -> list of mask numbers
    grouped_errors = defaultdict(lambda: defaultdict(list))

    for entry in job_errors_sorted:
        config = entry["config"]
        subject = entry["subject"]
        try:
            maskNr = int(entry["maskNr"])
        except ValueError:
            continue  # skip if mask number is invalid

        grouped_errors[config][subject].append(maskNr)


    for config, subjects in grouped_errors.items():
        for subject, masks in sorted(subjects.items()):
            if len(masks)>1:
                
                masks_sorted = sorted(set(masks))
                print(f"thisConfig = '{config}'")
                print(f"subjectID = '{subject}'")
                print(f"partialMasks = {masks_sorted}\n")
                print('job_cluster.add_job(\n\t'\
            'SL_crossnobis_partial,\n\t'\
            'subjectID=subjectID,\n\t'\
            'maskNr=PermuteArgument(partialMasks),\n\t'\
            'config_class_name = thisConfig\n'\
        ')\n')
            else: # don't use "PermuteArguments" of there is only one single partial pask missing
                masks_sorted = sorted(set(masks))
                print(f"thisConfig = '{config}'")
                print(f"subjectID = '{subject}'")
                print(f"partialMask = {masks_sorted[0]}\n")
                print('job_cluster.add_job(\n\t'\
            'SL_crossnobis_partial,\n\t'\
            'subjectID=subjectID,\n\t'\
            'maskNr=partialMask,\n\t'\
            'config_class_name = thisConfig\n'\
        ')\n')


