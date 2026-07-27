'''
Copy mask files (located in output folder of MRI stats on MRI server) 
to SCC so we can access them locally in our other scripts because mounting stil doesn't work properly.

'''


#%%
import os
import subprocess
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')
from utils.subj import *

subjectIDs = get_MRI_subjects()

for subj in subjectIDs:

    print(f'\n--------------------------------\nProcessing subject: {subj}\n--------------------------------')
    if subj == '19930306sbpe':
        path = {
            'inDir': f'mri:/data_MRI/neurokog/NCC25/analyze_fin/{subj}/NCC/firstLevel_sensory_M1C', 
            'outDir': f'/home/reabt/experiments/ncc/MRI/data/masks/19930306sbeh/'}
    else:
        path = {
            'inDir': f'mri:/data_MRI/neurokog/NCC25/analyze_fin/{subj}/NCC/firstLevel_sensory_M1C', 
            'outDir': f'/home/reabt/experiments/ncc/MRI/data/masks/{subj}/'}


    inDir = path['inDir']
    outDir = path['outDir']

    print(f'    Next: {os.path.split(inDir)[-1]}')

    if subprocess.run(f'rclone lsd {inDir}', shell=True, capture_output=True, text=True).returncode == 0:
        rclone_cmd =(   f'rclone copy {f'{inDir}/mask.nii'} '
                            f'"{outDir}" '
                            '-v')
        # print(rclone_cmd)
        result = subprocess.run(rclone_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"        Error occurred while copying data for subject {subj}: \n        {result.stderr}")

    else:
        print("          --> does not exist on the mri server.\n")
# %%
