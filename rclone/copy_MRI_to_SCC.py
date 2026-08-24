
#%%
import os
import sys
sys.path.append(os.path.dirname('/home/scc_e_393956/ncc/rsa/'))
from configs.config2 import * # directories + constants
import subprocess


subjectIDs = [  
                # '19840930bigs',
                # '19880331igse',
                # '19910703eigl',
                # '19910823ssld',
                # '19921205crfi',
                # '19951227eipo',
                # '19960628gblm',
                # '19960630cahi',
                # '19961123crsh',
                # '19970218crpo',
                # '19970302urmr',
                # '19970520smsr',
                # '19991211mrbn',
                # '20000118sbnb',
                # '20010917rswg',
                # '20020123sbhp',
                # '20020705ttbr',
                # '20040627vrrj',
                # '20040630gbaf',
                # '20050204vrao',
                # '20070324hlti',
                # '19930306sbpe',
                # '20050610atbu', 
                # '20050615buea',
                '20021027sldn'
                ]



'''
[1] copy the trimmed functionals

[2] copy structurals

[3] copy the firstlevel data (M1B --> the data we used for the RSA so far (26.08.2025), but I discovered, 
    that there is a problem with one timestamps used in the fMRI first level analysis 
    (because the last Outro Screen is visible until the end of the MRI recording and not just 5 seconds)).
    The corrected version is the "M1C" data, that will be copied in the next step

[4] copy the firstlevel data (M1C --> corrected version with the right timestamps for the Outro Screen)
'''

for subj in subjectIDs:

    print(f'\n--------------------------------\nProcessing subject: {subj}\n--------------------------------')
              
    # paths = [
    #         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/orig_functionals_trimmed', 
    #         'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/orig_functionals_trimmed'},

    #         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/orig_structurals', 
    #         'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/orig_structurals'},

    #         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/NCC/firstLevel_sensory_M1B', 
    #         'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/firstLevel_sensory_M1B'},

    #         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/NCC/firstLevel_sensory_M1C', 
    #         'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/firstLevel_sensory_M1C'},

    #         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/NCC/prepro_V1B', 
    #         'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/prepro_V1B'}
    #         ]

    paths = [
        {'inDir': f'mri:/data_MRI/neurokog/NCC25/analyze_fin/{subj}/orig_functionals_trimmed', 
        'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/orig_functionals_trimmed'},

        {'inDir': f'mri:/data_MRI/neurokog/NCC25/analyze_fin/{subj}/orig_structurals', 
        'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/orig_structurals'},

        {'inDir': f'mri:/data_MRI/neurokog/NCC25/analyze_fin/{subj}/NCC/firstLevel_sensory_M1B', 
        'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/firstLevel_sensory_M1B'},

        {'inDir': f'mri:/data_MRI/neurokog/NCC25/analyze_fin/{subj}/NCC/firstLevel_sensory_M1C', 
        'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/firstLevel_sensory_M1C'},

        {'inDir': f'mri:/data_MRI/neurokog/NCC25/analyze_fin/{subj}/NCC/prepro_V1B', 
        'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/prepro_V1B'}
        ]


    for n, path in enumerate(paths):
        inDir = path['inDir']
        outDir = path['outDir']

        print(f'    Next: {os.path.split(inDir)[-1]}')

        if subprocess.run(f'rclone lsd {inDir}', shell=True, capture_output=True, text=True).returncode == 0:

            if n == 4: # create the target directory for the prepro_V1B data, because it does not exist yet
                rclone_cmd =(   f'rclone copy {inDir} '
                                f'"{outDir}" '
                                '--include "swaud*" '
                                '-v')

            else:
                rclone_cmd =(   f'rclone copy {inDir} '
                                f'"{outDir}" '
                                '-v')
                
            result = subprocess.run(rclone_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"        Error occurred while copying data for subject {subj}: \n        {result.stderr}")

        else:
            print("          --> does not exist on the mri server.\n")
#%% Parallel version (faster but less readable bercause we can't track progress properly)
# import concurrent.futures

# def copy_subject_data(subj):
#     paths = [
#         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/orig_functionals_trimmed', 
#          'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/orig_functionals_trimmed'},

#         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/orig_structurals', 
#          'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/orig_structurals'},

#         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/NCC/firstLevel_sensory_M1B', 
#          'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/firstLevel_sensory_M1B'},

#         {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/NCC/firstLevel_sensory_M1C', 
#          'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/firstLevel_sensory_M1C'},

#          {'inDir': f'mri:/data/neurokog/NCC25/analyze_fin/{subj}/NCC/prepro_V1B', 
#          'outDir': f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/prepro_V1B'}
#     ]

#     for path in paths:
#         inDir = path['inDir']
#         outDir = path['outDir']
#         print(f'\nProcessing subject: {subj}')
#         print(f'    From: {inDir}')
#         print(f'    To:   {outDir}')

#         if subprocess.run(f'rclone lsd {inDir}', shell=True, capture_output=True, text=True).returncode == 0:
#             rclone_cmd = (f'rclone copy {inDir} '
#                           f'"{outDir}" '
#                           '-v')
#             result = subprocess.run(rclone_cmd, shell=True, capture_output=True, text=True)
#             if result.returncode != 0:
#                 print(f"        Error occurred while copying data for subject {subj}: \n        {result.stderr}")
#         else:
#             print(f"        Source directory {inDir} does not exist on the mri server.")

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     executor.map(copy_subject_data, subjectIDs)