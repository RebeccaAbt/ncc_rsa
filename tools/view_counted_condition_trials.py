#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import numpy as np
import joblib
from utils.subj import *
from utils.rsa import adjust_descriptors
import pandas as pd
from pymatreader import read_mat
from IPython.display import display

#%%

event_keys = ['auditory/hit/1', 'auditory/hit/2', 'auditory/hit/3', 'auditory/hit/4',
              'auditory/miss/1', 'auditory/miss/2', 'auditory/miss/3', 'auditory/miss/4',
              'somato/hit/1', 'somato/hit/2', 'somato/hit/3', 'somato/hit/4',
              'somato/miss/1', 'somato/miss/2', 'somato/miss/3', 'somato/miss/4',
              'visual/hit/1', 'visual/hit/2', 'visual/hit/3', 'visual/hit/4',
              'visual/miss/1', 'visual/miss/2', 'visual/miss/3', 'visual/miss/4']

event_values = [111,112,113,114,
                101,102,103,104,
                211,212,213,214,
                201,202,203,204,
                311,312,313,314,
                301,302,303,304]

event_dict = dict(zip(event_values, event_keys))


#%% =========================================== MEG 
print('MEG trials:\n------------------\n')

# all_subj = get_fusion_subjects()
all_subj = get_MEG_subjects()
all_files = os.listdir(os.path.join(MEG_DATA_DIR,'event_count_after_preproc'))

print(f' number of fusion subjects: {len(all_subj)} , number of event count files: {len(all_files)}')

rows = []
ids = []
for subjectID in all_subj:
    thisFile = os.path.join(MEG_DATA_DIR,'event_count_after_preproc',f'{subjectID}_event_counts.pkl')
    counts = joblib.load(thisFile)

    if len(counts['event_matched'])<24:
        print('\n----------------------------------------------------------------------------------------\n',
              f"{24-len(counts['event_matched'])} condition(s) missing for subject {subjectID}:\n {set(event_dict.keys()) - set(counts['event_matched'].keys())}.",
              '\n----------------------------------------------------------------------------------------\n')
    
    rows.append(counts['event_matched'])   # each is already a dict: {event: count}
    ids.append(subjectID)

df = pd.DataFrame(rows, index=ids).T   # transpose to match your original orientation

display(df)
print(f'Sorted number of trials to check for extreme small values: \n\n {np.sort(np.stack(np.array(df),0).reshape(-1))}')

#%% =========================================== MRI

print('MRI trials:\n------------------\n')

all_subj = get_MRI_subjects()
all_files = os.listdir(os.path.join(MEG_DATA_DIR,'event_count_after_preproc'))

rows = []
ids = []
for subjectID in all_subj:

    thisFile = os.path.join(
           MEG_DATA_DIR, 
            f'{subjectID}.mat'
        )

    mat_Data = read_mat(thisFile)['data']

    modality = np.stack(mat_Data['StimuliOrder1'],0).reshape(-1)
    stimulus = np.stack(mat_Data['StimuliOrder2'],0).reshape(-1)
    condition = np.stack(mat_Data['StimuliOrder3'],0).reshape(-1)
    responses = np.stack(mat_Data['response'],0).reshape(-1)
    assert(len(modality) == len(stimulus) == len(condition) == len(responses))

    select_idx = np.where((~np.isnan(responses)) & (condition == 1))[0] # only NT trials and trials with button presses

    modality = modality[select_idx]
    stimulus = stimulus[select_idx]
    condition = condition[select_idx]
    responses = responses[select_idx].astype(int)
    assert(len(modality) == len(stimulus) == len(condition) == len(responses))

    combined = [int(''.join(map(str, lst))) for lst in list(zip(modality, responses, stimulus))]
    unique_combined, counts = np.unique(combined, axis=0, return_counts=True)

    d = dict(zip(unique_combined, counts))
    d2 = dict((event_dict.get(k, k), v) for (k, v) in d.items()) # match with event names

    if len(unique_combined)<24:
        print('\n----------------------------------------------------------------------------------------\n',
            f"{24-len(unique_combined)} condition(s) missing for subject {subjectID}:\n {set(event_dict.values()) - set(d2.keys())}.",
            '\n----------------------------------------------------------------------------------------\n')

    rows.append(d2)   # each is already a dict: {event: count}
    ids.append(subjectID)

df_MRI = pd.DataFrame(rows, index=ids).T   # transpose to match your original orientation
display(df_MRI)

print(f'Sorted number of trials to check for extreme small values: \n\n {np.sort(np.stack(np.array(df_MRI),0).reshape(-1))}')




#%% Count valid MEG conditions (based on number of trials)


all_subj = get_MEG_subjects()

subjectID = all_subj[4]

def which_conditions_enough_MEG_trials(subjectID, all_conditions, min_trls=10):

    
    thisFile = os.path.join(
        MEG_DATA_DIR,'event_count_after_preproc',
        f'{subjectID}_event_counts.pkl'
    )

    counts = joblib.load(thisFile)['event_matched']

    [counts.pop(k) for k in [key for key, value in counts.items() if value < min_trls]]

    if len(counts) < 24:
        missing_counts = list(set(all_conditions) - set(counts.keys()))
        print(f'\n subjectID {subjectID} has only {len(counts)} valid conditions. mising conditions: \n\t{missing_counts} \n')

    return adjust_descriptors(counts.keys())

for subjectID in all_subj:
    which_conditions_enough_MEG_trials(subjectID, event_keys, min_trls=15)
# %%
