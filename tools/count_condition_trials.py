'''
Counts number of trials in each of the 24 Conditions --> AFTER <-- preprocessing 
(because some trials will be removed turing the preprocessing),
Saves the counts in a .pkl file for each subject.
'''


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import numpy as np
import joblib
from mne import read_epochs
from utils.load_cfg import load_MEG_config_instance
from utils.subj import *
#%%

# all_subj = get_fusion_subjects()

# all_subj = ['19930306sbeh']

all_subj = get_MEG_subjects()

#%%

for subjectID in all_subj[2:]:

    print(f'Processing subject {subjectID}...')
    config_class_name = 'MEGconfig_Eb'
    cfg = load_MEG_config_instance(config_class_name, subjectID) 
    MEG_file = cfg.MEG_inFile[0]


    os.path.join(cfg.dataDir, cfg.dataFolder)

    event_count_dir = os.path.join(cfg.dataDir, cfg.dataFolder)
    event_count_file = os.path.join(event_count_dir, f'{subjectID}_event_counts.pkl')

    if os.path.exists(event_count_file):
        continue

      # not relevant, if e.g. "E1" or "E2" because searchlight radius not relevant here",
    

    if MEG_file.endswith('.pkl') or MEG_file.endswith('.dat'):
        epochs = (joblib.load(MEG_file)['epochs_meg']
                #.filter(l_freq=1, h_freq=None, n_jobs=-1)
                #.resample(100)
                )
    elif MEG_file.endswith('.fif'):
        epochs = read_epochs(MEG_file)['NT']#.filter(l_freq=1, h_freq=None, n_jobs=-1)

    event_id = epochs.event_id
    unique_events, counts = np.unique(epochs.events[:,-1], return_counts=True)
    event_counts = dict([[u.item(), c.item()] for u, c in zip(unique_events, counts)])
    event_matched = {k: event_counts[v] for k, v in event_id.items()}

    joblib.dump({'event_counts': event_counts,
                 'event_matched': event_matched}, event_count_file)
    

