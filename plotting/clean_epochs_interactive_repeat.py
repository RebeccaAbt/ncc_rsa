#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants
import mne
import matplotlib
import matplotlib.pyplot as plt
mne.viz.set_browser_backend('qt')
# %matplotlib inline
# mne.viz.set_browser_backend('TkAgg')
#%%
# load  MANUALLY CLEANED DATA AGAIN! Check and save them AGAIN!!

subjects = ['20050204vrao', '19971028mrhs', '19970302urmr', '19910823ssld', '19970520smsr', '20050610atbu']
#%% NORMAL  workflow: load data after ICA:
# subjectID = subjects[0]
for subjectID in subjects:
# for subjectID in all_subjects_new:
# for subjectID in block_subjects:

	print(f"loading data of subject {subjectID}", flush=True)
	epochsFile = f'{MEG_CLEAN_EPOCHS_DIR}/{subjectID}/{subjectID}_maxfilter_True__ica_True__0.5-40Hz__fs_100__[-1.5_1.5]s_detrend_None_clean_meg-epo.fif'

	epochs = mne.read_epochs(epochsFile, preload=True) #.filter(l_freq=None, h_freq=35)# filter just for checking! remove before running script for saving cleaned epochs!

	epochs.plot(block=True, butterfly=False, n_epochs = 10, group_by = 'original', n_channels = 306, scalings = dict(mag=1e-12, grad=2e-11))   # dict(mag=1e-12, grad=4e-11)

	print(epochs.drop_log_stats())
	print(epochs.plot_drop_log())


	print(f"Saving epochs to {epochsFile}", flush=True)
	epochs.save(epochsFile, overwrite=True)

