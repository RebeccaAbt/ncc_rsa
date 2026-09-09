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

subjects = ['20050204vrao']
#%% NORMAL  workflow: load data after ICA:
subjectID = subjects[0]
# for subjectID in subjects:
# for subjectID in all_subjects_new:
# for subjectID in block_subjects:

print(f"loading data of subject {subjectID}", flush=True)
# epochsFile_ica = f'/home/reabt/experiments/ncc/MEG/data/epochs_clean/ica/{subjectID}/{subjectID}_maxfilter_True__ica_True__0.5-99Hz__fs_1000__[-1.5_1.5]s_meg-epo.fif'
# epochsFile_ica = f'{MEG_DATA_DIR}/epochs_clean/ica/{subjectID}/{subjectID}_maxfilter_True__ica_True__1-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1_meg-epo.fif'
epochsFile_ica = f'{MEG_DATA_DIR}/epochs_clean2/ica/{subjectID}/{subjectID}_maxfilter_True__ica_True__0.5-40Hz__fs_100__[-1.5_1.5]s_detrend_None_meg-epo.fif'

#%%
epochs = mne.read_epochs(epochsFile_ica, preload=True) #.filter(l_freq=None, h_freq=35)# filter just for checking! remove before running script for saving cleaned epochs!

epochs.plot(block=True, butterfly=False, n_epochs = 10, group_by = 'original', n_channels = 90, scalings = dict(mag=1e-12, grad=2e-11))   # dict(mag=1e-12, grad=4e-11)

print(epochs.drop_log_stats())
print(epochs.plot_drop_log())

#%%
outFolder = f'{MEG_CLEAN_EPOCHS_DIR}/{subjectID}'
os.makedirs(outFolder, exist_ok=True)

# outFile = epochsFile_ica.split('meg-epo.fif')[0]+ 'clean_meg-epo.fif'

outFile = os.path.join(outFolder, os.path.split(epochsFile_ica)[1]).split('meg-epo.fif')[0]+ 'clean_meg-epo.fif'


print(f"Saving epochs to {outFile}", flush=True)
epochs.save(outFile, overwrite=False)


#%% TEMPORARY: Load cleaned data just for viewing, not saving


# for subjectID in subjects:
# # for subjectID in all_subjects_new:
# # for subjectID in block_subjects:

# 	print(f"loading data of subject {subjectID}", flush=True)

# 	outFolder = f'/home/reabt/experiments/ncc/MEG/data/epochs_clean/manual_finish/{subjectID}'
# 	outFile = f'{outFolder}/{subjectID}_clean-epo.fif'

# 	epochs = mne.read_epochs(outFile, preload=True) #.filter(l_freq=None, h_freq=35)# filter just for checking! remove before running script for saving cleaned epochs!

# 	epochs.plot(block=True, butterfly=False, n_epochs = 10, group_by = 'original', n_channels = 90, scalings = dict(mag=1e-12, grad=2e-11))   # dict(mag=1e-12, grad=4e-11)

