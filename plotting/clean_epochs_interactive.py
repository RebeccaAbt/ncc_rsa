#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants
import mne
import matplotlib
import matplotlib.pyplot as plt
# mne.viz.set_browser_backend('qt')
%matplotlib inline
# mne.viz.set_browser_backend('TkAgg')
#%%
# all_subjects_new = [
#  '19800616mrgu',
#  '19840930bigs',
#  '19880331igse',
#  '19920917gbse',
#  '19921205crfi',
#  '19930306sbeh',
#  '19942803fbjm',
#  '19950623ajrn',
#  '19951227eipo',
#  '19960531hibu', --> still missing --> hhatte komische Daten --> ICA blockwise??
#  '19960628gblm',
#  '19961123crsh',
#  '19970520smsr',
#  '19970605btre',
#  '19970801cabd',
#  '19980223zlde', 
#  '19981005gndd',
#  '20000118sbnb', --> still missing --> hatte komische Daten, noch blink Artefakte?? --> ICA blockwise??
#  '20010917rswg', # tmp in other script
#  '20020705ttbr', # tmp in other script
#  '20021027sldn', # tmp,,,
#  '20040627vrrj', # tmp ...
#  '20040630gbaf', does not exist yet
#  '20040819knee',--> still missing --> hhatte komische Daten --> ICA blockwise??
#  '20050204vrao',
#  '20050610atbu',
#  ]

# subjects = [
	# '19960531hibu',
	# '20000118sbnb',
	# '20040627vrrj',
	# '20040819knee',
	# '20050204vrao',
	# '20050610atbu',
	# '20040630gbaf'
# ]

# subjects = [
# # '19970302urmr',
# '20040819knee',
# '19910823ssld',
# '19930306sbeh',
# '20021027sldn',
# '19950623ajrn',
# '20031022ekse',
# '19990810mrkh',
# '19942803fbjm',
# '19961123crsh',
# '19921205crfi'

# ]

subjects = ['19880331igse']
#%% NORMAL  workflow: load data after ICA:

for subjectID in subjects:
# for subjectID in all_subjects_new:
# for subjectID in block_subjects:

	print(f"loading data of subject {subjectID}", flush=True)
	# epochsFile_ica = f'/home/reabt/experiments/ncc/MEG/data/epochs_clean/ica/{subjectID}/{subjectID}_maxfilter_True__ica_True__0.5-99Hz__fs_1000__[-1.5_1.5]s_meg-epo.fif'
	epochsFile_ica = f'{MEG_DATA_DIR}/epochs_clean/ica/{subjectID}/{subjectID}_maxfilter_True__ica_True__1-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1_meg-epo.fif'
	epochs = mne.read_epochs(epochsFile_ica, preload=True) #.filter(l_freq=None, h_freq=35)# filter just for checking! remove before running script for saving cleaned epochs!

	epochs.plot(block=True, butterfly=False, n_epochs = 10, group_by = 'original', n_channels = 90, scalings = dict(mag=1e-12, grad=2e-11))   # dict(mag=1e-12, grad=4e-11)

	print(epochs.drop_log_stats())
	print(epochs.plot_drop_log())

	#%%
	outFolder = f'{MEG_CLEAN_EPOCHS_DIR}/{subjectID}'
	os.makedirs(outFolder, exist_ok=True)
	# outFile = f'{outFolder}/{subjectID}_clean-epo.fif'

	# outFile = f'{outFolder}/{subjectID}_maxfilter_ica__1-99Hz__fs1000__[-1.5_1.5]s_detrend_1_clean-epo.fif'
	outFile = epochsFile_ica.split('meg-epo.fif')[0]+ 'clean_meg-epo.fif'
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

