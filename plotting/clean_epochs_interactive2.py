#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import mne
import matplotlib
import matplotlib.pyplot as plt
mne.viz.set_browser_backend('qt')
# mne.viz.set_browser_backend('TkAgg')


# for subjectID in all_subjects_new:
# for subjectID in block_subjects_new:

subjects = [
	# '19960531hibu',
	# '20000118sbnb',
	# '20040627vrrj',
	'20040819knee',
	'20050204vrao',
	'20050610atbu',
]

for subjectID in subjects:


	print(f"loading data of subject {subjectID}", flush=True)
	epochsFile_ica = f'{MEG_ICA_DIR}/{subjectID}/{subjectID}_maxfilter_True__ica_True__0.5-99Hz__fs_1000__[-1.5_1.5]s_meg-epo.fif'
	epochs = mne.read_epochs(epochsFile_ica, preload=True) #.filter(l_freq=None, h_freq=35)# filter just for checking! remove before running script for saving cleaned epochs!

	
	epochs.plot(block=True, butterfly=False, n_epochs = 10, group_by = 'original', n_channels = 100, scalings = dict(mag=1e-12, grad=2e-11))   # dict(mag=1e-12, grad=4e-11)

	print(epochs.drop_log_stats())
	print(epochs.plot_drop_log())

	#%%
	outFolder = f'{MEG_CLEAN_EPOCHS_DIR}/{subjectID}'
	os.makedirs(outFolder, exist_ok=True)
	# outFile = f'{outFolder}/{subjectID}_clean-epo.fif'
	outFile = epochsFile_ica.split('meg-epo.fif')[0]+ 'clean_meg-epo.fif'
	print(f"Saving epochs to {outFile}", flush=True)
	epochs.save(outFile, overwrite=True)



