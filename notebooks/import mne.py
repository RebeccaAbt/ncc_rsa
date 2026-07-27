import mne
import os


subjects = []

for subjectID in subjects:

	print(f"loading data of subject {subjectID}", flush=True)

	inFolder_clean = f'/home/reabt/experiments/ncc/MEG/data/epochs_clean/manual_finish2/{subjectID}'
	inFile_clean = f'{inFolder_clean}/{subjectID}_clean-epo.fif'

	epochs_clean_old = mne.read_epochs(inFile_clean, preload=True) #.filter(l_freq=None, h_freq=35)# filter just for checking! remove before running script for saving cleaned epochs!

