'''
This script was written before I implemented the new pipeline, 
where we first cmopute epochs on [0.5 99] Hz filterred data and then clean the epochs afterwards.

This script creates epochs, filters the, and then applies the potato algorithm to clean them.

But we will probably not use this script anymore.
'''



#%% imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants


from genericpath import isfile
import joblib
import mne
from pymatreader import read_mat

from plus_slurm import Job


from utils.raw import Raw
from utils.epochs import get_epochs_R
from utils.clean_epochs import *

os.environ["FONTCONFIG_PATH"] = "/etc/fonts"

def get_suffix(h_pass, l_pass, fs, epochs_settings):
	suffix =  f'MF__ICA__filter_{str(h_pass).replace('.','_')}-{l_pass}__fs_{fs}__{epochs_settings['tmin']}-{epochs_settings['tmax']}s'
	return suffix

class SavePotato(Job):

	def run(self, 
			subjectID,
		 	job_data_folder = 'epochs_potato', # <============= !!!
			potato_setting={
				'estimator' : "lwf",
				'use_field': False,
				'picks': "mag",
				'n_potatoes': 1, # only if use_field
				'z_threshold': 3,
				'p_threshold': 0.001,}
			):
		
		potatoFolder = f'/home/reabt/experiments/ncc/MEG/data/epochs_clean/potato/{subjectID}'
		os.makedirs(potatoFolder, exist_ok=True)
		epochsFile = f'/home/reabt/experiments/ncc/MEG/data/epochs_clean/ica/{subjectID}/{subjectID}_maxfilter_True__ica_True__0.5-99Hz__fs_1000__[-1.5_1.5]s_meg-epo.fif'
		epochs = mne.read_epochs(epochsFile)

		print(f"Running potato with settings: {potato_setting}")
		epochs_clean, info = reject_bad_epochs_with_potato(
								epochs,
								**potato_setting,
								return_details=True,
								reject_epochs = True,
								tmin = -1, 
								tmax = 1
								)


		potato_outputs  = dict(settings = potato_setting, info = info)

		outFile_epochs = f'{potatoFolder}/{subjectID}_potato-epo.fif/'
		outFile_info = f'{potatoFolder}/{subjectID}_potato_info.pkl'

		print(f"Saving epochs to {outFile_epochs}", flush=True)
		epochs_clean.save(outFile_epochs, overwrite=True)

		print(f"Saving info to {outFile_info}", flush=True)
		joblib.dump(potato_outputs, outFile_info)

