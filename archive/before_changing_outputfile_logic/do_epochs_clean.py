#%% imports
from genericpath import isfile
import os
import sys
import joblib
import mne
from pymatreader import read_mat

from plus_slurm import Job

sys.path.append('/home/reabt/experiments/ncc/MRI/code/utils/')
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.raw import Raw
from utils.epochs import get_epochs_R
from utils.clean_epochs import *

os.environ["FONTCONFIG_PATH"] = "/etc/fonts"

def get_suffix(h_pass, l_pass, fs, epochs_settings):
	suffix =  f'MF__ICA__filter_{str(h_pass).replace('.','_')}-{l_pass}__fs_{fs}__{epochs_settings['tmin']}-{epochs_settings['tmax']}s'
	return suffix

class SaveEpochs(Job):

	def run(self, 
			subject_id,
		 	job_data_folder = 'epochs_potato', # <============= !!!
			maxfilter=True,
			ica=True,
			save_raw=False,
			l_pass=40,
			h_pass=0.1,
			src_type='beamformer',
			downsample_f = 100,
			epochs_settings={
							 'tmin': -1.,
							 'tmax': 1.,
							 'baseline': None,
							 'preload': True},
		 	suffix = 'clean',
			potato_setting={
				'estimator' : "oas",
				'use_field': True,
				'picks': "grad",
				'n_potatoes': 2, # only if use_field
				'z_threshold': 3,
				'p_threshold': 0.01,}


			):
		
		sep = '---------------------------------------------' # separator when printung stuff

		if potato_setting['use_field']:
			potato_suffix = f'potatoField__estim_{potato_setting['estimator']}__thres_n{potato_setting['n_potatoes']}_z{potato_setting['z_threshold']}'
		else:
			potato_suffix = f'potato4{potato_setting['picks']}__estim_{potato_setting['estimator']}__thres_z{potato_setting['z_threshold']}'
		
		outDir = f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}'
		os.makedirs(outDir, exist_ok=True)
		
		# don't do the full preprocessing if we already did it before
		suffix = get_suffix(h_pass, l_pass, fs=1000, epochs_settings = epochs_settings)
		fileName_baseEpochs = f'{outDir}/{subject_id}_{suffix}_meg-epo.fif'
		
		if not os.path.isfile(fileName_baseEpochs):

			print(f"{sep}\n Running preprocessing for {subject_id} \n{sep}")

			preproc_settings = {'maxfilter': maxfilter,
								'downsample_f': None,
								'notch': True,
								'l_pass': l_pass,
								'h_pass': h_pass}
			
			subject_id_short = subject_id[8:]
			event_info = read_mat(f'/home/reabt/experiments/ncc/MEG/data/behav/{subject_id_short}.mat')['data']

			print(f"{sep}\n Now doing: Raw.run_cleaner \n{sep}")

			data_raw = Raw.run_cleaner(subject_id,
									   event_info,
									   ica=ica,
									   ica_threshold=0.4,
									   **preproc_settings
									   )
			
			meg_raw = data_raw.pick(['meg', 'stim'])  

			print(f"{sep}\n Now doing:  epochs_meg \n{sep}", flush=True)

			epochs_meg, events = get_epochs_R(meg_raw, event_info, epochs_settings)

			print('--------------------\n saving uncleaned 1000Hz Epochs .... \n--------------------\n', flush=True) 
			# 1a ~~~~~~~~~~~~~~~~ save uncleaned epochs with orignal 1000 Hz fs
			epochs_meg.save(fileName_baseEpochs)
				
		else:
			print(f"{sep}\n Loading saved epochs for {subject_id} from file {fileName_baseEpochs}\n{sep}", flush=True)
			epochs_meg = mne.read_epochs(fileName_baseEpochs, preload=True)

		print(f"{sep}\n Now doing:  Potato \n{sep}", flush=True)

		epochs_meg_clean, potato_info = reject_bad_epochs_with_potato(
			epochs_meg,
			**potato_setting,
		)

		print_potato_summary(epochs_meg, epochs_meg_clean, potato_info)

		
		# 1b ~~~~~~~~~~~~~~~~ save uncleaned epochs after downsampling
		# print('--------------------\n saving uncleaned downsampled Epochs .... \n--------------------\n', flush=True) 
		# suffix = get_suffix(h_pass, l_pass, fs=downsample_f, epochs_settings = epochs_settings)
		# fileName = f'{outDir}/{subject_id}_{suffix}_meg-epo.fif'
		# if not os.path.isfile(fileName):
		# 	epochs_meg.save(fileName)

		# joblib.dump(potato_info, f'{outDir}/{subject_id}_{suffix}_{potato_suffix}_potato_info.pkl')

		# # 2a ~~~~~~~~~~~~~~~~ save cleaned epochs with orignal 1000 Hz fs
		# print('--------------------\n saving cleaned 1000Hz Epochs .... \n--------------------\n', flush=True) # cleaned epochs with orignal 1000 Hz fs
		# fileName = f'{outDir}/{subject_id}_{suffix}_{potato_suffix}_meg_clean-epo.fif'
		# if not os.path.isfile(fileName):
		# 	epochs_meg_clean.save(fileName)

		# 3 ~~~~~~~~~~~~~~~~ downsample and save cleaned epochs
		print('--------------------\n saving cleaned downsampled Epochs .... \n--------------------\n', flush=True)
		suffix = get_suffix(h_pass, l_pass, fs=downsample_f, epochs_settings = epochs_settings)
		fileName = f'{outDir}/{subject_id}_{suffix}_{potato_suffix}_meg_clean-epo.fif'
		if not os.path.isfile(fileName):
			epochs_meg_clean.resample(downsample_f).save(fileName)


		# old syntax for saving stzuff / creating filenames:
'''		print('--------------------\n saving uncleaned 1000Hz Epochs .... \n--------------------\n', flush=True) # uncleaned epochs with orignal 1000 Hz fs
		fs = 1000
		suffix =  f'maxfilter_{maxfilter}__ica_{ica}__l_pass_{l_pass}__downsample_f_{fs}__h_pass_{h_pass}__{epochs_settings['tmin']}-{epochs_settings['tmax']}s'# _meg-epo.dat'
		epochs_meg.save(f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_meg-epo.fif')
		joblib.dump(potato_info, f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_potato_info.pkl')

		print('--------------------\n saving cleaned 1000Hz Epochs .... \n--------------------\n', flush=True) # cleaned epochs with orignal 1000 Hz fs
		suffix =  f'maxfilter_{maxfilter}__ica_{ica}__l_pass_{l_pass}__downsample_f_{fs}__h_pass_{h_pass}__{epochs_settings['tmin']}-{epochs_settings['tmax']}s__useField_{potato_setting['use_field']}'# _meg-epo.dat'
		epochs_meg_clean.save(f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_{potato_suffix}_meg_clean-epo.fif')

		# downsampled cleaned epochs
		fs = downsample_f
		if downsample_f != 1000:

			print('--------------------\n saving cleaned downsampled Epochs .... \n--------------------\n', flush=True)
			suffix =  f'maxfilter_{maxfilter}__ica_{ica}__l_pass_{l_pass}__downsample_f_{downsample_f}__h_pass_{h_pass}__{epochs_settings['tmin']}-{epochs_settings['tmax']}s__useField_{use_field}'# _meg-epo.dat'
			epochs_meg_clean.resample(downsample_f).save(f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_meg_clean-epo.fif')

'''
