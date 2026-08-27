#%% imports

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

sys.path.append('/home/reabt/experiments/ncc/MEG/Fabi/utils/')
sys.path.append('/home/reabt/experiments/ncc/MEG/Fabi/')

os.environ["FONTCONFIG_PATH"] = "/etc/fonts"

# from clusterjobs.meta_job import Job
from plus_slurm import Job
import joblib
import numpy as np
from utils.raw import Raw
from utils.epochs import  get_epochs_R
from utils.ica import get_outFilePaths, get_blockwise_outFilePaths
from utils.provenance import configure_subject_logging, flatten_integer_values, record_artifact
from pymatreader import read_mat
import neurokit2 as nk
import mne


#%%

class SaveEpochs(Job):
	
	job_data_folder = 'epochs_icaTest'

	def _load_rejected_components(self, rejection_file):
		with open(rejection_file, encoding='utf-8') as file:
			return sorted(set(flatten_integer_values(json.load(file))))

	def _load_blockwise_rejected_components(self, rejection_file):
		with open(rejection_file, encoding='utf-8') as file:
			rejection_data = json.load(file)

		return {
			block: sorted(set(flatten_integer_values(block_data)))
			for block, block_data in rejection_data.items()
		}

	#%% the run method starts here
	def run(self, 
			subjectID,
			run_part = 0, # 0: both oarts, 1: only part 1, 2: only part 2
		 	epochsDir = MEG_EPOCHS_DIR,
		 	blockwise_ica=False,
			ica=True,
			ica_out_root= ICA_DIR,
			overwrite=True,
			preproc_settings = {'maxfilter': True,
								'downsample_f': None,
								'notch': True,
								'l_pass': 99,
								'h_pass': 0.5},

			epochs_settings={
							 'tmin': -1.5,
							 'tmax': 1.5,
							 'baseline': None,
							 'preload': True,
							 'h_freq': None, 	# not passed to mne.Epochs, but used on continuous data before	
							 'detrend': 1,		# 0=constant, 1=linear, None=no detrending
							 'fs': 1000			# not passed to mne.Epochs, but used on continuous data before		
							 }, 

		 	ica_settings = {'ica_method': "picard",
							'fit_params': None,
							'ica_threshold': 0.35,
							'n_components': 50,
							'eog': True,
							'ecg': True,							
							'train_thresh': 2,
							'train_freq': 16.7},
			use_mean_headpos = False
			):
		epoching_options = dict(epochs_settings)
		preprocessing_options = dict(preproc_settings)

		ica_options = dict(ica_settings)
		h_freq = epochs_settings.pop('h_freq')
		fs 	   = epochs_settings.pop('fs')
		_, ica_outFiles = get_outFilePaths(subjectID, ica_out_root)
		_, ica_outFiles_blockwise = get_blockwise_outFilePaths(subjectID, ica_out_root)
		print('ica_outFiles', ica_outFiles)
		print('ica_outFiles_blockwise', ica_outFiles_blockwise)
		print(f'---------------------------------------------\n Overwrite is set to {overwrite}. \n---------------------------------------------', flush=True)

		# os.makedirs(f'{MEG_DATA_DIR}/{job_data_folder}/{subjectID}/', exist_ok=True)
		subjIcaDir = os.path.join(epochsDir, 'ica', subjectID)
		os.makedirs(subjIcaDir, exist_ok=True)


		# if os.path.isfile(meg_outfile) and not overwrite:
		# 	print(f"Epochs file already exists for {subject_id} with settings: {suffix}. Skipping computation.")
		# 	return
		
		print(f"Running preprocessing for {subjectID} in preproc.py")

		
		subject_id_short = subjectID[8:]
		event_info = read_mat(f'{MEG_DATA_DIR}/behav/{subject_id_short}.mat')['data']

		if run_part == 1:
			print(f"---------------------------------------------\n Now doing: ICA PART 1 \n---------------------------------------------", flush=True)
			if blockwise_ica:
				data_raw = Raw.run_cleaner_blockwise_ica_part1(
					subjectID,
					ica=ica,
					ica_out_root=ica_out_root,
					**ica_settings,
					**preproc_settings,
					overwrite=overwrite,
					use_mean_headpos = use_mean_headpos
				)
			else:
				data_raw = Raw.run_cleaner_ica_part1(subjectID,
										   ica=ica,
										   ica_out_root = ica_out_root,
										   **ica_settings, 
										   **preproc_settings,
										   overwrite = overwrite,
										   use_mean_headpos = use_mean_headpos
										   )
		elif run_part == 2:
			print(f"---------------------------------------------\n Now doing: ICA PART 2 \n---------------------------------------------", flush=True)
			if blockwise_ica:
				if not ica_outFiles_blockwise['file_raw_clean'].exists() or overwrite:
					data_raw = Raw.run_cleaner_blockwise_ica_part2(
						subjectID,
						ica_out_root=ica_out_root,
						downsample_f=None,
						use_manual=True,
						use_automatic=False,
						overwrite=overwrite,
					)
				else:
					print(f'ICA already done. \nloading {ica_outFiles_blockwise['file_raw_clean']}')
					data_raw = mne.io.read_raw_fif(ica_outFiles_blockwise['file_raw_clean'], preload=True)
			else:
				if not ica_outFiles['file_raw_clean'].exists() or overwrite:
					data_raw = Raw.run_cleaner_ica_part2(subjectID,
											   ica_out_root = ica_out_root,
											   downsample_f=None,
											   overwrite = overwrite
										   )
				else:
					print(f'ICA already done. \nloading {ica_outFiles['file_raw_clean']}')
					data_raw = mne.io.read_raw_fif(ica_outFiles['file_raw_clean'], preload=True)

		# if run_part in [0, 2]:
			#%% do bio preprocessing
			print(f"---------------------------------------------\n Now doing: BIO PROCESSING \n---------------------------------------------", flush=True)
			if 'MISC001' in data_raw.ch_names:
					data_raw.set_channel_types({'MISC001': 'resp'})
					mne.rename_channels(data_raw.info, {'MISC001': 'rsp'})
					chan_list_bio = ['ecg', 'eog', 'resp', 'stim']
			else:
					chan_list_bio = ['ecg', 'eog', 'stim']

			bio_raw = data_raw.copy().pick(chan_list_bio)
			bio_df = bio_raw.to_data_frame()
			
			bio_clean = nk.bio_process(	ecg=bio_df['ECG003'], 
										rsp=bio_df['rsp'] if 'rsp' in data_raw.ch_names else None, 
										eog=bio_df['EOG001'], 
										sampling_rate=bio_raw.info['sfreq'])[0]
			
			ch_names_bio = bio_clean.columns.tolist() + bio_df.columns[-17:].tolist()
			ch_types_bio = np.tile('bio', len(bio_clean.columns)).tolist() + \
						   np.tile('stim', len(bio_df.columns[-17:])).tolist()
			
			bio_info = mne.create_info(ch_names=ch_names_bio, 
									   sfreq=bio_raw.info['sfreq'],
									   ch_types=ch_types_bio)
			
			bio_data = np.concatenate([bio_clean.to_numpy(),
									   bio_df[bio_df.columns[-17:]].to_numpy()], axis=1).T
									   
			bio_clean_raw = mne.io.RawArray(bio_data, bio_info)

			print(f"---------------------------------------------\n Now doing:  epochs_meg \n---------------------------------------------")
			
			meg_raw = data_raw.pick(['meg', 'stim']) 
			
			if h_freq:
				meg_filter_options = {
					    "l_freq": None,
					    "h_freq": h_freq,
					    "picks": "meg",
					    "method": "fir",
					    "fir_design": "firwin",
					    "fir_window": "hamming",
					    "l_trans_bandwidth": "auto",
					    "h_trans_bandwidth": "auto",
					    "phase": "zero",
					    "n_jobs": -1,
					}
				
				meg_raw.filter(**meg_filter_options)

			if fs != meg_raw.info['sfreq']:
				meg_raw.resample(fs)	

								
			epochs_meg, events = get_epochs_R(meg_raw, event_info, epochs_settings)
			# epochsFile = f'{MEG_DATA_DIR}/{job_data_folder}/{subjectID}/{subjectID}_{suffix}_meg-epo.fif'
			# suffix =  f'maxfilter_{preproc_settings['maxfilter']}__ica_{ica}__{preproc_settings['h_pass']}-{h_freq}Hz__fs_{fs}__[{epochs_settings['tmin']}_{epochs_settings['tmax']}]s_detrend_{epochs_settings['detrend']}'# _meg-epo.dat'
			
			suffix = (
				f'maxfilter_{preproc_settings['maxfilter']}__ica_{ica}__' +
				f'{epochs_meg.info['highpass']}-{int(epochs_meg.info['lowpass'])}Hz__' +
				f'fs_{int(epochs_meg.info['sfreq'])}__' +
				f'[{epochs_settings['tmin']}_{epochs_settings['tmax']}]s_' +
				f'detrend_{epochs_settings['detrend']}'
					)# _meg-epo.dat'
			
			epochsFile = os.path.join(subjIcaDir,f'{subjectID}_{suffix}_meg-epo.fif')

			# ~~~ Loggin stuff
			if blockwise_ica:
				rejection_file = str(ica_outFiles_blockwise['file_rejected'])
				ica_log = os.path.splitext(str(ica_outFiles_blockwise['file_metadata']))[0] + '.log'
				clean_raw_file = str(ica_outFiles_blockwise['file_raw_clean'])
				rejected_components = self._load_blockwise_rejected_components(rejection_file)
			else:
				rejection_file = str(ica_outFiles['file_rejected'])
				ica_log = os.path.splitext(str(ica_outFiles['file_ica']))[0] + '.log'
				clean_raw_file = str(ica_outFiles['file_raw_clean'])
				rejected_components = self._load_rejected_components(rejection_file)

			logger, _ = configure_subject_logging(
				epochsFile,
				subjectID,
				upstream_log=ica_log if os.path.isfile(ica_log) else None,
			)
			logger.info('Starting MEG epoching')
			logger.info('ICA part 2 rejected components: %s', rejected_components)
			logger.info('Processing before epoching: filter_hz=%s resample_hz=%s', h_freq, fs)
			logger.info('Epoching options: %s', epoching_options)
			# ~~~ Loggin stuff end

			print(f'--------------------\n saving Epochs to {epochsFile}.... \n--------------------\n', flush=True)
			epochs_meg.save(epochsFile, overwrite = True)
			joblib.dump(events, os.path.join(epochsDir, 'ica', f'{subjectID}_events.pkl'))

			# ~~~ Loggin stuff
			record_artifact(
				output_path=epochsFile,
				operation_name='SaveEpochs.run MEG epoching',
				parameters={
					'subjectID': subjectID,
					'run_part': run_part,
					'blockwise_ica': blockwise_ica,
					'ica': ica,
					'preprocessing': {
						'preprocessing_options': preprocessing_options,
						'filtering_continuous':  meg_filter_options,
						'epochs_info': {
							'l_freq': epochs_meg.info['highpass'], 
							'h_freq': epochs_meg.info['lowpass'], 
							'fs': epochs_meg.info['sfreq']}
					},
					'ica_options': ica_options,
					'ica_rejected_components': rejected_components,
					'epoching_options': epoching_options,
					'shape': list(epochs_meg.get_data(copy=False).shape),
				},
				input_paths=[
					path for path in [clean_raw_file, rejection_file,
						f'{MEG_DATA_DIR}/behav/{subject_id_short}.mat']
					if os.path.exists(path)
				],
			)
			logger.info('Wrote MEG epochs and provenance: %s', epochsFile)
			# ~~~ Loggin stuff end

			# doing bio epochs after meg epochs, because we need to remove the "detrend" that causes errors if applied to bio data only

			print(f"---------------------------------------------\n Now doing:  epochs_bio\n---------------------------------------------", flush=True)
			epochs_settings.pop('detrend')
			epochs_bio, events = get_epochs_R(bio_clean_raw, event_info, epochs_settings)
			print('--------------------\n saving Epochs .... \n--------------------\n', flush=True)

			# epochs_bio.save(f'{MEG_DATA_DIR}/{job_data_folder}/{subjectID}/{subjectID}_{suffix}_bio-epo.fif', overwrite = True)
			epochs_bio.save(os.path.join(subjIcaDir, f'{subjectID}_{suffix}_bio-epo.fif'), overwrite = True)
			# joblib.dump(events, f'{MEG_DATA_DIR}/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_bio-events.pkl')


