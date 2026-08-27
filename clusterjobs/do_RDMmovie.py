#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import joblib
import rsatoolbox as rsa
import numpy as np
import mne

from plus_slurm import Job

from utils.load_cfg import load_MEG_config_instance
from utils.rsa import adjust_descriptors
from utils.rsa_meg import create_run_idx, count_events_per_run, calc_rdm_movie_parallel
from utils.provenance import configure_subject_logging, record_artifact

#%% 

class RDMmovie(Job):
	def run(self,
			subjectID = '19910823ssld',
			config_class_name = 'MEGconfig_E',  # not relevant, if e.g. "E1" or "E2" because searchlight radius not relevant here",
			overwrite = True
			):

		print('\n[1] load config & set up paths for MEG --> this loads all the settings for the computations')

		cfg = load_MEG_config_instance(config_class_name, subjectID) 
		cfg.print_summary()
		cfg.configure_paths()

		MEG_file = cfg.MEG_inFile[0]
		fileName_movie = cfg.get_outFile_names()['movie']
		logger, _ = configure_subject_logging(fileName_movie, subjectID)
		logger.info("Starting RDM movie; config=%s", config_class_name)
		logger.info("Input epochs file: %s", MEG_file)

		if not overwrite and os.path.exists(fileName_movie):
			print(f'\n--------------------------------------------\n Skipped subject, because file {fileName_movie} already existis!\n--------------------------------------------\n')    
			return

		if cfg.cv_descriptor is not None:
			if str(cfg.cv_descriptor) not in fileName_movie:
				raise ValueError(f'cv_descriptor "{cfg.cv_descriptor}" not in output file name "{fileName_movie}". Please check! \nAlways define the cv_descxriptor at the beginning of the script so the paths are properly configured')
			
		print('[2] Create dataset for MEG movie computation\n')

		# load epochs

		if MEG_file.endswith('.pkl') or MEG_file.endswith('.dat'):
			epochs = (joblib.load(MEG_file)['epochs_meg'])

		elif MEG_file.endswith('.fif'):
			epochs = mne.read_epochs(MEG_file, preload=True)['NT']

		epochs.pick(cfg.channels)
		# epochs.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, n_jobs=-1)
		epochs.crop(tmin = cfg.t_min, tmax = cfg.t_max)
		# epochs.resample(sfreq = cfg.fs_new)
		
		count_events_per_run(cfg, epochs) # <-------------------------------------- This is new! Need to check if it works!

		ch_names = epochs.ch_names
		event_ids = epochs.event_id
		times = epochs.times

		epoch_array = epochs.get_data(copy=True) #(n_events, n_channels, n_times)

		n_events, n_channels, n_times = epoch_array.shape #(n_events, n_channels, n_times)

		#% get noise
		residuals = np.zeros_like(epoch_array)

		#% get all possible event names
		rev_event_id = {v: k for k, v in epochs.event_id.items()}
		event_names = np.array([rev_event_id[i] for i in epochs.events[:, 2]])

		# loop over event_id and compute residuals
		for event_id, trigger in epochs.event_id.items():

			events_data = epochs[event_id].get_data(copy=True) # n_repetitions x n_channels x n_times
			residuals[event_id == event_names, :, :] = events_data - np.mean(events_data, axis=0, keepdims=True)

		reshaped_residuals = np.swapaxes(residuals, 1, 2).reshape(-1, n_channels)
		prec = rsa.data.prec_from_residuals(reshaped_residuals, method='shrinkage_diag')

		print(f'\n{residuals.shape=} = (n_events, n_channels, n_timepoints)', flush=True)
		print(f'{reshaped_residuals.shape=} = (n_events * n_timepoints, n_channels)', flush=True)
		print(f'{prec.shape=} = (n_channels, n_channels)\n', flush=True)

		# assemble information to get data in rsatoolbox format
		all_events = event_names #everything really

		all_events = adjust_descriptors(all_events) # change the descriptors to match naming format  of fMRI data

		if event_names[0].split('/')[0] in ['NT', 'HI', 'catch']: # in some files, the trial type (NT/HI/catch) is part of the name. If so, we need to correct the indexing
			idx_correction = 1
		else: 
			idx_correction = 0

		modality = np.array([event_name.split('/')[0+idx_correction] for event_name in event_names]) #auditory, somato & visual
		hits = np.array([event_name.split('/')[1+idx_correction] for event_name in event_names]) #hits 
		stim_type = np.array([event_name.split('/')[2+idx_correction] for event_name in event_names]) #stim numbers..
		run_idx = create_run_idx(modality)

		obs_des = {'condition': all_events, #everything really
			'modality': modality,
			'hits': hits,
			'stim_type': stim_type,
			'run_idx': run_idx} # observation descriptor --> can be vigorously extended


		chn_des = {'channels': ch_names} # channel descriptors
		tim_des = {'time': times}
		des = {'session': 0} #made up stuff TODO: check if needed
		des['subject'] = str(MEG_file).split('/')[-2] #before ix

		data = rsa.data.TemporalDataset(epoch_array,
										descriptors=des,
										obs_descriptors=obs_des,
										channel_descriptors=chn_des,
										time_descriptors=tim_des)

		print('[3] Compute RDM movie for MEG data \n', flush=True)

		descriptor = 'condition'

		rdm_movie = calc_rdm_movie_parallel(
								data,
								method = cfg.RDMmethod, 
								descriptor = descriptor,
								cv_descriptor = cfg.cv_descriptor,
								noise = prec,
								unbalanced=True
								)
		
		print('\n----------------------------------------------------\n',
			  f"RDM descriptors before reordering: \n\t condition: {rdm_movie.pattern_descriptors['condition']}\n\t index: {rdm_movie.pattern_descriptors['index']}",
			  '\n--------------------------------------------\n', flush=True)

		rdm_movie.rdm_descriptors['time'] = data.time_descriptors['time']
		rdm_movie.reorder(np.argsort(rdm_movie.pattern_descriptors['condition']))
		print(rdm_movie.dissimilarities.shape)
		print(f'{rdm_movie.dissimilarities.shape=} = (n_subjects * n_timepoints, n_conditions * (n_conditions-1) / 2)')
		print(f'number of NaN values in results: {np.sum(np.isnan(rdm_movie.dissimilarities))}')

		os.makedirs(os.path.split(fileName_movie)[0], exist_ok=True)

		joblib.dump(rdm_movie, fileName_movie)
		record_artifact(
			output_path=fileName_movie,
			operation_name="RDMmovie.run",
			parameters={
				"subjectID": subjectID,
				"config_class_name": config_class_name,
				"overwrite": overwrite,
				"config": vars(cfg),
				"input_shape": [n_events, n_channels, n_times],
				"n_conditions": len(np.unique(all_events)),
				"n_searchlight_workers": -1,
			},
			input_paths=[MEG_file],
		)
		logger.info("Wrote RDM movie and provenance manifest: %s", fileName_movie)

#%%