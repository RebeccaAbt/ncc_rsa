# -*- coding: utf-8 -*-
"""
This Class takes different parts of the experiment from sinuhe
and specifies corresponding triggers
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

sys.path.append(FABI_DIR)

from obob_mne.raw import Raw as RawTemplate
import mne
import numpy as np
from obob_mne.events import read_events_from_analogue
from utils.events import fix_events_from_analogue
from utils.ica import *
from copy import deepcopy


def _crop_end_of_block(raw, padding=1, padding_stim=3):
	# If last trigger is a button press,  "padding" will be used to determine crop-time.
	# If last trigger is a stimulus, "padding_stim" will be used to determine crop-time.

	# padding/padding_stim:  seconds after last trigger, when block should be cropped
	splitline = '------------------------------------------'
	print( f'\n{splitline}\n cropping raw data:' +
		   f'\n\tpadding before first trigger: {padding_stim} ' +
		   f'\n\tpassing after last trigger: ' +
		   f'\n\t\t{padding} if last trigger = button ' +
		   f'\n\t\t{padding_stim} if last trigger = stimulus \n{splitline}\n')
	fs = raw.info['sfreq']

	cut_start = ((raw.events[0][0]-raw.first_samp)/fs)-padding_stim
	if cut_start < 0:
		print(f'\n{splitline}\n WARNING: block is not long enough for cropping \n{splitline}\n')
		cut_start = 0

	last_event_type = raw.events[-1][2]
	if last_event_type == 1:
		cut_end = ((raw.events[-1][0]-raw.first_samp)/fs)+padding_stim 
	else:
		cut_end = ((raw.events[-1][0]-raw.first_samp)/fs)+padding

	if cut_end > (raw.last_samp-raw.first_samp)/fs: # block not long enough for croping
		print(f'\n{splitline}\n WARNING: block is not long enough for cropping \n{splitline}\n')
		return raw
	else:
		return raw.crop(tmin=cut_start, tmax=cut_end)


def _apply_filters(raw, h_pass, l_pass, notch):

	print(f"---High-pass + low-pass FIR filter: {h_pass}-{l_pass} Hz, automatic transition width---")
	raw.filter(
		l_freq=h_pass,
		h_freq=l_pass,
		picks="meg",
		method="fir",
		fir_design="firwin",
		fir_window="hamming",
		l_trans_bandwidth='auto',
		h_trans_bandwidth='auto',
		phase="zero",
		n_jobs=-1,
	)

	if notch:
		print("---Band-stop filter: 49-51 Hz, 4th-order Butterworth, forward-backward---")
		raw.filter(
			l_freq=51,
			h_freq=49,
			picks="meg",
			method="iir",
			iir_params=dict(
				order=4,
				ftype="butter",
			),
			phase="zero",
			n_jobs=-1,
		)
	
	return raw

def _apply_maxwell(raw_tmp, subject_id, mean_d_idx=None):

	#cal & cross talk files specific to system
	calibration_file =f'{FABI_DIR}/utils/sss_cal.dat'
	cross_talk_file = f'{FABI_DIR}/utils/ct_sparse.fif'
	#+1 needed as blocks are indexed starting by 1 as destin_file = f'{FABI_DIR}/utils/ct_sparse.fif'ation for maxfilter
	if mean_d_idx:
		destination = Raw.get_fif_filename(subject_id=subject_id, run_nr=mean_d_idx+1) # <------------------- This is used, when we use the mean head position (within each subject) as destination
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n using SUBJ BLOCK MEAN HEAD POSITION \n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
	else:
		destination = (0., 0., 0.04)
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n using ELEKTA STANDARD HEAD POSITION \n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	print(f"maxfilter destination: {destination}")  

	#find bad channels first
	print(f"---Now doing: find bad channels first---")  
	noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw_tmp,
																	calibration=calibration_file,
																	cross_talk=cross_talk_file,  # noqa
																	)
	raw_tmp.info['bads'] = noisy_chs + flat_chs                   

	#correct everything here
	print(f"---Now doing: correct everything here---")
	raw_tmp = mne.preprocessing.maxwell_filter(raw_tmp,
												calibration=calibration_file,
												cross_talk=cross_talk_file,
												destination=destination,  # noqa
											   )
	return raw_tmp

def _rename_bio_channels(raw):
	raw.set_channel_types({	'BIO001': 'eog',
					   		'BIO002': 'eog',
					   		'BIO003': 'ecg',})

	mne.rename_channels(raw.info, {'BIO001': 'EOG001',
								   'BIO002': 'EOG002',
								   'BIO003': 'ECG003',})
	return raw

class Raw(RawTemplate):
	sinuhe_root = SINUHE_DATA
	study_acronym = SINUHE_STUDY
	file_glob_patterns = ['%s_block%02d.fif',
						  '%s_block%d.fif']

	def run_cleaner_ica_part1(subject_id, 
					maxfilter = True,
					ica = True,
					ica_out_root = ICA_DIR,
					ica_method = "picard",
					fit_params = None,
					ica_threshold = 0.5,
					eog = True,
					ecg = True,
					train_thresh = 2,
					train_freq = 16.7,
					n_components = 50,
					notch = True,
					downsample_f=None,
					l_pass = 99,
					h_pass = 0.5,
					overwrite = True, 
					use_mean_headpos = False):
		'''
		Run only part of the cleaer: get ICA compoents, save them but don't apply them. 
		Goal: Check all components, then add missing components manually to the list and then apply ICA in a second step. 
		'''
		print(f"---------------------------------------------\n now doing: Raw.run_cleaner_ica_part1 \n---------------------------------------------", flush=True)
		
		_, ica_outFiles = get_outFilePaths(subject_id, ica_out_root)

		if ica_outFiles['file_raw'].exists(): # and not overwrite:
			
			print(f"Continuous raw file already exists for {subject_id} at {ica_outFiles['file_raw']}. Skipping maxfilter and filtering steps.", flush=True)
			raw = mne.io.read_raw(ica_outFiles['file_raw'], preload=True)

		else:
			if subject_id == '19970520smsr':
				print(f"Using the adapted pipeline for subject '19970520smsr' because the behavioral data is missing for one block:", flush=True)
				n_blocks = 11
				block_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12] # 11th block is missing behav data
			else:
				n_blocks = Raw.get_number_of_runs(subject_id)
				block_indices = np.arange(1, n_blocks + 1)
			print('block_indices:', block_indices, flush=True)
			#get average head pos
			block_pos_l = []

			if use_mean_headpos:
				for block in block_indices:
					raw = Raw(subject_id, block_nr=block, preload=False)

					block_pos_l.append(raw.info["dev_head_t"]['trans'][:3, 3])

				blocks_pos = np.array(block_pos_l)
				print(f'Blocks positions: {blocks_pos}', flush=True)
				all_distances = np.sqrt(blocks_pos[:,0]**2 + blocks_pos[:,1]**2 + blocks_pos[:,2]**2)
				mean_distance = np.median(all_distances)
				mean_d_idx = (np.abs(all_distances - mean_distance)).argmin()
			else:
				mean_d_idx = None
					
			raw_all, first_samples = [], []

			for block in block_indices:
				raw_tmp = Raw(subject_id, block_nr=block, preload=True)
				raw_tmp = _crop_end_of_block(raw_tmp, padding=1, padding_stim=3)	

				if maxfilter:
					raw_tmp = _apply_maxwell(raw_tmp, subject_id, mean_d_idx)
					first_samples.append(raw_tmp.first_samp)

				# append and concatenate files
				raw_all.append(raw_tmp) 
				print('length of raw_all:', len(raw_all), flush=True)
			raw = mne.concatenate_raws(raw_all, on_mismatch='warn')

			if 'BIO003' in raw.ch_names: #sometimes this information is not correctly saved
				raw = _rename_bio_channels(raw)

			raw = _apply_filters(raw, h_pass, l_pass, notch)

		raw, ica_obj, ica_components_dict = run_my_ica_part1(
			raw=raw,
			subject_id=subject_id,
			out_root=ica_out_root,
			n_components=n_components,
			method=ica_method,
			fit_params = fit_params,
			random_state=42,
			ica_resample_freq=200,
			ica_hp_freq=1.0,
			ica_lp_freq=45.0,
			eog = eog,
			ecg = ecg,
			eog_corr_thresh=ica_threshold,
			ecg_corr_thresh=ica_threshold,
			train_freq=train_freq,
			train_thresh=train_thresh,
			surrogate_eog_chs=None,
			overwrite=overwrite,
		)

		return raw

	def run_cleaner_ica_part2(subject_id, 
					ica_out_root = ICA_DIR,
					downsample_f=None, 
					overwrite=True):

		print(f"---------------------------------------------\n now doing: Raw.run_cleaner_ica_part2 \n---------------------------------------------", flush=True)
		
		raw, ica_obj, ica_components_dict = run_my_ica_part2(
				subject_id=subject_id,
				out_root=ica_out_root,
				overwrite=overwrite,
			)

		if downsample_f != None:
			raw.resample(downsample_f, npad="auto")

		return raw
	

#%% ============================================================ blockwise



# ============================================================
# Add this to utils/raw.py inside or near your Raw class
# ============================================================

def _get_ncc_block_indices(subject_id):
	if subject_id == "19970520smsr":
		print(
			"Using adapted block list for 19970520smsr because behavioral data is missing for one block.",
			flush=True,
		)
		return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

	n_blocks = Raw.get_number_of_runs(subject_id)
	return list(np.arange(1, n_blocks + 1))


def _get_maxfilter_destination_idx(subject_id, block_indices):
	block_pos_l = []

	for block in block_indices:
		raw = Raw(subject_id, block_nr=block, preload=False)
		block_pos_l.append(raw.info["dev_head_t"]["trans"][:3, 3])

	blocks_pos = np.array(block_pos_l)
	print(f"Blocks positions: {blocks_pos}", flush=True)

	all_distances = np.sqrt(
		blocks_pos[:, 0] ** 2 +
		blocks_pos[:, 1] ** 2 +
		blocks_pos[:, 2] ** 2
	)

	median_distance = np.median(all_distances)
	median_list_idx = int((np.abs(all_distances - median_distance)).argmin())

	return median_list_idx


def _load_crop_maxfilter_filter_blocks(
	subject_id,
	maxfilter=True,
	notch=True,
	l_pass=99,
	h_pass=0.5,
	use_mean_headpos = False

):
	block_indices = _get_ncc_block_indices(subject_id)
	if use_mean_headpos:
		mean_d_idx = _get_maxfilter_destination_idx(subject_id, block_indices)
	else:
		mean_d_idx = None

	raw_blocks = []

	for block in block_indices:
		print(f"---------------------------------------------\n Loading block {block}\n---------------------------------------------", flush=True)

		raw_tmp = Raw(subject_id, block_nr=block, preload=True)
		raw_tmp = _crop_end_of_block(raw_tmp, padding=1, padding_stim=3)

		if maxfilter:
			raw_tmp = _apply_maxwell(raw_tmp, subject_id, mean_d_idx)

		if "BIO003" in raw_tmp.ch_names:
			raw_tmp = _rename_bio_channels(raw_tmp)

		raw_tmp = _apply_filters(raw_tmp, h_pass=h_pass, l_pass=l_pass, notch=notch)

		raw_blocks.append(raw_tmp)

	return raw_blocks, block_indices


def run_cleaner_blockwise_ica_part1(
	subject_id,
	maxfilter=True,
	ica=True,
	ica_out_root=ICA_BLOCK_DIR,
	ica_method="picard",
	fit_params=None,
	ica_threshold=0.5,
	eog = True,
	ecg = True,
	train_thresh = 2,
	train_freq = 16.7,
	n_components=50,
	notch=True,
	downsample_f=None,
	l_pass=99,
	h_pass=0.5,
	overwrite=True,
	use_mean_headpos = False
):
	print(f"---------------------------------------------\n now doing: Raw.run_cleaner_blockwise_ica_part1 \n---------------------------------------------", flush=True)

	raw_blocks, block_indices = _load_crop_maxfilter_filter_blocks(
		subject_id=subject_id,
		maxfilter=maxfilter,
		notch=notch,
		l_pass=l_pass,
		h_pass=h_pass,
		use_mean_headpos = use_mean_headpos
	)

	if ica:
		run_my_blockwise_ica_part1(
			raw_blocks=raw_blocks,
			subject_id=subject_id,
			block_indices=block_indices,
			out_root=ica_out_root,
			n_components=n_components,
			method=ica_method,
			fit_params=fit_params,
			random_state=42,
			ica_resample_freq=200,
			ica_hp_freq=1.0,
			ica_lp_freq=45.0,
			eog_corr_thresh=ica_threshold,
			ecg_corr_thresh=ica_threshold,
			train_freq=train_freq,
			train_thresh=train_thresh,
			surrogate_eog_chs=None,
			overwrite=overwrite,
		)

	return raw_blocks


def run_cleaner_blockwise_ica_part2(
	subject_id,
	ica_out_root=ICA_BLOCK_DIR,
	downsample_f=None,
	use_manual=True,
	use_automatic=False,
	overwrite=True,
):
	print(f"---------------------------------------------\n now doing: Raw.run_cleaner_blockwise_ica_part2 \n---------------------------------------------", flush=True)

	block_indices = _get_ncc_block_indices(subject_id)

	raw = run_my_blockwise_ica_part2(
		subject_id=subject_id,
		block_indices=block_indices,
		out_root=ica_out_root,
		use_manual=use_manual,
		use_automatic=use_automatic,
		overwrite=overwrite,
	)

	if downsample_f is not None:
		raw.resample(downsample_f, npad="auto")

	return raw


Raw.run_cleaner_blockwise_ica_part1 = staticmethod(run_cleaner_blockwise_ica_part1)
Raw.run_cleaner_blockwise_ica_part2 = staticmethod(run_cleaner_blockwise_ica_part2)
