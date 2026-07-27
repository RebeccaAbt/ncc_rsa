#%% imports
import sys
sys.path.append('/home/reabt/experiments/ncc/MEG/Fabi/utils/')
sys.path.append('/home/reabt/experiments/ncc/MEG/Fabi/')

import os
os.environ["FONTCONFIG_PATH"] = "/etc/fonts"

# from clusterjobs.meta_job import Job
from plus_slurm import Job
import joblib
import numpy as np
from utils.raw import Raw
from utils.epochs import  get_epochs_R
from pymatreader import read_mat
import neurokit2 as nk
import mne


#%%

class SaveEpochs(Job):
	
	job_data_folder = 'epochs_icaTest'

	#%% the run method starts here
	def run(self, 
			subject_id,
			run_part = 0, # 0: both oarts, 1: only part 1, 2: only part 2
		 	job_data_folder = 'epochs_icaTest', # <============= !!!
			maxfilter=True,
			ica=True,
			ica_out_root = "/home/reabt/experiments/ncc/MEG/data/ica_qc",
			l_pass=99,
			h_pass=0.5,
			epochs_settings={
							 'tmin': -1.5,
							 'tmax': 1.5,
							 'baseline': None,
							 'preload': True},
			):
		
		_, ica_outFile = get_outFilePaths(subject_id, ica_out_root)
		
		os.makedirs(f'/home/reabt/experiments/ncc/MEG/data/{self.job_data_folder}/{subject_id}/', exist_ok=True)

		suffix =  f'maxfilter_{maxfilter}__ica_{ica}__{h_pass}-{l_pass}Hz__fs_1000__[{epochs_settings['tmin']}_{epochs_settings['tmax']}]s'# _meg-epo.dat'
		meg_outfile = f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_meg-epo.fif'

		if os.path.isfile(meg_outfile):
			print(f"Epochs file already exists for {subject_id} with settings: {suffix}. Skipping computation.")
			return
		
		print(f"Running preprocessing for {subject_id} in preproc.py")

		preproc_settings = {'maxfilter': maxfilter,
						'downsample_f': None,
						'notch': True,
						'l_pass': l_pass,
						'h_pass': h_pass}
		
		subject_id_short = subject_id[8:]
		event_info = read_mat(f'/home/reabt/experiments/ncc/MEG/data/behav/{subject_id_short}.mat')['data']

		print(f"---------------------------------------------\n Now doing: Raw.run_cleaner \n---------------------------------------------")

		# data_raw = Raw.run_cleaner_andiStyle(subject_id,
		# 						   ica=ica,
		# 						   ica_threshold=0.4,
		# 						   **preproc_settings
		# 						   )

		data_raw = Raw.run_cleaner_new(subject_id,
								   ica=ica,
								   ica_threshold=0.4,
								   **preproc_settings
								   )
		
		print(f"---------------------------------------------\n finished doing: Raw.run_cleaner \n---------------------------------------------", flush=True)
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
		
		bio_clean = nk.bio_process(ecg=bio_df['ECG003'], 
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


		print(f"---------------------------------------------\n Now doing:  epochs_bio\n---------------------------------------------", flush=True)

		epochs_bio, events = get_epochs_R(bio_clean_raw, event_info, epochs_settings)
		print('--------------------\n saving Epochs .... \n--------------------\n', flush=True)

		epochs_bio.save(f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_bio-epo.fif')
		# joblib.dump(events, f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_bio-events.pkl')


		meg_raw = data_raw.pick(['meg', 'stim'])        
		print(f"---------------------------------------------\n Now doing:  epochs_meg \n---------------------------------------------")

		epochs_meg, events = get_epochs_R(meg_raw, event_info, epochs_settings)
		print('--------------------\n saving Epochs .... \n--------------------\n', flush=True)
		epochs_meg.save(f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}/{subject_id}_{suffix}_meg-epo.fif')
		joblib.dump(events, f'/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}_events.pkl')



