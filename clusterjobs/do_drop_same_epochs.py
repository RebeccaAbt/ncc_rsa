
#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import joblib
import mne
import numpy as np
from plus_slurm import Job
from utils.provenance import artifact_log_path, configure_subject_logging, record_artifact
#%%


def _saved_epochs_path(output_base):
	"""Return the main FIF path created by MNE for an output base path."""
	if os.path.isfile(output_base):
		return output_base
	if os.path.isfile(f'{output_base}.fif'):
		return f'{output_base}.fif'
	raised = FileNotFoundError(f'MNE did not create the expected output: {output_base}')
	raise raised

class DropEpochsFromEpo(Job):
	def run(self,
		 	subjectID, 
			cleanDir = MEG_CLEAN_EPOCHS_DIR,
			icaDir = os.path.join(MEG_CLEAN_EPOCHS_DIR, '../ica'),
			outDir =  MEG_CLEAN_EPOCHS_DIR,
			oldSuffix = 'clean-epo',
			newSuffix = 'maxfilter_True__ica_True__0.1-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1',
			newSuffix2 = '_meg-epo'
			):
		
		outSuffix = f'{newSuffix}_clean{newSuffix2}'
		inFile_old_clean = f'{cleanDir}/{subjectID}/{subjectID}_{oldSuffix}.fif'
		inFile_new_epochs = f'{icaDir}/{subjectID}/{subjectID}_{newSuffix}{newSuffix2}.fif'

		outSubjectDir = f'{outDir}/{subjectID}'
		os.makedirs(outSubjectDir, exist_ok=True)	# only relevant, if outDir is different than inDir	
		outFile_new_epos = f'{outDir}/{subjectID}/{subjectID}_{outSuffix}'
		logger, _ = configure_subject_logging(
			f'{outFile_new_epos}.fif',
			subjectID,
			upstream_log=artifact_log_path(inFile_new_epochs),
		)
		logger.info('Starting noisy-trial removal from epoch data')
		logger.info('Source of exclusions: old epochs file %s', inFile_old_clean)
		logger.info('Epochs to clean: %s', inFile_new_epochs)

		print(f"processing subject {subjectID}:\n\t loading old epochs:\t {inFile_old_clean} \n\t loading new epochs:\t {inFile_new_epochs}\n\t destination:\t\t {outFile_new_epos}")
		epochs_old_clean = mne.read_epochs(inFile_old_clean, preload=False, verbose=False)
		epochs_new = mne.read_epochs(inFile_new_epochs, preload=False, verbose=False)

		old_keep = epochs_old_clean.selection
		new_selection = epochs_new.selection

		excluded_trial_ids = np.setdiff1d(new_selection, old_keep)
		drop_positions = np.flatnonzero(np.isin(new_selection, excluded_trial_ids))

		epochs_new_clean = epochs_new.drop(drop_positions, reason='man_cleaned')


		print(
			f'{subjectID}: '
			f'old clean = {len(epochs_old_clean)}, '
			f'new before = {len(epochs_new)}, '
			f'new after = {len(epochs_new_clean)}'
		)

		epochs_new_clean.save(outFile_new_epos, overwrite=True)
		saved_output = _saved_epochs_path(outFile_new_epos)
		record_artifact(
			output_path=saved_output,
			operation_name='DropEpochsFromEpo.run',
			parameters={
				'subjectID': subjectID,
				'cleanDir': cleanDir,
				'icaDir': icaDir,
				'outDir': outDir,
				'oldSuffix': oldSuffix,
				'newSuffix': newSuffix,
				'newSuffix2': newSuffix2,
				'number_before': len(epochs_new),
				'number_after': len(epochs_new_clean),
				'excluded_trial_ids': excluded_trial_ids,
				'excluded_event_rows': epochs_new.events[drop_positions],
			},
			input_paths=[inFile_old_clean, inFile_new_epochs],
		)
		logger.info('Excluded original trial IDs: %s', excluded_trial_ids.tolist())
		logger.info('Wrote cleaned epochs and provenance: %s', saved_output)

class DropEpochsFromTxt(Job):
	def run(self,
		 	subjectID, 
			cleanDir = MEG_CLEAN_EPOCHS_DIR,
			icaDir = os.path.join(MEG_CLEAN_EPOCHS_DIR, '../ica'),
			outDir =  MEG_CLEAN_EPOCHS_DIR,
			epoSuffix = 'maxfilter_True__ica_True__0.1-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1', # 'maxfilter_True__ica_True__0.1-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1_bio-epo.fif''#
			epoSuffix2 = '_meg-epo',
			inSuffix = 'clean_epochs_idx.txt'
			):

		txtFile = os.path.join(cleanDir, f'{subjectID}_{inSuffix}')
		clean_epochs_idx = np.loadtxt(txtFile, dtype=int)

		inFile_new_epochs = f'{icaDir}/{subjectID}/{subjectID}_{epoSuffix}{epoSuffix2}.fif'
		epochs_new = mne.read_epochs(inFile_new_epochs, preload=False, verbose=False)

		outSubjectDir = f'{outDir}/{subjectID}'
		os.makedirs(outSubjectDir, exist_ok=True)	# only relevant, if outDir is different than inDir	
		outFile_new_epos = f'{outDir}/{subjectID}/{subjectID}_{epoSuffix}_clean{epoSuffix2}.fif'
		
		logger, _ = configure_subject_logging(
			outFile_new_epos,
			subjectID,
			upstream_log=artifact_log_path(inFile_new_epochs),
		)
		logger.info('Starting noisy-trial removal from exclusion index file')
		logger.info('Exclusion index file: %s', txtFile)
		logger.info('Epochs to clean: %s', inFile_new_epochs)
		
		# for some subjects, some epochs are already dropped in the ICA epochs, since we marked some artifacts in the continuous data that were automatically excluded from epoching
		current_original_idx = epochs_new.selection  # Original epoch indices that are still present after automatic dropping
		original_to_drop = np.setdiff1d(current_original_idx, clean_epochs_idx) # Original epoch indices that should be removed based on the previous manual cleaning
		drop_positions = np.flatnonzero(np.isin(current_original_idx, original_to_drop)) # Convert original epoch indices into positions in the current Epochs object

		epochs_new_clean = epochs_new.copy().drop(drop_positions,reason="man_cleaned")

		print(
		    f"{subjectID}: "
		    f"old clean = {len(clean_epochs_idx)}, "
		    f"new before = {len(epochs_new)}, "
		    f"new after = {len(epochs_new_clean)}, "
		    f"already auto-dropped = {len(epochs_new.drop_log) - len(epochs_new)}"
		)

		epochs_new_clean.save(outFile_new_epos, overwrite=True)
		saved_output = _saved_epochs_path(outFile_new_epos)
		record_artifact(
			output_path=saved_output,
			operation_name='DropEpochsFromTxt.run',
			parameters={
				'subjectID': subjectID,
				'cleanDir': cleanDir,
				'icaDir': icaDir,
				'outDir': outDir,
				'epoSuffix': epoSuffix,
				'epoSuffix2': epoSuffix2,
				'inSuffix': inSuffix,
				'number_before': len(epochs_new),
				'number_after': len(epochs_new_clean),
				'excluded_trial_ids': original_to_drop,
				'excluded_event_rows': epochs_new.events[drop_positions],
			},
			input_paths=[txtFile, inFile_new_epochs],
		)
		logger.info('Excluded original trial IDs: %s', original_to_drop.tolist())
		logger.info('Wrote cleaned epochs and provenance: %s', saved_output)