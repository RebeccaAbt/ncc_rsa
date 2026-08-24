
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from genericpath import isdir
from obob_mne.raw import Raw as RawTemplate
import re
from utils.load_cfg import *

# Notiz an mich: files mit wildcard einfacher finden: files2load = list(Path(indir).glob('*/*_downsample_f_100__h_pass_1.dat'))

def _get_bad_subj():
	with open(f"{CODE_DIR}/utils/bad_subj.txt", "r") as f:
		return f.read().strip() #strip: removes '\n' 

def _unlist_cfg_name(MEG_config):
	if isinstance(MEG_config, list):
		return MEG_config[0]
	else:
		return MEG_config

def _remove_bad_subj(subjects, do_print=True):
	bad_subj = _get_bad_subj()
	for bs in bad_subj.split(','):
		if bs in subjects:
			subjects.remove(bs)
			if do_print: print(f'Removed bad subject: {bs}') # nicht sicher, ob es auch funktioniert, wenn inb dem file mehrere subjbects sind
	remove_sub = []
	for subj in subjects:
		if len(subj) != 12:
			remove_sub.append(subj)

	if len(remove_sub)>0:
		[subjects.remove(s) for s in remove_sub]
		if do_print: print(f'Removed bad subject: {remove_sub}')
	return subjects


def get_all_subjects(remove_bad=True, do_print=False, inDir = MRI_RAW_DIR):
	"""
	All subjects of which we have fMRI data
	But this only checks if the subject foilder exists. If thr first level data is not yet in the folder, this will cause problems.
	use "get_MRI_subjects()" instead
	"""  
	all_subjects = [name for name in os.listdir(inDir) if os.path.isdir(os.path.join(inDir, name))]

	if remove_bad:
		all_subjects = _remove_bad_subj(all_subjects)

	if do_print:
		print(f'all subjects: {all_subjects}')
	return all_subjects

# --------------------------------------------------------------------------------------------- MRI

def get_MRI_subjects(remove_bad=True, do_print=True, inDir = MRI_RAW_DIR):
	"""
	Get subjects of which we have MRI first-level data.
	"""

	
	all_subjects = get_all_subjects()

	MRI_subjects = []

	for n, subj in enumerate(all_subjects):
		thisDirectory = os.path.join(inDir, subj, 'NCC')

		if os.path.isdir(thisDirectory):

			contains_firstLevel = any(
				name.startswith('firstLevel') and os.path.isdir(os.path.join(thisDirectory, name))
				for name in os.listdir(thisDirectory)
			)
			if contains_firstLevel:
				MRI_subjects.append(subj)
	
	if do_print: print(f'\nall MRI subjects: {MRI_subjects}')

	if remove_bad:
		MRI_subjects = _remove_bad_subj(MRI_subjects)

	return MRI_subjects

def get_new_MRI_subjects(MRI_config, baseDir = f'{MRI_DATA_DIR}/rsa', remove_bad=True):
	"""
	Get subjects of which we have MRI first-level data but haven't computed the RSA of a given config yet.
	"""

	MRI_config = _unlist_cfg_name(MRI_config)
	outDir = os.path.join(baseDir, MRI_config)
	
	os.makedirs(outDir, exist_ok=True)

	old_subj = set([name for name in os.listdir(outDir) if os.path.isdir(os.path.join(outDir, name))])
	MRI_subj = set(get_MRI_subjects())

	subjectIDs = list(MRI_subj.difference(old_subj))
	print(f'\nnew MRI subjects: {subjectIDs}')

	if remove_bad:
		subjectIDs = _remove_bad_subj(subjectIDs)

	return subjectIDs

def get_old_MRI_subjects(config, baseDir = f'{MRI_DATA_DIR}/rsa', remove_bad=True):
	"""
	Get subjects of which have already computed the RSA of a given config.
	This is useful if there were problems with the cluster etc. and we are not sure 
	if all partial masks of RSA data of the old subjects were computed and we want to rerun them again.
	Since we expect most of the data to actually be there, most masks will be skipped and the duration 
	for the execution of those subjobs will be very short 
	
	So if we select a very short run time, the subjobs for partial masks were the data is actually 
	missing will have the timeout error, since actual computing needs to take place. 
	This is the behaviour we want to see!
	 
	Because we can then run the "analyse_errorLogs.py" script to check which jobs have timed out 
	(=files that were actually missing) and we can then use the output to compute those single missing 
	files using the "run_timeout_SL_crossnobis_partial.py" script.

	This approach should make it easier to keep an overview which jobs we are running for new subjects 
	and which jobs are more of a sanity check, or which jobs compute missing data
	"""
	outDir = os.path.join(baseDir, config)
	subjectIDs = [name for name in os.listdir(outDir) if os.path.isdir(os.path.join(outDir, name))]
	print(subjectIDs)
	print(f'\nold MRI subjects: {subjectIDs}')

	if remove_bad:
		subjectIDs = _remove_bad_subj(subjectIDs)

	return subjectIDs

# --------------------------------------------------------------------------------------------- MEG

class Raw(RawTemplate): # This needs to be adapted since I am now on the alpha SCC
	sinuhe_root = '/home/reabt/mnt/data/'
	study_acronym = 'aw_ncc'
	file_glob_patterns = ['%s_block%02d.fif',
						  '%s_block%d.fif']

def get_MEG_raw_subjects(remove_bad=True):
	all_subjects = Raw.get_all_subjects()
	if remove_bad:
			all_subjects = _remove_bad_subj(all_subjects)
			all_subjects = list(all_subjects)
			all_subjects.sort()
	return all_subjects
	
def get_new_MEG_raw_subjects():
	all_subjects = set(get_MEG_raw_subjects())
	processed_subj = set(get_MEG_subjects())
	subjectIDs = list(all_subjects.difference(processed_subj))
	return subjectIDs

def get_MEG_subjects(MEG_config='MEGconfig_C', remove_bad=True, do_print = True):
	"""
	Get subjects of which we have preprocessed MEG data.
	"""

	MEG_config = _unlist_cfg_name(MEG_config)
	cfg = load_MEG_config_instance(MEG_config)
	MEG_subjects = os.listdir(os.path.join(cfg.dataDir, cfg.dataFolder))

	if remove_bad:
		MEG_subjects = _remove_bad_subj(MEG_subjects)

	if do_print: print(f'\nMEG subjects: {MEG_subjects}')
	return MEG_subjects

def get_new_MEG_subjects(MEG_config, for_fusion = True, remove_bad=True):
	"""
	Get new MEG subjects that have preprocessed data but no RSA data
	"""

	MEG_config = _unlist_cfg_name(MEG_config)
	cfg = load_MEG_config_instance(MEG_config)

	MEG_files = os.listdir(os.path.join(cfg.dataDir, 'movie_RDMs', cfg.prefix))
	old_subj = set([re.search(r'\d{8}[A-Za-z]{4}', f).group(0) for f in MEG_files])
	if remove_bad:
			old_subj = _remove_bad_subj(old_subj)

	if for_fusion:
		print('Only including subjects for fusion...')
		all_subj = set(get_fusion_subjects())

	else:
		print('including all subjects, also non-fusion subjects')
		all_subj = set(get_MEG_subjects())
	
	subjectIDs = list(all_subj.difference(old_subj))

	print(f'\nnew MEG subjects: {subjectIDs}')

	return subjectIDs


def get_missing_MEG_subjects(remove_bad=True): 
	"""
	Get subjects that have MRI data but are missing MEG data.
	"""
	MRI_subjects = get_all_subjects(do_print = False)
	fusion_subjects = get_fusion_subjects()

	missing_MEG_subjects = []
	for subj in MRI_subjects:
		
		if subj not in fusion_subjects:
			missing_MEG_subjects.append(subj)

	if remove_bad:
		missing_MEG_subjects = _remove_bad_subj(missing_MEG_subjects)

	return missing_MEG_subjects
	
# --------------------------------------------------------------------------------------------- Fusion

def get_fusion_subjects(remove_bad=True, do_print=True):
	"""
	Get subjects that have both MRI and MEG data.
	"""
	MRI_subjects = get_all_subjects(do_print = False)
	MEG_subjects = get_MEG_subjects(do_print = False)

	fusion_subjects = []
	for subj in MRI_subjects:
		
		if subj in MEG_subjects:
			fusion_subjects.append(subj)

	if do_print: print(f'\nfusion subjects: {fusion_subjects}')

	if remove_bad:
		fusion_subjects = _remove_bad_subj(fusion_subjects, do_print=do_print)

	return fusion_subjects

def get_new_fusion_subjects(fusion_config, meanMEG, baseDir = FUSION_DATA_DIR, remove_bad=True):
	"""
	Get subjects where fusion is still missing 
	"""
	
	fusion_config = _unlist_cfg_name(fusion_config)
	cfg = load_fusion_config_instance(fusion_config)

	all_subj = set(get_fusion_subjects(do_print = False))
	fusionDir = os.path.join(baseDir, fusion_config, 'commonalities')
	os.makedirs(fusionDir, exist_ok=True)
	fusion_files = os.listdir(fusionDir)

	if meanMEG:
		pattern = r'(\d{8}[A-Za-z]{4})_fusion_mean\.pkl$'
	else:
		pattern = r'(\d{8}[A-Za-z]{4})_fusion\.pkl$'

	old_subj = [m.group(1) for f in fusion_files if (m := re.search(pattern, f))]
	subjectIDs = list(all_subj.difference(old_subj))

	missing_MEG = [s for s in subjectIDs if not os.path.exists(cfg.MEG_input.replace('*', '{}').format(s))]
	missing_MRI = [s for s in subjectIDs if not os.path.exists(cfg.MRI_input.replace('*', '{}').format(s))]

	valid_subj = [s for s in subjectIDs if s not in missing_MEG and s not in missing_MRI]
	
	if missing_MEG: print(f'\nmissing necessary MEG data: {missing_MEG}')
	if missing_MRI: print(f'\nmissing necessary MRI data: {missing_MRI}')

	print(f'\nvalid new subjects: {valid_subj}\n')

	return valid_subj