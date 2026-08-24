import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import re
from glob import glob
import warnings
import joblib
import numpy as np

from utils.load_cfg import load_MRI_config_instance
from utils.compile import compile_SL_rdms_files
from utils.plots import *
from utils.subj import *
from utils.rsa import *

#%%

all_subjects = get_MRI_subjects()
#%%

def compile_rdms(
		subjectID = '19910823ssld',
		config_class_name = 'MRIconfig_C2'):


	print('[1] Loading configuration...')
	cfg = load_MRI_config_instance(config_class_name, subjectID)
	cfg.configure_paths()
	outFiles = cfg.get_outFile_names()
	inDir = cfg.outDir

	in_file = os.path.join(cfg.outDir, "empty_masks.txt")
	with open(in_file) as f:
		empty_masks = [int(line.strip()) for line in f if line.strip()]
	all_masks = list(map(int, np.arange(1, 61)))
	# ---------------------------------------
	# Locate partial result files
	# ---------------------------------------

	print('[2] Matching SL_rdms and eval_results files...')
	sl_rdms_pattern = re.sub(rf'({re.escape(cfg.prefix)}_partial_)\d{{1,2}}', r'\1*', os.path.basename(outFiles['SL_rdms_partial']))
	print(f"sl_rdms_pattern: {sl_rdms_pattern}")
	SL_rdms_files = sorted(glob(os.path.join(inDir, sl_rdms_pattern)))

	print(f'    -> Found {len(SL_rdms_files)} SL_Rdms files.')

	# Extract and validate mask numbers from filenames
	mask_numbers = []
	pattern = re.compile(rf'{cfg.prefix}_partial_(\d{{1,2}})')
	for file in SL_rdms_files:
		match = pattern.search(os.path.basename(file))
		if match:
			mask_numbers.append(int(match.group(1)))
		else:
			warnings.warn(f"Could not extract mask number from file: {file}")

	# Match and sort by mask number
	paired = sorted(zip(mask_numbers, SL_rdms_files))
	paired = [(m, s) for m, s in paired if m is not None]
	mask_numbers_sorted, SL_rdms_files_sorted = zip(*paired)

	# -------------------------------------------------
	# Verify that every mask is accounted for
	# -------------------------------------------------

	computed_masks = set(mask_numbers_sorted)
	empty_masks = set(empty_masks)
	all_masks = set(range(1, 61))

	missing_masks = all_masks - computed_masks - empty_masks

	if missing_masks:
		raise RuntimeError(
			f"The following partial masks are missing. "
			f"They are neither present as result files nor listed in "
			f"'empty_masks.txt': {sorted(missing_masks)}"
		)
	else:
		print("All partial masks are accounted for.")

	# (Optional) sanity checks
	unexpected_files = computed_masks - all_masks
	if unexpected_files:
		raise RuntimeError(
			f"Found result files for invalid mask numbers: {sorted(unexpected_files)}"
		)
	unexpected_empty = empty_masks - all_masks
	if unexpected_empty:
		raise RuntimeError(
			f"'empty_masks.txt' contains invalid mask numbers: {sorted(unexpected_empty)}"
		)

	all_SL_rdms = compile_SL_rdms_files(SL_rdms_files_sorted)

	# output from SL analysis before comparison with model RDMs (--> save outside of model loop)
	joblib.dump(all_SL_rdms, outFiles['SL_rdms'])

	save_RSA_outputs(cfg)
					 
	print('[Done] ')


for subjectID in all_subjects:
	thisConfig = 'MRIconfig_C2'
	compile_rdms(subjectID, thisConfig)