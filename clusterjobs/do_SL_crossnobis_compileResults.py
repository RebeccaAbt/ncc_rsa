#%%
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import re
from glob import glob
from collections import defaultdict
import warnings
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
from nilearn import plotting
import rsatoolbox as rsa

# Extend sys.path to import project modules

from utils.load_cfg import load_MRI_config_instance
from utils.compile import compile_SL_rdms_files, find_empty_masks
from utils.plots import *
from utils.rsa import *
from plus_slurm import Job
from utils.provenance import configure_subject_logging, record_artifact

#%%

class SL_crossnobis_compileResults(Job):
	def run(self,
			subjectID = '19910823ssld',
			config_class_name = 'MRIconfig_C'):
	
		print(
			" ----------------------------------------------------------------------------\n",
			"Looping over all partial masks and find searchlights, so we know which masks \n are supposed to be empty and which aren't\n",
			"----------------------------------------------------------------------------"
		)
		find_empty_masks(subjectID, config_class_name) # saves the empty mask indeices to a file

		print('[1] Loading configuration...')
		cfg = load_MRI_config_instance(config_class_name, subjectID)
		cfg.configure_paths()
		outFiles = cfg.get_outFile_names()
		inDir = cfg.outDir
		logger, _ = configure_subject_logging(outFiles['SL_rdms'], subjectID)

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
		if paired:
			mask_numbers_sorted, SL_rdms_files_sorted = zip(*paired)
		else:
			mask_numbers_sorted, SL_rdms_files_sorted = (), ()

		partial_searchlights = {}
		partial_config = None
		for mask_number, partial_file in paired:
			manifest_file = f'{partial_file}.provenance.json'
			if os.path.isfile(manifest_file):
				with open(manifest_file, encoding='utf-8') as file:
					manifest = json.load(file)
				partial_searchlights[mask_number] = manifest.get('parameters', {}).get('n_searchlights')
				if partial_config is None:
					partial_config = manifest.get('parameters', {}).get('config')
			else:
				partial_searchlights[mask_number] = None

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

		logger.info('Compiled fMRI crossnobis searchlights')
		logger.info('Configuration class: %s', config_class_name)
		logger.info('Configuration values from partial outputs: %s', partial_config or vars(cfg))
		logger.info('Searchlights per partial mask: %s', partial_searchlights)
		record_artifact(
			output_path=outFiles['SL_rdms'],
			operation_name='SL_crossnobis_compileResults.run',
			parameters={
				'config_class_name': config_class_name,
				'config': partial_config or vars(cfg),
				'subjectID': subjectID,
				'partial_searchlights': partial_searchlights,
				'empty_masks': sorted(empty_masks),
				'n_partial_files': len(SL_rdms_files_sorted),
			},
			input_paths=list(SL_rdms_files_sorted) + [in_file],
		)
						 
		print('[Done] ')