#%% improved ICA block
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import re
from almkanal.almkanal_steps.ica_utils import run_ica

import matplotlib
matplotlib.use("Agg")
import mne
from pathlib import Path
import json
import matplotlib.pyplot as plt

def get_outFilePaths(subject_id, out_root=ICA_DIR):
	out_dir = Path(out_root)
	base_data_dir = Path(ICA_DIR)
	file1_raw = base_data_dir / 'step1_raw_continuous'/ f"{subject_id}_raw.fif"
	file2_ica = out_dir / 'step2_ica_comps' / f"{subject_id}_ica.fif"
	file3_icaData = out_dir / 'step2_ica_tmp_data' / f"{subject_id}_ica_data.fif"
	file4_rejected = out_dir / 'step3_rejected_comps' / f"{subject_id}_ica_rejected_comps.json"
	file5_topos = out_dir / 'step3_comps_topos' / f"{subject_id}_ica_topos.png"
	file6_metadata = out_dir / 'step4_ica_metadata' / f"{subject_id}_ica_metadata.json"
	file7_raw_clean = out_dir / 'step5_raw_after_ica' / f"{subject_id}_after-ica_raw.fif"

	for fname in [file1_raw, file2_ica, file3_icaData, file4_rejected, file5_topos, file6_metadata, file7_raw_clean]:
		fname.parent.mkdir(parents=True, exist_ok=True)
	
	return out_dir, {'file_raw': file1_raw,
						'file_ica': file2_ica,
						'file_icaData': file3_icaData,
						'file_rejected': file4_rejected,
						'file_topos': file5_topos,
						'file_metadata': file6_metadata,
						'file_raw_clean': file7_raw_clean}


def _as_unique_int_list(x):
	if x is None:
		return []
	return sorted(set([int(v) for v in x]))

def run_my_ica(
	raw,
	subject_id,
	out_root=ICA_DIR,
	n_components=50,
	method="picard",
	random_state=42,
	ica_resample_freq=200,
	ica_hp_freq=1.0,
	ica_lp_freq=45.0,
	eog=True,
	ecg=True,
	eog_corr_thresh=0.5,
	ecg_corr_thresh=0.5,
	train_freq=16.7,
	train_thresh=3.0,
	surrogate_eog_chs=None,
	overwrite=True,
):

	out_dir, outFiles = get_outFilePaths(subject_id, out_root)

	# Save the original raw object before ICA
	raw.save(outFiles['file_raw'], overwrite=overwrite)

	tmp_ica_raw = raw.copy().filter(l_freq = ica_hp_freq, h_freq=ica_lp_freq).resample(200, npad="auto")
	tmp_ica_raw.save(outFiles['file_icaData'], overwrite=overwrite)
	del tmp_ica_raw

	# Fit ICA and detect EOG/ECG/train components using AlmKanal logic.
	# fit_only=True prevents run_ica from applying automatically.
	_, ica, components_dict, eog_scores, ecg_scores = run_ica(
		raw,
		fit_only=True,
		n_components=n_components,
		method=method,
		random_state=random_state,
		fit_params=None,
		resample_freq=ica_resample_freq,
		ica_hp_freq=ica_hp_freq,
		ica_lp_freq=ica_lp_freq,
		eog=eog,
		surrogate_eog_chs=surrogate_eog_chs,
		eog_corr_thresh=eog_corr_thresh,
		ecg=ecg,
		ecg_corr_thresh=ecg_corr_thresh,
		emg=False,
		train=True,
		train_freq=train_freq,
		train_thresh=train_thresh,
	)

	components_dict = {
		key: _as_unique_int_list(vals)
		for key, vals in components_dict.items()
	}

	rejected_components = sorted(
		set(int(c)
			for comps in components_dict.values()
			for c in comps
		))

	ica.exclude = rejected_components

	# 3) Save ICA object
	ica.save(outFiles['file_ica'], overwrite=overwrite)

	# Save metadata in a human-readable format
	metadata = {
		"subject_id": subject_id,
		"file1_raw": str(outFiles['file_raw']),
		"file2_ica": str(outFiles['file_ica']),
		"file7_raw_clean": str(outFiles['file_raw_clean']),
		"n_components": n_components,
		"method": method,
		"random_state": random_state,
		"ica_resample_freq": ica_resample_freq,
		"ica_hp_freq": ica_hp_freq,
		"ica_lp_freq": ica_lp_freq,
		"eog_corr_thresh": eog_corr_thresh,
		"ecg_corr_thresh": ecg_corr_thresh,
		"train_freq": train_freq,
		"train_thresh": train_thresh,
		"components_dict": components_dict,
		"rejected_components": rejected_components,
	}

	with open(outFiles['file_metadata'], "w") as f:
		json.dump(metadata, f, indent=4)

	compFile_text = {
		# "automatic": components_dict,
		"manual": components_dict | {"other": [],}}

	text = json.dumps(compFile_text, indent=4)
	text = re.sub(r'\[\s*([\d,\s]+)\s*\]', lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text,) # Collapse lists containing only integers

	with open(outFiles["file_rejected"], "w") as f:
		f.write(text)


	# Apply ICA to the original raw object
	raw.info["description"] = (
		f"# excluded components: {len(rejected_components)}; "
		f"excluded ICA: {rejected_components}; "
		f"components by type: {components_dict}"
	)

	ica.apply(raw, exclude=rejected_components)

	# Save cleaned raw
	raw.save(outFiles['file_raw_clean'], overwrite=overwrite)

	return raw, ica, components_dict, rejected_components


def run_my_ica_part1(
	raw,
	subject_id,
	out_root=ICA_DIR,
	n_components=50,
	method="picard",
	fit_params = None,
	random_state=42,
	ica_resample_freq=200,
	ica_hp_freq=1.0,
	ica_lp_freq=45.0,
	eog = True,
	ecg = True,
	eog_corr_thresh=0.5,
	ecg_corr_thresh=0.5,
	train_freq=16.7,
	train_thresh=3.0,
	surrogate_eog_chs=None,
	overwrite=True,
):
	
	out_dir, outFiles = get_outFilePaths(subject_id, out_root)

	if outFiles['file_ica'].exists() and not overwrite:
		print(f"ICA file already exists for subject {subject_id} at {outFiles['file_ica']}. Skipping ICA fitting.", flush=True)
		return None, None, None
	
	# 1) Save the original raw object before ICA

	if not outFiles['file_raw'].exists():# or overwrite:
		raw.save(outFiles['file_raw'], overwrite=overwrite)

	# tmp_ica_raw = raw.copy().load_data().filter(l_freq = None, h_freq=40).resample(200, npad="auto")
	tmp_ica_raw = raw.copy()
	tmp_ica_raw.load_data()
	tmp_ica_raw.filter(l_freq=ica_hp_freq, h_freq=ica_lp_freq)
	tmp_ica_raw.resample(200, npad="auto")

	print('----------------------------------------------\n saving tmp raw data for ICA fitting. \n---------------------------------------------', flush=True)
	tmp_ica_raw.load_data().save(outFiles['file_icaData'], overwrite=True)
	print('----------------------------------------------\n deleting tmp raw data for ICA fitting. \n---------------------------------------------', flush=True)
	del tmp_ica_raw

	print(f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n train_thresh: {train_thresh} \n train_freq: {train_freq} \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n')
	
	# 3 + 4) Fit ICA and detect EOG/ECG/train components using AlmKanal logic.
	# fit_only=True prevents run_ica from applying automatically.
	_, ica, components_dict, eog_scores, ecg_scores = run_ica(
		raw,
		fit_only=True,
		n_components=n_components,
		method=method,
		random_state=random_state,
		fit_params=fit_params,
		resample_freq=ica_resample_freq,
		ica_hp_freq=ica_hp_freq,
		ica_lp_freq=ica_lp_freq,
		eog=eog,
		surrogate_eog_chs=surrogate_eog_chs,
		eog_corr_thresh=eog_corr_thresh,
		ecg=ecg,
		ecg_corr_thresh=ecg_corr_thresh,
		emg=False,
		train=True,
		train_freq=train_freq,
		train_thresh=train_thresh,
	)
	print('components dict:', components_dict)
	print('eog_scores:', eog_scores)
	print('ecg_scores:', ecg_scores)
	components_dict = {
		key: _as_unique_int_list(vals)
		for key, vals in components_dict.items()
	}

	rejected_components = sorted(
		set(
			int(c)
			for comps in components_dict.values()
			for c in comps
		)
	)

	ica.exclude = rejected_components

	# 3) Save ICA object
	ica.save(outFiles['file_ica'], overwrite=overwrite)

	topo_plots = ica.plot_components(nrows = 10, ncols = 5)
	fig_path = outFiles['file_topos']
	topo_plots.savefig(fig_path, dpi=500)
	plt.close(topo_plots)

	# Save metadata in a human-readable format
	metadata = {
		"subject_id": subject_id,
		"file1_raw": str(outFiles['file_raw']),
		"file2_ica": str(outFiles['file_ica']),
		"file5_raw_clean": str(outFiles['file_raw_clean']),
		"n_components": n_components,
		"method": method,
		"random_state": random_state,
		"ica_resample_freq": ica_resample_freq,
		"ica_hp_freq": ica_hp_freq,
		"ica_lp_freq": ica_lp_freq,
		"eog_corr_thresh": eog_corr_thresh,
		"ecg_corr_thresh": ecg_corr_thresh,
		"train_freq": train_freq,
		"train_thresh": train_thresh,
		"components_dict": components_dict,
		"rejected_components": rejected_components,
	}

	with open(outFiles['file_metadata'], "w") as f:
		json.dump(metadata, f, indent=4)


	compFile_text = {
		# "automatic": components_dict,
		"manual": components_dict | {"other": [],}}

	text = json.dumps(compFile_text, indent=4)
	text = re.sub(r'\[\s*([\d,\s]+)\s*\]', lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text,)	# Collapse lists containing only integers

	with open(outFiles["file_rejected"], "w") as f:
		f.write(text)
		
	return raw, ica, components_dict#, rejected_components


def run_my_ica_part2(
	subject_id,
	out_root=ICA_DIR,
	overwrite=True,
):
	
	out_dir, outFiles = get_outFilePaths(subject_id, out_root)

	# Load data:
	raw = mne.io.read_raw_fif(outFiles['file_raw'], preload=True)	# 1) Load raw
	ica = mne.preprocessing.read_ica(outFiles['file_ica'])		# 2) load ICA
	with open(outFiles['file_rejected'], "r") as f:		# 3) load  rejected components dict
		components_dict = json.load(f)["manual"]

	rejected_components = sorted(
		set(
			int(c)
			for comps in components_dict.values()
			for c in comps
		))

	ica.exclude = rejected_components

	# Save ICA object with newly updated rejected components
	# ica.save(outFiles['file_ica'], overwrite=overwrite)

	# Apply ICA to the original raw object
	raw.info["description"] = (
		f"# excluded components: {len(rejected_components)}; "
		f"excluded ICA: {rejected_components}; "
		f"components by type: {components_dict}"
	)

	ica.apply(raw, exclude=rejected_components)

	# # Save cleaned raw
	raw.save(outFiles['file_raw_clean'], overwrite=overwrite)

	return raw, ica, components_dict

#%% =================================================================================== blockwise

# ============================================================
# Add this to utils/ica.py
# ============================================================

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from almkanal.almkanal_steps.ica_utils import run_ica


def _as_unique_int_list(x):
	if x is None:
		return []
	return sorted(set([int(v) for v in x]))


def _block_key(block):
	return f"block_{int(block):02d}"


def get_blockwise_outFilePaths(
	subject_id,
	out_root=ICA_BLOCK_DIR,
	block_indices=None,
):
	out_dir = Path(out_root)

	files = {
		"file_rejected": out_dir / "step3_rejected_comps" / f"{subject_id}_blockwise_ica_rejected_comps.json",
		"file_metadata": out_dir / "step4_ica_metadata" / f"{subject_id}_blockwise_ica_metadata.json",
		"file_raw_clean": out_dir / "step5_raw_after_ica" / f"{subject_id}_blockwise_after-ica_raw.fif",
		"blocks": {},
	}

	if block_indices is not None:
		for block in block_indices:
			key = _block_key(block)
			files["blocks"][key] = {
				"file_raw": out_dir / "step1_raw_continuous" / f"{subject_id}_{key}_raw.fif",
				"file_ica": out_dir / "step2_ica_comps" / f"{subject_id}_{key}_ica.fif",
				"file_icaData": out_dir / "step2_ica_tmp_data" / f"{subject_id}_{key}_ica_data.fif",
				"file_topos": out_dir / "step3_comps_topos" / f"{subject_id}_{key}_ica_topos.png",
				"file_raw_clean": out_dir / "step5_raw_after_ica_blocks" / f"{subject_id}_{key}_after-ica_raw.fif",
			}

	for fname in [files["file_rejected"], files["file_metadata"], files["file_raw_clean"]]:
		fname.parent.mkdir(parents=True, exist_ok=True)

	for block_files in files["blocks"].values():
		for fname in block_files.values():
			fname.parent.mkdir(parents=True, exist_ok=True)

	return out_dir, files


def _empty_component_template():
	return {
		# "automatic": {
		# 	"eog": [],
		# 	"ecg": [],
		# 	"train": [],
		# },
		"manual": {
			"eog": [],
			"ecg": [],
			"train": [],
			"other": [],
		},
	}


def _load_or_create_blockwise_json(json_file, block_indices):
	if json_file.exists():
		with open(json_file, "r") as f:
			data = json.load(f)
	else:
		data = {}

	for block in block_indices:
		key = _block_key(block)
		if key not in data:
			data[key] = _empty_component_template()

		for source in ["manual"]: # ["automatic", "manual"]:
			if source not in data[key]:
				data[key][source] = {}

			for comp_type in ["eog", "ecg", "train"]:
				if comp_type not in data[key][source]:
					data[key][source][comp_type] = []

	return data



def _flatten_component_dict(component_dict):
	return sorted(
		set(
			int(component)
			for components in component_dict.values()
			for component in components
		)
	)

def run_my_blockwise_ica_part1(
	raw_blocks,
	subject_id,
	block_indices,
	out_root=ICA_BLOCK_DIR,
	n_components=50,
	method="picard",
	fit_params=None,
	random_state=42,
	ica_resample_freq=200,
	ica_hp_freq=1.0,
	ica_lp_freq=45.0,
	eog = True,
	ecg = True,
	eog_corr_thresh=0.5,
	ecg_corr_thresh=0.5,
	train_freq=16.7,
	train_thresh=2.0,
	surrogate_eog_chs=None,
	overwrite=True,
):
	_, outFiles = get_blockwise_outFilePaths(
		subject_id=subject_id,
		out_root=out_root,
		block_indices=block_indices,
	)

	compFile_text = _load_or_create_blockwise_json(
		outFiles["file_rejected"],
		block_indices,
	)

	metadata = {
		"subject_id": subject_id,
		"out_root": str(out_root),
		"n_components": n_components,
		"method": method,
		"random_state": random_state,
		"ica_resample_freq": ica_resample_freq,
		"ica_hp_freq": ica_hp_freq,
		"ica_lp_freq": ica_lp_freq,
		"eog_corr_thresh": eog_corr_thresh,
		"ecg_corr_thresh": ecg_corr_thresh,
		"train_freq": train_freq,
		"train_thresh": train_thresh,
		"blocks": {},
	}

	for raw, block in zip(raw_blocks, block_indices):
		key = _block_key(block)
		block_files = outFiles["blocks"][key]

		if block_files["file_ica"].exists() and not overwrite:
			print(f"ICA already exists for {subject_id}, {key}. Skipping.", flush=True)
			continue

		print(f"---------------------------------------------\n Fitting ICA for {subject_id}, {key}\n---------------------------------------------", flush=True)

		# raw.save(block_files["file_raw"], overwrite=overwrite)

		tmp_ica_raw = raw.copy().load_data()
		tmp_ica_raw.filter(l_freq=ica_hp_freq, h_freq=ica_lp_freq)
		tmp_ica_raw.resample(ica_resample_freq, npad="auto")
		tmp_ica_raw.save(block_files["file_icaData"], overwrite=overwrite)
		del tmp_ica_raw

		_, ica, components_dict, eog_scores, ecg_scores = run_ica(
			raw,
			fit_only=True,
			n_components=n_components,
			method=method,
			random_state=random_state,
			fit_params=fit_params,
			resample_freq=ica_resample_freq,
			ica_hp_freq=ica_hp_freq,
			ica_lp_freq=ica_lp_freq,
			eog=eog,
			surrogate_eog_chs=surrogate_eog_chs,
			eog_corr_thresh=eog_corr_thresh,
			ecg=ecg,
			ecg_corr_thresh=ecg_corr_thresh,
			emg=False,
			train=True,
			train_freq=train_freq,
			train_thresh=train_thresh,
		)

		print(f"--------------------------------- \n EOG : {eog} \n ECG: {ecg} \n---------------------------------", flush=True)
		components_dict = {
			comp_type: _as_unique_int_list(components)
			for comp_type, components in components_dict.items()
		}

		automatic_rejected = _flatten_component_dict(components_dict)
		ica.exclude = automatic_rejected

		ica.save(block_files["file_ica"], overwrite=overwrite)

		topo_fig = ica.plot_components(nrows=12, ncols=5)
		topo_fig.savefig(block_files["file_topos"], dpi=500)
		plt.close(topo_fig)

		# compFile_text[key]["automatic"] = {
		# 	"eog": components_dict.get("eog", []),
		# 	"ecg": components_dict.get("ecg", []),
		# 	"train": components_dict.get("train", []),
		# }

		compFile_text[key]["manual"] = {
			"eog": components_dict.get("eog", []),
			"ecg": components_dict.get("ecg", []),
			"train": components_dict.get("train", []),
		}


		# if "manual" not in compFile_text[key]:
		# 	compFile_text[key]["manual"] = {
		# 		"eog": [],
		# 		"ecg": [],
		# 		"train": [],
		# 	}

		metadata["blocks"][key] = {
			"block": int(block),
			"file_raw": str(block_files["file_raw"]),
			"file_ica": str(block_files["file_ica"]),
			"file_icaData": str(block_files["file_icaData"]),
			"file_topos": str(block_files["file_topos"]),
			# "automatic_components": compFile_text[key]["automatic"],
			"automatic_rejected_components": automatic_rejected,
		}

	with open(outFiles["file_rejected"], "w") as f:
		json.dump(compFile_text, f, indent=4)

	text = json.dumps(compFile_text, indent=4)
	text = re.sub(r'\[\s*([\d,\s]+)\s*\]', lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text,) # Collapse lists containing only integers

	with open(outFiles["file_rejected"], "w") as f:
		f.write(text)

	with open(outFiles["file_metadata"], "w") as f:
		json.dump(metadata, f, indent=4)

	return raw_blocks


def run_my_blockwise_ica_part2(
	subject_id,
	block_indices,
	out_root=ICA_BLOCK_DIR,
	use_manual=True,
	use_automatic=False,
	overwrite=True,
):
	_, outFiles = get_blockwise_outFilePaths(
		subject_id=subject_id,
		out_root=out_root,
		block_indices=block_indices,
	)

	with open(outFiles["file_rejected"], "r") as f:
		compFile_text = json.load(f)

	clean_blocks = []

	for block in block_indices:
		key = _block_key(block)
		block_files = outFiles["blocks"][key]

		print(f"---------------------------------------------\n Applying ICA for {subject_id}, {key}\n---------------------------------------------", flush=True)

		raw = mne.io.read_raw_fif(block_files["file_raw"], preload=True)
		ica = mne.preprocessing.read_ica(block_files["file_ica"])

		components_by_type = {
			"eog": [],
			"ecg": [],
			"train": [],
		}

		block_dict = compFile_text[key]

		rejected_components = []

		# if use_automatic:
		# 	rejected_components.extend(
		# 		int(c)
		# 		for comps in block_dict.get("automatic", {}).values()
		# 		for c in comps
		# 	)

		if use_manual:
			rejected_components.extend(
				int(c)
				for comps in block_dict.get("manual", {}).values()
				for c in comps
			)

		rejected_components = sorted(set(rejected_components))

		ica.exclude = rejected_components

		raw.info["description"] = (
			f"blockwise ICA; {key}; "
			f"# excluded components: {len(rejected_components)}; "
			f"excluded ICA: {rejected_components}; "
			f"manual components: {block_dict.get('manual', {})}; "
			# f"automatic components: {block_dict.get('automatic', {})}"
		)

		ica.apply(raw, exclude=rejected_components)

		raw.save(block_files["file_raw_clean"], overwrite=overwrite)
		clean_blocks.append(raw)

	raw_clean = mne.concatenate_raws(clean_blocks, on_mismatch="warn")
	raw_clean.save(outFiles["file_raw_clean"], overwrite=overwrite)

	return raw_clean
