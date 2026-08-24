#%% improved ICA block

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from almkanal.almkanal_steps.ica_utils import run_ica

import matplotlib
import mne
matplotlib.use("Agg")
from pathlib import Path
import json
import matplotlib.pyplot as plt


def _as_unique_int_list(x):
	if x is None:
		return []
	return sorted(set([int(v) for v in x]))


def _save_ica_figures(raw_for_ica, raw_cleaned, ica, components_dict, out_dir, subject_id):
	out_dir = Path(out_dir)
	fig_dir = out_dir / "ica_figures"
	fig_dir.mkdir(parents=True, exist_ok=True)

	all_bad_components = sorted(
		set(
			int(c)
			for comps in components_dict.values()
			for c in comps
		)
	)

	if len(all_bad_components) == 0:
		print("No ICA components were marked as bad. No component-specific ICA figures saved.", flush=True)
		return

	fig = ica.plot_components(
		picks=all_bad_components,
		show=False,
		title=f"{subject_id}: rejected ICA component topographies",
	)

	if isinstance(fig, list):
		for i, cur_fig in enumerate(fig):
			cur_fig.savefig(fig_dir / f"{subject_id}_ica_rejected_topographies_page-{i:02d}.png", dpi=200)
			plt.close(cur_fig)
	else:
		fig.savefig(fig_dir / f"{subject_id}_ica_rejected_topographies.png", dpi=200)
		plt.close(fig)

	ica_sources = ica.get_sources(raw_for_ica)
	source_data = ica_sources.get_data()
	times = ica_sources.times

	for artifact_name, components in components_dict.items():
		components = sorted(set(int(c) for c in components))

		for comp in components:
			fig = ica.plot_components(
				picks=[comp],
				show=False,
				title=f"{subject_id}: ICA {comp} marked as {artifact_name}",
			)

			if isinstance(fig, list):
				fig = fig[0]

			fig.savefig(fig_dir / f"{subject_id}_ica-{comp:03d}_{artifact_name}_topography.png", dpi=200)
			plt.close(fig)

			fig, ax = plt.subplots(figsize=(12, 3))
			ax.plot(times, source_data[comp])
			ax.set_title(f"{subject_id}: ICA {comp} source time course ({artifact_name})")
			ax.set_xlabel("Time (s)")
			ax.set_ylabel("ICA source amplitude")
			fig.tight_layout()
			fig.savefig(fig_dir / f"{subject_id}_ica-{comp:03d}_{artifact_name}_source.png", dpi=200)
			plt.close(fig)

			try:
				figs = ica.plot_properties(
					raw_for_ica,
					picks=[comp],
					show=False,
				)

				for i, fig in enumerate(figs):
					fig.savefig(fig_dir / f"{subject_id}_ica-{comp:03d}_{artifact_name}_properties-{i:02d}.png", dpi=200)
					plt.close(fig)

			except Exception as err:
				print(
					f"Could not save plot_properties for ICA {comp} ({artifact_name}): {err}",
					flush=True,
				)

	try:
		fig = raw_cleaned.compute_psd().plot(show=False)
		fig.savefig(fig_dir / f"{subject_id}_raw_psd.png", dpi=200)
		plt.close(fig)

	except Exception as err:
		print(f"Could not save PSD figure: {err}", flush=True)



#%%

def run_my_ica_part1(
	raw,
	subject_id,
	out_root=ICA_DIR,
	n_components=50,
	method="picard",
	random_state=42,
	ica_resample_freq=200,
	ica_hp_freq=1.0,
	ica_lp_freq=40.0,
	eog_corr_thresh=0.5,
	ecg_corr_thresh=0.5,
	train_freq=16,
	train_thresh=3.0,
	surrogate_eog_chs=None,
	overwrite=True,
):
	
	out_dir = Path(out_root)

	raw_pre_ica_fname = out_dir / 'step1_raw_continuous'/ f"{subject_id}_raw.fif"
	raw_ica_fit_fname = out_dir / 'step2_raw_for_ica' / f"{subject_id}_filtered_ds_raw.fif"
	ica_fname = out_dir / 'step3_ica_comps' / f"{subject_id}_ica.fif"
	rejectedComps_fname = out_dir / 'step4_rejected_comps' / f"{subject_id}_ica_rejected_comps.json"
	metadata_fname = out_dir / 'step5_ica_metadata' / f"{subject_id}_ica_metadata.json"
	raw_after_ica_fname = out_dir / 'step6_raw_after_ica' / f"{subject_id}_after-ica_raw.fif"

	for fname in [raw_pre_ica_fname, raw_ica_fit_fname, ica_fname, rejectedComps_fname, metadata_fname, raw_after_ica_fname]:
		fname.parent.mkdir(parents=True, exist_ok=True)
	
	# 1) Save the original raw object before ICA
	raw.save(raw_pre_ica_fname, overwrite=overwrite)

	# 3 + 4) Fit ICA and detect EOG/ECG/train components using AlmKanal logic.
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
		eog=True,
		surrogate_eog_chs=surrogate_eog_chs,
		eog_corr_thresh=eog_corr_thresh,
		ecg=True,
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
		set(
			int(c)
			for comps in components_dict.values()
			for c in comps
		)
	)

	ica.exclude = rejected_components

	# 3) Save ICA object
	ica.save(ica_fname, overwrite=overwrite)

	# Save metadata in a human-readable format
	metadata = {
		"subject_id": subject_id,
		"raw_pre_ica_fname": str(raw_pre_ica_fname),
		"raw_ica_fit_fname": str(raw_ica_fit_fname),
		"ica_fname": str(ica_fname),
		"raw_after_ica_fname": str(raw_after_ica_fname),
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

	with open(metadata_fname, "w") as f:
		json.dump(metadata, f, indent=4)

	with open(rejectedComps_fname, "w") as f:
		json.dump(components_dict, f, indent=4)

	return raw, ica, components_dict#, rejected_components


def run_my_ica_part2(
	raw,
	subject_id,
	out_root=ICA_DIR,
	overwrite=True,
):
	
	out_dir = Path(out_root)

	raw_pre_ica_fname = out_dir / 'step1_raw_continuous'/ f"{subject_id}_raw.fif"
	raw_ica_fit_fname = out_dir / 'step2_raw_for_ica' / f"{subject_id}_filtered_ds_raw.fif"
	ica_fname = out_dir / 'step3_ica_comps' / f"{subject_id}_ica.fif"
	rejectedComps_fname = out_dir / 'step4_rejected_comps' / f"{subject_id}_ica_rejected_comps.json"
	metadata_fname = out_dir / 'step5_ica_metadata' / f"{subject_id}_ica_metadata.json"
	raw_after_ica_fname = out_dir / 'step6_raw_after_ica' / f"{subject_id}_after-ica_raw.fif"

	# ---------------------------------------------------
	
	raw = mne.io.read_raw_fif(raw_pre_ica_fname)	# 1) Load raw
	ica = mne.preprocessing.read_ica(ica_fname)		# 2) load ICA
	with open(rejectedComps_fname, "r") as f:		# 3) load  rejected components dict
		components_dict = json.load(f)
	# ---------------------------------------------------

	rejected_components = sorted(
		set(
			int(c)
			for comps in components_dict.values()
			for c in comps
		))

	ica.exclude = rejected_components

	# Save ICA object
	ica.save(ica_fname, overwrite=overwrite)

	# Apply ICA to the original raw object
	raw.info["description"] = (
		f"# excluded components: {len(rejected_components)}; "
		f"excluded ICA: {rejected_components}; "
		f"components by type: {components_dict}"
	)

	ica.apply(raw, exclude=rejected_components)

	# # Save cleaned raw
	raw.save(raw_after_ica_fname, overwrite=overwrite)

	return raw, ica, components_dict
