#%% imports
import os
os.environ["FONTCONFIG_PATH"] = "/etc/fonts"

import sys
from pathlib import Path
import joblib
import numpy as np
import mne
import neurokit2 as nk
from attrs import define
from plus_slurm import Job
from pymatreader import read_mat

sys.path.append("/home/reabt/experiments/ncc/MEG/Fabi/utils/")
sys.path.append("/home/reabt/experiments/ncc/MEG/Fabi/")

from obob_mne.raw import Raw as RawTemplate
from utils.epochs import get_epochs_R

from almkanal import AlmKanal, AlmKanalStep
from almkanal.almkanal_steps.ica_utils import ICA

#%% raw loader template
class NCCRaw(RawTemplate):
	sinuhe_root = "/home/reabt/mnt/data/"
	study_acronym = "aw_ncc"
	file_glob_patterns = [
		"%s_block%02d.fif",
		"%s_block%d.fif",
	]


#%% plus_slurm job
class SaveEpochsAlmKanal(Job):
	job_data_folder = "epochs_almkanal"

	def run(
		self,
		subject_id,
		job_data_folder="epochs_almkanal",
		maxfilter=True,
		ica=True,
		l_pass=99,
		h_pass=0.5,
		notch=True,
		downsample_f=None,
		epochs_settings={
			"tmin": -1.5,
			"tmax": 1.5,
			"baseline": None,
			"preload": True,
		},
		ica_threshold=0.4,
		overwrite=False,
	):
		out_dir = Path(f"/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}")
		out_dir.mkdir(parents=True, exist_ok=True)

		suffix = (
			f"maxfilter_{maxfilter}"
			f"__ica_{ica}"
			f"__{h_pass}-{l_pass}Hz"
			f"__fs_1000"
			f"__[{epochs_settings['tmin']}_{epochs_settings['tmax']}]s"
		)

		meg_outfile = out_dir / f"{subject_id}_{suffix}_meg-epo.fif"

		if meg_outfile.is_file() and not overwrite:
			print(
				f"Epochs file already exists for {subject_id} with settings: "
				f"{suffix}. Skipping computation.",
				flush=True,
			)
			return

		subject_id_short = subject_id[8:]
		event_info = read_mat(
			f"/home/reabt/experiments/ncc/MEG/data/behav/{subject_id_short}.mat"
		)["data"]

		raw_pre_ica_fname = out_dir / f"{subject_id}_{suffix}_before-ica_raw.fif"
		raw_after_ica_fname = out_dir / f"{subject_id}_{suffix}_after-ica_raw.fif"

		loader = LoadNCCBlocks(subject_id=subject_id, preload=True)
		load_result = loader.run(data=None, info={})

		raw_blocks = load_result["data"]

		for raw in raw_blocks:
			raw._inst_type = "raw"

		maxwell_destination = load_result["load_info"]["destination"]

		raw_steps = [
			NCCMaxwell(enabled=maxfilter, destination=maxwell_destination),
			NCCConcatenateBlocks(on_mismatch="warn"),
			NCCRenameBioChannels(),
			NCCBandpassFilter(h_pass=h_pass, l_pass=l_pass),
			NCCNotchFilter(enabled=notch),
		]

		# raw_steps = [
		# 	LoadNCCBlocks(subject_id=subject_id, preload=True),
		# 	NCCMaxwell(enabled=maxfilter),
		# 	NCCConcatenateBlocks(on_mismatch="warn"),
		# 	NCCRenameBioChannels(),
		# 	NCCBandpassFilter(h_pass=h_pass, l_pass=l_pass),
		# 	NCCNotchFilter(enabled=notch),
		# ]

		if ica:
			raw_steps.extend([
				NCCSaveRaw(fname=str(raw_pre_ica_fname), overwrite=True),
				ICA(
					fit_only=False,
					n_components=50,
					method="picard",
					random_state=42,
					fit_params=None,
					ica_hp_freq=1.0,
					ica_lp_freq=40.0,
					resample_freq=200,
					eog=True,
					eog_corr_thresh=ica_threshold,
					surrogate_eog_chs=None,
					ecg=True,
					ecg_corr_thresh=ica_threshold,
					emg=False,
					train=True,
					train_freq=16,
					train_thresh=3.0,
				),
				NCCSaveRaw(fname=str(raw_after_ica_fname), overwrite=True),
			])

		if downsample_f is not None:
			raw_steps.append(
				NCCResampleRaw(sfreq=downsample_f)
			)

		raw_pipeline = AlmKanal(
			steps=raw_steps,
			pick_params=None,
		)

		print("---------------------------------------------")
		print("Now running raw AlmKanal pipeline")
		print("---------------------------------------------", flush=True)

		# data_raw, raw_report = raw_pipeline.run(data=None)
		data_raw, raw_report = raw_pipeline.run(raw_blocks)

		raw_report_fname = out_dir / f"{subject_id}_{suffix}_raw_report.html"
		raw_json_fname = out_dir / f"{subject_id}_{suffix}_raw_pipeline.json"

		raw_report.save(
			raw_report_fname,
			overwrite=True,
			open_browser=False,
		)
		raw_pipeline.generate_json(path=raw_json_fname)

		print("---------------------------------------------")
		print("Now running BIO AlmKanal pipeline")
		print("---------------------------------------------", flush=True)

		bio_pipeline = AlmKanal(
			steps=[
				NCCPhysioCleaner(),
				NCCEpochFromBehavior(
					event_info=event_info,
					epochs_settings=epochs_settings,
					name="epochs_bio",
				),
			],
			pick_params=None,
		)

		epochs_bio, bio_report = bio_pipeline.run(data_raw.copy())

		bio_fif, bio_html, bio_json = save_pipeline_outputs(
			pipeline=bio_pipeline,
			data=epochs_bio,
			report=bio_report,
			out_dir=out_dir,
			subject_id=subject_id,
			suffix=suffix,
			data_kind="bio",
			overwrite=True,
		)

		print("---------------------------------------------")
		print("Now running MEG AlmKanal pipeline")
		print("---------------------------------------------", flush=True)

		meg_pipeline = AlmKanal(
			steps=[
				NCCPickChannels(picks=["meg", "stim"]),
				NCCEpochFromBehavior(
					event_info=event_info,
					epochs_settings=epochs_settings,
					name="epochs_meg",
				),
			],
			pick_params=None,
		)

		epochs_meg, meg_report = meg_pipeline.run(data_raw.copy())

		meg_fif, meg_html, meg_json = save_pipeline_outputs(
			pipeline=meg_pipeline,
			data=epochs_meg,
			report=meg_report,
			out_dir=out_dir,
			subject_id=subject_id,
			suffix=suffix,
			data_kind="meg",
			overwrite=True,
		)

		events = meg_pipeline.info["steps_info"]["NCCEpochFromBehavior"]["epoch_info"]["events"]
		joblib.dump(
			events,
			f"/home/reabt/experiments/ncc/MEG/data/{job_data_folder}/{subject_id}_events.pkl",
		)

		print("Saved outputs:", flush=True)
		print(f"MEG epochs: {meg_fif}", flush=True)
		print(f"BIO epochs: {bio_fif}", flush=True)
		print(f"Raw report: {raw_report_fname}", flush=True)
		print(f"MEG report: {meg_html}", flush=True)
		print(f"BIO report: {bio_html}", flush=True)
		
		
#%% AlmKanal steps
@define
class LoadNCCBlocks(AlmKanalStep):
	subject_id: str
	preload: bool = True

	must_be_before: tuple = (
		"NCCMaxwell",
		"NCCRenameBioChannels",
		"NCCBandpassFilter",
		"NCCNotchFilter",
		"ICA",
	)
	must_be_after: tuple = ()

	def _get_block_indices(self):
		if self.subject_id == "19970520smsr":
			print(
				"Using adapted block list for subject '19970520smsr' because "
				"behavioral data is missing for one block.",
				flush=True,
			)
			return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

		n_blocks = NCCRaw.get_number_of_runs(self.subject_id)
		return list(np.arange(1, n_blocks + 1))

	def run(self, data, info):
		block_indices = self._get_block_indices()
		raw_blocks = [NCCRaw(self.subject_id, block_nr=block, preload=self.preload) for block in block_indices]
		block_pos = np.array([raw.info["dev_head_t"]["trans"][:3, 3]for raw in raw_blocks])

		all_distances = np.sqrt(block_pos[:, 0] ** 2 + block_pos[:, 1] ** 2 + block_pos[:, 2] ** 2)

		median_distance = np.median(all_distances)
		median_distance_idx = int(np.abs(all_distances - median_distance).argmin())

		destination_run_nr = median_distance_idx + 1
		destination = NCCRaw.get_fif_filename(
			subject_id=self.subject_id,
			run_nr=destination_run_nr,
		)

		print(f"Blocks positions: {block_pos}", flush=True)
		print(f"Maxwell destination: {destination}", flush=True)

		return {
			"data": raw_blocks,
			"load_info": {
				"subject_id": self.subject_id,
				"block_indices": block_indices,
				"block_positions": block_pos,
				"all_distances": all_distances,
				"median_distance": median_distance,
				"median_distance_idx": median_distance_idx,
				"destination_run_nr": destination_run_nr,
				"destination": destination,
			},
		}

	def reports(self, data, report, info):
		for ix, raw in enumerate(data):
			report.add_raw(
				raw,
				butterfly=False,
				psd=True,
				title=f"raw_block_{ix + 1:02d}",
			)

# @define
# class NCCMaxwell(AlmKanalStep):
# 	enabled: bool = True
# 	calibration_file: str = "/home/reabt/experiments/ncc/MEG/Fabi/utils/sss_cal.dat"
# 	cross_talk_file: str = "/home/reabt/experiments/ncc/MEG/Fabi/utils/ct_sparse.fif"

# 	must_be_before: tuple = (
# 		"NCCConcatenateBlocks",
# 		"NCCRenameBioChannels",
# 		"NCCBandpassFilter",
# 		"NCCNotchFilter",
# 		"ICA",
# 	)
# 	must_be_after: tuple = ("LoadNCCBlocks",)

# 	def run(self, data, info):
# 		if not self.enabled:
# 			return {
# 				"data": data,
# 				"maxwell_info": {
# 					"enabled": False,
# 				},
# 			}

# 		destination = info["LoadNCCBlocks"]["load_info"]["destination"]

# 		raw_max_blocks = []
# 		bad_channels_by_block = {}

# 		for ix, raw in enumerate(data):
# 			print(f"--- Maxwell block {ix + 1} / {len(data)} ---", flush=True)

# 			noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(
# 				raw,
# 				calibration=self.calibration_file,
# 				cross_talk=self.cross_talk_file,
# 			)

# 			raw.info["bads"] = noisy_chs + flat_chs
# 			bad_channels_by_block[ix + 1] = {
# 				"noisy_chs": noisy_chs,
# 				"flat_chs": flat_chs,
# 			}

# 			raw = mne.preprocessing.maxwell_filter(
# 				raw,
# 				calibration=self.calibration_file,
# 				cross_talk=self.cross_talk_file,
# 				destination=destination,
# 			)

# 			raw_max_blocks.append(raw)

# 		return {
# 			"data": raw_max_blocks,
# 			"maxwell_info": {
# 				"enabled": True,
# 				"calibration_file": self.calibration_file,
# 				"cross_talk_file": self.cross_talk_file,
# 				"destination": destination,
# 				"bad_channels_by_block": bad_channels_by_block,
# 			},
# 		}

# 	def reports(self, data, report, info):
# 		for ix, raw in enumerate(data):
# 			report.add_raw(
# 				raw,
# 				butterfly=False,
# 				psd=True,
# 				title=f"maxwell_block_{ix + 1:02d}",
# 			)
@define
class NCCMaxwell(AlmKanalStep):
	enabled: bool = True
	destination: str | None = None
	calibration_file: str = "/home/reabt/experiments/ncc/MEG/Fabi/utils/sss_cal.dat"
	cross_talk_file: str = "/home/reabt/experiments/ncc/MEG/Fabi/utils/ct_sparse.fif"

	must_be_before: tuple = (
		"NCCConcatenateBlocks",
		"NCCRenameBioChannels",
		"NCCBandpassFilter",
		"NCCNotchFilter",
		"ICA",
	)
	must_be_after: tuple = ()

	def run(self, data, info):
		if not self.enabled:
			return {
				"data": data,
				"maxwell_info": {
					"enabled": False,
					"destination": self.destination,
				},
			}

		destination = self.destination

		raw_max_blocks = []
		bad_channels_by_block = {}

		for ix, raw in enumerate(data):
			print(f"--- Maxwell block {ix + 1} / {len(data)} ---", flush=True)

			noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(
				raw,
				calibration=self.calibration_file,
				cross_talk=self.cross_talk_file,
			)

			raw.info["bads"] = noisy_chs + flat_chs
			bad_channels_by_block[ix + 1] = {
				"noisy_chs": noisy_chs,
				"flat_chs": flat_chs,
			}

			raw = mne.preprocessing.maxwell_filter(
				raw,
				calibration=self.calibration_file,
				cross_talk=self.cross_talk_file,
				destination=destination,
			)

			raw_max_blocks.append(raw)

		return {
			"data": raw_max_blocks,
			"maxwell_info": {
				"enabled": True,
				"calibration_file": self.calibration_file,
				"cross_talk_file": self.cross_talk_file,
				"destination": destination,
				"bad_channels_by_block": bad_channels_by_block,
			},
		}
	

@define
class NCCConcatenateBlocks(AlmKanalStep):
	on_mismatch: str = "warn"

	must_be_before: tuple = (
		"NCCRenameBioChannels",
		"NCCBandpassFilter",
		"NCCNotchFilter",
		"ICA",
	)
	must_be_after: tuple = ("LoadNCCBlocks",)

	def run(self, data, info):
		raw = mne.concatenate_raws(data, on_mismatch=self.on_mismatch)

		return {
			"data": raw,
			"concat_info": {
				"on_mismatch": self.on_mismatch,
				"n_blocks": len(data),
			},
		}

	def reports(self, data, report, info):
		report.add_raw(
			data,
			butterfly=False,
			psd=True,
			title="raw_concatenated",
		)


@define
class NCCRenameBioChannels(AlmKanalStep):
	must_be_before: tuple = (
		"NCCBandpassFilter",
		"NCCNotchFilter",
		"ICA",
		"NCCPhysioCleaner",
	)
	must_be_after: tuple = ("NCCConcatenateBlocks",)

	def run(self, data, info):
		renamed = False

		if "BIO003" in data.ch_names:
			data.set_channel_types({
				"BIO001": "eog",
				"BIO002": "eog",
				"BIO003": "ecg",
			})

			mne.rename_channels(data.info, {
				"BIO001": "EOG001",
				"BIO002": "EOG002",
				"BIO003": "ECG003",
			})

			renamed = True

		if "MISC001" in data.ch_names:
			data.set_channel_types({"MISC001": "resp"})
			mne.rename_channels(data.info, {"MISC001": "rsp"})

		return {
			"data": data,
			"bio_channel_info": {
				"renamed_bio001_bio002_bio003": renamed,
				"has_rsp": "rsp" in data.ch_names,
				"has_ecg": "ECG003" in data.ch_names,
				"has_eog": "EOG001" in data.ch_names,
			},
		}

	def reports(self, data, report, info):
		report.add_html(
			f"""
			<p>BIO channel status:</p>
			<ul>
				<li>ECG003 present: {"ECG003" in data.ch_names}</li>
				<li>EOG001 present: {"EOG001" in data.ch_names}</li>
				<li>EOG002 present: {"EOG002" in data.ch_names}</li>
				<li>rsp present: {"rsp" in data.ch_names}</li>
			</ul>
			""",
			title="BIO channel renaming",
		)


@define
class NCCBandpassFilter(AlmKanalStep):
	h_pass: float = 0.5
	l_pass: float = 99.0

	must_be_before: tuple = ("NCCNotchFilter", "ICA")
	must_be_after: tuple = ("NCCRenameBioChannels",)

	def run(self, data, info):
		print(
			f"--- High-pass + low-pass FIR filter: {self.h_pass}-{self.l_pass} Hz ---",
			flush=True,
		)

		data.filter(
			l_freq=self.h_pass,
			h_freq=self.l_pass,
			picks="meg",
			method="fir",
			fir_design="firwin",
			fir_window="hamming",
			l_trans_bandwidth="auto",
			h_trans_bandwidth="auto",
			phase="zero",
		)

		return {
			"data": data,
			"filter_info": {
				"l_freq": self.h_pass,
				"h_freq": self.l_pass,
				"picks": "meg",
				"method": "fir",
				"fir_design": "firwin",
				"fir_window": "hamming",
				"phase": "zero",
			},
		}

	def reports(self, data, report, info):
		report.add_raw(
			data,
			butterfly=False,
			psd=True,
			title=f"raw_filtered_{self.h_pass}-{self.l_pass}Hz",
		)


@define
class NCCNotchFilter(AlmKanalStep):
	enabled: bool = True

	must_be_before: tuple = ("ICA",)
	must_be_after: tuple = ("NCCBandpassFilter",)

	def run(self, data, info):
		if not self.enabled:
			return {
				"data": data,
				"notch_info": {
					"enabled": False,
				},
			}

		print(
			"--- Band-stop filter: 49-51 Hz, 4th-order Butterworth, forward-backward ---",
			flush=True,
		)

		data.filter(
			l_freq=51,
			h_freq=49,
			picks="meg",
			method="iir",
			iir_params=dict(
				order=4,
				ftype="butter",
			),
			phase="zero",
		)

		return {
			"data": data,
			"notch_info": {
				"enabled": True,
				"l_freq": 51,
				"h_freq": 49,
				"picks": "meg",
				"method": "iir",
				"iir_params": {
					"order": 4,
					"ftype": "butter",
				},
				"phase": "zero",
			},
		}

	def reports(self, data, report, info):
		report.add_raw(
			data,
			butterfly=False,
			psd=True,
			title="raw_after_49-51Hz_notch",
		)


@define
class NCCSaveRaw(AlmKanalStep):
	fname: str
	overwrite: bool = True

	must_be_before: tuple = ()
	must_be_after: tuple = ()

	def run(self, data, info):
		Path(self.fname).parent.mkdir(parents=True, exist_ok=True)
		data.save(self.fname, overwrite=self.overwrite)

		return {
			"data": data,
			"save_raw_info": {
				"fname": self.fname,
				"overwrite": self.overwrite,
			},
		}

	def reports(self, data, report, info):
		report.add_html(
			f"<p>Saved raw file:<br><code>{self.fname}</code></p>",
			title="Saved raw",
		)


@define
class NCCPhysioCleaner(AlmKanalStep):
	must_be_before: tuple = ()
	must_be_after: tuple = ("NCCRenameBioChannels",)

	def run(self, data, info):
		if "rsp" in data.ch_names:
			chan_list_bio = ["ecg", "eog", "resp", "stim"]
		else:
			chan_list_bio = ["ecg", "eog", "stim"]

		bio_raw = data.copy().pick(chan_list_bio)
		bio_df = bio_raw.to_data_frame()

		bio_clean = nk.bio_process(
			ecg=bio_df["ECG003"],
			rsp=bio_df["rsp"] if "rsp" in data.ch_names else None,
			eog=bio_df["EOG001"],
			sampling_rate=bio_raw.info["sfreq"],
		)[0]

		stim_cols = bio_df.columns[-17:].tolist()

		ch_names_bio = bio_clean.columns.tolist() + stim_cols
		ch_types_bio = (
			np.tile("bio", len(bio_clean.columns)).tolist() +
			np.tile("stim", len(stim_cols)).tolist()
		)

		bio_info = mne.create_info(
			ch_names=ch_names_bio,
			sfreq=bio_raw.info["sfreq"],
			ch_types=ch_types_bio,
		)

		bio_data = np.concatenate(
			[
				bio_clean.to_numpy(),
				bio_df[stim_cols].to_numpy(),
			],
			axis=1,
		).T

		bio_clean_raw = mne.io.RawArray(bio_data, bio_info)

		return {
			"data": bio_clean_raw,
			"physio_info": {
				"chan_list_bio": chan_list_bio,
				"stim_cols": stim_cols,
				"used_ecg": "ECG003",
				"used_eog": "EOG001",
				"used_rsp": "rsp" if "rsp" in data.ch_names else None,
			},
		}

	def reports(self, data, report, info):
		report.add_raw(
			data,
			butterfly=False,
			psd=True,
			title="bio_clean_raw",
		)


@define
class NCCPickChannels(AlmKanalStep):
	picks: list

	must_be_before: tuple = ()
	must_be_after: tuple = ()

	def run(self, data, info):
		data = data.copy().pick(self.picks)

		return {
			"data": data,
			"pick_info": {
				"picks": self.picks,
				"ch_names": data.ch_names,
			},
		}

	def reports(self, data, report, info):
		report.add_raw(
			data,
			butterfly=False,
			psd=True,
			title=f"picked_{'_'.join(self.picks)}",
		)


@define
class NCCEpochFromBehavior(AlmKanalStep):
	event_info: dict
	epochs_settings: dict
	name: str = "epochs"

	must_be_before: tuple = ()
	must_be_after: tuple = ()

	def run(self, data, info):
		epochs, events = get_epochs_R(
			data,
			self.event_info,
			self.epochs_settings,
		)

		return {
			"data": epochs,
			"epoch_info": {
				"events": events,
				"epochs_settings": self.epochs_settings,
				"name": self.name,
			},
		}

	def reports(self, data, report, info):
		report.add_epochs(
			data,
			psd=True,
			title=self.name,
		)


def save_pipeline_outputs(
	pipeline,
	data,
	report,
	out_dir,
	subject_id,
	suffix,
	data_kind,
	overwrite=True,
):
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	fif_fname = out_dir / f"{subject_id}_{suffix}_{data_kind}-epo.fif"
	html_fname = out_dir / f"{subject_id}_{suffix}_{data_kind}_report.html"
	json_fname = out_dir / f"{subject_id}_{suffix}_{data_kind}_pipeline.json"

	data.save(fif_fname, overwrite=overwrite)
	report.save(html_fname, overwrite=overwrite, open_browser=False)
	pipeline.generate_json(path=json_fname)

	return fif_fname, html_fname, json_fname


@define
class NCCResampleRaw(AlmKanalStep):
	sfreq: float
	npad: str = "auto"

	must_be_before: tuple = ()
	must_be_after: tuple = ()

	def run(self, data, info):
		data.resample(self.sfreq, npad=self.npad)

		return {
			"data": data,
			"resample_info": {
				"sfreq": self.sfreq,
				"npad": self.npad,
			},
		}

	def reports(self, data, report, info):
		report.add_raw(
			data,
			butterfly=False,
			psd=True,
			title=f"raw_resampled_{self.sfreq}Hz",
		)