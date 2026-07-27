#%% imports
import os
import sys
import subprocess

import joblib
import numpy as np
import neurokit2 as nk
import mne
from os.path import join
from pymatreader import read_mat
from pyriemann.clustering import Potato, PotatoField
from pyriemann.utils.covariance import covariances

from clusterjobs.meta_job import Job
from utils.raw import Raw
from utils.epochs import get_epochs, get_epochs_R
from utils.src_utils import data2source

sys.path.append('/home/reabt/experiments/ncc/MEG/Fabi/utils/')
sys.path.append('/home/reabt/experiments/ncc/MEG/Fabi/')

os.environ["FONTCONFIG_PATH"] = "/etc/fonts"

#%%

def reject_bad_epochs_with_potato(
	epochs,
	picks="data",
	threshold=3.0,
	use_field=False,
	return_details=True,
):
	"""
	Detect bad epochs with the Riemannian Potato and return a cleaned Epochs object.

	Parameters
	----------
	epochs : mne.Epochs
		Your epoched data.
	picks : str | list | None
		Channels to use for artifact detection.
		Typical choices:
		- "meg"       -> all MEG channels
		- "grad"      -> gradiometers only
		- "mag"       -> magnetometers only
		- ["MEG...", ...] -> custom channel list
	threshold : float
		z-threshold of the Potato. Smaller = stricter.
		3 is the pyRiemann default.
	use_field : bool
		If False: use a single Potato.
		If True:  use PotatoField with separate potatoes for mag and grad.
				  This can be useful when channel groups have different properties.
	return_details : bool
		If True, also return masks / scores / fitted model.

	Returns
	-------
	epochs_clean : mne.Epochs
		Cleaned epochs.
	info : dict
		Extra information if return_details=True.
	"""

	# work on a copy so the original epochs remain unchanged
	epochs_work = epochs.copy()

	if use_field:
		# practical setup for MEG:
		# build separate covariance sets for grads and mags, then combine with PotatoField
		epochs_grad = epochs_work.copy().pick("grad")
		epochs_mag = epochs_work.copy().pick("mag")

		X_grad = epochs_grad.get_data(copy=True)   # shape: (n_epochs, n_grad, n_times)
		X_mag = epochs_mag.get_data(copy=True)     # shape: (n_epochs, n_mag, n_times)

		C_grad = covariances(X_grad, estimator="oas")
		C_mag = covariances(X_mag, estimator="oas")

		potato = PotatoField(
			n_potatoes=2,
			z_threshold=threshold,
			p_threshold=0.01,
			metric="riemann",
		)

		# fit on all epochs, then predict clean/bad
		potato.fit([C_grad, C_mag])
		clean_mask = potato.predict([C_grad, C_mag]).astype(bool)

		# optional diagnostic outputs
		z_scores = potato.transform([C_grad, C_mag])          # shape: (n_epochs, 2)
		clean_prob = potato.predict_proba([C_grad, C_mag])    # shape: (n_epochs,)
		details = {
			"clean_mask": clean_mask,
			"bad_idx": np.where(~clean_mask)[0],
			"z_scores": z_scores,
			"clean_probability": clean_prob,
			"model": potato,
		}

	else:
		# simpler version: single Potato on one chosen channel set
		epochs_sel = epochs_work.copy().pick(picks)
		X = epochs_sel.get_data(copy=True)  # shape: (n_epochs, n_channels, n_times)

		# covariance per epoch
		C = covariances(X, estimator="oas")

		potato = Potato(
			metric="riemann",
			threshold=threshold,
			n_iter_max=100,
		)

		potato.fit(C)
		clean_mask = potato.predict(C).astype(bool)

		# transform gives standardized log-distance to the centroid
		z_scores = potato.transform(C)
		clean_prob = potato.predict_proba(C)

		details = {
			"clean_mask": clean_mask,
			"bad_idx": np.where(~clean_mask)[0],
			"z_scores": z_scores,
			"clean_probability": clean_prob,
			"model": potato,
		}

	# keep only clean epochs
	good_idx = np.where(clean_mask)[0]
	epochs_clean = epochs[good_idx]

	if return_details:
		return epochs_clean, details
	return epochs_clean

'''
# ------------------------------------------------------------------
# EXAMPLE 1: apply after get_epochs(...) and before save(...)
# ------------------------------------------------------------------

# epochs_meg, events = get_epochs(meg_raw, event_info, epochs_settings)

# simplest option:
# use only grad OR only mag first, so you can test the behavior more cleanly
# epochs_meg_clean, potato_info = reject_bad_epochs_with_potato(
#     epochs_meg,
#     picks="grad",
#     threshold=3.0,
#     use_field=False,
# )

# more MEG-specific option:
# use separate potatoes for grad and mag and combine them
# epochs_meg_clean, potato_info = reject_bad_epochs_with_potato(
#     epochs_meg,
#     threshold=3.0,
#     use_field=True,
# )

# print(f"Rejected {len(potato_info['bad_idx'])} / {len(epochs_meg)} epochs")
# print("Bad epoch indices:", potato_info["bad_idx"])

# save cleaned epochs instead of the original ones
# epochs_meg_clean.save(
#     f"/home/reabt/experiments/ncc/MEG/data/epochs/{subject_id}/{subject_id}_{suffix}_meg_potato-epo.fif"
# )


# ------------------------------------------------------------------
# EXAMPLE 2: same idea for BIO epochs
# ------------------------------------------------------------------

# epochs_bio_clean, potato_info_bio = reject_bad_epochs_with_potato(
#     epochs_bio,
#     picks="data",      # all non-stim channels in the Epochs object
#     threshold=3.0,
#     use_field=False,
# )

# print(f"Rejected {len(potato_info_bio['bad_idx'])} / {len(epochs_bio)} BIO epochs")

# epochs_bio_clean.save(
#     f"/home/reabt/experiments/ncc/MEG/data/epochs/{subject_id}/{subject_id}_{suffix}_bio_potato-epo.fif"
# )


# ------------------------------------------------------------------
# OPTIONAL: inspect which epochs were marked bad
# ------------------------------------------------------------------
'''
def print_potato_summary(epochs_before, epochs_after, potato_info):
	n_before = len(epochs_before)
	n_after = len(epochs_after)
	n_bad = len(potato_info["bad_idx"])
	print(f"Epochs before: {n_before}")
	print(f"Epochs after : {n_after}")
	print(f"Rejected     : {n_bad}")
	print(f"Rejected idx : {potato_info['bad_idx']}")


# print_potato_summary(epochs_meg, epochs_meg_clean, potato_info)


class DebugEpochs(Job):
	job_data_folder = 'epochs' 

	#%% the run method starts here
	def run(self):
		print("Loading the data...", flush=True)
		loadedData = joblib.load('/home/reabt/experiments/ncc/MEG/Fabi/20050610atbu_variables_saved_before_epoching.pkl')
		subject_id = '20050610atbu'
		suffix = 'Test'
		data_raw =  loadedData['data_raw']
		meg_raw =  loadedData['meg_raw']
		event_info =  loadedData['event_info']
		preproc_settings =  loadedData['preproc_settings']
		epochs_settings =  loadedData['epochs_settings']
		print("Finished Loading the data!", flush=True)
		
		epochs_meg, events = get_epochs_R(meg_raw, event_info, epochs_settings)


			# simplest option:
		epochs_meg_clean, potato_info = reject_bad_epochs_with_potato(
			epochs_meg,
			picks="grad",
			threshold=3.0,
			use_field=False,
		)

		print_potato_summary(epochs_meg, epochs_meg_clean, potato_info)

		print('--------------------\n saving Epochs .... \n--------------------\n', flush=True)
		epochs_meg_clean.save(f'/home/reabt/experiments/ncc/MEG/data/epochs/{subject_id}/{subject_id}_{suffix}_meg-epo_clean.fif')
		joblib.dump(events, f'/home/reabt/experiments/ncc/MEG/data/epochs/{subject_id}/{subject_id}_{suffix}_meg-events_clean.pkl')
		joblib.dump(potato_info, f'/home/reabt/experiments/ncc/MEG/data/epochs/{subject_id}/{subject_id}_{suffix}_potato.pkl')


