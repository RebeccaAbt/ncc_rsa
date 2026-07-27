
#%% imports
import numpy as np
from pyriemann.clustering import Potato, PotatoField
from pyriemann.utils.covariance import covariances

# sys.path.append('/home/reabt/experiments/ncc/MRI/code/utils/')
# sys.path.append('/home/reabt/experiments/ncc/MRI/code/')
#%%

def reject_bad_epochs_with_potato(
	epochs,
	picks="data",
	# threshold=3.0,
	n_potatoes = 2,
	z_threshold=3,
	p_threshold=0.01,
	use_field=False,
	estimator="oas",
	return_details=True,
	reject_epochs = True,
	tmin = -1, 
	tmax = 1
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
	print(f"Copying epochs and cropping to tmin={tmin}s, tmax={tmax}s for potato processing", flush=True)
	epochs_work = epochs.copy().crop(tmin = tmin, tmax = tmax)

	if use_field:
		# practical setup for MEG:
		# build separate covariance sets for grads and mags, then combine with PotatoField
		epochs_grad = epochs_work.copy().pick("grad")
		epochs_mag = epochs_work.copy().pick("mag")

		X_grad = epochs_grad.get_data(copy=True)   # shape: (n_epochs, n_grad, n_times)
		X_mag = epochs_mag.get_data(copy=True)     # shape: (n_epochs, n_mag, n_times)

		C_grad = covariances(X_grad, estimator=estimator)
		C_mag = covariances(X_mag, estimator=estimator)

		potato = PotatoField(
			n_potatoes=n_potatoes,
			z_threshold=z_threshold,
			p_threshold=p_threshold,
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
		C = covariances(X, estimator=estimator)

		potato = Potato(
			metric="riemann",
			threshold=z_threshold,
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
	# print(f'clean mask: {clean_mask}', flush=True)
	# good_idx = np.where(clean_mask)[0]
	bad_idx = np.where(clean_mask==0)[0]
	# print(f'good_idx: {good_idx}', flush=True)
	# epochs_clean = epochs[good_idx]

	epochs_clean = epochs.copy().drop(bad_idx, reason='potato')

	if return_details:
		return epochs_clean, details
	return epochs_clean


def print_potato_summary(epochs_before, epochs_after, potato_info):
	n_before = len(epochs_before)
	n_after = len(epochs_after)
	n_bad = len(potato_info["bad_idx"])
	print(f"Epochs before: {n_before}")
	print(f"Epochs after : {n_after}")
	print(f"Rejected     : {n_bad}")
	print(f"Rejected idx : {potato_info['bad_idx']}")