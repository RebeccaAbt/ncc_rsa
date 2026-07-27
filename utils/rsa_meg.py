	
import numpy as np
import os
import joblib
from joblib import Parallel, delayed

from collections.abc import Iterable

import numpy as np
from rsatoolbox.rdm.rdms import concat
from rsatoolbox.rdm import calc_rdm, calc_rdm_unbalanced

def create_run_idx(modality_array):

	'''
	Create a run index based on modality changes in the dataset.
	3 blocks (= every modality once) = 1 run.
	This is used to define the crossvalidation folds. 
	Since we have 12 blocks (3 modalities, repeated 4 times), 
	we will have 4-fold crossvalidation, if we use "run_idx" as cv_descriptor for the rsa.
	'''
	unique_modalities, modality_numbers = np.unique(modality_array, return_inverse=True)

	# Find indices where the modality changes (block boundaries)
	block_boundaries = np.where(np.diff(modality_numbers) != 0)[0] + 1
	# Add start and end indices
	block_starts = np.concatenate(([0], block_boundaries))
	block_ends = np.concatenate((block_boundaries, [len(modality_numbers)]))

	# Each group consists of len(unique_modalities) consecutive blocks
	group_size = len(unique_modalities)
	run_idx = np.zeros_like(modality_numbers, dtype=int)

	for group_idx, start in enumerate(range(0, len(block_starts), group_size), 1):
		for block in range(group_size):
			idx = start + block
			if idx < len(block_starts):
				run_idx[block_starts[idx]:block_ends[idx]] = group_idx

	return run_idx


def count_events_per_run(cfg, epochs):
	'''
	Count the number of events in each run (defined by run_idx_array).
	This is useful to check if the runs are balanced in terms of number of events, which is important for crossvalidation.
	'''
	
	subjectID = cfg.subjectID

	# MEG_cfg = MEG_cfg
	if hasattr(cfg, "MEG_config"): # input cfg is a fusion config
		event_count_dir = os.path.join(cfg.MEG_cfg.dataDir, cfg.MEG_cfg.dataFolder)
	else: # input cfg is a MEG config
		event_count_dir = os.path.join(cfg.dataDir, cfg.dataFolder)


	fileName = os.path.splitext(os.path.basename(cfg.MEG_inFile[0]))[0] + '_event_counts.pkl'
	event_count_file = os.path.join(event_count_dir, fileName)
	# event_count_file = os.path.join(event_count_dir, f'{subjectID}_event_counts.pkl')
	
	event_id = epochs.event_id
	unique_events, counts = np.unique(epochs.events[:,-1], return_counts=True)
	event_counts = dict([[u.item(), c.item()] for u, c in zip(unique_events, counts)])
	event_matched = {k: event_counts[v] for k, v in event_id.items()}


	print('\n-----------------------------------------------------------\n',
		  f'Saving event counts for subject {subjectID} in file {event_count_file}',
		  '\n-----------------------------------------------------------\n')

	joblib.dump({'event_counts': event_counts,
				 'event_matched': event_matched}, event_count_file)


# ------------------------------------------------------------------ v  added by me
def calc_rdm_movie_parallel(
        dataset, method='euclidean', descriptor=None, noise=None,
        cv_descriptor=None, prior_lambda=1, prior_weight=0.1,
        time_descriptor='time', bins=None, unbalanced=False):
    """
    calculates an RDM movie from an input TemporalDataset

    Args:
        dataset (rsatoolbox.data.dataset.TemporalDataset):
            The dataset the RDM is computed from
        method (String):
            a description of the dissimilarity measure (e.g. 'Euclidean')
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            used only for Mahalanobis and Crossnobis estimators
            defaults to an identity matrix, i.e. euclidean distance
        time_descriptor (String): descriptor key that points to the time
            dimension in dataset.time_descriptors. Defaults to 'time'.
        bins (array-like): list of bins, with bins[i] containing the vector
            of time-points for the i-th bin. Defaults to no binning.
        unbalanced (bool): if set to True use calc_rdm_unbalanced,
            else and by default use calc_rdm

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with RDM movie
    """

    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat, ds_i in enumerate(dataset):
            if noise is None:
                rdms.append(calc_rdm_movie_parallel(
                    ds_i, method=method,
                    descriptor=descriptor))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm_movie_parallel(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm_movie_parallel(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise[i_dat]))
        rdm = concat(rdms)
    else:
        if bins is not None:
            binned_data = dataset.bin_time(time_descriptor, bins)
            splited_data = binned_data.split_time(time_descriptor)
            time = binned_data.time_descriptors[time_descriptor]
        else:
            splited_data = dataset.split_time(time_descriptor)
            time = dataset.time_descriptors[time_descriptor]

        def process(data): # for doing the computation in parallel
            dat_single = data.time_as_observations(time_descriptor)
            if unbalanced:
                return calc_rdm_unbalanced(
                    dat_single, method=method,
                    descriptor=descriptor, noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda,
                    prior_weight=prior_weight)
            else:
                return calc_rdm(
                    dat_single, method=method,
                    descriptor=descriptor, noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda,
                    prior_weight=prior_weight)

        rdms = Parallel(n_jobs=-1, verbose=11)(
            delayed(process)(data) for data in splited_data)

        rdm = concat(rdms)
        rdm.rdm_descriptors[time_descriptor] = time
    rdm.dissimilarity_measure = method
    return rdm
# ------------------------------------------------------------------ ^