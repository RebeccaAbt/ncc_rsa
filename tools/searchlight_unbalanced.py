
#%%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import numpy as np
from tqdm import tqdm
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.calc_unbalanced import calc_rdm_unbalanced
from rsatoolbox.rdm.rdms import RDMs

def get_searchlight_RDMs_unbalanced(data_2d, centers, neighbors, events,
                         method='crossnobis',
                         descriptor='events',
                         cv_descriptor='run_number',
                         verbose=True):
    """
    Calculates searchlight RDMs for **unbalanced datasets** using cross-validated dissimilarities.

    Args:
        data_2d (2D np.array): Brain data array (n_observations x n_voxels)
        centers (1D np.array): Center voxel indices from get_volume_searchlight
        neighbors (list of arrays): Neighbor voxel indices for each center
        events (1D np.array): Condition labels per observation
        method (str): Dissimilarity method ('crossnobis', etc.)
        descriptor (str): Observation descriptor used to group trials (default 'events')
        cv_descriptor (str): Cross-validation descriptor to separate partitions (e.g., 'run_number')
        verbose (bool): Whether to print progress bars

    Returns:
        RDMs: rsatoolbox.rdm.RDMs object with searchlight RDMs
    """
    data_2d, centers = np.array(data_2d), np.array(centers)
    n_centers = centers.shape[0]

    # For memory reasons, we chunk the data if we have a lot of centers
    if n_centers > 1000:
        chunked_centers = np.split(np.arange(n_centers),
                                   np.linspace(0, n_centers, 101, dtype=int)[1:-1])

        n_conditions = len(np.unique(events))
        RDM_array = np.zeros((n_centers, n_conditions * (n_conditions - 1) // 2))

        for chunk in tqdm(chunked_centers, desc='Calculating unbalanced RDMs...', disable=not verbose):
            center_data = []
            for c in chunk:
                center = centers[c]
                center_neighbors = neighbors[c]
                ds = Dataset(
                    measurements=data_2d[:, center_neighbors],
                    descriptors={'center': center},
                    obs_descriptors={descriptor: events},
                    channel_descriptors={'voxels': center_neighbors}
                )
                center_data.append(ds)

            RDM_chunk = calc_rdm_unbalanced(center_data, method=method,
                                            descriptor=descriptor,
                                            cv_descriptor=cv_descriptor)
            RDM_array[chunk, :] = RDM_chunk.dissimilarities
    else:
        center_data = []
        for c in range(n_centers):
            center = centers[c]
            nb = neighbors[c]
            ds = Dataset(
                measurements=data_2d[:, nb],
                descriptors={'center': c},
                obs_descriptors={descriptor: events},
                channel_descriptors={'voxels': nb}
            )
            center_data.append(ds)

        RDM_array = calc_rdm_unbalanced(center_data, method=method,
                                        descriptor=descriptor,
                                        cv_descriptor=cv_descriptor).dissimilarities

    # Build RDMs object
    SL_rdms = RDMs(
        RDM_array,
        rdm_descriptors={'voxel_index': centers},
        dissimilarity_measure=method
    )

    return SL_rdms
