import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import numpy as np
import pandas as pd
import joblib

from pingouin import partial_corr
from copy import deepcopy
from joblib import Parallel, delayed
from scipy.stats import rankdata

from utils.subj import get_fusion_subjects
from utils.load_cfg import load_config_instance


def compare_descriptors(dataRDM_descriptors, model_descriptors):

    modality_map = {
        'auditory': 'aud',
        'somato': 'tac',
        'visual': 'vis'
    }

    # transform MEG descriptors into same format as fMRI descriptors
    normalized_events = []
    for e in dataRDM_descriptors:
        parts = e.split('/')  # e.g., ['auditory', 'hit', '1']
        modality = modality_map[parts[0]]
        state = parts[1]
        number = parts[2]
        normalized = f"{modality}_{state}_{number}"
        normalized_events.append(normalized)

    assert normalized_events == model_descriptors, "Mismatch: order of MEG descriptors does not match order of fMRI descriptors!" # check uif o


def adjust_descriptors(descriptors):
    modality_map = {
            'auditory': 'aud',
            'somato': 'tac',
            'visual': 'vis'
        }

        # transform MEG descriptors into same format as fMRI descriptors
    renamed_events = []
    for e in descriptors:
        parts = e.split('/')  # e.g., ['auditory', 'hit', '1']
        modality = modality_map[parts[0]]
        state = parts[1]
        number = parts[2]
        normalized = f"{modality}_{state}_{number}"
        renamed_events.append(normalized)
    return renamed_events


def get_mean_movie():
    '''
    Compute the mean of the RDM-movies of sujects, that are currently included 
    in the fusion analysis (=subjects where MEG + fMRI data exists).
    Outputs a variable of the class "RDMs"
    '''

    all_subj = get_fusion_subjects()
    cfg = load_config_instance('SetupConfig_C1', all_subj[0]) # Because this config uses crossnobis distance & the correct RSA measure 'corr_cov'

    all_RDMs = []
    for subj in all_subj:

        cfg.subjectID = subj

        movieFile = cfg.get_outFile_names()['movie']

        if os.path.exists(movieFile):
            rdm_movie = joblib.load(movieFile)
            all_RDMs.append(rdm_movie.get_vectors())

    mean_rdms = np.mean(np.array(all_RDMs), axis = 0)
    mean_movie = deepcopy(rdm_movie)
    mean_movie.dissimilarities = mean_rdms # put the data back in necessary RDMs structure
    return mean_movie

##% =================================================================================================== pandas-based (slow)

def commonality_coeff_pd(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec, method = 'spearman'):
    '''
    Uses pd.DataFrage as input. This works fine, but is slow when trying parallel computing stuff
    :param fMRI_vec: Description
    :param MEG_vec: Description
    :param sensory_vec: Description
    :param suprasensory_vec: Description
    :param method: Description
    '''
    df=pd.DataFrame({'fMRI':fMRI_vec,
                    'MEG':MEG_vec,
                    'sensory': sensory_vec,
                    'suprasensory': suprasensory_vec})


    semipartial_both = (partial_corr(data=df, x='MEG', y='fMRI', y_covar=ALL_MODELS, method=method)['r'].values[0])**2
    semipartial_sensory = (partial_corr(data=df, x='MEG', y='fMRI', y_covar='sensory', method=method)['r'].values[0])**2
    semipartial_suprasensory = (partial_corr(data=df, x='MEG', y='fMRI', y_covar='suprasensory', method=method)['r'].values[0])**2

    unique_sensory = semipartial_suprasensory - semipartial_both
    unique_suprasensory = semipartial_sensory - semipartial_both

    return unique_sensory, unique_suprasensory


def fuse_timepoint_pd(t, SL_rdms, sensory_vec, suprasensory_vec): # this was before parallelisation
    MEG_vec = t.get_vectors()[0]
    timepoint = t.rdm_descriptors['index'][0]
    # voxel_index_py = []
    sensory = []
    suprasensory = []

    for SL in SL_rdms:
        fMRI_vec = SL.get_vectors()[0]
        u_sensory, u_suprasensory = commonality_coeff_pd(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec)
        sensory.append(u_sensory)
        suprasensory.append(u_suprasensory)

    return {
        'time': timepoint,
        'sensory': sensory,
        'suprasensory': suprasensory
    }

def fuse_timepoint_pd_parallel(t, SL_rdms, sensory_vec, suprasensory_vec):
    MEG_vec = t.get_vectors()[0]
    timepoint = t.rdm_descriptors['index'][0]

    def process_SL(SL):
        fMRI_vec = SL.get_vectors()[0]
        return commonality_coeff_pd(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec)

    results = Parallel(n_jobs=-1)(delayed(process_SL)(SL) for SL in SL_rdms)
    sensory, suprasensory = zip(*results)

    return {
        'time': timepoint,
        'sensory': sensory,
        'suprasensory': suprasensory
    }

##% =================================================================================================== numpy-based (slow)


def _pcorr_from_precision(V):
    """Compute (partial) correlation matrix from covariance matrix via precision.
    Pseudoinverse of covariance; 'hermitian=True' is faster/more stable for symmetric matrices (NumPy>=1.17)
    """
    try:
        Vi = np.linalg.pinv(V, hermitian=True)
    except TypeError:
        Vi = np.linalg.pinv(V)
    Vi_diag = np.diag(Vi)
    D = np.diag(np.sqrt(1.0 / Vi_diag))
    pcor = - (D @ Vi @ D)  # partial correlation matrix
    return pcor, Vi, Vi_diag

def _spcorr_from_precision(pcor, V, Vi, Vi_diag):
    
    """Compute semi-partial correlation matrix as in pingouin.partial_corr.
    pingouin code (ported):
    spcor = pcor / sqrt(diag(V))[..., None] / sqrt(abs(Vi_diag - Vi**2 / Vi_diag[..., None])).T
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        denom1 = np.sqrt(np.diag(V))[:, None]                               # shape (p, 1), scales columns
        denom2 = np.sqrt(np.abs(Vi_diag - (Vi**2) / Vi_diag[:, None])).T    # shape (p, p), scales rows
        spcor = pcor / denom1 / denom2
    return spcor

def partial_corr_r_numpy(x, y, y_covar=None, method="spearman"):
    """
    mothod: "pearson" or "spearman"
    NumPy equivalent for pingouin.partial_corr's 'r' (no CI/p-values).
    --> faster computation because we don't need to construct pd. data frames
    Remove only from y (semi-partial on y)  <-- should be the same as using "partial_corr(data=df, x='MEG', y='fMRI', y_covar=..."
    """
    X = np.column_stack([x, y, y_covar])

    if method == "spearman":    # rank-transform columns before covariance
        X = np.apply_along_axis(rankdata, 0, X)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    V = np.cov(X, rowvar=False, bias=False)                 # covariance  
    pcor, Vi, Vi_diag = _pcorr_from_precision(V)            # precision -> partial correlation matrix
    spcor = _spcorr_from_precision(pcor, V, Vi, Vi_diag)    # semi-partial correlations

    return float(spcor[0, 1])

def commonality_coeff_np(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec, method='spearman'):

    both_cov = np.column_stack([sensory_vec, suprasensory_vec])
    r_both = partial_corr_r_numpy(MEG_vec, fMRI_vec, y_covar=both_cov, method=method) ** 2
    r_sens = partial_corr_r_numpy(MEG_vec, fMRI_vec, y_covar=sensory_vec, method=method) ** 2
    r_supr = partial_corr_r_numpy(MEG_vec, fMRI_vec, y_covar=suprasensory_vec, method=method) ** 2

    unique_sensory = r_supr - r_both
    unique_suprasensory = r_sens - r_both

    return unique_sensory, unique_suprasensory

def fuse_timepoint_np(t, SL_rdms, sensory_vec, suprasensory_vec): # this was before parallelisation
    
    MEG_vec = t.get_vectors()[0]
    timepoint = t.rdm_descriptors['index'][0]
    # voxel_index_py = []
    sensory = []
    suprasensory = []

    for SL in SL_rdms:
        fMRI_vec = SL.get_vectors()[0]
        u_sensory, u_suprasensory = commonality_coeff_np(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec)
        sensory.append(u_sensory)
        suprasensory.append(u_suprasensory)

    return {
        'time': timepoint,
        'sensory': sensory,
        'suprasensory': suprasensory}

def fuse_timepoint_np_parallel(t, SL_rdms, sensory_vec, suprasensory_vec):
    MEG_vec = t.get_vectors()[0]
    timepoint = t.rdm_descriptors['index'][0]

    def process_SL(SL):
        fMRI_vec = SL.get_vectors()[0]
        return commonality_coeff_np(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec)

    results = Parallel(n_jobs=-1)(delayed(process_SL)(SL) for SL in SL_rdms)
    sensory, suprasensory = zip(*results)

    return {
        'time': timepoint,
        'sensory': sensory,
        'suprasensory': suprasensory}

