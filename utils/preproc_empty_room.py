import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import mne
import numpy as np
from datetime import datetime
from os import listdir
from os.path import join

from raw import Raw

#from rm_train import rm_train_ica


def preproc_data(subject_id,
                 cur_data, 
                 maxfilter=True,
                 notch=False,
                 l_pass=None,
                 h_pass=None,  
                 downsample_f=None):
    
    """
    Minimal preprocessing function. Only contains some maxfiltering.
    I might add an automatic ICA and autoreject, riemannian potato option
    """

    if isinstance(cur_data, str):
        cur_data = mne.io.read_raw_fif(cur_data, preload=True, verbose=False, on_split_missing='warn')

    if maxfilter:
        print('Running maxfilter')
        calibration_file = f'{FABI_DIR}/utils/sss_cal.dat'
        cross_talk_file = f'{FABI_DIR}/utils/ct_sparse.fif'
        destination = Raw.get_fif_filename(subject_id=subject_id, run_nr=2)

        # find bad channels first
        noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(cur_data,
                                                                          coord_frame='meg',
                                                                          calibration=calibration_file,
                                                                          cross_talk=cross_talk_file  # noqa
                                                                          )
        cur_data.info['bads'] = noisy_chs + flat_chs

        cur_data = mne.preprocessing.maxwell_filter(cur_data,
                                                    calibration=calibration_file,
                                                    cross_talk=cross_talk_file,
                                                    coord_frame='meg',
                                                    #destination=destination,  #NOTE: set to (0, 0, 0.04) if you want to compare sensor data
                                                    st_fixed=False)

        #% make sure that if channels are set as bio that they get added correctly
    if 'BIO003' in cur_data.ch_names:
        cur_data.set_channel_types({'BIO001': 'eog',
                                    'BIO002': 'eog',
                                    'BIO003': 'ecg',})

        mne.rename_channels(cur_data.info, {'BIO001': 'EOG001',
                                            'BIO002': 'EOG002',
                                            'BIO003': 'ECG003',})

    if np.logical_or(l_pass != None, h_pass != None):
        cur_data.filter(l_freq=h_pass, h_freq=l_pass)

    if notch:
        cur_data.notch_filter(np.arange(50, 351, 50), filter_length='auto', phase='zero')

    if downsample_f != None:
        cur_data.resample(downsample_f, npad="auto")

    return cur_data