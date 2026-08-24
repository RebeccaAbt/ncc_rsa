#%%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

sys.path.append(FABI_DIR)

import mne
from utils.ica import *
from utils.ica import _block_key
from utils.raw import _get_ncc_block_indices
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
mne.viz.set_browser_backend('qt')
#%%

subject_id = '19970302urmr'

block_indices = _get_ncc_block_indices(subject_id)

_, outFiles = get_blockwise_outFilePaths(
	subject_id=subject_id,
	out_root=ICA_BLOCK_DIR,
	block_indices=block_indices,
)	


for block in block_indices:
	print('block:', block)
	key = _block_key(block)
	block_files = outFiles["blocks"][key]

	raw_file = block_files["file_raw"]

	raw = mne.io.read_raw_fif(raw_file, preload=True)


	raw.plot(block=True, butterfly=False,group_by = 'original', n_channels = 180)#, scalings = dict(mag=1e-12, grad=2e-11))   # dict(mag=1e-12, grad=4e-11)

	print(raw)

	print('writing file...', flush=True)
	raw.save(raw_file, overwrite=True)

#%%

