
#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import joblib
import numpy as np


thisDir = os.path.join(MEG_DATA_DIR, 'movie_RDMs/SetupConfig_C')

file_list = [f for f in os.listdir(thisDir) if f.endswith('.pkl')]
for fname in file_list:
    print(f'\nProcessing {fname}')
    fpath = os.path.join(thisDir, fname)
    rdm_movie = joblib.load(fpath)
    rdm_movie.reorder(np.argsort(rdm_movie.pattern_descriptors['condition']))
    joblib.dump(rdm_movie, fpath)

