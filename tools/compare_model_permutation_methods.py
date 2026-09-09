#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import joblib
import mne
import nibabel as nib
import numpy as np
from nilearn import datasets, image, plotting

from utils.load_cfg import load_fusion_config_instance, load_MEG_config_instance, load_MRI_config_instance
from utils.subj import *
from utils.fusion_stat import *
from utils.rsa import *
from utils.model import get_random_model_like
from utils.rsa import data3d_to_masked1d
from utils.rsa_eval import *
mne.viz.set_browser_backend('matplotlib')

from rsatoolbox.rdm.rdms import permute_rdms


#%%

all_subjects = get_MRI_subjects()
subjectID = '19840930bigs'
config_class_name = 'MRIconfig_E2' # not relevant, if e.g. "E1" or "E2" because searchlight radius not relevant here",

cfg = load_MRI_config_instance(config_class_name, subjectID)

common_centers_mask, mask = get_common_centers_masks(config_class_name)
mask_1d = common_centers_mask.flatten()


#%%
models = cfg.get_model_RDM()

model = models[2]
option = 3

if option == 1:
	rand1 = permute_rdms(model.rdm_obj)
	rand2 = permute_rdms(model.rdm_obj)
	rand3 = permute_rdms(model.rdm_obj)

elif option == 2:
	rand1 = get_random_model_like(n=1, like_model=model, method='random')
	rand2 = get_random_model_like(n=1, like_model=model, method='random')
	rand3 = get_random_model_like(n=1, like_model=model, method='random')

elif option == 3:
	rand1 = get_random_model_like(n=1, like_model=model, method='label')
	rand2 = get_random_model_like(n=1, like_model=model, method='label')
	rand3 = get_random_model_like(n=1, like_model=model, method='label')

if option in [1, 2, 3]:
	plot_rdm(rand1)
	plot_rdm(rand2)
	plot_rdm(rand3)