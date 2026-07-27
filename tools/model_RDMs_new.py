
#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from utils.files import fix_spm_rawdata_paths
import utils.plots
import joblib
import pandas as pd
import numpy as np
import rsatoolbox as rsa
# outDir: 

modelsDir = MODELS_DIR
os.makedirs(modelsDir, exist_ok=True)

#%% 
# ~~~ model RDMs for 6 conditionds ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
concept_df = pd.DataFrame({'Modality': [0, 0, 1, 1, 2, 2], 'Perceived': [0, 1, 0, 1, 0, 1]})
n_cond = len(concept_df)
ConsciousMat = (concept_df['Perceived'].values[:, None] != concept_df['Perceived'].values).astype(float)
SensoryMat = (concept_df['Modality'].values[:, None] != concept_df['Modality'].values).astype(float)
models_6 = [
    rsa.model.ModelFixed('sensory', SensoryMat),
    rsa.model.ModelFixed('conscious', ConsciousMat)
]
utils.plots.plot_rdm(models_6[0])
utils.plots.plot_rdm(models_6[1])
models_6_file = os.path.join(modelsDir, 'models_6.joblib')
joblib.dump(models_6, models_6_file)
#%% 
# ~~~ model RDMs for 24 conditions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# hits vs misses -> i.e. supramodal conscious access
# Modalities: Auditory, Somatosensory, Visual
modality_vec = np.concatenate([np.zeros((1,8)), np.ones((1,8)), np.ones((1,8))*2]).flatten()
perc_vec = np.tile(np.concatenate([np.zeros((1,4)), np.ones((1,4))]).flatten(), 3)
stim_vec = np.tile(np.array(np.array(np.arange(1,5))).flatten(), 6)

# RDM descriptors:
descr_modality_vec = np.concatenate([['aud_']*8, ['tac_']*8, ['vis_']*8])
descr_perc_vec = np.tile(np.concatenate([['hit']*4, ['miss']*4]), 3)
descr_stim_vec = np.tile([('_1'), ('_2'), ('_3'), ('_4') ], 6)
RDM_descriptor = descr_modality_vec + descr_perc_vec + descr_stim_vec
RDM_descriptor = [str(s) for s in RDM_descriptor]


data = {'Modality': modality_vec,
        'Perceived': perc_vec,
        'Stimulus': stim_vec}
concept_df = pd.DataFrame(data)
ConsciousMat = np.zeros((len(concept_df), len(concept_df)))
SensoryMat = np.zeros((len(concept_df), len(concept_df)))
for ii in range(0,len(concept_df)):
    for kk in range(0,len(concept_df)):
        
        #make the supramodal model
        if concept_df["Perceived"].iloc[ii] == concept_df["Perceived"].iloc[kk]:
            ConsciousMat[ii, kk] = 0
        else:
            ConsciousMat[ii, kk] = 1
        
        #make the sensory model    
        if concept_df["Modality"].iloc[ii] == concept_df["Modality"].iloc[kk]:
            SensoryMat[ii, kk] = np.abs(concept_df["Stimulus"].iloc[ii] - concept_df["Stimulus"].iloc[kk]) / 4
        else:
            SensoryMat[ii, kk] = 1
model_names = ['sensory', 'conscious']
models_24 = [rsa.model.ModelFixed('sensory', SensoryMat), rsa.model.ModelFixed('conscious', ConsciousMat) ]

models_24[0].rdm_obj.pattern_descriptors['condition'] = RDM_descriptor
models_24[1].rdm_obj.pattern_descriptors['condition'] = RDM_descriptor


utils.plots.plot_rdm(models_24[0])
utils.plots.plot_rdm(models_24[1])

models_24_file = os.path.join(modelsDir, 'models_24.joblib')
joblib.dump(models_24, models_24_file)


