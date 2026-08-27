
#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.files import fix_spm_rawdata_paths
import utils.plots
import joblib
import pandas as pd
import numpy as np
import rsatoolbox as rsa
import matplotlib.pyplot as plt

# outDir: 

modelsDir = MODELS_DIR
os.makedirs(modelsDir, exist_ok=True)

#%% 
# ~~~ model RDMs for 6 conditionds ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
concept_df = pd.DataFrame({'Modality': [0, 0, 1, 1, 2, 2], 'Perceived': [0, 1, 0, 1, 0, 1]})
n_cond = len(concept_df)
SuprasensMat = (concept_df['Perceived'].values[:, None] != concept_df['Perceived'].values).astype(float)
SensoryMat = (concept_df['Modality'].values[:, None] != concept_df['Modality'].values).astype(float)
models_6 = [
    rsa.model.ModelFixed('sensory', SensoryMat),
    rsa.model.ModelFixed('suprasensory', SuprasensMat)
]
utils.plots.plot_rdm(models_6[0])
utils.plots.plot_rdm(models_6[1])
models_6_file = os.path.join(modelsDir, 'models_6.joblib')
# joblib.dump(models_6, models_6_file)
#%% 

def plot_rdm2(RDM, kwargs = {}):
    
    thisRDM = RDM.predict_rdm()
    rsa.vis.show_rdm(thisRDM, **kwargs)
    plt.show(thisRDM)


# ~~~ model RDMs for 24 conditions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# hits vs misses -> i.e. supramodal suprasensory access
# Modalities: Auditory, Somatosensory, Visual
modality_vec = np.concatenate([np.zeros((1,8)), np.ones((1,8)), np.ones((1,8))*2]).flatten()
perc_vec = np.tile(np.concatenate([np.zeros((1,4)), np.ones((1,4))]).flatten(), 3)
stim_vec = np.tile(np.array(np.array(np.arange(1,5))).flatten(), 6)

# RDM descriptors:
descr_modality_vec = np.concatenate([['aud_']*8, ['tac_']*8, ['vis_']*8])
descr_perc_vec = np.tile(np.concatenate([['hit']*4, ['miss']*4]), 3)
descr_stim_vec = np.tile([('_1'), ('_2'), ('_3'), ('_4')], 6)
RDM_DESCRIPTOR = [f"{m}{p}{s}" for m, p, s in zip(descr_modality_vec, descr_perc_vec, descr_stim_vec)]


data = {'Modality': modality_vec,
        'Perceived': perc_vec,
        'Stimulus': stim_vec}

concept_df = pd.DataFrame(data)
SuprasensMat = np.zeros((len(concept_df), len(concept_df)))
SensoryMat = np.zeros((len(concept_df), len(concept_df)))
for ii in range(0,len(concept_df)):
    for kk in range(0,len(concept_df)):
     
        #make the supramodal model
        if concept_df["Perceived"].iloc[ii] == concept_df["Perceived"].iloc[kk]:
            SuprasensMat[ii, kk] = 0
        else:
            SuprasensMat[ii, kk] = 1
        
        #make the sensory model    
        if concept_df["Modality"].iloc[ii] == concept_df["Modality"].iloc[kk]:
            SensoryMat[ii, kk] = np.abs(concept_df["Stimulus"].iloc[ii] - concept_df["Stimulus"].iloc[kk]) / 4
        else:
            SensoryMat[ii, kk] = 1
model_names = ['sensory', 'suprasensory']
models_24 = [rsa.model.ModelFixed('sensory', SensoryMat), rsa.model.ModelFixed('suprasensory', SuprasensMat) ]

models_24[0].rdm_obj.pattern_descriptors['condition'] = RDM_DESCRIPTOR
models_24[1].rdm_obj.pattern_descriptors['condition'] = RDM_DESCRIPTOR

plot_rdm2(models_24[0], dict(show_colorbar = 'panel', pattern_descriptor = 'condition'))
plot_rdm2(models_24[1], dict(show_colorbar = 'panel', pattern_descriptor = 'condition'))


models_24_file = os.path.join(modelsDir, 'models_24.joblib')
# joblib.dump(models_24, models_24_file)

#%% save models separately

model_names = ['sensory', 'suprasensory']

sensory_model = rsa.model.ModelFixed('sensory', SensoryMat)
sensory_model.rdm_obj.pattern_descriptors['condition'] = RDM_DESCRIPTOR
sensory_model_file = os.path.join(modelsDir, 'model_sensory.joblib')
joblib.dump(sensory_model, sensory_model_file)

suprasensory_model = rsa.model.ModelFixed('suprasensory', SuprasensMat)
suprasensory_model.rdm_obj.pattern_descriptors['condition'] = RDM_DESCRIPTOR
suprasensory_model_file = os.path.join(modelsDir, 'model_suprasensory.joblib')
joblib.dump(suprasensory_model, suprasensory_model_file)

models_24[0].rdm_obj.pattern_descriptors['condition'] = RDM_DESCRIPTOR
models_24[1].rdm_obj.pattern_descriptors['condition'] = RDM_DESCRIPTOR

#%% trying to make differnt models

# SuprasensMat = np.zeros((len(concept_df), len(concept_df)))
SensoryMat2 = np.zeros((len(concept_df), len(concept_df)))
for ii in range(0,len(concept_df)):
    for kk in range(0,len(concept_df)):
        if concept_df["Modality"].iloc[ii] == concept_df["Modality"].iloc[kk]:
            SensoryMat2[ii, kk] = 0 # np.abs(concept_df["Stimulus"].iloc[ii] - concept_df["Stimulus"].iloc[kk]) / 4
        else:
            SensoryMat2[ii, kk] = 1

sensory_model2 = rsa.model.ModelFixed('sensory2', SensoryMat2)
sensory_model2.rdm_obj.pattern_descriptors['condition'] = RDM_DESCRIPTOR
sensory_model2_file = os.path.join(modelsDir, 'model_sensory2.joblib')
joblib.dump(sensory_model2, sensory_model2_file)

plot_rdm2(sensory_model2, dict(show_colorbar = 'panel', pattern_descriptor = 'condition'))

#%% create from full matrix
# shuffle the lower triangle and mirror it to the upper triangle
rng = np.random.default_rng()
lower_idx = np.tril_indices_from(SensoryMat2, k=-1)
flat_lower = SensoryMat2[lower_idx].copy()
rng.shuffle(flat_lower)
shuffled = np.zeros_like(SensoryMat2)
shuffled[lower_idx] = flat_lower
shuffled = shuffled + shuffled.T
np.fill_diagonal(shuffled, 0)
SensoryMat_shuffeled = shuffled

fig, ax = plt.subplots(figsize=(13, 3), ncols=1)
pos = ax.imshow(SensoryMat_shuffeled, interpolation='none')
cbar = fig.colorbar(pos, ax=ax)
plt.show()

#%% create from model
# Shuffle modeldata, a 1D vector containing the lower triangle, and mirror it.

modeldata = np.asarray(sensory_model2.rdm_obj.get_vectors()).ravel()
n_conditions = int((1 + np.sqrt(1 + 8 * modeldata.size)) / 2)
if n_conditions * (n_conditions - 1) // 2 != modeldata.size:
    raise ValueError('modeldata must contain exactly the lower triangle data.')

rng = np.random.default_rng()
rng.shuffle(modeldata)
lower_idx = np.tril_indices(n_conditions, k=-1)
SensoryMat_shuffeled = np.zeros((n_conditions, n_conditions), dtype=modeldata.dtype)
SensoryMat_shuffeled[lower_idx] = modeldata
SensoryMat_shuffeled += SensoryMat_shuffeled.T
np.fill_diagonal(SensoryMat_shuffeled, 0)

fig, ax = plt.subplots(figsize=(13, 3), ncols=1)
pos = ax.imshow(SensoryMat_shuffeled, interpolation='none')
cbar = fig.colorbar(pos, ax=ax)
plt.show()


#%%

# check if the correlation between the two model RDMs is significant
mask = np.triu(np.ones_like(SuprasensMat, dtype=bool), k=1)
Suprasens_vec = SuprasensMat[mask]
Sensory_vec = SensoryMat2[mask]

from scipy import stats
r, p = stats.pearsonr(Suprasens_vec, Sensory_vec)
print(f'Correlation between SuprasensMat and SensoryMat: r = {r:.4f}, p = {p:.6g}')
if p < 0.05:
    print('The correlation is statistically significant (p < 0.05).')
else:
    print('The correlation is not statistically significant (p >= 0.05).')

# SuprasensMat
# SensoryMat


#%%

sensory_model2.rdm