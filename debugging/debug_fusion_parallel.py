'''
Uses the code from "do_fusion_np_parallel.py" to debug and verify the results of the fusion 
procedure. 
- We load a fusion results file
- then we compute the sensory/suprasensory commanily manually for one timepoint/Searchlight
- then we compare the results to the fusion results file
- then we use matlab to verify the results
'''



import os
import sys

import joblib
import nibabel as nib
import numpy as np

sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import *
from utils.rsa import reorder_rdms
from utils.fusion import *

#%%

subjectID = '19910823ssld'
config_class_name = 'FusionConfig_C2'
meanMEG = False


print(f'\n-------------------------------------\n'\
        f'using the mean MEG RDM-movie: {meanMEG}\n'\
            '-------------------------------------\n')

print('[1] load config & data\n')

cfg = load_fusion_config_instance(config_class_name, subjectID) 
cfg.print_summary()
cfg.configure_paths()

outFiles = cfg.get_outFile_names()

print(f'Loading RDM-movie file {os.path.split(cfg.MEG_input)[1]}...\n')
rdm_movie = joblib.load(cfg.MEG_input) # load MEG RDM-movie of current subject

print(f'Loading SL_rdms file {os.path.split(cfg.MRI_input)[1]}...\n')
SL_rdms = joblib.load(cfg.MRI_input)

models = cfg.get_model_RDM()
mask = nib.load(cfg.maskFile) # we need this only for fransforming c-indiuces to fortran indices because linear indexing in matlab works differently 
s = mask.shape

#%%
print('[2] Reorder data\n')

SL_rdms, models = reorder_rdms(SL_rdms, models) # adjust model, if conditions were missing in fMRI data

subset_descriptors = models[0].rdm_obj.pattern_descriptors['condition']
rdm_movie = rdm_movie.subset_pattern('condition', subset_descriptors) # adjust MEG data if condition was missing in fMRI data

#%%
print('[3] Calculate commonality\n')

sensory_vec = models[0].rdm_obj.get_vectors()[0]
suprasensory_vec = models[1].rdm_obj.get_vectors()[0]


#%% --- starting here where we add some stuff for debugging/verifying the results

t_idx = 130

fusionFile = '/home/reabt/Desktop/ncc/fusion/FusionConfig_C2/commonalities/C2_19910823ssld_fusion.pkl'
good_clu_file = '/home/reabt/Desktop/ncc/fusion/FusionConfig_C2/stats/C2_good_clusters_7_subj_thres_0_05_sensory.pkl'

loaded_fusion_data = joblib.load(fusionFile)
good_clusters = joblib.load(good_clu_file)

fusion_t130 = loaded_fusion_data['data'][t_idx] 

#%% --- get one index of a good cluster and use it to find the correct index for the SL-rdm data:

cl_idx = np.where(np.array(good_clusters) == 1)
idx = 15795 # some index in the good cluster we use to accesss the 4d index tuple (I chosse one ast the timepoint 30 (which is tp 130 of the total time scale, of which we loaded the data as "t130"))
this_idx = (cl_idx[0][idx], cl_idx[1][idx], cl_idx[2][idx], cl_idx[3][idx]) # tuple of one 4d index 
idx_3d = this_idx[1:] # full-brain 3D index of one point from a good cluster
linear_idx = np.ravel_multi_index(idx_3d,good_clusters.shape[1:]) # turn this 3d index into a linear index
full_2d_indices = loaded_fusion_data['voxel_index_py'] # this is a list of the full-brain linear indices corresponding to the voxels we have searchlight data for 
SL_idx = np.where(full_2d_indices == linear_idx)[0][0] # find the list indx in voxel-info data, that corresponds to the full-brain voxel index, so we can ultimately access the SL_rdm data 


#%% --- manually use the function to compute the commonality and compare it to the results stores in the fusion data:

fMRI_vec = SL_rdms[SL_idx].get_vectors()[0]
MEG_vec = rdm_movie[t_idx].get_vectors()[0]

unique_sensory_pd, unique_suprasensory_pd = commonality_coeff_pd(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec, method = 'spearman')
unique_sensory_np, unique_suprasensory_np = commonality_coeff_np(fMRI_vec, MEG_vec, sensory_vec, suprasensory_vec, method='spearman')

print(f'unique_sensory_pd: {unique_sensory_pd}\n'
      f'unique_sensory_np: {unique_sensory_np}\n'
      f'unique_suprasensory_pd: {unique_suprasensory_pd}\n'
      f'unique_suprasensory_np: {unique_suprasensory_np}\n')

fusion_data_sensory = loaded_fusion_data['data'][t_idx]['sensory'][SL_idx]
fusion_data_suprasensory = loaded_fusion_data['data'][t_idx]['suprasensory'][SL_idx]

print(f'fusion_data_sensory: {fusion_data_sensory}\n'
      f'fusion_data_suprasensory: {fusion_data_suprasensory}')


#%% MATLAB code I ran to verify the results
'''
addpath('C:\Users\mrsre\Documents\docs papers & literature\NCC\Supplementary Material\Hebart et al (2018)  supplementary\helper_functions')

fMRI_vec = [-1.78229763e-05 -9.73027974e-05 6.36024647e-05 -5.48315729e-05 -5.32970199e-05 -1.96552062e-05 8.25579567e-06 1.08810096e-06 -8.43398228e-05 -9.61705350e-06 3.19831176e-05 -1.45009751e-06 -4.01163259e-05 4.51624230e-05 -1.96970369e-05 9.49120813e-06 1.25757988e-04 1.85181946e-04 2.32931297e-04 2.03913814e-04 1.25325262e-04 -7.71504651e-06 5.52962322e-05 -9.12716162e-05 -4.08622615e-05 -5.14621707e-05 -9.27683620e-06 -1.43060404e-05 1.42940801e-05 8.96918639e-05 -1.24645495e-05 1.02805576e-04 -4.39647402e-06 8.00987341e-05 1.77832931e-05 3.78187816e-05 1.08137993e-04 2.04936602e-05 7.40702337e-05 1.99619191e-04 2.26561288e-04 2.48586764e-04 1.86816409e-04 1.00766381e-04 1.24270202e-04 -4.78276204e-05 -5.46814109e-05 -8.73400003e-05 -1.03324275e-05 -3.84712307e-05 -1.20801166e-05 -1.45925396e-04 -2.08965372e-05 -4.76897523e-05 -8.98424376e-06 -5.06974613e-05 3.35897041e-05 -8.84905186e-06 -1.17214787e-04 -2.93666783e-05 1.34154044e-05 5.67240471e-05 6.75660379e-05 4.07439046e-05 -8.01722709e-05 -5.60197032e-06 -1.05917360e-04 3.67859466e-05 1.24081947e-05 6.98389692e-05 2.62638987e-05 5.77185692e-05 6.59783237e-05 7.67080157e-06 6.19326327e-05 -4.24622815e-05 2.19460795e-05 7.39131286e-05 1.64966723e-05 6.34174350e-05 1.35552773e-04 6.77457143e-05 8.18436412e-05 1.51525936e-04 5.04341953e-05 9.40862057e-05 -4.90958496e-05 -2.11068697e-05 2.30166629e-05 6.84445961e-06 -1.76997231e-05 3.37747613e-05 -3.50713768e-05 -8.35543788e-06 -4.36681204e-05 -7.62662444e-05 1.14439388e-05 -5.10926853e-05 3.05658975e-05 1.27407147e-04 1.35060451e-04 1.20268842e-04 1.05201493e-04 -2.32234449e-05 -4.33498604e-06 -1.86377660e-05 -1.87212509e-05 2.28820458e-05 -8.28415277e-05 5.50441441e-05 2.01596625e-05 6.55371654e-05 2.65386422e-06 1.33173080e-04 7.37496487e-05 5.92003618e-05 1.13879117e-04 1.56062657e-04 2.00922003e-04 2.54364748e-04 1.39339350e-04 2.55177645e-06 6.99303452e-05 4.46720980e-05 3.31354177e-05 -9.30140293e-06 6.10030724e-05 3.78444989e-05 8.44288720e-05 -1.83519654e-05 6.05228179e-05 8.06440958e-05 8.35653744e-05 3.37032679e-05 1.82729117e-04 1.97411467e-04 3.14969269e-04 1.07206332e-04 8.17849666e-05 9.48948588e-05 7.10709188e-05 -1.89088531e-05 4.73767149e-05 -3.94406655e-05 7.75368147e-05 -5.28334895e-05 1.59765143e-05 -3.52397594e-06 1.19502013e-04 8.30593124e-05 2.20756099e-04 3.17222804e-04 3.16705272e-04 1.32967407e-04 8.67750210e-05 2.01033165e-04 3.00751149e-05 -1.91543735e-05 7.37176667e-05 1.07278785e-04 -1.03793654e-04 1.47901811e-05 -2.78714523e-05 8.57061546e-05 1.57922772e-05 1.39301702e-04 1.13037445e-04 1.14237342e-04 2.81006075e-05 -1.71138306e-05 -5.63460981e-06 8.72356254e-05 9.85131195e-05 3.59710096e-05 -2.22255994e-05 9.93040910e-05 2.36731361e-05 8.21725556e-05 1.07605435e-04 1.76511159e-04 2.15986189e-04 3.01575850e-04 1.40500137e-04 8.02502232e-05 1.38373519e-04 5.19395541e-05 1.81437255e-04 -5.66065015e-05 1.08818522e-05 3.66238251e-06 1.48345719e-05 4.09163738e-05 1.37109842e-04 1.20769721e-04 7.90157144e-07 -7.59035443e-07 -4.52916336e-05 -3.42176588e-05 7.38634213e-05 -4.13230235e-05 6.73639084e-05 3.21012693e-05 1.76775045e-04 1.05050764e-04 2.58164875e-04 2.68265574e-04 2.43359283e-04 1.76751958e-04 9.23488230e-05 2.04469387e-04 2.57002829e-05 1.27199317e-04 -6.01659567e-05 1.43671830e-04 1.78628310e-04 2.10366180e-04 2.87025652e-04 3.36956884e-04 2.76908493e-04 1.53657723e-04 1.54121560e-04 -1.13740017e-04 -7.64927675e-05 1.35414432e-04 -4.28068434e-05 1.61105564e-04 1.40555264e-04 1.40674010e-04 -2.20728930e-05 -2.48307518e-05 6.64451655e-06 4.76133800e-08 1.09245315e-04 1.21265661e-04 2.66814874e-04 2.14359806e-04 1.08725606e-04 1.71449216e-04 5.93588886e-05 3.96295121e-05 4.68421077e-05 4.17210249e-05 1.14555308e-04 2.04656678e-04 1.61372737e-04 6.49249477e-05 7.81915173e-05 1.41842182e-05 2.74105037e-05 -3.51450640e-05 1.69699911e-05 6.63385633e-05 7.43377548e-06 1.04850928e-04 1.57503640e-05 2.53561952e-05 5.29562947e-05 1.06958795e-04 1.40581270e-05 3.24644142e-05 -1.96605383e-05 1.32989622e-05 1.52736722e-04 -3.44255201e-06 8.08646017e-05 -3.47724840e-05 8.61184841e-05 -1.47295450e-06 1.09436074e-04 1.43220426e-05 -2.61524558e-05 7.51966484e-05 3.06209362e-05 -2.37782194e-05 -2.70917360e-05 -3.87079315e-05]';
MEG_vec = [-3.06555341e-04 4.96359350e-04 2.42283121e-04 -2.54277765e-03 -3.06437587e-03 -2.88594071e-03 1.12797070e-03 2.55469373e-03 -2.65988272e-03 -9.16128574e-04 -3.21965642e-04 -6.90764053e-04 -1.40705822e-03 -8.94428155e-04 -2.16061966e-03 -2.49014427e-03 1.84226577e-04 1.63995511e-03 -2.16946529e-04 -5.95370659e-04 -1.57520114e-03 -3.23346929e-03 -3.52725820e-03 -1.83749685e-04 1.38370825e-03 4.48049395e-03 4.92698690e-03 1.40906625e-03 1.37400624e-03 7.06126901e-03 3.82897123e-03 6.54952687e-03 8.74687299e-03 4.27137732e-03 3.81118287e-03 3.65595168e-03 6.68604093e-03 8.73646442e-03 6.34914182e-03 9.27576779e-03 9.59743603e-03 2.91970552e-03 9.11763646e-03 6.34998886e-04 4.14482334e-03 -6.52924500e-04 4.07890761e-03 2.48702290e-03 3.50989456e-03 1.56832104e-03 5.14136831e-03 3.58849638e-03 5.27092256e-03 7.68661277e-03 3.77117297e-03 4.40472421e-03 4.48804243e-03 2.30595309e-03 4.80381474e-03 4.33330022e-03 3.18387931e-03 7.03465253e-03 5.32595490e-03 5.68040881e-03 2.88408977e-03 2.15321566e-03 5.07400351e-03 2.37513545e-03 5.41379178e-03 5.91893491e-03 7.55872207e-03 5.54944659e-03 5.53734354e-03 5.75475115e-03 7.09663439e-03 5.29373909e-03 3.65528138e-03 3.61692793e-03 6.99924004e-03 5.47378387e-03 4.17779663e-03 5.81723115e-03 2.21920194e-03 7.85153579e-03 1.87809736e-03 4.93979286e-03 -3.07835764e-04 1.40117583e-03 3.45685395e-03 2.62929770e-03 -5.72778121e-05 9.71978895e-04 2.70047544e-03 1.06830685e-03 2.28838494e-03 1.87543534e-03 -8.89673422e-04 2.18391132e-03 2.87549399e-03 3.01439769e-03 1.83390976e-03 4.70720412e-03 1.17542323e-03 -1.04539956e-03 -5.43762971e-05 1.10213092e-03 2.08522992e-03 1.25904951e-03 1.53605653e-03 -1.33988139e-03 2.33498834e-03 1.43101616e-03 1.77173818e-03 -1.09207510e-03 -4.97528299e-04 1.00136588e-03 1.79747918e-03 1.44247137e-03 -6.34724344e-04 -3.71056969e-04 1.46994516e-03 -5.34871318e-04 1.38999066e-03 5.28989514e-03 3.83604576e-03 2.87139511e-03 2.09825725e-03 6.20419303e-03 5.91162566e-04 1.50513034e-03 1.11667808e-03 3.02739019e-03 4.34077912e-03 3.40700147e-03 5.27654255e-03 4.72471542e-03 2.10167461e-03 2.98281448e-03 -1.41114560e-03 6.69111477e-04 4.96422220e-03 1.32477403e-03 4.92444394e-03 6.22731768e-03 2.30504271e-03 2.30522130e-03 2.78969649e-03 5.96942382e-04 5.38584199e-03 2.52041094e-03 7.08603423e-04 3.70381987e-03 4.32163497e-04 3.10430053e-03 1.84869004e-03 4.94515214e-03 1.53205681e-04 -1.31487717e-04 1.62876932e-03 2.62557736e-03 2.45387319e-03 2.74573831e-03 1.98461807e-03 2.68507608e-03 5.58176864e-03 7.50027218e-03 3.99914554e-03 2.98207311e-03 2.81654617e-03 2.70099703e-03 3.69053219e-03 -1.45576961e-03 -8.20045071e-04 1.93579754e-03 -5.90181005e-04 1.00789787e-03 -4.59036527e-04 2.71357612e-03 1.47895009e-03 2.54464587e-03 -5.07492763e-04 2.02142712e-03 1.57735347e-03 -2.73514908e-04 2.60141022e-03 1.45185895e-03 2.17948865e-03 1.20076006e-03 -4.76960828e-04 2.82795226e-04 1.53886195e-03 2.11027484e-03 5.19733209e-03 1.07973091e-03 1.38971080e-03 2.84930701e-03 1.31035392e-03 7.17741279e-04 4.89315214e-03 2.96150712e-03 3.79209377e-03 8.93822963e-04 4.83912286e-03 4.37350457e-03 5.61713561e-03 4.45546127e-03 4.13622154e-03 3.65934212e-03 2.13335283e-03 7.11115487e-03 1.09509444e-03 3.84903289e-03 -1.13927891e-04 3.01828167e-03 2.67254586e-03 3.16368711e-03 1.44287774e-03 2.92928216e-04 1.39850606e-03 3.63574530e-04 -2.82743317e-05 4.49738385e-04 1.38948574e-03 2.24275102e-03 3.03230336e-03 3.90314595e-03 2.86790127e-03 1.12869593e-03 3.47127078e-03 -1.22018746e-04 1.98622035e-03 1.30914749e-03 2.82190674e-03 3.37876329e-03 4.22577351e-03 1.39744041e-03 9.59486187e-04 9.53956910e-04 -4.13089387e-04 2.20898206e-03 1.35504270e-03 -1.32132935e-04 1.64384079e-03 2.40395594e-04 4.75540559e-04 -1.28160157e-03 4.81166107e-04 1.22404929e-03 8.61630812e-04 3.80209621e-03 -5.58272549e-04 4.26286592e-03 2.62265580e-04 4.32099600e-03 9.27240278e-04 -4.35183312e-04 2.83240299e-04 2.00789959e-03 -8.38856612e-04 3.10630910e-03 3.07035522e-03 1.19644850e-03 1.30871109e-03 2.88147908e-03 2.47885182e-03 3.75189438e-03 2.88983699e-03 1.14587983e-03 2.54696144e-05 1.73267732e-03 4.63397657e-03 7.01462633e-04 -5.13124786e-04 1.96688353e-03 4.99415484e-03 4.18554748e-04]';	

sensory_vec = [0.25 0.5 0.75 0 0.25 0.5 0.75 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.25 0.5 0.25 0 0.25 0.5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.25 0.5 0.25 0 0.25 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.75 0.5 0.25 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.25 0.5 0.75 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.25 0.5 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.25 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.25 0.5 0.75 0 0.25 0.5 0.75 1 1 1 1 1 1 1 1 0.25 0.5 0.25 0 0.25 0.5 1 1 1 1 1 1 1 1 0.25 0.5 0.25 0 0.25 1 1 1 1 1 1 1 1 0.75 0.5 0.25 0 1 1 1 1 1 1 1 1 0.25 0.5 0.75 1 1 1 1 1 1 1 1 0.25 0.5 1 1 1 1 1 1 1 1 0.25 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.25 0.5 0.75 0 0.25 0.5 0.75 0.25 0.5 0.25 0 0.25 0.5 0.25 0.5 0.25 0 0.25 0.75 0.5 0.25 0 0.25 0.5 0.75 0.25 0.5 0.25]'; 
suprasensory_vec = [0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0]';

% this was from the original code:
% rMEG_MRItaskobj = correlate([y xMRI xtask xobj],'type','spearman','method','semipartialcorr');
% rMEG_MRIobj = correlate([y xMRI xobj],'type','spearman','method','semipartialcorr');
% rMEG_MRItask = correlate([y xMRI xtask],'type','spearman','method','semipartialcorr');
% 
% CMEGMRItask(i_time,j_roi) = rMEG_MRIobj(2,1).^2-rMEG_MRItaskobj(2,1).^2;
% CMEGMRIobj(i_time,j_roi) = rMEG_MRItask(2,1).^2-rMEG_MRItaskobj(2,1).^2;


% This is the computation with our data: 
rMEG_MRI_sens_supra = correlate([MEG_vec fMRI_vec sensory_vec suprasensory_vec],'type','spearman','method','semipartialcorr');
rMEG_MRI_supra = correlate([MEG_vec fMRI_vec suprasensory_vec],'type','spearman','method','semipartialcorr');
rMEG_MRI_sens = correlate([MEG_vec fMRI_vec sensory_vec],'type','spearman','method','semipartialcorr');

% originale Schreibweise
% unique_sensory = rMEG_MRI_supra(2,1).^2-rMEG_MRI_sens_supra(2,1).^2;
% unique_suprasensory = rMEG_MRI_sens(2,1).^2-rMEG_MRI_sens_supra(2,1).^2;

% more readable, but the same as above:

semi_both = rMEG_MRI_sens_supra(2,1).^2;
semi_sensory = rMEG_MRI_sens(2,1).^2;
semi_suprasensory = rMEG_MRI_supra(2,1).^2;

unique_sensory = semi_suprasensory - semi_both
unique_suprasensory = semi_sensory - semi_both

'''