
import os
import sys
import re
from glob import glob
import seaborn as sns

import joblib
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rsatoolbox as rsa

sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_config_instance
from utils.compile import compile_SL_rdms_files
from plus_slurm import Job
import warnings
from collections import defaultdict
#%%

subjectID = '19910823ssld'
config_class_name = 'SetupConfig1'
maskNr = 1

# 1) get config 
print('1) getting config')

cfg = load_config_instance(config_class_name, subjectID) 
outFiles = cfg.get_outFile_names()

compiled_Prefix = os.path.join(cfg.outDir, f'{cfg.filePrefix_compiled}') # add suffix before saving
inDir = cfg.outDir

# 2) match output files of partial data with partial brain mask file
print('2) match output files of partial data with partial brain mask file')
      
sl_rdms_pattern = re.sub(rf'({re.escape(cfg.prefix)}_)\d{{1,2}}', r'\1*', os.path.basename(outFiles['SL_rdms']))
eval_results_pattern = re.sub(rf'({re.escape(cfg.prefix)}_)\d{{1,2}}', r'\1*', os.path.basename(outFiles['eval_results']))

SL_rdms_files = sorted(glob(os.path.join(inDir, sl_rdms_pattern)))
eval_results_files = sorted(glob(os.path.join(inDir, eval_results_pattern)))


print(f"Found {len(eval_results_files)} files:")

mask_numbers = []
pattern = re.compile(rf'{cfg.prefix}_(\d{{1,2}})') 

# Extract mask numbers for each file
for file in eval_results_files:
    match = pattern.search(os.path.basename(file))
    if match:
        mask_numbers.append(int(match.group(1)))
    else:
        # In case pattern doesn't match
        msg = f"Could not extract mask number from file: {file}. \n File will be left out in the full-brain compilation"
        warnings.warn(msg)

paired = sorted(zip(mask_numbers, SL_rdms_files, eval_results_files))
paired = [(num, SL_file, eval_file) for num, SL_file, eval_file in paired if num is not None] # Remove any pairs where mask number is None

# Unzip sorted pairs
mask_numbers_sorted, SL_rdms_files_sorted, eval_results_files_sorted = zip(*paired)

# 3) Load and append eval_results variables
print('3) Load and append eval_results variables ')


compiled_data = []
for num, SL_file, eval_file in zip(mask_numbers_sorted, SL_rdms_files_sorted, eval_results_files_sorted):
    # if num > 1:
    #     break
    SL_rdms = joblib.load(SL_file)
    eval_results = joblib.load(eval_file)
    '''
    for plotting:
    eval_results_bool = np.zeros(eval_results.shape)
    eval_results_bool[eval_results > 0] = 1
    RDM_img = nib.Nifti1Image(eval_results_bool, affine = nib.load(cfg.maskFile).affine)
    plot_overlay_on_mask(RDM_img, nib.load(cfg.maskFile))
    '''
    eval_score = [float(e.evaluations.item()) for e in eval_results]

    compiled_data.append({'mask_number': num, 
                        'eval_score': eval_score,
                        'voxel_index': SL_rdms.rdm_descriptors['voxel_index']
                        })
    

# 3.2) 

all_SL_rdms = compile_SL_rdms_files(SL_rdms_files_sorted)




#%%

# 4) check for duplicats and see if eval_scores at duplicate voxels match
print('4) check for duplicats and see if eval_scores at duplicate voxels match')
# Combine all eval_score and voxel_index rows in order
all_eval_scores = []
all_voxel_indices = []
for d in compiled_data:
    all_eval_scores.extend(d['eval_score'])
    all_voxel_indices.extend(d['voxel_index'])

# Check for duplicate voxel indices and compare their eval_scores
voxel_to_indices = defaultdict(list)
for idx, voxel in enumerate(all_voxel_indices):
    voxel_to_indices[voxel].append(idx)

duplicates = {voxel: inds for voxel, inds in voxel_to_indices.items() if len(inds) > 1}

all_equal = True
for voxel, inds in duplicates.items():
    values = [all_eval_scores[i] for i in inds]
    if not all(v == values[0] for v in values):
        all_equal = False
        print(f"Voxel index {voxel} has different eval_scores at indices {inds}: values={values}")

if all_equal:
    print(f"All duplicate voxel indices have the same eval_score values.")
else:
    print(f"Some duplicate voxel indices have different eval_score values!")

# Find indices of unique voxel indices (keep first occurrence)
_, unique_indices = np.unique(all_voxel_indices, return_index=True)
unique_indices_sorted = np.sort(unique_indices)


'''sanity check to verify nothing goes wrong the way the data is filtered
________________________________________________________________________________

all_voxel_indices_new = [all_voxel_indices[i] for i in unique_indices_sorted]
all_eval_scores_new = [all_eval_scores[i] for i in unique_indices_sorted]

thisIdx = 50000

old_indx = all_voxel_indices[thisIdx]
old_val = all_eval_scores[thisIdx]

# 1) index where new variable has same voxel index
new_indx = np.where(all_voxel_indices_new == old_indx)[0][0]
# 2) now check if eval score at this index is the sam as in the old variable 
new_val = all_eval_scores_new[new_indx]

old_val == new_val
________________________________________________________________________________
'''


# Filter all_voxel_indices and all_eval_scores to remove duplicates

all_voxel_indices = [all_voxel_indices[i] for i in unique_indices_sorted]
all_eval_scores = [all_eval_scores[i] for i in unique_indices_sorted]

# get the maximum and the voxel index of the maximum

max_eval_scores_idx = np.argmax(all_eval_scores) # find index of highest value in all_eval_scores
max_voxel_idx = all_voxel_indices[max_eval_scores_idx] # use max_eval_scores_idx to index max_eval_scores_idx so we now the actual voxel index of the maximum
max_SL_rdms_idx = np.where(all_SL_rdms.rdm_descriptors['voxel_index'] == max_voxel_idx) # SL_rdm with highest model match


#%%
# 5) Create Nifti image from combined results

print('5) Create Nifti image from combined results')

mask = nib.load(cfg.maskFile) 
RDM_brain = np.zeros(np.prod(mask.shape))
RDM_brain[all_voxel_indices] = all_eval_scores
RDM_brain = RDM_brain.reshape(mask.shape)
RDM_brain_img = nib.Nifti1Image(RDM_brain, affine = mask.affine)
nib.save(RDM_brain_img, f'{compiled_Prefix}_RDM_brain.nii' )

# 6) plotting
print('6) plotting')

coords = range(-20, 50, 5)

# ~~~ Plot Histogramm
print('     - Histogramm')

fig = plt.figure()
sns.histplot(all_eval_scores, kde=True)
plt.title(f'compiled SL RSA results: {cfg.subjectID}', size=18)
plt.ylabel('Occurance')
plt.xlabel(f'{cfg.RSAmethod}')
sns.despine()
plt.show()
plt.savefig(f'{compiled_Prefix}_eval_score_hist.png', dpi=300)
plt.close()

# ~~~ Plot eval_score on Brain normal
print('     - eval_score on Brain')

fig = plt.figure(figsize=(12, 2))
plotting.plot_stat_map(
    RDM_brain_img, 
    colorbar=True, 
    threshold=0,
    cut_coords=coords,
    display_mode='z', 
    draw_cross=False, 
    figure=fig,
    title=f'{cfg.plot2_title}', 
    # cmap= mcolors.ListedColormap(['black', 'red', 'blue']),
    # cmap = 'viridis',
    black_bg=False, 
    annotate=False,
    vmin = -0.9,
    vmax = 0.9
)
plt.show()
plt.savefig(f'{compiled_Prefix}_eval_score_brain.png', dpi=300)
plt.close()

# ~~~ plot eval score on brain + fill the gaps where zeros were

zero_indices = [i for i, val in enumerate(all_eval_scores) if val == 0] # Indices where all_eval_scores is zero
zero_voxels = [all_voxel_indices[i] for i in zero_indices]
overlay_data = np.zeros(mask.shape)
overlay_data[np.unravel_index(zero_voxels, mask.shape)] = 1  # Mark centers
overlay_img = nib.Nifti1Image(overlay_data, affine = mask.affine)

fig = plt.figure(figsize=(12, 2))

display = plotting.plot_stat_map(
    stat_map_img = RDM_brain_img, 
    # bg_img=mask,
    colorbar=True, 
    threshold=0,
    cut_coords=coords,
    display_mode='z', 
    draw_cross=False, 
    figure=fig,
    title=f'{cfg.plot2_title}', 
    black_bg=False, 
    annotate=False,
    vmin = -0.9,
    vmax = 0.9
)

display.add_overlay(
    overlay_img
)

plt.show()
plt.savefig(f'{compiled_Prefix}_eval_score_brain_fill0.png', dpi=300)
plt.close()

# ~~~ plot only the zeros on top of the rest of the data

fig = plt.figure(figsize=(12, 2))

display = plotting.plot_stat_map(
    stat_map_img = RDM_brain_img, 
    # bg_img=mask,
    colorbar=True, 
    threshold=0,
    cut_coords=coords,
    display_mode='z', 
    draw_cross=False, 
    figure=fig,
    title=f'{cfg.plot2_title}', 
    cmap= mcolors.ListedColormap('blue'),
    black_bg=False, 
    annotate=False,
    vmin = -0.9,
    vmax = 0.9
)

display.add_overlay(
    overlay_img,
    cmap=mcolors.ListedColormap(['red'])
)

plt.show()
plt.savefig(f'{compiled_Prefix}_eval_score_brain_show0.png', dpi=300)
plt.close()

# ~~~ plot RDM with maximum model match
print('     - RDM with maximum model match')


fig = rsa.vis.show_rdm(
    all_SL_rdms[max_SL_rdms_idx], 
    pattern_descriptor = 'condition', 
    num_pattern_groups=4,
    cmap = 'viridis',
    figsize = (5, 5),
    show_colorbar='panel',
    rdm_descriptor = f'RDM at max. {cfg.modelType} model match: {cfg.subjectID}\n', 
    )

outFile = os.path.join(cfg.outDir, f'{cfg.filePrefix}_RDM_maxFit_{cfg.modelType}.png')
plt.savefig(outFile, dpi='figure',  bbox_inches='tight')
plt.close()

# ~~~ plot coordinates of maximum model match
print('     - coordinates of maximum model match')

max_coord = np.unravel_index(np.argmax(RDM_brain), RDM_brain.shape) # Get coordinates of maximum value in RDM_brain
print(f"Coordinates of maximum value in RDM_brain: {max_coord}")

maxPoint = np.zeros(mask.shape)
maxPoint[max_coord] = 1
maxPoint_img = nib.Nifti1Image(maxPoint, affine = mask.affine)

plotting.plot_stat_map(maxPoint_img, 
                    #    draw_cross = False, 
                    cmap = mcolors.ListedColormap(['black', 'red']),
                    colorbar = False,
                    title = f'Point of max. {cfg.modelType} model match: {cfg.subjectID}')

outFile = os.path.join(f'{compiled_Prefix}_maxFit_coords_{cfg.modelType}.png')
plt.savefig(outFile, dpi='figure',  bbox_inches='tight')
plt.close()

# 7) save output variables

joblib.dump(RDM_brain, outFiles['RDM_brain_compiled'])
joblib.dump(all_SL_rdms, outFiles['SL_rdms_compiled'])

joblib.dump({'eval_scores': all_eval_scores, 
            'voxel_indices': all_voxel_indices}, 
            f'{compiled_Prefix}_eval_scores_and_indices.pkl')
