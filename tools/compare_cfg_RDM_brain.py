#%%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

from utils.plots import plot_overlay_on_mask
import nibabel as nib
import joblib

#%%
maskNr = 47
subjectID = '19910823ssld'
RDM_brain_1 = joblib.load(f'{MRI_DATA_DIR}/rsa_old/{subjectID}/test_partial_{maskNr}_M4B_model2_crossnobis_spearman_2_RDM_brain.pkl')

RDM_brain_2 = joblib.load(f'{MRI_RSA_DIR}/SetupConfig1/{subjectID}/SL_partial_{maskNr}_M4B_6Cond_suprasensory_crossnobis_spearman_r2_thr0_5_RDM_brain.pkl')

mask = nib.load(f'{MRI_MASKS_DIR}/{subjectID}/mask.nii')
overlay_1 = nib.Nifti1Image(RDM_brain_1, affine = mask.affine)
overlay_2 = nib.Nifti1Image(RDM_brain_2, affine = mask.affine)

plot_overlay_on_mask(overlay_1, mask)
# plot_overlay_on_mask(overlay_2, mask)
# %%
# maskNr = 48
# subjectID = '19910823ssld'
# RDM_brain_1 = joblib.load(f'/home/reabt/Desktop/ncc/MRI/data/rsa_old/{subjectID}/test_partial_{maskNr}_M4B_model2_crossnobis_spearman_2_RDM_brain.pkl')
# RDM_brain_2 = joblib.load(f'/home/reabt/Desktop/ncc/MRI/data/rsa/SetupConfig1/{subjectID}/SL_partial_{maskNr}_M4B_6Cond_suprasensory_crossnobis_spearman_r2_thr0_5_RDM_brain.pkl')

# mask = nib.load(f'/home/reabt/Desktop/ncc/MRI/data/masks/{subjectID}/mask.nii')
# overlay_1 = nib.Nifti1Image(RDM_brain_1, affine = mask.affine)
# overlay_2 = nib.Nifti1Image(RDM_brain_2, affine = mask.affine)

# plot_overlay_on_mask(overlay_1, mask)
# plot_overlay_on_mask(overlay_2, mask)
# %%
from nilearn import plotting
plotting.plot_stat_map(
    overlay_1,
    bg_img=mask,  # Use the original image as the background
    title="Centers Visualization",
    threshold=0.1,  # Lower threshold to ensure visibility
    display_mode="ortho",  # Orthogonal view
    colorbar=True,
    vmax=3,  # Maximum value for the colorbar
    alpha = 0.9
)
plotting.show()