import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

import numpy as np
import seaborn as sns
import rsatoolbox as rsa
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
import nibabel as nib
import matplotlib.colors as mcolors


def plot_rdm(RDM):
    
    thisRDM = RDM.predict_rdm()
    rsa.vis.show_rdm(thisRDM)
    plt.show(thisRDM)

def plot_nifti(image, title = None):
    '''
    simply plot a nifti image. output can be saved using "display.safefig('filename.png')"
    '''

    fig = plt.figure(figsize=(12, 3))

    display = plotting.plot_stat_map(
            image, 
            colorbar=True,
            display_mode='z', 
            draw_cross=False, 
            figure=fig,
            black_bg=False, 
            annotate=False,
            cut_coords = 10,
            title = title)
    plt.show()
    return display


def plot_overlay_on_mask(overlay, mask):

    # # Define a custom colormap
    # cmap = mcolors.ListedColormap(['black', 'red'])  # Black for background, red for centers, blue for neighbors
    # bounds = [0, 0.5, 1.5]  # Define boundaries for the colormap
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the overlay on top of the mask
    plotting.plot_stat_map(
        overlay,
        bg_img=mask,  # Use the original image as the background
        title="Centers Visualization",
        threshold=0.1,  # Lower threshold to ensure visibility
        display_mode="ortho",  # Orthogonal view
        colorbar=True,
        cmap='viridis',
        vmax=3,  # Maximum value for the colorbar
        alpha = 0.9
    )
    plotting.show()


def plot_RDMbrain(RDM_brain, eval_score, subjectID = EXAMPLE_SUBJ_1, percentile = 90):

    tmp_img = nib.load(f'{MRI_DATA_DIR}/{MRI_RAW_FOLDER}/{subjectID}/{MRI_1ST_LEVEL_FOLDER}/beta_0001.nii')
    threshold = np.percentile(eval_score, percentile)
    plot_img = new_img_like(tmp_img, RDM_brain)

    cmap = _RDMcolormapObject()

    # coords = range(-20, 40, 5)
    fig = plt.figure(figsize=(12, 3))

    display = plotting.plot_stat_map(
            plot_img, colorbar=True, threshold=threshold,
            display_mode='z', draw_cross=False, figure=fig,
            title=f'suprasensory', cmap=cmap,
            black_bg=False, annotate=False)
    plt.show()

def plot_centers(centers, subjectID=EXAMPLE_SUBJ_1):
    from nilearn import plotting
    import nibabel as nib
    import matplotlib.colors as mcolors

    mask = nib.load(f'{MRI_DATA_DIR}/{MRI_RAW_FOLDER}/{subjectID}/{MRI_1ST_LEVEL_FOLDER}/mask.nii')

    overlay_data = np.zeros(mask.shape)
    overlay_data[np.unravel_index(centers, mask.shape)] = 1  # Mark centers
    overlay_img = nib.Nifti1Image(overlay_data, affine = mask.affine)

    # Define a custom colormap
    cmap = mcolors.ListedColormap(['black', 'red'])  # Black for background, red for centers, blue for neighbors
    bounds = [0, 0.5, 1.5]  # Define boundaries for the colormap
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the overlay on top of the mask
    plotting.plot_stat_map(
        overlay_img,
        bg_img=mask,  # Use the original image as the background
        title="Centers Visualization",
        threshold=0.1,  # Lower threshold to ensure visibility
        display_mode="ortho",  # Orthogonal view
        colorbar=True,
        cmap=cmap,
        vmax=1,  # Maximum value for the colorbar
        alpha = 0.9,
        black_bg="False"
    )
    plotting.show()


def plot_rsa_hist(eval_score, xlab_Method = 'Pearson Correlation', title = 'title'):
    sns.histplot(eval_score, kde=True)
    plt.title(title, size=18)
    plt.ylabel('Occurance')
    plt.xlabel(f'{xlab_Method} correlation')
    sns.despine()
    plt.show()


def _RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = mcolors.LinearSegmentedColormap.from_list("", cs)
    return cmap


def plot_brain_map(cfg, mask, RDM_brain, eval_score):
    def RDMcolormapObject(direction=1):
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow'] if direction else ['yellow', 'red', 'gray', 'turquoise', 'blue']
        return mcolors.LinearSegmentedColormap.from_list("", cs)

    threshold = np.percentile(eval_score, cfg.resultsPlot_thr)
    plot_img = new_img_like(mask, RDM_brain)
    cmap = RDMcolormapObject()

    fig = plt.figure(figsize=(12, 3))
    plotting.plot_stat_map(
        plot_img, colorbar=True, 
        # threshold=threshold,
        cut_coords=[-30, -20, 0, 20, 40, 60],
        display_mode='z', draw_cross=False, figure=fig,
        title=f'{cfg.plot2_title} - {cfg.subjectID}', cmap=cmap,
        black_bg=False, annotate=False
    )
    print(f'plot_brain_map: File saved as {cfg.ResultsPlotFile}')
    plt.savefig(cfg.ResultsPlotFile, dpi=300)
    plt.close()


def eval_score_histogram(cfg, eval_score):
    # Plot 1: histogram
    sns.histplot(eval_score, kde=True)
    plt.title(f'{cfg.plot1_title} - {cfg.subjectID}', size=18)
    plt.ylabel('Occurance')
    plt.xlabel(f'RSA method für comparison: {cfg.RDMmethod} -> {cfg.RSAmethod}')
    sns.despine()
    print(f'eval_score_histogram: File saved as {cfg.DistPlotFile}')
    plt.savefig(cfg.DistPlotFile, dpi=300)
    plt.close()


def plot_max_modelFit_rdm(cfg, SL_rdms, eval_score):
    voxel_indices = SL_rdms.rdm_descriptors['voxel_index']

    max_value = np.max(eval_score) # find max eval_score
    max_idx = int(np.argmax(eval_score)) # find max eval_score
    max_voxel = voxel_indices[max_idx] # find voxel index of the max eval score
    max_rdm_idx = np.where(SL_rdms.rdm_descriptors['voxel_index'] == max_voxel)[0][0] # find indx of voxel index in compiles SL_rdms

    if 'conditions' in SL_rdms.pattern_descriptors:
        pattern_descriptor = 'conditions'
    else:
        pattern_descriptor = 'index'

    print(f'maximum eval score: {max_value} at voxel {max_voxel} (index {max_idx})')
    fig = rsa.vis.show_rdm(
        SL_rdms[max_rdm_idx], 
        pattern_descriptor=pattern_descriptor,
        rdm_descriptor=f'Max match: {cfg.subjectID}', 
        num_pattern_groups=4,
        cmap = 'viridis',
        figsize = (5, 5),
        show_colorbar='panel'
    )

    if cfg.maskNr > 0:
        fileName = os.path.join(cfg.outDir, f'{cfg.filePrefix}_RDM_maxFit_{cfg.modelType}.png') # partial mask prefix
    else:
        fileName = os.path.join(cfg.outDir, f'{cfg.filePrefix_fullBrain}_RDM_maxFit_{cfg.modelType}.png') # full brain prefix
    print(f'plot_max_modelFit_rdm: File saved as {fileName}')
    plt.savefig(fileName) # full brain prefix

    plt.close()
