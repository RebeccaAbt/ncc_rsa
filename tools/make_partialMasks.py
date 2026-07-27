'''
Use the "mask.nii" files that are located in the first-level directory of the fMRI analysis folders 
(that are ssynced from the MRI server) To create partial masks that can be used for the 
searchlight analysis. 

We don't need them if we use euclidean distance. But if we use crossnobis, this speeds up the computation
because we can run the pipeline on the partial files and merge them afterwards.
'''


#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from pathlib import Path
import numpy as np
import nibabel as nib

import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting

from utils.subj import *

#%%
subjects = get_MRI_subjects() # <- same subject list

options = [ # <- same prefixes and margins as MATLAB
{'prefix': '24cond_mask_part_', 'margin': 0},
{'prefix': '24cond_SL_marg1_mask_part_', 'margin': 1},
{'prefix': '24cond_SL_marg2_mask_part_', 'margin': 2},
{'prefix': '24cond_SL_marg3_mask_part_', 'margin': 3},
{'prefix': '24cond_SL_marg4_mask_part_', 'margin': 4},
{'prefix': '24cond_SL_marg5_mask_part_', 'margin': 5},
{'prefix': '24cond_SL_marg6_mask_part_', 'margin': 6},
{'prefix': '24cond_SL_marg7_mask_part_', 'margin': 7},
]

# rclone-equivalent locations (no syncing; we read/write here directly)

# inDir = '/home/reabt/experiments/ncc/MRI/data/sync/'        # <-- use this when we access locally saved data
inDir = MRI_RAW_DIR # <-- use this when we access data directly on the MRI server. We need to have the MRI server mounted for this

inFolder = f'{MRI_1ST_LEVEL_FOLDER}/mask.nii' 

maskDir = f'{MRI_DATA_DIR}/masks/'


# grid definition (same as MATLAB)

xParts, yParts, zParts = 3, 4, 5

# ---------------- helpers ----------------

def get_cube_coordinates(dim_len: int, n_parts: int, slice_margin: int):

    '''
    For slicing the 3D mask file into 60 partial masks. We use this to do the crossnobois 
    serachlight analysis on partial data to speed up the computation.
    '''

    if dim_len % n_parts != 0:
    # fallback that still mirrors contiguous, approximately equal chunks
    # (MATLAB used exact division; we warn but continue deterministically)
        sizes = np.diff(np.linspace(0, dim_len, n_parts + 1).astype(int))
    else:
        sizes = np.full(n_parts, dim_len // n_parts, dtype=int)

    coords = []
    start = 0
    for i in range(n_parts):
        end = start + sizes[i] - 1  # inclusive
        if i == 0:
            s = 0
            e = min(dim_len - 1, end + slice_margin)
        elif i == n_parts - 1:
            s = max(0, start - slice_margin)
            e = dim_len - 1
        else:
            s = max(0, start - slice_margin)
            e = min(dim_len - 1, end + slice_margin)
        coords.append(np.arange(s, e + 1, dtype=int))
        start = end + 1
    return coords


for subj in subjects:
    print(f'\nCurrent subject: {subj}')
    src_mask_file = os.path.join(inDir, subj, inFolder) # /data/neurokog/NCC25/analyze_fin/<subj>/NCC/firstLevel_sensory_M1B/mask.nii
    out_dir = os.path.join(maskDir, subj) # /home/reabt/Desktop/data/masks/<subj>
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(src_mask_file):
        print("Couldn't find mask file... skipping subject.")
        continue
        # raise FileNotFoundError(f'Mask not found: {src_mask_file}')

    # load mask once
    mask_img = nib.load(str(src_mask_file))
    mask_hdr = mask_img.header.copy()
    mask_aff = mask_img.affine
    mask_data = mask_img.get_fdata()  # float64; we'll preserve dtype on write
    orig_dtype = mask_img.get_data_dtype()

    # dimensions (prefer reading from image instead of hardcoding 63×76×55)
    nx, ny, nz = mask_data.shape
    print(f'  Loaded mask with shape: {mask_data.shape}')

    # for each option (prefix+margin) create partial masks
    for opt in options:
        out_prefix = opt['prefix']

        if os.path.isfile(os.path.join(out_dir, f'{out_prefix}1.nii')):
            print(f'  Skipping existing files with prefix {out_prefix} for subject {subj}')
            continue
        else:
            slice_margin = int(opt['margin'])
            print(f'  Current Option: margin {slice_margin}')

            # get coordinates for each axis (same decomposition as MATLAB: equal chunks + margin)
            xCoords = get_cube_coordinates(nx, xParts, slice_margin)
            yCoords = get_cube_coordinates(ny, yParts, slice_margin)
            zCoords = get_cube_coordinates(nz, zParts, slice_margin)

            # build cube coordinate triplets in MATLAB loop order (i over x, j over y, k over z)
            cube_coords = []
            for i in range(len(xCoords)):
                for j in range(len(yCoords)):
                    for k in range(len(zCoords)):
                        cube_coords.append((i, j, k))

            empty_masks = []  # track empty mask indices for margin==0

            # generate, check and save each partial mask
            for iCube, (ix, iy, iz) in enumerate(cube_coords, start=1):
                xi = xCoords[ix]
                yi = yCoords[iy]
                zi = zCoords[iz]

                # pull the original data for this cube
                cube_data = mask_data[np.ix_(xi, yi, zi)]

                # create full-volume mask with zeros, then insert cube (exactly like MATLAB)
                new_mask_data = np.zeros_like(mask_data)
                new_mask_data[np.ix_(xi, yi, zi)] = cube_data

                # check empties (sum==0), log only for zero-margin option (but still write the file)
                if slice_margin == 0 and np.isclose(new_mask_data.sum(), 0.0):
                    print(f'    partial mask nr. {iCube} is empty')
                    empty_masks.append(iCube)

                # write NIfTI (.nii, not .nii.gz) with original affine & dtype; keep header
                out_file = os.path.join(out_dir, f'{out_prefix}{iCube}.nii')
                nib.save(nib.Nifti1Image(new_mask_data.astype(orig_dtype), mask_aff, mask_hdr), str(out_file))

            # write empty mask info only for the zero-margin case (same as MATLAB)
            if slice_margin == 0 and len(empty_masks) > 0:
                log_file = os.path.join(out_dir, 'empty_masks.txt')
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write('Empty mask indices:\n')
                    for idx in empty_masks:
                        f.write(f'{idx}\n')
                print(f'  Wrote empty mask indices to {log_file}')

# %% =========================================================================== PLOTTING


def plot_masks_overlay(subjDir_PC: str, outFile_prefix: str, maskFile: str,
                       display_mode: str = "ortho", cut_coords=None, alpha: float = 0.6,
                       figure_title: str = "Partial masks overlay"):

    # Gather files to load (mirrors `dir([subjDir_PC '\' outFile_prefix '*'])` in MATLAB)
    files2load = sorted(glob.glob(os.path.join(subjDir_PC, f"{outFile_prefix}*")))
    print(f"Found {len(files2load)} files with prefix '{outFile_prefix}' in {subjDir_PC}")
    if len(files2load) == 0:
        raise FileNotFoundError(f"No files found with prefix '{outFile_prefix}' in {subjDir_PC}")

    # Background image (like `spm_image('Display', maskFile)`)
    if not os.path.exists(maskFile):
        raise FileNotFoundError(f"Background maskFile not found: {maskFile}")

    display = plotting.plot_anat(
        bg_img=maskFile,
        display_mode=display_mode,
        cut_coords=cut_coords,
        title=figure_title
    )

    colors = np.random.rand(60, 3) 
    # Overlay each mask in a random color (similar idea to addcolouredblobs)
    for n, mask_path in enumerate(files2load):
        try:
            img = nib.load(mask_path)
            data = img.get_fdata()
        except Exception as e:
            print(f"Skipping {mask_path} (load error: {e})")
            continue

        # Skip empty masks (mimics checking for empty partial masks)
        if not np.any(data):
            print(f"Skipping empty mask: {os.path.basename(mask_path)}")
            continue

        # Binarize to ensure a crisp overlay and keep original geometry
        bin_data = (data > 0).astype(np.uint8)
        overlay_img = nib.Nifti1Image(bin_data, img.affine, img.header)

        # Random color with specified alpha; first entry is fully transparent for background (0)
        color = np.random.rand(3)
        cmap = ListedColormap([colors[n][0], colors[n][1], colors[n][2]])

        # Add overlay to the same display (like adding another blob layer)
        display.add_overlay(overlay_img, cmap=cmap, alpha=alpha)

    plt.show()

subj = subjects[0] 

subjDir_PC = maskDir + subjects[0]  # first subject in the list; adjust as needed

outFile_prefix = options[4]['prefix'] 

# maskFile = f'/home/reabt/experiments/ncc/MRI/data/sync/{subj}/NCC/firstLevel_sensory_M1B/mask.nii' # background mask (SPM-like)
maskFile = f'{get_spm_dir(subj)}/mask.nii' # background mask (SPM-like)

plot_masks_overlay(
    subjDir_PC=subjDir_PC,
    outFile_prefix=outFile_prefix,
    maskFile=maskFile,
    display_mode="ortho",
    cut_coords=None,
    alpha=0.6,
    figure_title="Partial masks overlay (Python)"
)


# %%
