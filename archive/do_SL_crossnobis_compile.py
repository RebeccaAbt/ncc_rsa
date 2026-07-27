
'''
    Warum kein Overlap, aber bei anderem Script schon ?? 
    --> nächster Test: Config mit thr = 1; 
    dann nochmal anschauen
'''


#%%
import os
import sys
import re
from glob import glob
from nilearn.image import new_img_like
import seaborn as sns

import joblib
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rsatoolbox as rsa

sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.plots import plot_RDMbrain, plot_rsa_hist
from utils.compile import get_compiled_centers
from utils.load_cfg import load_config_instance

from plus_slurm import Job
import warnings
#%%

class SL_crossnobis_compile(Job):
    def run(self,
            subjectID = '19910823ssld',
            config_class_name = 'SetupConfig2',
            maskNr = 0):
    
    # 1) get config 

        cfg = load_config_instance(config_class_name, subjectID) 

        compiled_Prefix = os.path.join(cfg.outDir, f'{cfg.filePrefix_compiled}') # add suffix before saving
        inDir = cfg.outDir

    # 2) match output files of partial data with partial brain mask file

        rdm_brain_files = sorted(glob(os.path.join(inDir, '*_RDM_brain.pkl')))
        print(f"Found {len(rdm_brain_files)}RDM brain files:")
    
        mask_numbers = []
        pattern = re.compile(rf'{cfg.prefix}_(\d{{1,2}})') 

        # Extract mask numbers for each file
        for file in rdm_brain_files:
            match = pattern.search(os.path.basename(file))
            if match:
                mask_numbers.append(int(match.group(1)))
            else:
                # In case pattern doesn't match
                msg = f"Could not extract mask number from file: {file}. \n File will be left out in the full-brain compilation"
                warnings.warn(msg)

        paired = sorted(zip(mask_numbers, rdm_brain_files))
        paired = [(num, file) for num, file in paired if num is not None] # Remove any pairs where mask number is None

        # Unzip sorted pairs
        mask_numbers_sorted, rdm_brain_files_sorted = zip(*paired)

    # 3) Load and append RDM_brain variables
        rdm_brain_data = []
        for num, file in zip(mask_numbers_sorted, rdm_brain_files_sorted):
            RDM_brain = joblib.load(file)
            '''
            for plotting:
            RDM_brain_bool = np.zeros(RDM_brain.shape)
            RDM_brain_bool[RDM_brain > 0] = 1
            RDM_img = nib.Nifti1Image(RDM_brain_bool, affine = nib.load(cfg.maskFile).affine)
            plot_overlay_on_mask(RDM_img, nib.load(cfg.maskFile))
            '''
            rdm_brain_data.append({'mask_number': num, 'RDM_brain': RDM_brain})
            
    # 4) Find coordinates where multiple RDM_brain arrays have nonzero values

        rdm_arrays = [d['RDM_brain'] for d in rdm_brain_data]
        mask_numbers = [d['mask_number'] for d in rdm_brain_data]
        shape = rdm_arrays[0].shape
        
        stacked = np.stack(rdm_arrays, axis=0) # Stack all arrays to shape (n_masks, x, y, z)
        nonzero_counts = np.count_nonzero(stacked, axis=0) # find non-zero coords
        coords_multi_nonzero = np.where(nonzero_counts >= 2) # coords with multiple non-zeros
        print(f"Found {coords_multi_nonzero[0].shape[0]} coordinates with nonzero values in two or more RDM_brain arrays.")

    # 5) check at coordinates with multiple values if values are the same

        # For each such coordinate, collect the values and mask numbers where value is nonzero
        multi_nonzero_values = []
        all_equal = True
        for idx in zip(*coords_multi_nonzero):
            values = []
            masks = []
            for i, arr in enumerate(rdm_arrays):
                val = arr[idx]
                if val != 0:
                    values.append(val)
                    masks.append(mask_numbers[i])
            if len(values) >= 2:
                multi_nonzero_values.append({
                    'indices': idx,
                    'mask_numbers': masks,
                    'values': values
                })
                # Check if all values are the same
                if not all(v == values[0] for v in values):
                    all_equal = False
                    print(f"Different values at {idx}: mask_numbers={masks}, values={values}")

        print('-----------------------------------------------------------------------')
        if all_equal:
            print(f"All of the {coords_multi_nonzero[0].shape[0]} multi-nonzero coordinates have the same value across masks.")
        else:
            raise ValueError("Some coordinates have different values across masks! ")
        print('-----------------------------------------------------------------------')

    # 6) Combine partial RDM_brain arrays. Only keep one value if voxel covered by multiple partial masks

        RDM_brain_combined = np.zeros(shape, dtype=stacked.dtype)

        # For each mask, fill in values only where combined is still zero
        for arr in stacked:
            mask = (RDM_brain_combined == 0) & (arr != 0)
            RDM_brain_combined[mask] = arr[mask]

    # 7) Also compile the partial centers

        compiled_centers, compiled_centers_img = get_compiled_centers(cfg)

    # 8) Plots

        # ~~~ Plot centers

        cmap = mcolors.ListedColormap(['black', 'green', 'yellow', 'orange' 'red', 'magenta', 'purple', 'cyan', 'blue'])  # Black for background, red for centers, blue for neighbors
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Define boundaries for the colormap
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig = plt.figure(figsize=(12, 3))
        # Plot the overlay on top of the mask
        plotting.plot_stat_map(
            compiled_centers_img,
            bg_img=nib.load(cfg.maskFile),  # Use the original image as the background
            title=f'compiled Searchlight centers - {cfg.subjectID}',
            threshold=0.1,  # Lower threshold to ensure visibility
            display_mode="ortho",  # Orthogonal view
            colorbar=True,
            cmap=cmap,
            vmax=8,  # Maximum value for the colorbar
            alpha = 0.9
        )

        plotting.show()
        plt.savefig(f'{compiled_Prefix}_centers.png', dpi=300)
        plt.close()

        # ~~~ Plot RSA results on brain

        mask = nib.load(cfg.maskFile)
        plot_img = new_img_like(mask, RDM_brain_combined)
        coords = range(-20, 50, 5)

        fig = plt.figure(figsize=(12, 3))
        plotting.plot_stat_map(
            plot_img, 
            colorbar=True, 
            # threshold=threshold,
            cut_coords=coords,
            display_mode='z', 
            draw_cross=False, 
            figure=fig,
            title=f"compiled SL RSA results: {cfg.subjectID}",
            # cmap=cmap,
            black_bg=False, 
            annotate=False,
            vmin = -0.9,
            vmax = 0.9
        )
        plt.show()
        plt.savefig(f'{compiled_Prefix}_eval_score_brain.png', dpi=300)
        plt.close()

        # ~~~ Plot  histogram of eval_score

        allCentersMask_bool = compiled_centers_img.get_fdata() > 0
        eval_score = RDM_brain_combined[allCentersMask_bool] # compute eval_score from RDM_brain cmobined, so we don't have to compile all of the eval_score for each partial mask separately

        fig = plt.figure()
        sns.histplot(eval_score, kde=True)
        plt.title(f'compiled SL RSA results: {cfg.subjectID}', size=18)
        plt.ylabel('Occurance')
        plt.xlabel(f'{cfg.RSAmethod}')
        sns.despine()
        plt.show()

        plt.savefig(f'{compiled_Prefix}_eval_score_hist.png', dpi=300)
        plt.close()

        # # ~~~ plot RDM with maximum model match

        # max_idx = np.argmax(eval_score)
        # fig = rsa.vis.show_rdm(
        #     SL_rdms[max_idx], 
        #     pattern_descriptor = 'condition', 
        #     num_pattern_groups=4,
        #     cmap = 'viridis',
        #     figsize = (5, 5),
        #     show_colorbar='panel',
        #     rdm_descriptor = f'RDM at max. {cfg.modelType} model match: {cfg.subjectID}\n', 
        #     )

        # outFile = os.path.join(cfg.outDir, f'{cfg.filePrefix}_RDM_maxFit_{cfg.modelType}.png')
        # plt.savefig(outFile, dpi='figure',  bbox_inches='tight')
        # plt.close()

        # # ~~~ plot coordinates of maximum model match

        # max_coord = np.unravel_index(np.argmax(RDM_brain), RDM_brain.shape) # Get coordinates of maximum value in RDM_brain
        # print(f"Coordinates of maximum value in RDM_brain: {max_coord}")

        # maxPoint = np.zeros(mask.shape)
        # maxPoint[max_coord] = 1
        # maxPoint_img = nib.Nifti1Image(maxPoint, affine = mask.affine)

        # plotting.plot_stat_map(maxPoint_img, 
        #                     #    draw_cross = False, 
        #                     cmap = mcolors.ListedColormap(['black', 'red']),
        #                     colorbar = False,
        #                     title = f'Point of max. {cfg.modelType} model match: {cfg.subjectID}')

        # outFile = os.path.join(cfg.outDir, f'{cfg.filePrefix}_maxFit_coords_{cfg.modelType}.png')
        # plt.savefig(outFile, dpi='figure',  bbox_inches='tight')
        # plt.close()



# %%
        # save the outputs
        joblib.dump(RDM_brain_combined, f'{compiled_Prefix}_SL_rdms.pkl')
        joblib.dump(compiled_centers, f'{compiled_Prefix}_centers.pkl')
        joblib.dump(eval_score, f'{compiled_Prefix}_eval_score.pkl')