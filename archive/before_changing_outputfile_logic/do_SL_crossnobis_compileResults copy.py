import os
import sys
import re
from glob import glob
from collections import defaultdict
import warnings
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
from nilearn import plotting
import rsatoolbox as rsa

# Extend sys.path to import project modules
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')
from utils.load_cfg import load_config_instance
from utils.compile import compile_SL_rdms_files
from plus_slurm import Job

#%%

class SL_crossnobis_compileResults(Job):
    def run(self,
            subjectID = '19910823ssld',
            config_class_name = 'SetupConfig1',
            maskNr = 0):
    

        print('[1] Loading configuration...')
        cfg = load_config_instance(config_class_name, subjectID)
        outFiles = cfg.get_outFile_names()
        compiled_Prefix = os.path.join(cfg.outDir, f'{cfg.filePrefix_compiled}')
        inDir = cfg.outDir

        # ---------------------------------------
        # Locate partial result files
        # ---------------------------------------
        print('[2] Matching SL_rdms and eval_results files...')
        sl_rdms_pattern = re.sub(rf'({re.escape(cfg.prefix)}_)\d{{1,2}}', r'\1*', os.path.basename(outFiles['SL_rdms']))
        eval_results_pattern = re.sub(rf'({re.escape(cfg.prefix)}_)\d{{1,2}}', r'\1*', os.path.basename(outFiles['eval_results']))
        SL_rdms_files = sorted(glob(os.path.join(inDir, sl_rdms_pattern)))
        eval_results_files = sorted(glob(os.path.join(inDir, eval_results_pattern)))
        print(f'    -> Found {len(eval_results_files)} result files.')

        # Extract and validate mask numbers from filenames
        mask_numbers = []
        pattern = re.compile(rf'{cfg.prefix}_(\d{{1,2}})')
        for file in eval_results_files:
            match = pattern.search(os.path.basename(file))
            if match:
                mask_numbers.append(int(match.group(1)))
            else:
                warnings.warn(f"Could not extract mask number from file: {file}")

        # Match and sort by mask number
        paired = sorted(zip(mask_numbers, SL_rdms_files, eval_results_files))
        paired = [(m, s, e) for m, s, e in paired if m is not None]
        mask_numbers_sorted, SL_rdms_files_sorted, eval_results_files_sorted = zip(*paired)

        # ---------------------------------------
        # Load and compile eval scores and rdms
        # ---------------------------------------
        print('[3] Loading eval_results and SL_rdms...')
        compiled_data = []
        for num, sl_file, eval_file in zip(mask_numbers_sorted, SL_rdms_files_sorted, eval_results_files_sorted):
            SL_rdms = joblib.load(sl_file)
            eval_results = joblib.load(eval_file)
            '''
            for plotting:
            eval_results_bool = np.zeros(eval_results.shape)
            eval_results_bool[eval_results > 0] = 1
            RDM_img = nib.Nifti1Image(eval_results_bool, affine = nib.load(cfg.get_mask_file()).affine)
            plot_overlay_on_mask(RDM_img, nib.load(cfg.get_mask_file()))
            '''
            eval_score = [float(e.evaluations.item()) for e in eval_results]
            compiled_data.append({
                'mask_number': num,
                'eval_score': eval_score,
                'voxel_index': SL_rdms.rdm_descriptors['voxel_index']
            })

        print('    -> Merging SL_rdms...')
        all_SL_rdms = compile_SL_rdms_files(SL_rdms_files_sorted)

        # ---------------------------------------
        # Deduplicate by voxel index
        # ---------------------------------------
        print('[4] Checking for duplicate voxel indices...')
        all_voxel_indices = []
        all_eval_scores = []
        for entry in compiled_data:
            all_voxel_indices.extend(entry['voxel_index'])
            all_eval_scores.extend(entry['eval_score'])

        voxel_map = defaultdict(list)
        for i, voxel in enumerate(all_voxel_indices):
            voxel_map[voxel].append(i)

        duplicates = {v: i for v, i in voxel_map.items() if len(i) > 1}

        all_equal = True
        for voxel, inds in duplicates.items():
            values = [all_eval_scores[i] for i in inds]
            if not all(v == values[0] for v in values):
                all_equal = False
                # print(f"Voxel index {voxel} has different eval_scores at indices {inds}: values={values}")

        print("    -> All duplicates match." if all_equal else "    -> Some duplicates differ!")


        # Filter for unique voxels
        _, unique_indices = np.unique(all_voxel_indices, return_index=True)
        unique_indices = sorted(unique_indices)
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

        all_voxel_indices = [all_voxel_indices[i] for i in unique_indices_sorted]
        all_eval_scores = [all_eval_scores[i] for i in unique_indices_sorted]

        # Determine max voxel and its index in RDM
        max_idx = int(np.argmax(all_eval_scores)) # find max eval_score
        max_voxel = all_voxel_indices[max_idx] # find voxel index of the max eval score
        max_rdm_idx = np.where(all_SL_rdms.rdm_descriptors['voxel_index'] == max_voxel)[0][0] # find indx of voxel index in compiles SL_rdms

        # ---------------------------------------
        # Save brain image
        # ---------------------------------------
        print('[5] Creating NIfTI image frrom full-brain results...')
        mask = nib.load(cfg.get_mask_file())
        RDM_brain = np.zeros(mask.shape).flatten()
        RDM_brain[all_voxel_indices] = all_eval_scores
        RDM_brain = RDM_brain.reshape(mask.shape)
        RDM_img = nib.Nifti1Image(RDM_brain, affine=mask.affine)
        nib.save(RDM_img, f'{compiled_Prefix}_RDM_brain.nii')

        # ---------------------------------------
        # Plotting
        # ---------------------------------------
        print('[6] Plotting...')
        coords = range(-20, 50, 5)

        # Histogram
        print('    - Histogram')
        sns.histplot(all_eval_scores, kde=True)
        plt.title(f'Subject {cfg.subjectID} RSA Score Distribution')
        plt.xlabel('RSA Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'{compiled_Prefix}_eval_score_hist.png')
        plt.close()

        # Brain map
        print('    - RDM brain map')
        fig = plotting.plot_stat_map(
            stat_map_img=RDM_img, cut_coords=coords,
            display_mode='z', title=cfg.plot2_title,
            threshold=0, colorbar=True, vmin=-0.9, vmax=0.9
        )
        plt.savefig(f'{compiled_Prefix}_eval_score_brain.png')
        plt.close()

        # Max voxel overlay
        print('    - Max voxel overlay')
        max_coord = np.unravel_index(np.argmax(RDM_brain), RDM_brain.shape)
        highlight = np.zeros(mask.shape)
        highlight[max_coord] = 1
        max_overlay = nib.Nifti1Image(highlight, affine=mask.affine)
        plotting.plot_stat_map(max_overlay, title='Max RSA Match')
        plt.savefig(f'{compiled_Prefix}_maxFit_coords_{cfg.modelType}.png')
        plt.close()

        # RDM matrix at max voxel
        print('    - RDM at max voxel')

        if 'conditions' in all_SL_rdms.pattern_descriptors:
            pattern_descriptor = 'conditions'
        else:
            pattern_descriptor = 'index'

        fig = rsa.vis.show_rdm(
            all_SL_rdms[max_rdm_idx], 
            pattern_descriptor=pattern_descriptor,
            rdm_descriptor=f'Max match: {cfg.subjectID}', 
            num_pattern_groups=4,
            cmap = 'viridis',
            figsize = (5, 5),
            show_colorbar='panel'
        )

        plt.savefig(f'{compiled_Prefix}_RDM_maxFit_{cfg.modelType}.png')
        plt.close()

        # ---------------------------------------
        # Save outputs
        # ---------------------------------------
        print('[7] Saving output...')

        joblib.dump(RDM_brain, outFiles['RDM_brain_compiled'])
        joblib.dump(all_SL_rdms, outFiles['SL_rdms_compiled'])
        joblib.dump({'eval_scores': all_eval_scores, 'voxel_indices': all_voxel_indices},
                    f'{compiled_Prefix}_eval_scores_and_indices.pkl')

        print('[Done] ')
