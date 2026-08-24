# %%

import os
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_config_instance
from utils.files import fix_spm_rawdata_paths
from utils.plots import plot_overlay_on_mask
from utils.rsa import get_searchlight_RDMs_crossnobis

import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

from nilearn import plotting
from nilearn.image import new_img_like

import rsatoolbox as rsa
from rsatoolbox.io.spm import SpmGlm
from rsatoolbox.data import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.data.ops import merge_datasets
from rsatoolbox.rdm import calc_rdm, calc_rdm_unbalanced, RDMs
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.descriptor_utils import dict_to_list
from rsatoolbox.util.searchlight import (
    get_volume_searchlight,
    get_searchlight_RDMs,
    evaluate_models_searchlight
)

from plus_slurm import Job
from configs.config2 import BaseConfig as cfg
import importlib
import configs.config as config
importlib.reload(config)

#%%

class SL_crossnobis_partial(Job):
    def run(self,
            subjectID = '19910823ssld',
            maskNr = 1, 
            config_class_name = 'SetupConfig1'):
        
        print('----------------------------------------------------')
        print('running...')
        print(f'     - subject:       {subjectID}')
        print(f'     - maskNr:        {maskNr}')
        print(f'     - configuration: {config_class_name}')
        print('----------------------------------------------------')

    # 1) load instance of selected class object --> this loads all the settings for the computations

        cfg = load_config_instance(config_class_name, subjectID, maskNr) 
        cfg.print_summary()
        cfg.save_summary()

        outFiles = cfg.get_outFile_names()

        model = cfg.get_model_RDM()
        
        print(f'Running crossnobis Searchlight part {maskNr}', flush=True)
        os.makedirs(cfg.outDir, exist_ok=True)

    # 2) defining and loading some more variables
        spm = SpmGlm(cfg.spmDir)
        spm.get_info_from_spm_mat()

        print('    - Renaming rawdata file paths in SPM.mat...', flush=True)
        fix_spm_rawdata_paths(spm, cfg.dataDir + 'sync')        
        

    # 3) loading the betas and info

        print('    - Loading betas and info...', flush=True)
        _, _, info = spm.get_betas(cfg.maskFile)
        info = pd.DataFrame(info)
        reg_mask = info['reg_name'].str.contains('_hit') | info['reg_name'].str.contains('_miss')
        info = info[reg_mask].reset_index(drop=True)

        info['modality'] = info['reg_name'].str.extract(r'^(aud|tac|vis)')
        info['awareness'] = info['reg_name'].str.extract(r'_(hit|miss)')
        info['condition'] = info['modality'] + '_' + info['awareness']

        mask = nib.load(cfg.maskFile)
        mask_data = mask.get_fdata()
        mask_bool = mask_data > 0

    # 4) finding Searchlights
        print(f'    - Getting searchlight centers and neighbors of mask nr. {maskNr} with radius {cfg.SLradius} voxels and threshold {cfg.SLthr}')

        centers, neighbors = get_volume_searchlight(mask_bool, radius = cfg.SLradius, threshold = cfg.SLthr)

        print('    - Computing searchlight RDMs...')

    # 5) computing RDMs for every searchlight
        SL_rdms = get_searchlight_RDMs_crossnobis(
            spm,
            centers, 
            neighbors, 
            mask,
            reg_mask, 
            info, 
            'condition', 
            'crossnobis')

        # Save SL_rdms using joblib
        joblib.dump(SL_rdms, outFiles['SL_rdms'])
        joblib.dump(centers, outFiles['centers'])
        joblib.dump(neighbors, outFiles['neighbors'])

    # 6) compare Searchlight RDMs with model RDM
        print('    - Evaluating RDMs...')

        eval_results = evaluate_models_searchlight(SL_rdms, model, eval_fixed, method = cfg.RSAmethod)
        eval_score = [float(e.evaluations) for e in eval_results]

        joblib.dump(eval_results, outFiles['eval_results'])

        x, y, z = mask.shape
        RDM_brain = np.zeros([x*y*z])
        RDM_brain[SL_rdms.rdm_descriptors['voxel_index']] = eval_score
        RDM_brain = RDM_brain.reshape(mask.shape)

        joblib.dump(RDM_brain, outFiles['RDM_brain'])

        # Plot 1: histogram
        sns.histplot(eval_score, kde=True)
        plt.title(cfg.plot1_title, size=18)
        plt.ylabel('Occurance')
        plt.xlabel(f'{cfg.RSAmethod} correlation')
        sns.despine()
        plt.savefig(cfg.DistPlotFile, dpi=300)
        plt.close()

        # Plot 2: brain map
        def RDMcolormapObject(direction=1):
            cs = ['blue', 'turquoise', 'gray', 'red', 'yellow'] if direction else ['yellow', 'red', 'gray', 'turquoise', 'blue']
            return matplotlib.colors.LinearSegmentedColormap.from_list("", cs)

        threshold = np.percentile(eval_score, cfg.resultsPlot_thr)
        plot_img = new_img_like(mask, RDM_brain)
        cmap = RDMcolormapObject()

        fig = plt.figure(figsize=(12, 3))
        plotting.plot_stat_map(
            plot_img, 
            colorbar=True, 
            threshold=threshold,
            cut_coords=[-30, -20, 0, 20, 40, 60],
            display_mode='z', draw_cross=False, figure=fig,
            title=f'{cfg.plot2_title} | threshold: {cfg.resultsPlot_thr}%', cmap=cmap,
            black_bg=False, annotate=False
        )
        plt.savefig(cfg.ResultsPlotFile, dpi=300)
        plt.close()

