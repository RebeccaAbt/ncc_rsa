
#%% 
import os
import sys
import joblib

import nibabel as nib
import numpy as np
from pqdm.processes import pqdm

import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
from PIL import Image

sys.path.append('/home/reabt/experiments/ncc/MRI/code/')
from utils.load_cfg import load_fusion_config_instance
from plus_slurm import Job
#%%
class Movie(Job):
    def run(self,
    config_class_name = 'FusionConfig_C5',
    meanMEG = False,
    thres = 0.1,    
    n_subj = 7):

        subjectID = '19910823ssld' # dummy subject
        cfg = load_fusion_config_instance(config_class_name, subjectID)
        models = ALL_MODELS

        for model in models:
            
            cfg.modelType = model

            thres_str = str(thres).replace('.','_')

            print(f'\t [4] Saving variables...', flush=True)

            file_note_clu = f'{n_subj}_subj_{model}' # anottate number of subjects & thresold used for defining "good" clusters
            file_note_good_clu = f'{n_subj}_subj_thres_{thres_str}_{model}'

            if meanMEG == False:
                fileName_clu = cfg.get_outFile_names()['cp'].replace('cp', f'cp_{file_note_clu}')
                fileName_good_clu = cfg.get_outFile_names()['good_clusters'].replace('clusters', f'clusters_{file_note_good_clu}')
            else:
                fileName_clu = cfg.get_outFile_names()['cp_mean'].replace('clusters', f'clusters{file_note_clu}')
                fileName_good_clu = cfg.get_outFile_names()['good_clusters_mean'].replace('clusters', f'clusters_{file_note_good_clu}')

            clu = joblib.load(fileName_clu)
            good_clusters = joblib.load(fileName_good_clu)

            # now plot it as movie
            if meanMEG == True:
                movieFile = f'cluster_outputs/glass_slow_{config_class_name}_movie_{n_subj}_subj_thres_{thres_str}_{model}_mean.gif'
            else:
                movieFile = f'cluster_outputs/glass_slow_{config_class_name}_movie_{n_subj}_subj_thres_{thres_str}_{model}.gif'
        
            mask_img = nib.load(cfg.get_mask_file())
            n_times = good_clusters.shape[0]

            frames = []

            time_array = np.arange(0, 1000, 10)  # includes 1000
            # for t in range(n_times):
            def pil_imgs(mask_img, clusters_t, t):

                fig = plt.figure(figsize=(6, 6))
                cluster_img = new_img_like(mask_img, clusters_t)

                # plotting.plot_stat_map(
                plotting.plot_glass_brain(
                    cluster_img,
                    display_mode='ortho',
                    cut_coords=(0, 0, 0),
                    draw_cross=False,
                    black_bg=True,
                    figure=fig,
                    cmap = 'Greens',
                    annotate=False,
                    title=f'time point: {time_array[t]} ms'
                )

                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape((height, width, 4))
                frame = buf[:, :, [1, 2, 3]]  # convert ARGB → RGB

                plt.close(fig)
                return frame

            def pil_imgs_wrapper(args):
                return pil_imgs(*args)    

            args = [(mask_img, good_clusters[t], t) for t  in range(n_times)]
            frames = pqdm(args, pil_imgs_wrapper, n_jobs=5)  # adjust n_jobs as needed
            frames = [Image.fromarray(frame) for frame in frames]

            print('saving movie as gif...')
            print(dir(frames))
            frames[0].save(
                movieFile,
                save_all=True,
                append_images=frames[1:],
                # duration=100,
                fps = 1,
                loop=2
            )
