

#%% 
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants

import shutil
import joblib

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
from pqdm.processes import pqdm
from nilearn import plotting
from nilearn.image import new_img_like

from utils.load_cfg import load_fusion_config_instance
from plus_slurm import Job

#%%

def pil_imgs(mask_img, clusters_t, t, time_array):
    fig = plt.figure(figsize=(6, 6))
    cluster_img = new_img_like(mask_img, clusters_t)

    plotting.plot_glass_brain(
        cluster_img,
        display_mode="ortho",
        cut_coords=(0, 0, 0),
        draw_cross=False,
        black_bg=False,
        figure=fig,
        cmap="Greens",
        threshold = 0.1,
        annotate=False,
        colorbar = False,
        title=f"time point: {time_array[t]} ms",
    )

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape((height, width, 4))

    frame = buf[:, :, [1, 2, 3]]  # ARGB -> RGB
    plt.close(fig)
    return frame

def pil_imgs_wrapper(args):
    return pil_imgs(*args)


all_configs =[
			"FusionConfig_E2",
			"FusionConfig_E5",
			"FusionConfig_C2",
			"FusionConfig_C5",
	]


# config_class_name = [all_configs[c] for c in [0, 2, 4, 6]]#  4 and 5 missing for now


# meanMEG = [True, False]
# meanMEG = [True]
meanMEG = [False]
thres = [0.05 ]#, 0.05, 0.01]  
n_subj = 22
models = ALL_MODELS
tail = 1 # tail = 2
cluster_def_method = 'p' # cluster_def_method = 't'
# cp_variables = ['X_pre', 'X_post'] #['X_pre', 'X_post', 'X_diff']
cp_variables = ['X_post'] 


# for model in models:
model = models[0]
subjectID = '19910823ssld' # dummy subject
for thisConfig in all_configs:
    print(f"\nProcessing config {thisConfig}\n")
    for meanOpt in meanMEG:
        # if meanOpt == False:
        #     continue
        for thr in thres:
            for model in models:
                for X_def in cp_variables:
                                        
                    cfg = load_fusion_config_instance(thisConfig, subjectID)
                    cfg.modelType = model

                    thres_str = str(thr).replace('.','_')

                    print(f'\t Loading variables...', flush=True)

                    # file_note_clu = f'{n_subj}_subj_{model}' # anottate number of subjects & thresold used for defining "good" clusters
                    # file_note_good_clu = f'{n_subj}_subj_thres_{thres_str}_{model}'

                    file_note_clu = f'{n_subj}_subj_{model}_{X_def}_{cluster_def_method}Thres_{tail}tailed'
                    file_note_good_clu = f'{n_subj}_subj_{model}_{X_def}_{cluster_def_method}Thres_{tail}tailed_thres_{thres_str}'


                    if meanOpt == False:
                        fileName_clu = cfg.get_outFile_names()['cp'].replace('cp', f'cp_{file_note_clu}')
                        fileName_good_clu = cfg.get_outFile_names()['good_clusters'].replace('clusters', f'clusters_{file_note_good_clu}')
                    else:
                        fileName_clu = cfg.get_outFile_names()['cp_mean'].replace('clusters', f'clusters{file_note_clu}')
                        fileName_good_clu = cfg.get_outFile_names()['good_clusters_mean'].replace('clusters', f'clusters_{file_note_good_clu}')
                    # print(f'\ntrying to load file:\n {fileName_good_clu}\n')
                    # clu = joblib.load(fileName_clu)
                    if os.path.exists(fileName_good_clu):
                        good_clusters = joblib.load(fileName_good_clu)
                        cfg_suffix = thisConfig.split('_', 1)[1]
                        # now plot it as movie
                        if meanOpt == True:
                            movieFile = f'fusion_outputs/glass_{cfg_suffix}_{file_note_good_clu}_mean.mp4'
                        else:
                            movieFile = f'fusion_outputs/glass_{cfg_suffix}_{file_note_good_clu}.mp4'

                        mask_img = nib.load(cfg.maskFile)
                        n_times = good_clusters.shape[0]

                        time_array = np.arange(0, 1000, 10)  # includes 1000

                        args = [(mask_img, good_clusters[t], t, time_array) for t in range(n_times)]
                        frames = pqdm(args, pil_imgs_wrapper, n_jobs=5)

                        # # Optional: save GIF as before
                        # frames_pil = [Image.fromarray(frame) for frame in frames]
                        # frames_pil[0].save(
                        #     "animation.gif",
                        #     save_all=True,
                        #     append_images=frames_pil[1:],
                        #     duration=400,
                        #     loop=0,
                        # )

                        ffmpeg_path = shutil.which("ffmpeg")
                        fps = 2  # because your GIF duration=400 ms per frame -> 1000/400 = 2.5 fps

                        with imageio.get_writer(
                            movieFile,
                            fps=fps,
                            codec="libx264",
                            format="FFMPEG",
                            pixelformat="yuv420p",
                        ) as writer:
                            for frame in frames:
                                writer.append_data(frame)

                        print("done.")
                    else:
                        print(f'\tSkipping {thisConfig} {model} {thr}')




#%%

