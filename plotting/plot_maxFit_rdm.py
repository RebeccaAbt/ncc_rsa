#%%

import os
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from utils.load_cfg import load_MRI_config_instance
from utils.rsa import *
from utils.plots import *
from utils.subj import *

import joblib
import nibabel as nib
import matplotlib.pyplot as plt


import rsatoolbox as rsa

from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import (
	evaluate_models_searchlight
)

from configs.config import BaseConfig as cfg
import importlib
import configs.config as config
importlib.reload(config)
from configs.config import BaseConfig as cfg
import matplotlib
%matplotlib inline

#%%
# subjectID = '19921205crfi'
# subjectID = '19840930bigs'
thisConfig = 'MRIconfig_C2'
maskNr = 47
figsize = (5, 5)
lw = 2

for subjectID in get_MRI_subjects()[:10]:
	cfg = load_MRI_config_instance(thisConfig, subjectID, maskNr=maskNr) 
	modelNames = ALL_MODELS 

	outFiles = cfg.get_outFile_names()
	# use tmp old directory, because the new RDMs aren't computed yet
	for key, val in outFiles.items():
		if isinstance(val, str):
			outFiles[key] = val.replace('data/rsa/', 'data/rsa/old/')

	if cfg.maskNr == 0:
		SL_rdms      = joblib.load(outFiles['SL_rdms'])
		eval_results = joblib.load(outFiles['eval_results'])

	else:
		print(outFiles['SL_rdms_partial'])
		SL_rdms     = joblib.load(outFiles['SL_rdms_partial'])
		eval_results = joblib.load(outFiles['eval_results_partial'])

	info        = joblib.load(outFiles['info'])
	info_full   = joblib.load(f'{CODE_DIR}/resources/info.pkl')
	models      = cfg.get_model_RDM()
	mask        = nib.load(cfg.get_mask_file())

	SL_rdms, models = reorder_rdms(SL_rdms, models)
	
	# 8) extract and plot data separately for sensory & suprasensory model
	for model in modelNames:

	# model = modelNames[0]
		# ------------------------ v 
		cfg.modelType = model
		cfg.configure_paths()
		outFiles = cfg.get_outFile_names()
		# ------------------------ ^ important! for right prefix of files (sensory/suprasensory)

		eval_score = np.concatenate([e.evaluations[0][cfg.modelIdx] for e in eval_results])
		RDM_brain = get_RDM_brain(mask, SL_rdms, eval_score) # to display eval scores of model comparison in brain-shape
		  
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
			# pattern_descriptor=pattern_descriptor,
			# rdm_descriptor=f'Max match: {cfg.subjectID}', 
			# num_pattern_groups=4, # makes wonky lines thast are not exactly in the middle between categories
			gridlines = [3.5, 7.5, 11.5, 15.5, 19.5], # makes better lines, I think
			cmap = 'viridis',
			# figsize = (12, 12), # bigger size used, when I add the gridlines and change the linewidth to 5
			figsize = figsize,
			# show_colorbar='panel',
			linewidth = 40.1
		)
	
		real_fig = fig[0]

		import matplotlib.lines as mlines

		for line in real_fig.findobj(match=mlines.Line2D):
			if line.get_color() in ("w", "white", "#ffffff", (1, 1, 1), (1, 1, 1, 1)):
				line.set_linewidth(lw)
		edgecolor_objs = [obj for obj in real_fig.findobj()
							  if any(hasattr(obj, attr) for attr in ('get_edgecolor',
																	 'set_edgecolor',
																	 'edgecolor',
																	 'Edgecolor'))]
			
		for obj in edgecolor_objs:
			obj.set_edgecolor(None)

		real_fig.set_frameon(False)
		real_fig

		fig_opts = dict( dpi=300, bbox_inches='tight', pad_inches = 0)
		real_fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/rdm_{subjectID}_{thisConfig.split('_')[1]}_{model}_{maskNr}.png', transparent=True, format='png', **fig_opts)

# ------------------------- colorbar stuff
# real_fig = fig[0]

# ax_rdm = real_fig.axes[0]
# ax_cb = real_fig.axes[1]

# # left, bottom, width, height
# print(ax_cb.get_position())
# ax_cb.set_position([0.86, 0.18, 0.06, 0.64])
# ax_cb.tick_params(labelsize=12)

		real_fig.canvas.draw()
		plt.show()

#%%

lw = 5
real_fig = fig[0]

for obj in real_fig.findobj():
	if hasattr(obj, "get_color"):
		print(type(obj), obj.get_color())


import matplotlib.lines as mlines

for line in real_fig.findobj(match=mlines.Line2D):
	if line.get_color() in ("w", "white", "#ffffff", (1, 1, 1), (1, 1, 1, 1)):
		line.set_linewidth(lw)

real_fig