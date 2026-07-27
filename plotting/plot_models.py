#%%
import os
import sys
sys.path.append('/home/reabt/experiments/ncc/MRI/code/')

from configs.config_MRI import MRIconfig_C2
from utils.load_cfg import load_MRI_config_instance
from utils.files import fix_spm_rawdata_paths
from utils.plots import *
from utils.rsa import *
import rsatoolbox as rsa
import matplotlib
%matplotlib inline

#%%

cfg = load_MRI_config_instance('MRIconfig_C2') 

#%%
models = cfg.get_model_RDM()

modelNames = ALL_MODELS
for model, name in zip(models, modelNames):
	fig, _, _ = rsa.vis.rdm_plot.show_rdm(model.rdm_obj, 
									contour_symmetry=rsa.vis.rdm_plot.Symmetry.UPPER, 
									overlay_symmetry=rsa.vis.rdm_plot.Symmetry.UPPER, 
									cmap='Grays_r', # add '_r' for reversed color scale --> 'bone_r'
									# cmap='bone_r',
									vmin=0.05, vmax = 1, 
									nanmask= ~np.tri(24, 24,dtype=bool),
									contour_color = 'red',
									# gridlines = [3.5, 7.5, 11.5, 15.5, 19.5, 23.5],
									# num_pattern_groups = 4
									  )
	fig.set_frameon(False)
	# fig.set_edgecolor('red')
	edgecolor_objs = [obj for obj in fig.findobj()
	                  if any(hasattr(obj, attr) for attr in ('get_edgecolor',
	                                                         'set_edgecolor',
	                                                         'edgecolor',
	                                                         'Edgecolor'))]
	
	for obj in edgecolor_objs:
		obj.set_edgecolor(None)

	fig.show()
	# print(fig)


	fig_opts = dict( dpi=300, bbox_inches='tight', pad_inches = 0)
	fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/model_transparent_{name}.svg', transparent=True, format='svg', **fig_opts)
	fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/model_transparent_{name}.png', transparent=True, format='png', **fig_opts)
	fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/model_{name}.svg', transparent=False, format='svg', **fig_opts)
	fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/model_{name}.png', transparent=False, format='png', **fig_opts)
	
	fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/model_grid_transparent_{name}.svg', transparent=True, format='svg', **fig_opts)
	fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/model_grid_transparent_{name}.png', transparent=True, format='png', **fig_opts)
	fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/model_grid_{name}.svg', transparent=False, format='svg', **fig_opts)
	fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/model_grid_{name}.png', transparent=False, format='png', **fig_opts)

	# # a
	# # a

# %%
