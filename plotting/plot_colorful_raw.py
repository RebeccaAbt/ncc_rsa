#%%
import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

%matplotlib inline

#%%
subjects = ['19880331igse']
subjectID = subjects[0]

inFile = f'/home/reabt/experiments/ncc/MEG/data/ica_qc/step5_raw_after_ica/{subjectID}_after-ica_raw.fif'
raw = mne.io.read_raw_fif(inFile, preload=False) #.filter(l_freq=None, h_freq=35)# filter just for checking! remove before running script for saving cleaned epochs!
ch_names = raw.ch_names
#%%
plt.close('all')
n_channels = 90
start = 150 # start st second...
picks = 'grad'
order = list(np.random.choice(range(0,203), size=n_channels, replace=False)) # random order= randomly pick the channels

fig = raw.plot(block=False, 
		 butterfly=False, 
		#  group_by = 'original', 
		 n_channels = 90, 
		 scalings = dict(mag=1e-12, grad=2e-11),
		 start = start,
		 picks = picks,
		 order=order,
)   # dict(mag=1e-12, grad=4e-11)

colors = np.random.rand(len(raw.ch_names), 3)

print(dir(fig))
lines = [
	obj for obj in fig.findobj(match=mlines.Line2D)
	if len(obj.get_xdata()) > 2
]

for line, color in zip(lines[:len(raw.ch_names)], colors):
	line.set_color(color)
	line.set_linewidth(2)

fig.set_figheight(15)
fig.set_figwidth(17)
fig.canvas.draw()
plt.show(block=True)
fig


#%%

fig_opts = dict( dpi=600, bbox_inches='tight', pad_inches = 0)

fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/raw_example_2.png', 
	transparent=True, 
	format='png', **fig_opts)


