#%%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.datasets import *
#%%
start = -10
times = 4
step = 4
end = start+(times*step)
print(end)
cut_coords=np.arange(start, end, step)


# Load the standard MNI template
template = load_mni152_template()


# Create an empty statistical map
empty_data = np.zeros(template.shape, dtype=np.float32)
empty_img = nib.Nifti1Image(empty_data, template.affine)

fig = plt.figure(figsize=(20, 5))

# Plot only the anatomical template
plotting.plot_stat_map(
    empty_img,
    bg_img=template,
    display_mode="y",      # coronal slice
    cut_coords=cut_coords,        # y = 0 mm
    threshold=np.inf,      # hide the (empty) statistical map
    draw_cross=False,
    annotate=False,
    colorbar=False,
    black_bg=False,
	figure = fig,
    cmap="cold_hot"
)
for ax in fig.axes:
    ax.set_facecolor(None)

fig.savefig( '/home/reabt/experiments/ncc/MRI/code/plots_samba/brainTemplate_4.png', transparent=True, format='png', dpi=600, bbox_inches='tight')

#%%

from nilearn import plotting
from nilearn.datasets import load_mni152_template

template = load_mni152_template()

plotting.plot_anat(
    template,
    display_mode="y",
    cut_coords=[0],
    draw_cross=False,
    annotate=False,
    black_bg=False
)

plt.show()
#%%
# display

# # fig.savefig('/home/reabt/experiments/ncc/MRI/code/plots_ÖGP/evoked_meg_transparent2.png', transparent=True, format='png', dpi=300, bbox_inches='tight')

# plt.show()