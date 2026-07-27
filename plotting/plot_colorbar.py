
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl

#%%
fig = plt.figure(figsize=(12, 12))

ax = fig.add_axes([0.35, 0.05, 0.95, 0.10])  # left, bottom, width, height

norm = mpl.colors.Normalize(vmin=0, vmax=1)

cbar = mpl.colorbar.ColorbarBase(
    ax,
    cmap='Grays',
	# cmap = 'viridis',
    norm=norm,
    orientation='horizontal'
)

# Remove ticks and tick labels
cbar.set_ticks([])
cbar.ax.tick_params(length=0)

# Remove the outline (optional)
cbar.outline.set_visible(False)
# cbar.outline.set_visible(True)
# cbar.outline.set_linewidth(5)
cbar

plt.show()

#%%

fig_opts = dict( dpi=300, bbox_inches='tight', pad_inches = 0)
fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/colorbar_grays.png', 
			transparent=True, format='png', **fig_opts)
