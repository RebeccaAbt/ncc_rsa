#%%
import numpy as np
from matplotlib import pyplot as plt

#%%
def gabor_fn(sigma,theta,Lambda,psi,gamma):
    sigma_x = sigma
    sigma_y = float(sigma)/gamma

    # Bounding box
    nstds = 3
    xmax = max(abs(nstds*sigma_x*np.cos(theta)),abs(nstds*sigma_y*np.sin(theta)))
    xmax = np.ceil(max(1,xmax))
    ymax = max(abs(nstds*sigma_x*np.sin(theta)),abs(nstds*sigma_y*np.cos(theta)))
    ymax = np.ceil(max(1,ymax))
    xmin = -xmax; ymin = -ymax
    (x,y) = np.meshgrid(np.arange(xmin,xmax+1),np.arange(ymin,ymax+1 ))
    (y,x) = np.meshgrid(np.arange(ymin,ymax+1),np.arange(xmin,xmax+1 ))

    # Rotation
    x_theta=x*np.cos(theta)+y*np.sin(theta)
    y_theta=-x*np.sin(theta)+y*np.cos(theta)

    gb= np.exp(-.5*(x_theta**2/sigma_x**2+y_theta**2/sigma_y**2))*np.cos(2*np.pi/Lambda*x_theta+psi)
    return gb


data = gabor_fn(sigma=200.,theta=np.pi/2.,Lambda=200.,psi=90,gamma=1.)


fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(data, cmap='gray', interpolation='none')
ax.axis('off')
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

edgecolor_objs = [obj for obj in fig.findobj()
	                  if any(hasattr(obj, attr) for attr in ('get_edgecolor',
	                                                         'set_edgecolor',
	                                                         'edgecolor',
	                                                         'Edgecolor'))]
	
for obj in edgecolor_objs:
	obj.set_edgecolor(None)
fig.set_frameon(False)



fig_opts = dict( dpi=300, bbox_inches='tight', pad_inches = 0)
fig.savefig(f'/home/reabt/experiments/ncc/MRI/code/plots_samba/gabor.png', transparent=True, format='png', **fig_opts)