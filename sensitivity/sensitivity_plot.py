import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

mpl.rc('image', interpolation='nearest', origin='lower')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 13,
          #'legend.frameon': True,
          'figure.figsize': (9, 7),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize': 15,
         'ytick.labelsize': 15}
pylab.rcParams.update(params)

target_name='J0200-5105'
date='2021_10_20'
in_dir='/Users/data/SOFI/'+date+'/'+target_name+'/reduced'
# in_dir='/Users/s2224823/data/old_SOFI_data/myPSO318/reduced'

sensitivity_map=np.loadtxt(in_dir+'/sensitivity_map_prdthres_03.txt')
prd_grid=np.loadtxt(in_dir+'/prd_grid.txt')
amp_grid=np.loadtxt(in_dir+'/amp_grid.txt')

#make a plot of senstivity map
X, Y = np.meshgrid(prd_grid, amp_grid)
levels = np.array([0.2, 0.5, 0.7, 0.9])
# norm = cm.colors.Normalize(0, 1)
cmap = cm.RdBu
extent=(prd_grid.min(), prd_grid.max(), amp_grid.min()*100, amp_grid.max()*100)
fig, ax=plt.subplots(constrained_layout=True)
im=ax.imshow(sensitivity_map, vmax=1, vmin=0, cmap=cm.RdBu, aspect='auto', extent=extent)
# cs=ax.contourf(X, Y, sensit, levels, norm=norm, cmap=cm.get_cmap(cmap))
#plot contour lines
cs=ax.contour(sensitivity_map, levels, colors='black', linewidths=1.5, extent=extent)
#label contour lines
ax.clabel(cs, inline=True, fontsize=10)
ax.set_xlabel('Period [hr]')
ax.set_ylabel('Amplitude [%]')
ax.minorticks_on()
cbar=fig.colorbar(im, ax=ax)
plt.savefig(in_dir+'/plots/sensitivity_map.png', dpi=200)
plt.close()
