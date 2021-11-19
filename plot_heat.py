import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp2d
from lib.mapper import load_xyz, map_function

try:
    xyz = np.load('data/xyz.npy')
    XY = map_function(xyz)
    print('Load xyz array succeed!')
except:
    print('Load xyz array fail! Recompute and save!')
    xyz_file = 'data/Biosemi128OK.xyz'
    xyz, channel_name = load_xyz(xyz_file)
    np.save('data/xyz.npy', xyz)



'''
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(XY[:,0], XY[:,1], 'o', color='black')
for i in range(len(XY)):
    plt.annotate(channel_name[i], (XY[i,0], XY[i,1]))


ax.set_aspect('equal', adjustable='box')
plt.savefig('dot.png')

sys.exit()
'''

np.random.seed(42)
X_dat = XY[:,0]
Y_dat = XY[:,1]
Z_dat = np.random.rand(len(Y_dat))

# create x-y points to be used in heatmap
num_points = 2000
xi = np.linspace(X_dat.min(), X_dat.max(), num_points)
yi = np.linspace(Y_dat.min(), Y_dat.max(), num_points)
#print(xi)

# Interpolate for plotting
zi = griddata((X_dat, Y_dat), Z_dat, (xi[None,:], yi[:,None]), method='cubic')
#zi = zi-zi.min()+0.01


xv, yv = np.meshgrid(xi, yi)
circle = xv**2 + yv**2
zi[circle>1] = float("NAN")

fig, ax = plt.subplots()
#c = ax.pcolormesh(xi, yi, -zi, cmap='RdBu', vmin=-Z_dat.min(), vmax=-Z_dat.max())
c = ax.pcolormesh(xi, yi, zi, cmap='magma', vmin=Z_dat.min(), vmax=Z_dat.max())
#c = ax.pcolormesh(xi, yi, zi, cmap='seismic', vmin=Z_dat.min(), vmax=Z_dat.max()) 

# set the limits of the plot to the limits of the data
ax.axis([xi.min(), xi.max(), yi.min(), yi.max()])
fig.colorbar(c, ax=ax)

circle=plt.Circle((0,0),1)

ax.set_aspect('equal', adjustable='box')
plt.savefig('heat.png')