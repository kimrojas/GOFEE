#!/usr/bin/env python                                                                                                                         
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gofee.surrogate.descriptor.fingerprint import Fingerprint

from ase.io import read, write
from ase.visualize import view

import sys

runs_name = sys.argv[1]

try:
    Nruns2use = int(sys.argv[2])
except:
    Nruns2use = 12

descriptor = Fingerprint()

Ncols = 4
Nrows = int(Nruns2use/Ncols)

width = 5
height = 4

fig, axes = plt.subplots(3,4,figsize=(Ncols*width, Nrows*height))
for i in range(Nruns2use):
    i_col = i % Ncols
    i_row = i // Ncols
    ax = axes[i_row][i_col]
    ax.set_title(f'run {i}')
    ax.set_xlabel('feature space distance')
    ax.set_ylabel('Energy rel. to Emin')
    print('progress: {}/{}'.format(i+1,Nruns2use))
    try:
        traj = read(runs_name + '/run{}/structures.traj'.format(i), index=':')
        E = np.array([a.get_potential_energy() for a in traj])
        index_min = np.argmin(E)
        a_min = traj[index_min]
        f_min = descriptor.get_featureMat([a_min])
        f_all = descriptor.get_featureMat(traj)
        d = cdist(f_min, f_all, metric='euclidean').reshape(-1)
        dE = E - E[index_min]
        ax.loglog(d,dE, 'ko', alpha=0.3, label=f'Emin = {E[index_min]}')
        ax.legend()
    except Exception as error:
        print(error)
plt.tight_layout()
plt.savefig(f'rel_dist_{runs_name}.png')