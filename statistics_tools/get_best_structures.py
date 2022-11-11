#!/usr/bin/env python                                                                                                                         
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.visualize import view

import sys

runs_name = sys.argv[1]

try:
    Nmax = sys.argv[2]
except:
    Nmax = ''

best_structures = []
for i in range(100):
    try:
        traj_path = runs_name + '/run{}/structures.traj'.format(i)
        traj = read(traj_path, index=f':{Nmax}')

        E = np.array([a.get_potential_energy() for a in traj])
        index_best = np.argmin(E)
        Ebest = E[index_best]
        a_best = traj[index_best]

        Ncalc = len(traj)
        print('{}:'.format(i), traj_path)
        print('Ncalc:', Ncalc, 'Ebest:', Ebest)
        best_structures.append(a_best)
    except Exception as error:
        print(error)
        break
    
write('temp_best.traj', best_structures)
view(best_structures)
