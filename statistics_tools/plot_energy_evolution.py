#!/usr/bin/env python                                                                                                                         
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.visualize import view

import sys
import argparse


parser = argparse.ArgumentParser(description='Plot energy evolution of runs')
parser.add_argument('-i', type=str, help='folder containing GOFEE runs')
parser.add_argument('-dE', type=float, help='Energy range used for plotting')
parser.add_argument('-N', default=12, type=int, help='Maximum number of runs to plot')
parser.add_argument('-kappa', default=1, type=float, help='kappa in acquisition funciton')
parser.add_argument('-ref', type=str, help='reference traj, for which the energy is plottet as dashed line')
parser.add_argument('-name', default=None, type=str, help='reference traj, for which the energy is plottet as dashed line')
args = parser.parse_args()

runs_name = args.i
dE = args.dE
Nruns2use = args.N
kappa = args.kappa
ref = args.ref
name = args.name

if ref is not None:
    a_ref = read(ref, index='0')
    Eref = a_ref.get_potential_energy()
"""
runs_name = sys.argv[1]
try:
    dE = int(sys.argv[2])
except:
    dE = None

try:
    Nruns2use = int(sys.argv[3])
except:
    Nruns2use = 12
"""
E_all = []
Epred_all = []
Epred_std_all = []
Ebest_all = []
for i in range(Nruns2use):
    print('progress: {}/{}'.format(i+1,Nruns2use))
    try:
        traj = read(runs_name + '/run{}/structures.traj'.format(i), index=':')
        E = np.array([a.get_potential_energy() for a in traj])
        E_all.append(E)
        Epred = []
        Epred_std = []
        for a in traj:
            try:
                Epred_i = a.info['key_value_pairs']['Epred']
                Epred_std_i = a.info['key_value_pairs']['Epred_std']
                Epred.append(Epred_i)
                Epred_std.append(Epred_std_i)
            except Exception as err:
                #print(err)
                Epred.append(np.nan)
                Epred_std.append(np.nan)
        Epred_all.append(np.array(Epred))
        Epred_std_all.append(np.array(Epred_std))
        
        Ebest = np.min(E)
        Ebest_all.append(Ebest)
        print('Ebest={}'.format(Ebest))
    except Exception as error:
        print(error)

ncol = 1
nrow = np.int(np.ceil(Nruns2use / ncol))
width = 20 # ncol*5
height = nrow*5
fig, axes = plt.subplots(nrow, ncol, figsize=(width, height))
dy_plot = 0.25/nrow
plt.subplots_adjust(bottom=dy_plot, top=1-dy_plot, hspace=0.2)
for i,(E, Epred, Epred_std, Ebest) in enumerate(zip(E_all, Epred_all, Epred_std_all, Ebest_all)):
    x = np.arange(len(E))
    irow = i // ncol
    icol = i % ncol
    if nrow > 1:
        if ncol > 1:
            ax = axes[irow,icol]
        else:
            ax = axes[irow]
    else:
        if ncol > 1:
            ax = axes[icol]
        else:
            ax = axes
    ax.set_title(f'run {i} , Ebest={Ebest:.3f}')
    ax.set_xlabel('Evaluations in search')
    ax.set_ylabel('Energy [eV]')
    ax.plot(x, Epred, color='crimson', label='Predicted')
    ax.fill_between(x, Epred-kappa*Epred_std, Epred+kappa*Epred_std, color='crimson', alpha=0.3)
    ax.plot(x,E, 'k', lw=0.5, label='Evaluated')
    if ref is not None:
        ax.plot([x[0], x[-1]], [Eref, Eref], 'k:')
    ax.legend(loc='upper right')
    if dE is not None:
        if ref is not None:
            Emin = Eref - 1
        else:
            Emin = Ebest - 1
        ax.set_ylim([Emin, Emin+dE])
if name is not None:
    plt.savefig(f'./energyEvol_{name}.png')
else:
    plt.savefig(f'./energyEvol_{runs_name}.png')

