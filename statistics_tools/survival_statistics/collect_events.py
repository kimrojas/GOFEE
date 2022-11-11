import numpy as np
from scipy.spatial.distance import euclidean

from ase.io import read, write

import os

def collect_events(a_gm, runs_path, featureCalculator, dmax=0.1, dE_max=0.2, N=100, Ninit=0, traj_name=None, save_dir='stats', check_finalPop=False):
    f_gm = featureCalculator.get_feature(a_gm)
    E_gm = a_gm.get_potential_energy()
    times = []
    events = []
    structures = []
    iter_max = 0
    for n in range(N):
        found = False
        try:
            traj = read(runs_path + 'run{}/structures.traj'.format(n,n), index=':')
            run_length = len(traj)
            if run_length > iter_max:
                iter_max = len(traj)

            E_all = np.array([a.get_potential_energy() for a in traj])
            for i, (E, a) in enumerate(zip(E_all, traj)):
                if E < E_gm + dE_max:
                    f = featureCalculator.get_feature(traj[i])
                    d = euclidean(f, f_gm)
                    if d < dmax:
                        number_SP = i + Ninit
                        E = a.get_potential_energy()
                        print('run:', n, '\t iter_found:', number_SP, '\t dist:', d, '\t energy:', E)
                        times.append(i)
                        events.append(1)
                        structures.append(a)
                        found = True
                        break
            if not found:
                times.append(run_length)
                events.append(0)
        except Exception as err:
            print('Exception caught:', err)
        

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if traj_name is not None:
        write(save_dir+'/'+traj_name, structures)

    return times, events

def collect_events_energy(a_gm, runs_path, dE_max=0.3, Fmax=None, N=100, Ninit=0, traj_name=None, save_dir='stats', check_finalPop=False):
    E_gm = a_gm.get_potential_energy()
    times = []
    events = []
    structures = []
    iter_max = 0
    for n in range(N):
        found = False
        try:
            traj = read(runs_path + 'run{}/structures.traj'.format(n,n), index=':')
            run_length = len(traj)
            if run_length > iter_max:
                iter_max = len(traj)
            
            E_all = np.array([a.get_potential_energy() for a in traj])
            F_all = np.array([a.get_forces() for a in traj])
            for i, (E, F, a) in enumerate(zip(E_all, F_all, traj)):
                if Fmax is None:
                    Fmax_cond = True
                else:
                    Fmax_i = np.sqrt((F**2).sum(axis=1).max())
                    Fmax_cond = Fmax_i <= Fmax
                if E < E_gm + dE_max and Fmax_cond:
                    number_SP = i + Ninit
                    print('run:', n, '\t iter_found:', number_SP, '\t Energy:', E)
                    times.append(number_SP)
                    events.append(number_SP)
                    structures.append(a)
                    found = True
                    break
            if not found:
                times.append(run_length)
                events.append(0)            
        except Exception as err:
            print('Exception caught:', err)
        

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if traj_name is not None:
        write(save_dir+'/'+traj_name, structures)

    return times, events

def collect_events_old(a_gm, runs_path, featureCalculator, dmax=0.1, N=100, Ninit=0, traj_name=None, save_dir='stats', check_finalPop=False):
    f_gm = featureCalculator.get_feature(a_gm)
    times = []
    events = []
    structures = []
    iter_max = 0
    for n in range(N):
        found = False
        try:
            traj = read(runs_path + 'run{}/structures.traj'.format(n,n), index=':')
            run_length = len(traj)
            if run_length > iter_max:
                iter_max = len(traj)

            fMat = featureCalculator.get_featureMat(traj)
            for i, (f, a) in enumerate(zip(fMat, traj)):
                d = euclidean(f, f_gm)
                if d < dmax:
                    number_SP = i + Ninit
                    E = a.get_potential_energy()
                    print('run:', n, '\t iter_found:', number_SP, '\t dist:', d, '\t energy:', E)
                    times.append(i)
                    events.append(1)
                    structures.append(a)
                    found = True
                    break
            if not found:
                times.append(run_length)
                events.append(0)            
        except:
            pass
        

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if traj_name is not None:
        write(save_dir+'/'+traj_name, structures)

    return times, events
