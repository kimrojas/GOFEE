import numpy as np
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
from ase.data import covalent_radii
from ase.geometry import get_distances
from ase import Atoms
from ase.constraints import FixAtoms

from ase.visualize import view
from ase.io import read, write
from ase.calculators.calculator import Calculator

from copy import deepcopy

import time
import sys
import os

def check_valid_bondlengths(a, blmin=None, blmax=None, indices=None, indices_placed=None, check_too_close=True, check_isolated=True):
    """Calculates if the bondlengths between atoms with indices
    in 'indices' and all other atoms are valid. The validity is
    determined by blmin and blmax.

    Parameters:

    a: Atoms object

    blmin: The minimum allowed distance between atoms in units of
    the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: The maximum allowed distance, in units of the covalent 
    distance, from a single isolated atom to the closest atom. If
    blmax=None, no constraint is enforced on isolated atoms.

    indices: The indices of the atoms of which the bondlengths
    with all other atoms is checked. if indices=None all bondlengths
    are checked.
    """
    bl = get_distances_as_fraction_of_covalent(a,indices, indices_placed)
    
    # Filter away self interactions.
    #bl = bl[bl > 1e-6].reshape(bl.shape[0],bl.shape[1]-1)
    bl = bl[bl > 1e-6].reshape(bl.shape[0],-1)
    
    # Check if atoms are too close
    if blmin is not None and check_too_close:
        tc = np.any(bl < blmin)
    else:
        tc = False
        
    # Check if there are isolated atoms
    if blmax is not None and check_isolated:
        isolated = np.any(np.all(bl > blmax, axis=1))
    else:
        isolated = False
        
    is_valid = not tc and not isolated
    return is_valid
    
def get_distances_as_fraction_of_covalent(a, indices=None, indices_placed=None, covalent_distances=None):
    if indices is None:
        indices = np.arange(len(a))
        
    if covalent_distances is None:
        cd = get_covalent_distance_from_atom_numbers(a, indices=indices, indices_placed=indices_placed)
    else:
        cd = covalent_distances[indices,:]
    if indices_placed is None:
        _, d = get_distances(a[indices].positions,
                             a.positions,
                             cell=a.get_cell(),
                             pbc=a.get_pbc())
    else:
        _, d = get_distances(a[indices].positions,
                             a[indices_placed].positions,
                             cell=a.get_cell(),
                             pbc=a.get_pbc())

    bl = d/cd
    return bl

def get_covalent_distance_from_atom_numbers(a, indices=None, indices_placed=None):
    r_cov_all = np.array([covalent_radii[n] for n in a.get_atomic_numbers()])
    if indices_placed is None:
        r_cov = r_cov_all
    else:
        r_cov = r_cov_all[indices_placed]
    if indices is None:
        r_cov_sub = r_cov_all
    else:
        r_cov_sub = r_cov_all[indices]
    cd_mat = r_cov_sub.reshape(-1,1) + r_cov.reshape(1,-1)
    return cd_mat

def get_min_distances_as_fraction_of_covalent(a, indices=None, covalent_distances=None):
    bl = get_distances_as_fraction_of_covalent(a,indices)
    
    # Filter away self interactions.
    bl = bl[bl > 1e-6].reshape(bl.shape[0],bl.shape[1]-1)
    
    return np.min(bl), bl.min(axis=1).argmin()

def array_to_string(arr, unit='', format='0.4f', max_line_length=80):
    msg = ''
    line_length_counter = 0
    for i, x in enumerate(arr):
        string = f'{i} = {x:{format}}{unit},  '
        #string = f"{f'{i}={x:{format}}{unit},':15}"
        line_length_counter += len(string)
        if line_length_counter >= max_line_length:
            msg += '\n'
            line_length_counter = len(string)
        msg += string
    return msg

def rattle_N_spherical_shell(a, rattle_strength, Nrattle=15):
    """Help function for rattling within a sphere
    """
    for constraint in a.constraints:
        if isinstance(constraint, FixAtoms):
            indices_fixed = constraint.get_indices()
            indices = np.delete(np.arange(a.get_number_of_atoms()), indices_fixed)
            break
    else:
        indices = np.arange(a.get_number_of_atoms())

    Natoms_free = len(indices)
    Natoms_rattle = min(15, Natoms_free)

    indices_rattle = indices[np.random.permutation(Natoms_free)[:Natoms_rattle]]

    num_rattle = a.numbers[indices_rattle]
    r = rattle_strength*np.array([covalent_radii[n] for n in num_rattle]).reshape(-1,1)
    theta = np.random.uniform(low=0, high=2*np.pi, size=Natoms_rattle)
    phi = np.random.uniform(low=0, high=np.pi, size=Natoms_rattle)
    pos_add = r * np.c_[np.cos(theta)*np.sin(phi),
                        np.sin(theta)*np.sin(phi),
                        np.cos(phi)]
    pos = a.get_positions()
    pos[indices_rattle] += pos_add
    return pos

def get_screening_range_from_rattle(a, gpr, dx=0.05, Nrep=20):
    structures_rattle = []
    for i in range(Nrep):
        anew = a.copy()
        pos_new = rattle_N_spherical_shell(anew, rattle_strength=dx)
        anew.set_positions(pos_new)
        structures_rattle.append(anew)
    f0 = gpr.descriptor.get_featureMat([a])
    f_rattled = gpr.descriptor.get_featureMat(structures_rattle)
    d = cdist(f_rattled, f0, metric='euclidean')
    screening_range = np.mean(d)
    return screening_range

def get_minimum_point_by_energy_decent(f, f_all, E_all, Epred=None, Nclosest=5):
    # Go to closest point, and start from there
    d = cdist(f.reshape(1,-1), f_all, metric='euclidean').reshape(-1)
    if Epred is not None:
        f_current = f
        E_current = Epred
    else:
        idx_current = np.argmin(d)
        f_current = f_all[idx_current]
        E_current = E_all[idx_current]
    for i in range(300):
        # Find next point by going through the "N_closest" points in order
        # and accepting the first point that is lower in energy than the current.
        d = cdist(f_current.reshape(1,-1), f_all, metric='euclidean').reshape(-1)
        index_sorted = np.argsort(d)[:Nclosest]
        for idx in index_sorted:
            if E_all[idx] < E_current:
                idx_current = idx
                f_current = f_all[idx]
                E_current = E_all[idx]
                break
        else:
            if i == 0 and Epred is not None:
                return None
            else:
                break
    else:
        raise RuntimeError('Failed to reach minimum data-point using energy decent')        
    return idx_current

def get_force_points_by_energy_decent(f, f_all, E_all, Npoints=20, Nclosest=8):
    n_data = len(f_all)
    indices = []
    mask = np.ones(n_data, dtype=bool)
    Nsteps = min(Npoints, n_data)
    for i in range(Nsteps):
        if len(indices) > 0:
            f_new = f_all[indices[-1]]
        else:
            f_new = f
        d = cdist(f.reshape(1,-1), f_all[mask], metric='euclidean').reshape(-1)
        index_closest = np.argsort(d)[:Nclosest]
        index_closest = np.arange(n_data)[mask][index_closest]
        E_closest = E_all[index_closest]
        index_next = index_closest[np.argmin(E_closest)]
        indices.append(index_next)
        mask[index_next] = False
    return indices

class OperationConstraint():
    """ Class used to enforce constraints on the positions of
    atoms in mutation and crossover operations.

    Parameters:

    box: Box in which atoms are allowed to be placed. It should
    have the form [] [p0, vspan] where 'p0' is the position of
    the box corner and 'vspan' is a matrix containing the three
    spanning vectors.

    xlim: On the form [xmin, xmax], specifying, in the x-direction, 
    the lower and upper limit of the region atoms can be moved 
    within.

    ylim, zlim: See xlim.
    """
    def __init__(self, box=None, xlim=None, ylim=None, zlim=None):
        self.box = box
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

    def check_if_valid(self, positions):
        """ Returns whether positions are valid under the 
        constraints or not.
        """
        if np.ndim(positions) == 1:
            pos = positions.reshape(-1,3)
        else:
            pos = positions

        if self.box is not None:
            p0, V = self.box
            p_rel = pos - p0  # positions relative to box anchor.
            V_inv = np.linalg.inv(V)
            p_box = p_rel @ V_inv  # positions in box-vector basis.
            if (np.any(p_box < 0) or np.any(p_box > 1)):
                return False
        if self.xlim is not None:
            if (np.any(pos[:,0] < self.xlim[0]) or 
                    np.any(pos[:,0] > self.xlim[1])):
                return False
        if self.ylim is not None:
            if (np.any(pos[:,1] < self.ylim[0]) or 
                    np.any(pos[:,1] > self.ylim[1])):
                return False
        if self.zlim is not None:
            if (np.any(pos[:,2] < self.zlim[0]) or 
                    np.any(pos[:,2] > self.zlim[1])):
                return False

        return True

class backgroundCalculator(Calculator):
    """
    This calculator requires that you run a process in the baskground,
    which waits for a traj-file with the name "fname", does the calculation
    and saves the resulting atoms object as "fname[:-5] + '_done.traj'".
    """
    implemented_properties = ['energy', 'forces']
    default_parameters = {}

    def __init__(self, fname, world, **kwargs):
        self.fname = fname
        self.world = world
        self.fname_done = self.fname[:-5] + '_done.traj'
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties or 'forces' in properties:
            #if self.world.rank == 0:
            write(self.fname, atoms)
            for i in range(3600): # ~1 hour
                try:
                    atoms_done = read(self.fname_done, index='0')
                    self.world.barrier()
                    if self.world.rank == 0:
                        os.remove(self.fname_done)
                    E = atoms_done.get_potential_energy()
                    F = atoms_done.get_forces()
                    self.results['energy'] = E
                    self.results['forces'] = F
                    break
                except:
                    time.sleep(1)

class Kmeans():
    def __init__(self, n_clusters, n_init=10, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.centers = None


    def kpp_init(self, X):
        assert len(X) >= self.n_clusters
        n_data, n_dim = X.shape
        centers = np.zeros((self.n_clusters, n_dim))
        i_0 = np.random.randint(n_data)
        centers[0] = deepcopy(X[i_0])
        # Initialize mask
        mask = np.ones(n_data, dtype=bool)
        mask[[i_0]] = False
        for i_clust in range(1,self.n_clusters):
            d_sq = cdist(X[mask], centers[:i_clust], metric='sqeuclidean')
            d_sq_min = d_sq.min(axis=1)
            index_i = np.random.choice(np.arange(n_data)[mask], 1, p=d_sq_min/np.sum(d_sq_min))
            centers[i_clust] = deepcopy(X[index_i])
            mask[[index_i]] = False
        return centers

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        for _ in range(self.n_init):
            centers_old = self.kpp_init(X)
            centers_new = deepcopy(centers_old)
            for _ in range(self.max_iter):
                labels = self.get_labels(X, centers_old)
                for i_clust in range(self.n_clusters):
                    centers_new[i_clust] = np.mean(X[labels==i_clust], axis=0)
                error = np.linalg.norm(centers_new - centers_old)
                if error == 0:
                    break
            score = self.get_score(X, centers_new)
            if self.centers is None:
                self.centers = centers_new
                self.score = score
            elif score < self.score:
                self.centers = centers_new
                self.score = score
            
        self.labels_ = self.get_labels(X, self.centers)
        self.cluster_centers_ = self.centers
        return self

    def get_labels(self, X, centers):
        d = cdist(X, centers, metric='euclidean')
        labels = np.argmin(d, axis=1)
        return labels

    def get_score(self, X, centers):
        d = cdist(X, centers, metric='euclidean')
        score = np.min(d, axis=1).sum()
        return score

def get_sorted_dist_list(atoms, mic=False):
    """ Utility method used to calculate the sorted distance list
        describing the cluster in atoms. """
    numbers = atoms.numbers
    unique_types = set(numbers)
    pair_cor = dict()
    for n in unique_types:
        i_un = [i for i in range(len(atoms)) if atoms[i].number == n]
        d = []
        for i, n1 in enumerate(i_un):
            for n2 in i_un[i + 1:]:
                d.append(atoms.get_distance(n1, n2, mic))
        d.sort()
        pair_cor[n] = np.array(d)
    return pair_cor

class InteratomicDistanceComparator(object):

    """ An implementation of the comparison criteria described in
          L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012)

        Parameters:

        n_top: The number of atoms being optimized by the GA.
            Default 0 - meaning all atoms.

        pair_cor_cum_diff: The limit in eq. 2 of the letter.
        pair_cor_max: The limit in eq. 3 of the letter
        dE: The limit of eq. 1 of the letter
        mic: Determines if distances are calculated
        using the minimum image convention
    """
    def __init__(self, n_top=None, pair_cor_cum_diff=0.015,
                 pair_cor_max=0.7, dE=0.02, mic=False):
        self.pair_cor_cum_diff = pair_cor_cum_diff
        self.pair_cor_max = pair_cor_max
        self.dE = dE
        self.n_top = n_top or 0
        self.mic = mic

    def looks_like(self, a1, a2):
        """ Return if structure a1 or a2 are similar or not. """
        if len(a1) != len(a2):
            raise Exception('The two configurations are not the same size')

        # first we check the energy criteria
        dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
        if dE >= self.dE:
            return False

        # then we check the structure
        a1top = a1[-self.n_top:]
        a2top = a2[-self.n_top:]
        cum_diff, max_diff = self.__compare_structure__(a1top, a2top)

        return (cum_diff < self.pair_cor_cum_diff
                and max_diff < self.pair_cor_max)

    def __compare_structure__(self, a1, a2):
        """ Private method for calculating the structural difference. """
        p1 = get_sorted_dist_list(a1, mic=self.mic)
        p2 = get_sorted_dist_list(a2, mic=self.mic)
        numbers = a1.numbers
        total_cum_diff = 0.
        max_diff = 0
        for n in p1.keys():
            cum_diff = 0.
            c1 = p1[n]
            c2 = p2[n]
            assert len(c1) == len(c2)
            if len(c1) == 0:
                continue
            t_size = np.sum(c1)
            d = np.abs(c1 - c2)
            cum_diff = np.sum(d)
            max_diff = np.max(d)
            ntype = float(sum([i == n for i in numbers]))
            total_cum_diff += cum_diff / t_size * ntype / float(len(numbers))
        return (total_cum_diff, max_diff)