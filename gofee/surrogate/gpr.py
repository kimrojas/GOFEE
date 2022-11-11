import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import cdist, euclidean

from ase.constraints import FixAtoms

from gofee.surrogate.kernel import GaussKernel, DoubleGaussKernel
from gofee.surrogate.descriptor.fingerprint import Fingerprint
from gofee.surrogate.prior.prior import RepulsivePrior
from gofee.surrogate.gpr_calculator import gpr_calculator
from gofee.utils import get_force_points_by_energy_decent

from time import time
from copy import deepcopy

import sys


def remove_fixed_atoms_from_indices(atoms_list, indices_list):
    indices_list_cleaned = []
    for atoms, indices in zip(atoms_list, indices_list):
        atom_indices_use = []
        atom_indices_fixed = []
        for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    atom_indices_fixed = constraint.get_indices()
                    break
        for idx_pair in indices:
            # idx_pair[0] is atoms index ([1] is cartesian-direction)
            if idx_pair[0] not in atom_indices_fixed:
                atom_indices_use.append(tuple(idx_pair))
        indices_list_cleaned.append(tuple(atom_indices_use))
    # Make it a tuple to use for indexing
    return tuple(indices_list_cleaned)

def add_to_array(x_base, x_add):
    if x_base is None:
        x_base = x_add
    else:
        x_base = np.r_[x_base, x_add]
    return x_base

def get_largest_force_components(forces_list, N=3):
    force_indices_save = []
    for F in forces_list:
        indices = np.dstack(np.unravel_index(np.argsort(np.abs(F).ravel())[::-1], F.shape))[0, :N]
        force_indices_save.append(indices)
    return force_indices_save

class gpr_memory():
    """ Class for saving "expensive to calculate" data for
    the Gaussian Process Regression model.
    """
    def __init__(self, descriptor, prior, store_forces=False, **kwargs):
        self.descriptor = descriptor
        self.prior = prior
        self.store_forces=store_forces
        self.initialize_data()

    def initialize_data(self):
        self.structures = []
        self.energies = None
        self.features = None
        self.prior_energies = None

        self.forces = None
        self.feature_gradients = None
        self.prior_forces = None

        if self.store_forces:
            self.forces_all = []
            self.feature_gradients_all = []
            self.prior_forces_all = []
        
            self.Nforce_list = []
            self.saved_force_indices = []

    def get_data(self):
        return np.copy(self.energies), np.copy(self.features), np.copy(self.prior_energies)

    def get_feature_and_feature_gradients(self):
        all_idx = np.arange(len(self.Nforce_list))
        structure_idx_with_saved_forces = all_idx[np.array(self.Nforce_list) > 0]
        X_sub = self.features[structure_idx_with_saved_forces]
        Nforce_comp = np.max(self.Nforce_list)
        Nf = self.feature_gradients.shape[-1]
        return X_sub, self.feature_gradients.reshape(-1, Nforce_comp, Nf)

    def save_data(self, atoms_list, force_indices_save=None, add_data=True, save_folder=None):
        if not add_data:
            self.initialize_data()

        self.structures += atoms_list
        self.save_energies(atoms_list)
        self.save_features(atoms_list)
        self.save_prior_energies(atoms_list)

        # Extract and save force related stuff
        if self.store_forces:
            forces_list = [a.get_forces() for a in atoms_list]
            self.forces_all += forces_list
            if save_folder is not None:
                Nsaved = len(self.energies)-len(atoms_list)
                for i, a in enumerate(atoms_list):
                    i_save = Nsaved+i
                    np.save(f'{save_folder}/f_grad{i_save}.npy', self.descriptor.get_featureGradient(a))
            if self.prior is not None:
                prior_forces_list = [self.prior.forces(a) for a in atoms_list]
                self.prior_forces_all += prior_forces_list

            # Save desired force information
            if force_indices_save is not None:
                assert len(atoms_list) == len(force_indices_save)
                #force_indices_save = remove_fixed_atoms_from_indices(atoms_list, force_indices_save)
                self.saved_force_indices += force_indices_save
                self.Nforce_list += [len(indices) for indices in force_indices_save]
                self.save_force_components(forces_list, force_indices_save)
                self.save_feature_gradient_components(feature_gradients_list, force_indices_save)
                if self.prior is not None:
                    self.save_prior_force_components(prior_forces_list, force_indices_save)
                else:
                    self.prior_forces = 0

    def save_energies(self, atoms_list):
        energies_save = np.array([a.get_potential_energy() for a in atoms_list])
        self.energies = add_to_array(self.energies, energies_save)
    
    def save_features(self, atoms_list):
        features_save = self.descriptor.get_featureMat(atoms_list)
        self.features = add_to_array(self.features, features_save)

    def save_prior_energies(self, atoms_list):
        if self.prior is not None:
            prior_energies_save = np.array([self.prior.energy(a) for a in atoms_list])
            self.prior_energies = add_to_array(self.prior_energies, prior_energies_save)
        else:
            self.prior_energies = 0


    def save_force_components(self, force_list, force_indices_save=None, add=True):
        forces_save = None
        for F, indices in zip(force_list, force_indices_save):
            if len(indices) > 0:
                forces_save_i = np.array([F[i_atom, i_dim] for i_atom, i_dim in indices])
                forces_save = add_to_array(forces_save, forces_save_i)
        if add:
            self.forces = add_to_array(self.forces, forces_save)
        else:
            self.forces = forces_save

    def save_feature_gradient_components(self, feature_gradients_list, force_indices_save, add=True):
        feature_gradients_save = None
        for f_grad, indices in zip(feature_gradients_list, force_indices_save):
            if len(indices) > 0:
                feature_gradients_save_i = np.array([f_grad[3*i_atom+i_dim] for i_atom, i_dim in indices])
                feature_gradients_save = add_to_array(feature_gradients_save, feature_gradients_save_i)
        if add:
            self.feature_gradients = add_to_array(self.feature_gradients, feature_gradients_save)
        else:
            self.feature_gradients = feature_gradients_save

    def save_prior_force_components(self, prior_forces_list, force_indices_save, add=True):
        prior_forces_save = None
        for F_prior, indices in zip(prior_forces_list, force_indices_save):
            if len(indices) > 0:
                prior_forces_save_i = np.array([F_prior[3*i_atom+i_dim] for i_atom, i_dim in indices])
                prior_forces_save = add_to_array(prior_forces_save, prior_forces_save_i)
        if add:
            self.prior_forces = add_to_array(self.prior_forces, prior_forces_save)
        else:
            self.prior_forces = prior_forces_save

    def trim_memory(self, ref, Nmax, Nmax_force=None, Nforces=None, save_folder=None):
        """ Trim memory to only include the Nmax structures closest to
        the structure 'ref'.
        """
        assert self.energies is not None

        # Get data-indices to use
        f_ref = self.descriptor.get_feature(ref)
        d = cdist(f_ref.reshape(1,-1), self.features, metric='euclidean').reshape(-1)
        indices_use = np.argsort(d)[:Nmax]

        # Filter all data in memory
        self.energies = self.energies[indices_use]
        self.features = self.features[indices_use]
        if self.prior is not None:
            self.prior_energies = self.prior_energies[indices_use]

        if Nforces is not None:
            assert self.store_forces == True
            if Nmax_force is not None:
                indices_use_force = indices_use[:Nmax_force]
                #indices_use_force = indices_use[ get_force_points_by_energy_decent(f_ref, self.features, self.energies, Npoints=Nmax_force) ]
            else:
                indices_use_force = indices_use
            self.Nforce_list = [Nforces if i in indices_use_force else 0 for i in indices_use]

            forces_list = [self.forces_all[i] for i in indices_use_force]
            if save_folder is None:
                feature_gradients_list = [self.feature_gradients_all[i] for i in indices_use_force]
            else:
                feature_gradients_list = []
                for i in indices_use_force:
                    feature_gradients_i = np.load(f'{save_folder}/f_grad{i}.npy')
                    feature_gradients_list.append(feature_gradients_i)
            if self.prior is not None:
                prior_forces_list = [self.prior_forces_all[i] for i in indices_use_force]
            # Determine force indices to use
            self.saved_force_indices = get_largest_force_components(forces_list, Nforces)
            # Save relevant force components
            self.save_force_components(forces_list, self.saved_force_indices)
            self.save_feature_gradient_components(feature_gradients_list, self.saved_force_indices)
            if self.prior is not None:
                self.save_prior_force_components(prior_forces_list, self.saved_force_indices)
            else:
                self.prior_forces = 0
            
        else:
            self.forces = None
            self.feature_gradients = None
            self.prior_forces = None

    def get_trimmed_memory(self, ref, Nmax, Nmax_force=None, Nforces=None, save_folder=None):
        """ Trim memory to only include the Nmax structures closest to
        the structure 'ref'.
        """
        assert self.energies is not None

        # Determine the number of free atoms
        N_free_atoms = self.structures[0].get_number_of_atoms()
        for constraint in self.structures[0].constraints:
            if isinstance(constraint, FixAtoms):
                N_free_atoms = len(constraint.get_indices())
                break

        mem_new = gpr_memory(descriptor=self.descriptor, prior=self.prior)

        # Get data-indices to use
        f_ref = self.descriptor.get_feature(ref)
        d = cdist(f_ref.reshape(1,-1), self.features, metric='euclidean').reshape(-1)
        indices_use = np.argsort(d)[:Nmax]

        # Filter all data in memory
        mem_new.energies = np.copy(self.energies[indices_use])
        mem_new.features = np.copy(self.features[indices_use])
        if self.prior is not None:
            mem_new.prior_energies = np.copy(self.prior_energies[indices_use])

        # Get structure indices for which to use forces
        if Nforces is not None:
            assert self.store_forces == True
            Nforces = min(int(3*N_free_atoms), Nforces)
            if Nmax_force is not None:
                indices_use_force = indices_use[:Nmax_force]
                #indices_use_force = indices_use[ get_force_points_by_energy_decent(f_ref, self.features, self.energies, Npoints=Nmax_force) ]
            else:
                indices_use_force = indices_use
            mem_new.Nforce_list = deepcopy([Nforces if i in indices_use_force else 0 for i in indices_use])

            forces_list = [self.forces_all[i] for i in indices_use_force]
            if save_folder is None:
                feature_gradients_list = [self.feature_gradients_all[i] for i in indices_use_force]
            else:
                feature_gradients_list = []
                for i in indices_use_force:
                    feature_gradients_i = np.load(f'{save_folder}/f_grad{i}.npy')
                    feature_gradients_list.append(feature_gradients_i)
            if self.prior is not None:
                prior_forces_list = [self.prior_forces_all[i] for i in indices_use_force]
            # Determine force indices to use
            mem_new.saved_force_indices = get_largest_force_components(forces_list, Nforces)
            # Save relevant force components
            mem_new.save_force_components(forces_list, mem_new.saved_force_indices)
            mem_new.save_feature_gradient_components(feature_gradients_list, mem_new.saved_force_indices)
            if self.prior is not None:
                mem_new.save_prior_force_components(prior_forces_list, mem_new.saved_force_indices)
            else:
                mem_new.prior_forces = 0
            
        else:
            mem_new.forces = None
            mem_new.feature_gradients = None
            mem_new.prior_forces = None

        return mem_new


class GPR():
    """Gaussian Process Regression
    
    Parameters:
    
    descriptor:
        Descriptor defining the represention of structures. The Gaussian Process
        works with the representations.
    
    kernel:
        Kernel (or covariance) function used in the Gaussian Process.
    
    prior:
        Prior mean function used.

    n_restarts_optimizer: int
        Number of gradient decent restarts performed by each compute process
        during hyperparameter optimization.
    """
    def __init__(self, descriptor=None, kernel='double', prior=None, n_restarts_optimizer=1, template_structure=None, scale_reg=False, **kwargs):
        if descriptor is None:
            self.descriptor = Fingerprint()
        else:
            self.descriptor = descriptor
        Nsplit_eta = None
        if template_structure is not None:
            self.descriptor.initialize_from_atoms(template_structure)
            if hasattr(self.descriptor, 'use_angular'):
                if self.descriptor.use_angular:
                    Nsplit_eta = self.descriptor.Nelements_2body

        if kernel == 'single':
            self.kernel = GaussKernel(Nsplit_eta=Nsplit_eta)
        elif kernel == 'double':
            self.kernel = DoubleGaussKernel(Nsplit_eta=Nsplit_eta)
        else:
            self.kernel = kernel
            self.kernel.Nsplit_eta = Nsplit_eta
            self.kernel.set_theta_bounds(eta_bnd=(0.1,10))

        if prior is None:
            self.prior = RepulsivePrior()
        else:
            self.prior = prior

        self.n_restarts_optimizer = n_restarts_optimizer
        self.template_structure = template_structure
        self.scale_reg = scale_reg

        self.memory = gpr_memory(self.descriptor, self.prior, **kwargs)

    def predict_energy(self, a, eval_std=False):
        """Evaluate the energy predicted by the GPR-model.

        parameters:

        a: Atoms object
            The structure to evaluate.

        eval_std: bool
            In addition to the force, predict also force contribution
            arrising from including the standard deviation of the
            predicted energy.
        """
        x = self.descriptor.get_feature(a)

        k = self.kernel.kernel_vector(x, self.X)
        if self.memory.forces is not None:
            # Add appropriate kernel derivatives to kernel vector, if
            # forces have been used for training.
            Nf = x.shape[-1]
            k_jac = self.kernel.kernel_jacobian(self.X, x).reshape(-1,Nf)
            k_jac = k_jac.repeat(self.memory.Nforce_list, axis=0)
            k_jac = np.einsum('jk,jk->j', k_jac, self.memory.feature_gradients)
            k = np.r_[k, k_jac]

        E = np.dot(k,self.alpha) + self.bias + self.prior.energy(a)
        if eval_std:
            # Lines 5 and 6 in GPML
            vk = np.dot(self.K_inv, k)
            E_var = self.K0 - np.dot(k, vk)
            E_var = max(0, E_var)
            if E_var < 0:
                print(f'Evar = {E_var}')
            #assert g >= 0, f'g = {g}, Estd = {np.sqrt(self.K0 - np.dot(k, vk))}'
            E_std = np.sqrt(E_var)
            return E, E_std
        else:
            return E

    def predict_forces(self, a, eval_with_energy_std=False):
        """Evaluate the force predicted by the GPR-model.

        parameters:

        a: Atoms object
            The structure to evaluate.

        eval_with_energy_std: bool
            In addition to the force, predict also force contribution
            arrising from including the standard deviation of the
            predicted energy.
        """
        
        t0 = time()
        # Calculate descriptor and its gradient
        x = self.descriptor.get_feature(a)
        t1 = time()
        x_ddr = self.descriptor.get_featureGradient(a)
        t2 = time()

        # Calculate kernel and its derivative
        #k_ddx = self.kernel.kernel_jacobian(x, self.X, trim_shape=True)
        k_ddx = self.kernel.kernel_jacobian(x, self.X).reshape(len(self.X), -1)
        k_ddr = np.dot(k_ddx, x_ddr.T)
        if self.memory.forces is not None:
            # Add appropriate kernel derivatives to kernel vector, if
            # forces have been used for training.
            
            Nf = x.shape[-1]
            t3 = time()
            X_sub, X_grad_sub = self.memory.get_feature_and_feature_gradients()
            k_hess = self.kernel.kernel_hessian(X_sub, x, X_grad_sub, x_ddr.reshape(1,-1,Nf))
            t4 = time()

            k_ddr = np.r_[k_ddr, k_hess]

        F = -np.dot(k_ddr.T, self.alpha) + self.prior.forces(a)
        t6 = time()
        #print(f'f: {t1-t0:.1e}, fg: {t2-t1:.1e}, kj: {t3-t2:.1e}, kg1: {t4-t3:.1e}, kg2: {t5-t4:.1e}, F: {t6-t5:.1e}, tot: {t6-t0:.1e}')
        #print(f'make K: {t4-t3}, repeat: {t41-t4}, ein1: {t42-t41}, ein2: {t5-t42}')
        #print(f'f: {t1-t0:.1e}, fg: {t2-t1:.1e}, k: {t4-t3:.1e}, F: {t6-t4:.1e}, tot: {t6-t0:.1e}')
        if eval_with_energy_std:
            k = self.kernel.kernel_vector(x, self.X)
            if self.memory.forces is not None:
                Nf = x.shape[-1]
                k_jac = self.kernel.kernel_jacobian(self.X, x).reshape(-1,Nf)
                k_jac = k_jac.repeat(self.memory.Nforce_list, axis=0)
                k_jac = np.einsum('jk,jk->j', k_jac, self.memory.feature_gradients)
                k = np.r_[k, k_jac]
            vk = np.dot(self.K_inv, k)
            E_var = self.K0 - np.dot(k.T, vk)
            if E_var <= 0:
                F_std = np.zeros(F.shape)
            else:
                assert E_var >= 0, f'Evar = {E_var}'
                F_std = 1/np.sqrt(E_var) * np.dot(k_ddr.T, vk)
            return F.reshape((-1,3)), F_std.reshape(-1,3)
        else:
            return F.reshape(-1,3)

    def update_bias(self, indices_use=None):
        E = self.memory.energies
        E_min = np.min(E)
        indices_use = np.arange(len(E))[E < E_min+10]
        if indices_use is not None:
            self.bias = np.mean(self.memory.energies[indices_use] - self.memory.prior_energies[indices_use])
        else:
            self.bias = np.mean(self.memory.energies - self.memory.prior_energies)

    #def update_bias(self):
    #    self.bias = np.mean(self.memory.energies - self.memory.prior_energies)

    def prepare_kernel_hess(self):
        X_sub, X_grad_sub = self.memory.get_feature_and_feature_gradients()
        K_hess = self.kernel.kernel_hessian(X_sub, X_sub, X_grad_sub, X_grad_sub, with_noise=True)
        return K_hess

    def get_regularization_scaling(self):
        
        def get_scaling(x, x_min=10):
            scaling = 0.5*(x - x_min)
            scaling[scaling <=0 ] = 0
            return scaling**2

        if self.scale_reg:
            E, _, prior_energies = self.memory.get_data()
            E -= prior_energies
            E_min = np.min(E)
            E -= E_min
            scaling = get_scaling(E)
            return scaling
        else:
            return None

    def prepare_kernel_matrix(self):
        reg_scaling = self.get_regularization_scaling()
        K = self.kernel(self.X, reg_scaling=reg_scaling)
        if self.memory.forces is not None:
            # Train on both forces and energies.
            t0 = time()
            K_jac = self.kernel.kernel_jacobian(self.X, self.X)  # N_X x N_X x Nf
            t1 = time()
            K_jac = K_jac.repeat(self.memory.Nforce_list, axis=0)  #  N_X_rep x N_X x Nf
            t2 = time()
            # feat_grad shape: N_X_rep x Nf
            K_jac = np.einsum('ijk,ik->ij',K_jac,self.memory.feature_gradients)  # N_X_rep x N_X
            t3 = time()
            K_hess = self.prepare_kernel_hess()
            #print(f'k_jac: {t1-t0:.1e}, k_jac rep: {t2-t1:.1e}, k_jac ein: {t3-t2:.1e}, k_hess: {time()-t3:.1e}')
            K_hess_temp = K_hess
            """
            K_hess = self.kernel.kernel_hessian_old(self.X,self.X)  # N_X x N_X x Nf x Nf
            K_hess = K_hess.repeat(self.memory.Nforce_list, axis=0)
            K_hess = K_hess.repeat(self.memory.Nforce_list, axis=1)  # N_X_rep x N_X_rep x Nf x Nf
            K_hess = np.einsum('ijkl,ik->ijl',K_hess,self.memory.feature_gradients)
            K_hess = np.einsum('ijl,jl->ij',K_hess,self.memory.feature_gradients)
            K_hess += self.kernel.amplitude*self.kernel.noise*np.eye(K_hess.shape[0])
            """
            
            K = np.block([[K,       K_jac.T],
                          [K_jac,   K_hess]])
        return K

    def train(self, atoms_list=None, force_indices_save=None,  add_data=True):
        if atoms_list is not None:
            assert isinstance(atoms_list, list)
            if not len(atoms_list) == 0:
                self.memory.save_data(atoms_list, 
                                      force_indices_save=force_indices_save,
                                      add_data=add_data)
        t0 = time()
        self.update_bias()
        self.E, self.X, self.prior_energies = self.memory.get_data()
        self.Y = self.E - self.prior_energies - self.bias
        t1 = time()
        K = self.prepare_kernel_matrix()
        t2 = time()
        if self.memory.forces is not None:
            # Augment target energies with force components
            #Y_force = self.memory.forces - self.memory.prior_forces
            Y_grad = -(self.memory.forces - self.memory.prior_forces)
            self.Y = np.r_[self.Y, Y_grad]
            # Direct inversion insted of cholesky decomposition because
            # K is not possitive definite.
            self.K_inv = np.linalg.inv(K)
            self.alpha = np.dot(self.K_inv, self.Y)
        else:
            # Train on energies only.
            L = cholesky(K, lower=True)
            self.alpha = cho_solve((L, True), self.Y)
            self.K_inv = cho_solve((L, True), np.eye(K.shape[0]))
        t3 = time()
        #print(f'K: {t2-t1:.1e}, inv: {t3-t2:.1e}, tot: {t3-t0:.1e}')
        reg_scaling = self.get_regularization_scaling()
        self.K0 = self.kernel.kernel_value(self.X[0], self.X[0], reg_scaling=reg_scaling)
        self.min_uncertainty = np.exp(self.kernel.theta[0])*np.exp(self.kernel.theta[-2])

    def update_theta_bounds(self):
        f_use = self.memory.features[:300]
        N = f_use.shape[0]
        idx_triu = np.triu_indices(N,k=1)
        d = cdist(f_use, f_use, metric='euclidean')[idx_triu]
        d_p90 = np.percentile(d,90)
        #self.kernel.theta_bounds[1][0] = np.log(d_p98)
        self.kernel.theta_bounds[1] = np.log([5*d_p90, 25*d_p90])

    def optimize_hyperparameters(self, atoms_list=None, add_data=True, comm=None, maxiter=100, theta_guess=None):
        if self.n_restarts_optimizer == 0:
            self.train(atoms_list)
            return

        if atoms_list is not None:
            assert isinstance(atoms_list, list)
            if not len(atoms_list) == 0:
                self.memory.save_data(atoms_list, add_data=add_data)

        self.update_bias()
        self.E, self.X, self.prior_energies = self.memory.get_data()
        self.Y = self.E - self.prior_energies - self.bias

        #self.update_theta_bounds()

        results = []
        for i in range(self.n_restarts_optimizer):
            theta_initial = np.random.uniform(self.kernel.theta_bounds[:, 0],
                                              self.kernel.theta_bounds[:, 1])
            if i == 0:
                # Make sure that the previously currently choosen
                # hyperparameters are always tried as initial values.
                start_from_best = True
                if comm is not None:
                    if comm.rank > 0:
                        start_from_best = False
                if start_from_best:
                    if theta_guess is not None:
                        theta_initial = theta_guess
                    else:
                        theta_initial = self.kernel.theta
                        
            res = self.constrained_optimization(theta_initial, maxiter)
            results.append(res)
        index_min = np.argmin(np.array([r[1] for r in results]))
        result_min = results[index_min]
        
        if comm is not None:
        # Find best hyperparameters among all communicators and broadcast.
            results_all = comm.gather(result_min, root=0)
            if comm.rank == 0:
                index_all_min = np.argmin(np.array([r[1] for r in results_all]))
                result_min = results_all[index_all_min]
            else:
                result_min = None
            result_min = comm.bcast(result_min, root=0)
                
        self.kernel.theta = result_min[0]
        self.lml = -result_min[1]

        self.train()
    
    def neg_log_marginal_likelihood(self, theta=None, eval_gradient=True):
        if theta is not None:
            self.kernel.theta = theta

        reg_scaling = self.get_regularization_scaling()
        if eval_gradient:
            K, K_gradient = self.kernel(self.X, eval_gradient, reg_scaling=reg_scaling)
        else:
            K = self.kernel(self.X, reg_scaling=reg_scaling)

        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), self.Y)

        lml = -0.5 * np.dot(self.Y, alpha)
        lml -= np.sum(np.log(np.diag(L)))
        lml -= K.shape[0]/2 * np.log(2*np.pi)
        
        if eval_gradient:
            # Equation (5.9) in GPML
            K_inv = cho_solve((L, True), np.eye(K.shape[0]))
            tmp = np.einsum("i,j->ij", alpha, alpha) - K_inv

            lml_gradient = 0.5*np.einsum("ij,kij->k", tmp, K_gradient)
            return -lml, -lml_gradient
        else:
            return -lml

    def constrained_optimization(self, theta_initial, maxiter=100):
        theta_opt, func_min, convergence_dict = \
            fmin_l_bfgs_b(self.neg_log_marginal_likelihood,
                          theta_initial,
                          bounds=self.kernel.theta_bounds,
                          maxiter=maxiter)
        #print(f"# function calls: {convergence_dict['funcalls']}, nit: {convergence_dict['nit']}")
        #print(np.exp(theta_initial))
        return theta_opt, func_min

    def numerical_neg_lml(self, dx=1e-4):
        N_data = self.X.shape[0]
        theta = np.copy(self.kernel.theta)
        N_hyper = len(theta)
        lml_ddTheta = np.zeros((N_hyper))
        for i in range(N_hyper):
            theta_up = np.copy(theta)
            theta_down = np.copy(theta)
            theta_up[i] += 0.5*dx
            theta_down[i] -= 0.5*dx

            lml_up = self.neg_log_marginal_likelihood(theta_up, eval_gradient=False)
            lml_down = self.neg_log_marginal_likelihood(theta_down, eval_gradient=False)
            lml_ddTheta[i] = (lml_up - lml_down)/dx
        return lml_ddTheta

    def numerical_forces(self, a, dx=1e-4, eval_with_energy_std=False):
        Na, Nd = a.positions.shape
        if not eval_with_energy_std:
            F = np.zeros((Na,Nd))
            for ia in range(Na):
                for idim in range(Nd):
                    a_up = a.copy()
                    a_down = a.copy()
                    a_up.positions[ia,idim] += 0.5*dx
                    a_down.positions[ia,idim] -= 0.5*dx
                    
                    E_up = self.predict_energy(a_up)
                    E_down = self.predict_energy(a_down)
                    F[ia,idim] = -(E_up - E_down)/dx
            return F
        else:
            F = np.zeros((Na,Nd))
            Fstd = np.zeros((Na,Nd))
            for ia in range(Na):
                for idim in range(Nd):
                    a_up = a.copy()
                    a_down = a.copy()
                    a_up.positions[ia,idim] += 0.5*dx
                    a_down.positions[ia,idim] -= 0.5*dx
                    
                    E_up, Estd_up = self.predict_energy(a_up, eval_std=True)
                    E_down, Estd_down = self.predict_energy(a_down, eval_std=True)
                    F[ia,idim] = -(E_up - E_down)/dx
                    Fstd[ia,idim] = -(Estd_up - Estd_down)/dx
            return F, Fstd

    def get_state(self):
        return [deepcopy(self.descriptor),
                deepcopy(self.kernel),
                deepcopy(self.prior),
                deepcopy(self.n_restarts_optimizer),
                deepcopy(self.scale_reg),
                deepcopy(self.template_structure)]

    def get_local_model(self, ref, Nmax, Nmax_force=None, Nforces=None, save_folder=None, n_restarts_optimizer=1):
        mem_new = self.memory.get_trimmed_memory(ref, Nmax, Nmax_force, Nforces, save_folder)
        gpr_new = GPR(descriptor=self.descriptor,
                      kernel=deepcopy(self.kernel),
                      prior=self.prior,
                      n_restarts_optimizer=n_restarts_optimizer)
        gpr_new.memory = mem_new
        return gpr_new

    def get_calculator(self, kappa):
        return gpr_calculator(self, kappa)
    
