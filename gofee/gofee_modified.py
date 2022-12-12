""" Definition of GOFEE class.
"""
import math
import numpy as np
from scipy.spatial.distance import cdist, euclidean
import pickle
from os.path import isfile

from ase import Atoms
from ase.io import read, write, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.dftb import Dftb
from ase.calculators.calculator import Calculator, FileIOCalculator
from ase.optimize import BFGS
from ase.constraints import FixAtoms

from gofee.surrogate import GPR
from gofee.population import Population, ClusteringPopulation, LVPopulation
from gofee.utils import array_to_string, get_screening_range_from_rattle, get_minimum_point_by_energy_decent, backgroundCalculator
from gofee.parallel_utils import split, parallel_function_eval

from gofee.bfgslinesearch_constrained import relax

from gofee.candidates import CandidateGenerator
from gofee.candidates import RattleMutation

from mpi4py import MPI
world = MPI.COMM_WORLD

import traceback
import sys
from os import path, mkdir
from datetime import timedelta

from time import time
from copy import deepcopy

import os

class GOFEE():
    """
    GOFEE global structure search method.
        
    Parameters:

    structures: Atoms-object, list of Atoms-objects or None
        In initial structures from which to start the sesarch.
        If None, the startgenerator must be supplied.
        If less than Ninit structures is supplied, the remaining
        ones are generated using the startgenerator or by rattling
        the supplied structures, depending on wether the
        startgenerator is supplied.

    calc: ASE calculator
        Specifies the energy-expression
        with respect to which the atomic coordinates are
        globally optimized.

    gpr: GPR object
        The Gaussian Process Regression model used as the
        surrogate model for the Potential energy surface.
    
    startgenerator: Startgenerator object
        Used to generate initial random
        structures. Must be supplied if structures if structues=None.
        (This is the recommended way to initialize the search.)

    candidate_generator: OperationSelector object
        Object used to generate new candidates.

    trajectory: str
        Name of trajectory to which all structures,
        evaluated during the search, is saved.

    logfile: file object or str
        If *logfile* is a string, a file with that name will be opened.
        Use '-' for stdout.

    kappa: float
        How much to weigh predicted uncertainty in the acquisition
        function.

    max_steps: int
        Number of search steps.

    Ninit: int
        Number of initial structures. If len(structures) <
        Ninit, the remaining structures are generated using the
        startgenerator (if supplied) or by rattling the supplied
        'structures'.

    max_relax_dist: float
        Max distance (in Angstrom) that an atom is allowed to
        move during surrogate relaxation.

    Ncandidates: int
        Number of new cancidate structures generated and
        surrogate-relaxed in each search iteration.

    population_size: int
        Maximum number of structures in the population.

    dualpoint: boolean
        Whether to use dualpoint evaluation or not.

    min_certainty: float
        Max predicted uncertainty allowed for structures to be
        considdered for evaluation. (in units of the maximum possible
        uncertainty.)

    position_constraint: OperationConstraint object
        Enforces constraints on the positions of the "free" atoms
        in the search. The constraint is enforces both during
        mutation/crossover operations and during surrogate-relaxation.

    position_constraint: OperationConstraint object
        Enforces constraints on the positions of the "free" atoms
        in the search. The constraint is enforces both during
        mutation/crossover operations and during surrogate-relaxation.

    relax_scheme: None or string
        Possible choices are:
            None (default) : Same model for exploration and exploitation
            'exploit' : Relaxation is potentially finished without uncertainty included,
                        if candidate is determined to be exploitation.

    restart: str
        Filename for restart file.
    """
    def __init__(self, structures=None,
                 calc=None,
                 gpr=None,
                 startgenerator=None,
                 candidate_generator=None,
                 kappa=2,
                 max_steps=200,
                 Ninit=10,
                 max_relax_dist=4,
                 Ncandidates=30,
                 population_size=10,
                 dualpoint=True,
                 min_certainty=0.7,
                 position_constraint=None,
                 improved_structure_convergence={'Fmax': 0.1, 'std_threshold': 0.1, 'subtract_noise': False},
                 filter_optimized_structures=False,
                 relax_scheme={},
                 population_scheme={},
                 N_use_pop=1,
                 apply_centering=False,
                 trajectory='structures.traj',
                 logfile='search.log',
                 restart='restart.pickl',
                 reference_structure=None,
                 verbose=False,
                 random_seed=None,
                 similarity_thr=0.999):
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.verbose = verbose

        # Initialize timing
        self.t_start = time()

        # Define parallel communication
        self.comm = world.Dup()  # Important to avoid mpi-problems from call to ase.parallel in BFGS
        self.master = self.comm.rank == 0

        if structures is None:
            assert startgenerator is not None
            self.structures = None
        else:
            if isinstance(structures, Atoms):
                self.structures = [structures]
            elif isinstance(structures, list):
                assert isinstance(structures[0], Atoms)
                self.structures = structures
            elif isinstance(structures, str):
                self.structures = read(structures, index=':')
        
        if isinstance(calc, Calculator):
            self.calc = calc
        elif isinstance(calc, str):
            fname = calc
            self.calc = backgroundCalculator(fname, self.comm)
        else:
            assert structures is not None
            calc = structures[0].get_calculator()
            assert calc is not None and not isinstance(calc, SinglePointCalculator)
            self.calc = calc

        if startgenerator is None:
            assert structures is not None
            self.startgenerator = None
        else:
            self.startgenerator = startgenerator

        # Determine atoms to optimize
        if startgenerator is not None:
            self.n_to_optimize = len(self.startgenerator.stoichiometry)
        else:
            self.n_to_optimize = len(self.structures[0])
            for constraint in self.structures[0].constraints:
                if isinstance(constraint, FixAtoms):
                    indices_fixed = constraint.get_indices()
                    self.n_to_optimize -= len(indices_fixed)
                    break
        
        # Set up candidate-generator if not supplied
        if candidate_generator is not None:
            self.candidate_generator = candidate_generator
        else:
            rattle = RattleMutation(self.n_to_optimize,
                                    Nrattle=3,
                                    rattle_range=4)
            self.candidate_generator = CandidateGenerator([1.0],[rattle])

        self.kappa = kappa
        self.max_steps = max_steps
        self.Ninit = Ninit
        self.max_relax_dist = max_relax_dist
        self.Ncandidates = Ncandidates
        self.dualpoint = dualpoint
        self.min_certainty = min_certainty
        self.position_constraint = position_constraint
        self.improved_structure_convergence = improved_structure_convergence
        self.filter_optimized_structures = filter_optimized_structures
        self.relax_scheme = relax_scheme
        self.population_scheme = population_scheme
        self.N_use_pop = N_use_pop
        self.apply_centering = apply_centering
        self.restart = restart
        self.reference_structure = reference_structure
        self.similarity_thr = similarity_thr
        
        self.current_path =  os.getcwd() #SAM: To print trajectory to current path
        # Add position-constraint to candidate-generator
        #self.candidate_generator.set_constraints(position_constraint)

        if isinstance(trajectory, str):
            self.trajectory = Trajectory(filename=trajectory, mode='a', master=self.master)
            if self.restart:
                self.traj_name = trajectory

        if not self.master:
            logfile = None
        elif isinstance(logfile, str):
            if logfile == "-":
                logfile = sys.stdout
            else:
                logfile = open(logfile, "a")
        self.logfile = logfile
        self.log_msg = ''

        # Save feature_gradients if force_training is used
        if 'Nforces' in self.relax_scheme:
            self.feature_gradients_save_folder = 'feature_gradients'
            if self.master:
                mkdir(self.feature_gradients_save_folder)
            gpr_kwargs = {'store_forces': True}
        else:
            gpr_kwargs = {}

        # Initialize or restart search
        self.population_method = self.population_scheme.get('method', 'clustering')
        if restart is None or not path.exists(restart):
            self.initialize()

            if gpr is not None:
                self.gpr = gpr
            else:
                self.gpr = GPR(template_structure=self.structures[0], **gpr_kwargs)
            
            # Initialize population
            if self.population_method == 'clustering':
                self.population = ClusteringPopulation(population_size=population_size,
                                                       **population_scheme)
            elif self.population_method == 'similarity':
                self.population = Population(population_size=population_size, gpr=self.gpr, similarity2equal=0.9995)
            elif self.population_method == 'LV':
                self.population = LVPopulation(population_size=population_size,
                                               **population_scheme)
        else:
            self.read()
            self.comm.barrier()

        if self.improved_structure_convergence:
            self.relax_dict = {}
            self.relax_Fmax_dict = {}
            self.Fmax_lim = self.improved_structure_convergence.get('Fmax', 0.1)
            self.std_threshold = self.improved_structure_convergence.get('std_threshold', 0.1)
            self.IC_subtract_noise = self.improved_structure_convergence.get('subtract_noise', False)

        if self.filter_optimized_structures:
            self.sufficiently_optimized_structures = []
            if self.improved_structure_convergence:
                self.Fmax_filter = self.Fmax_lim
            else:
                self.Fmax_filter = 0.1

        self.BFGS_step_lengths = []

    def get_kappa(self):
        """22/08/12, SAM: Method to get kappa as a function if it is specify in the input."""
        if self.kappa == "decay":
            if self.max_steps < 201:
                kappa = 1 + 0.02 * self.max_steps * math.exp(- self.steps ** 2 / (0.25 * self.max_steps ** 2))
            else:
                kappa = 1 + 4 * math.exp(- self.steps ** 2 / (0.25 * self.max_steps ** 2))
        else:
            kappa = self.kappa
        return kappa

    def initialize(self):
        self.get_initial_structures()
        self.steps = 0

    def get_initial_structures(self):
        """Method to prepare the initial structures for the search.
        
        The method makes sure that there are atleast self.Ninit
        initial structures.
        These structures are first of all the potentially supplied
        structures. If more structures are required, these are
        generated using self.startgenerator (if supplied), otherwise
        they are generated by heavily rattling the supplied structures.
        """
        
        # Collect potentially supplied structures.
        if self.structures is not None:
            for a in self.structures:
                a.info = {'origin': 'PreSupplied'}
        else:
            self.structures = []
        
        Nremaining = self.Ninit - len(self.structures)
        
        if Nremaining > 0 and self.startgenerator is None:
            # Initialize rattle-mutation for all atoms.
            rattle = RattleMutation(self.n_to_optimize,
                                    Nrattle=self.n_to_optimize,
                                    rattle_range=2)

        # Generation of remaining initial-structures (up to self.Ninit).
        for i in range(Nremaining):
            if self.startgenerator is not None:
                a = self.startgenerator.get_new_candidate()
            else:
                # Perform two times rattle of all atoms.
                a0 = self.structures[i % len(self.structures)]
                a = rattle.get_new_candidate([a])
                a = rattle.get_new_candidate([a])
            self.structures.append(a)
                
    def evaluate_initial_structures(self):
        """ Evaluate energies and forces of all initial structures
        (self.structures) that have not yet been evaluated.
        """
        structures_init = []
        for a in self.structures:
            a = self.evaluate(a)
            structures_init.append(a)

        self.save_structures(structures_init)
        self.population.add(structures_init)

    def run(self, max_steps=None, restart=None):
        """ Method to run the search.
        """
        if restart is not None:
            self.read(restart)

        if max_steps is not None:
            self.max_steps = max_steps

        if self.steps == 0:
            self.evaluate_initial_structures()

        while self.steps < self.max_steps:
            self.log_msg += (f"\n##### STEPS: {self.steps} #####\n")
            self.log_msg += (f"Runtime: {timedelta(seconds=time()-self.t_start)}\n\n")
            t0 = time()
            self.train_surrogate()
            t1 = time()
            self.update_population()
            t2 = time()
            unrelaxed_candidates = self.generate_new_candidates()
            t3 = time()
            relaxed_candidates = self.relax_candidates_with_surrogate(unrelaxed_candidates)
            t4 = time()
            kappa = self.get_kappa()
            self.log_msg += (f"kappa = {kappa} \n")
            a_add = []
                
            for _ in range(5):
                try:
                    anew = self.select_with_acquisition(relaxed_candidates, kappa)
                    #idx_relax = anew.info['key_value_pairs'].get('index_closest_min')
                    #if idx_relax is not None:
                    #    anew = self.make_relax_step_in_target_potential(idx_relax)
                    #else:
                    if self.check_similarity_with_all_runs(anew):
                        anew = self.evaluate(anew)
                        self.save_structures([anew])
                        a_add.append(anew)
                        
                        if self.dualpoint:
                        #if idx_relax is not None:
                            # Continue BFGS on from "anew".
                          #  adp = self.make_relax_step_in_target_potential(len(self.gpr.memory.energies)-1)
                        #else:
                            adp = self.get_dualpoint(anew)
                            adp = self.evaluate(adp)
                            self.save_structures([adp])
                        a_add.append(adp)                                                                        
                                            
                    ###### testing ######
                    #self.write_BFGS_stats(anew0, idx_relax)  # for testing only
                    #####################
                    break
                except Exception as err:
                    kappa /=2
                    if self.master:
                        #print(f'Exception :\n{err}', file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
            else:
                raise RuntimeError('Evaluation failed repeatedly - It might help to constrain the atomic positions during search.')
            self.comm.barrier()
            # Add evaluated structure(s) to GPR-memory
            #self.save_structures(a_add)

            # log timing
            self.log_msg += "Timing:\n"
            if self.verbose:
                self.log_msg += f"E_true:\n{array_to_string([a.info['key_value_pairs']['Epred'] for a in relaxed_candidates], unit='eV')}\n\n"
            self.log_msg += f"{'Training':12}{'Relax pop.':12}{'Make cands.':15}{'Relax cands.':16}{'Evaluate':12}\n"
            self.log_msg += f"{t1-t0:<12.2e}{t2-t1:<12.2e}{t3-t2:<15.2e}{t4-t3:<16.2e}{time()-t4:<12.2e}\n\n"

            # Print status on sufficiently optimized structures
            self.status_sufficiently_optimized_structures(a_add)

            # Add structure to population
            try:
                index_lowest = np.argmin([a.get_potential_energy() for a in a_add])
                self.population.add([a_add[index_lowest]])
            except:
                pass
            
            self.log_msg += (f"Prediction:\nenergy = {anew.info['key_value_pairs']['Epred']:.5f}eV,  energy_std = {anew.info['key_value_pairs']['Epred_std']:.5f}eV\n")
            self.log_msg += (f"E_true:\n{array_to_string([a.get_potential_energy() for a in a_add], unit='eV')}\n\n")
            self.log_msg += (f"Energy of population:\n{array_to_string([a.get_potential_energy() for a in self.population.pop], unit='eV')}\n")
            if self.verbose:
                self.log_msg += (f"Max force of ML-relaxed population:\n{array_to_string([(a.get_forces()**2).sum(axis=1).max()**0.5 for a in self.population.get_refined_pop()], unit='eV/A')}\n")
            
            if self.reference_structure is not None:
                self.print_reference_comparison(a_add)

            self.steps += 1
            self.log()

            # Save search state
            self.save_state()

    def print_reference_comparison(self, a_add):
        Eref = self.gpr.predict_energy(self.reference_structure)
        structures_compare = [self.reference_structure] + a_add
        E_compare = []
        Estd_compare = []
        for a in structures_compare:
            E, Estd = self.gpr.predict_energy(a, eval_std=True)
            E_compare.append(E)
            Estd_compare.append(Estd)
        self.log_msg += (f"\nReference E= {E_compare}\n")
        self.log_msg += (f"Reference Estd= {Estd_compare}\n\n")


    def get_dualpoint(self, a, lmax=0.10, Fmax_flat=5):
        """Returns dual-point structure, i.e. the original structure
        perturbed slightly along the forces.
        
        lmax: The atom with the largest force will be displaced by
        this distance
        
        Fmax_flat: maximum atomic displacement. is increased linearely
        with force until Fmax = Fmax_flat, after which it remains
        constant as lmax.
        """
        F = a.get_forces()
        a_dp = a.copy()

        # Calculate and set new positions
        Fmax = np.sqrt((F**2).sum(axis=1).max())
        pos_displace = lmax * F*min(1/Fmax_flat, 1/Fmax)
        pos_dp = a.positions + pos_displace
        a_dp.set_positions(pos_dp)
        return a_dp

    def generate_new_candidates(self):
        """Method to generate a self.Ncandidates new candidates
        by applying the operations defined in self.candidate_generator
        to the structures currently in the population.
        The tasks are parrlelized over all avaliable cores.
        """
        Njobs = self.Ncandidates
        task_split = split(Njobs, self.comm.size)
        def func1():
            return [self.generate_candidate() for i in task_split[self.comm.rank]]
        candidates = parallel_function_eval(self.comm, func1)
        return candidates

    def relax_candidates_with_surrogate(self, candidates):
        """ Method to relax new candidates using the
        surrogate-model.
        The tasks are parrlelized over all avaliable cores.
        """
        Njobs = self.Ncandidates
        task_split = split(Njobs, self.comm.size)
        def func2():
            return [self.surrogate_relaxation(candidates[i], Fmax=0.1, steps=200, kappa=self.get_kappa()) #2022/10/29, SAM: to get call kappa
                    for i in task_split[self.comm.rank]]
        relaxed_candidates = parallel_function_eval(self.comm, func2)
        relaxed_candidates = self.certainty_filter(relaxed_candidates)
        if self.add_population_to_candidates():
            relaxed_candidates = self.population.get_refined_pop() + relaxed_candidates
        if self.filter_optimized_structures:
            relaxed_candidates = self.sufficiently_optimized_filter(relaxed_candidates)
        
        return relaxed_candidates

    def generate_candidate(self):
        """ Method to generate new candidate.
        """
        parents = self.population.get_structure_pair()
        a_mutated = self.candidate_generator.get_new_candidate(parents)
        return a_mutated

    def surrogate_relaxation(self, a, Fmax=0.1, steps=200, kappa=None):
        """ Method to carry out relaxations of new candidates in the
        surrogate potential.
        """
        t0 = time()
        kappa = self.relax_scheme.get('kappa', kappa)
        relax_method = self.relax_scheme.get('method', 'normal')
        std_threshold_relax = self.relax_scheme.get('std_threshold_relax')
        if relax_method == 'normal':  # when dict is empty
            calc = self.gpr.get_calculator(kappa)
            a_relaxed, termination_cause = relax(a, calc, Fmax=Fmax, steps_max=steps,
                                                max_relax_dist=self.max_relax_dist,
                                                position_constraint=self.position_constraint,
                                                Epred_std_max=std_threshold_relax)

            if termination_cause == 'predicted_uncertainty':
                kappa_final = self.relax_scheme.get('kappa_final')
                if kappa_final is not None:
                    calc = self.gpr.get_calculator(kappa_final)
                    a_relaxed, termination_cause = relax(a_relaxed, calc, Fmax=Fmax, steps_max=30,
                                                max_relax_dist=self.max_relax_dist,
                                                position_constraint=self.position_constraint)

        elif relax_method == 'forceModel':
            kappa = self.relax_scheme.get('kappa', kappa)
            calc_explore = self.gpr.get_calculator(kappa)
            a_relaxed, termination_cause = relax(a, calc_explore, Fmax=Fmax, steps_max=steps,
                                                max_relax_dist=self.max_relax_dist,
                                                position_constraint=self.position_constraint,
                                                Epred_std_max=std_threshold_relax)
            if termination_cause == 'predicted_uncertainty':
                # Make model for relaxation in exploitation region,
                # based on choosen relax_scheme.

                # Is new surrogate model required
                if ('noise' in self.relax_scheme or 
                    'Nmax_data' in self.relax_scheme or 
                    'Nforces' in self.relax_scheme):

                    gpr = gpr0.get_local_model(a,
                            Nmax=relax_scheme.get('Nmax_data'),
                            Nmax_force=relax_scheme.get('Nmax_force', 20),
                            Nforces=relax_scheme.get('Nforces'),
                            save_folder='feature_gradients')
                    gpr.kernel.dynamic_noise = False
                    if 'noise' in relax_scheme:
                        noise = relax_scheme['noise']
                        gpr.kernel.noise = noise
                        gpr.kernel.set_theta_bounds(noise_bnd=(noise,noise))

                    t1 = time()
                    # Optimize hyperparameters without forces
                    gpr.optimize_hyperparameters()
                    t2 = time()
                    # Train with forces (if desired)
                    gpr.train()
                    t3 = time()
                    calc_exploit = gpr.get_calculator(kappa=None)
                else:
                    t1 = time()
                    t2 = time()
                    t3 = time()
                    calc_exploit = self.gpr.get_calculator(kappa=None)

                # Relax in exploitation-model.
                a_relaxed, termination_cause = relax(a_relaxed, calc_exploit, Fmax=0.01, steps_max=steps,
                                                    max_relax_dist=1,
                                                    position_constraint=self.position_constraint)
                print(f'step: {self.steps}, rank: {self.comm.rank}, initial relax={t1-t0}, hyper_opt={t2-t1}, train={t3-t2} relax={time()-t3}')

        # Evaluate uncertainty
        E, Estd = self.gpr.predict_energy(a_relaxed, eval_std=True)


        if self.improved_structure_convergence:
            if self.IC_subtract_noise:
                std_threshold = self.std_threshold - self.gpr.min_uncertainty
            else:
                std_threshold = self.std_threshold
            a_relaxed.info['key_value_pairs'].pop('index_closest_min', None)
            if Estd < std_threshold:
                # Make BFGS relaxation step from best structure in this minimum
                index_min = get_minimum_point_by_energy_decent(self.gpr.descriptor.get_feature(a_relaxed),
                                                                self.gpr.memory.features,
                                                                self.gpr.memory.energies,
                                                                Epred=E)
                if index_min is not None:
                    a_relaxed.info['key_value_pairs']['index_closest_min'] = index_min

        # Save prediction in info-dict
        a_relaxed.info['key_value_pairs']['Epred'] = E
        a_relaxed.info['key_value_pairs']['Epred_std'] = Estd
        a_relaxed.info['key_value_pairs']['kappa'] = self.get_kappa()

        return a_relaxed
        
    def certainty_filter(self, structures):
        """ Method to filter away the most uncertain surrogate-relaxed
        candidates, which might otherewise get picked for first-principles
        evaluation, based on the very high uncertainty alone.
        """
        certainty = np.array([a.info['key_value_pairs']['Epred_std']
                              for a in structures]) / np.sqrt(self.gpr.K0)
        min_certainty = self.min_certainty
        for _ in range(5):
            filt = certainty < min_certainty
            if np.sum(filt.astype(int)) > 0:
                structures = [structures[i] for i in range(len(filt)) if filt[i]]
                break
            else:
                min_certainty = min_certainty + (1-min_certainty)/2
        return structures
    
    def sufficiently_optimized_filter(self, structures):
        """ Method to filter away new candidates that are ' roughly identical'
        to structures that have already been sufficiently optimized in the search.
        """
        structures_filtered = []
        for a in structures:
            use_structure = True
            if self.improved_structure_convergence:
                idx_relax = a.info['key_value_pairs'].get('index_closest_min')
                if idx_relax is not None:
                    if idx_relax in self.relax_Fmax_dict:
                        Fmax = self.relax_Fmax_dict[idx_relax][0]
                        if Fmax < self.Fmax_filter:
                            use_structure = False
            #else:
            #    Fmax = (a.get_forces()**2).sum(axis=1).max()**0.5
            #    if Fmax < self.Fmax_filter:
            #        use_structure = False
            if use_structure:
                structures_filtered.append(a)
        return structures_filtered     

    def update_population(self):
        """ Method to update the population with the new first-principles
        evaluated structures.
        """
        self.population.update(self.gpr.memory.structures,
                               self.gpr.memory.features)

        if self.population_method == 'clustering':
            Fmax, steps = 0.05, 50
        else:
            Fmax, steps = 0.01, 200

        cond_relax_pop = self.population_method != 'clustering' or self.add_population_to_candidates()

        if cond_relax_pop:
            Njobs = len(self.population.pop)
            task_split = split(Njobs, self.comm.size)
            func = lambda: [self.surrogate_relaxation(self.population.pop[i],
                                                    Fmax=Fmax, steps=steps, kappa=None)
                            for i in task_split[self.comm.rank]]
            self.population.pop_MLrelaxed = parallel_function_eval(self.comm, func)
            for a in self.population.pop_MLrelaxed:
                a.info['key_value_pairs']['origin'] = 'Population'
                del a.calc

    def add_population_to_candidates(self):
        if self.N_use_pop is not None:
            return self.steps % self.N_use_pop == 0
        else:
            return False

    def train_surrogate(self):
        """ Method to train the surrogate model.
        The method only performs hyperparameter optimization every
        ten training instance, as carrying out the hyperparameter
        optimization is significantly more expensive than the basic
        training.
        """
        # Train
        if self.steps < 50 or (self.steps % 10) == 0:
            self.gpr.optimize_hyperparameters(comm=self.comm)
            self.log_msg += (f"lml: {self.gpr.lml}\n")
            self.log_msg += (f"kernel optimized:\nTheta = {[f'{x:.2e}' for x in np.exp(self.gpr.kernel.theta)]}\n\n")
        else:
            self.gpr.train()
            self.log_msg += (f"kernel fixed:\nTheta = {[f'{x:.2e}' for x in np.exp(self.gpr.kernel.theta)]}\n\n")

    def select_with_acquisition(self, structures, kappa):
        """ Method to select single most "promizing" candidate 
        for first-principles evaluation according to the acquisition
        function min(E-kappa*std(E)).
        """
        Epred = np.array([a.info['key_value_pairs']['Epred']
                          for a in structures])
        Epred_std = np.array([a.info['key_value_pairs']['Epred_std']
                              for a in structures])
        acquisition = Epred - kappa*Epred_std
        index_select = np.argmin(acquisition)
        return structures[index_select]

    def status_sufficiently_optimized_structures(self, a_add):
        if self.filter_optimized_structures:
            for ai in a_add:
                Fmax = np.sqrt((ai.get_forces()**2).sum(axis=1).max())
                if Fmax < self.Fmax_filter:
                    self.sufficiently_optimized_structures.append(ai)
            self.log_msg += (f"relax_Fmax_dict:\n{self.relax_Fmax_dict}\n")
            if len(self.sufficiently_optimized_structures) > 0:
                self.log_msg += (f"Energy sufficiently optimized structures:\n{array_to_string([a.get_potential_energy() for a in self.sufficiently_optimized_structures], unit='eV')}\n")

    def evaluate(self, a):
        """ Method to evaluate the energy and forces of the selacted
        candidate.
        """
        if self.apply_centering:
            a.center()
        a = self.comm.bcast(a, root=0)
        a.wrap()

        #if isinstance(self.calc, Dftb):
        if isinstance(self.calc, FileIOCalculator):
            if self.master:
                try:
                    a.set_calculator(self.calc)
                    E = a.get_potential_energy()
                    F = a.get_forces()
                    results = {'energy': E, 'forces': F}
                    calc_sp = SinglePointCalculator(a, **results)
                    a.set_calculator(calc_sp)
                    success = True
                except:
                    success = False
            else:
                success = None
            success = self.comm.bcast(success, root=0)
            if success == False:
                raise RuntimeError('DFTB evaluation failed')
            a = self.comm.bcast(a, root=0)
        else:
            a.set_calculator(self.calc)
            E = a.get_potential_energy()
            F = a.get_forces()
            results = {'energy': E, 'forces': F}
            calc_sp = SinglePointCalculator(a, **results)
            a.set_calculator(calc_sp)

        self.write(a)

        return a
    
    def make_relax_step_in_target_potential(self, idx_relax):
        """ Method to evaluate the energy and forces of the selacted
        candidate.
        """
        def take_relax_step(a, idx_relax):
            if idx_relax in self.relax_dict:
                dyn = self.relax_dict[idx_relax]
            else:
                a.set_calculator(self.calc)
                dyn = BFGS(a)
                self.relax_dict[idx_relax] = dyn
            a0 = dyn.atoms.copy()
            self.relax_dict[len(self.gpr.memory.energies)] = dyn
            try:
                # In newer ASE versions Optimizer object, the run loop iterates over
                # self.nsteps, and does not reset it each time .run() is called.
                steps = dyn.nsteps+1
            except:
                # In older ASE versions, the Optimizer.run() method iterates over the
                # non-atribute variable step. step is set =0 each time .run() is called.
                steps = 1
            dyn.run(fmax = self.Fmax_lim, steps=steps)
            a = dyn.atoms
            E = a.get_potential_energy()
            F = a.get_forces()
            results = {'energy': E, 'forces': F}
            anew = a.copy()
            calc_sp = SinglePointCalculator(anew, **results)
            anew.set_calculator(calc_sp)
            return anew, a0
            
        #a = self.comm.bcast(a, root=0)
        a = self.gpr.memory.structures[idx_relax].copy()

        ### Testing BFGS-decent ###
        a0 = a.copy()
        ###########################


        #if isinstance(self.calc, Dftb):
        if isinstance(self.calc, FileIOCalculator):
            if self.master:
                try:
                    a, a0 = take_relax_step(a, idx_relax)
                    success = True
                except Exception as err:
                    print(f'Exception(dftb) :\n{err}', file=sys.stderr)
                    success = False
            else:
                success = None
            success = self.comm.bcast(success, root=0)
            if success == False:
                raise RuntimeError('DFTB evaluation failed')
            a = self.comm.bcast(a, root=0)
            a0 = self.comm.bcast(a0, root=0)
        else:
            a, a0 = take_relax_step(a, idx_relax)

        ### Testing BFGS-decent ###
        self.BFGS_step_lengths.append(self.get_featurespace_distance(a0,a))
        ###########################

        Fmax = (a.get_forces()**2).sum(axis=1).max()**0.5
        # Initialize new Fmax trajectory
        if idx_relax not in self.relax_Fmax_dict:
            self.relax_Fmax_dict[idx_relax] = [Fmax]
        # Link indices from same relax path
        idx_new = len(self.gpr.memory.energies)
        self.relax_Fmax_dict[idx_new] = self.relax_Fmax_dict[idx_relax]
        self.relax_Fmax_dict[idx_new][0] = Fmax

        a.info['key_value_pairs']['Epred'] = np.nan
        a.info['key_value_pairs']['Epred_std'] = np.nan
        a.info['key_value_pairs']['kappa'] = np.nan

        self.write(a)

        return a

    def save_structures(self, atoms_list_save):
        if 'Nforces' in self.relax_scheme:
            self.gpr.memory.save_data(atoms_list_save, save_folder=self.feature_gradients_save_folder)
        else:
            self.gpr.memory.save_data(atoms_list_save)

    def write(self, a):
        """ Method for writing new evaluated structures to file.
        """
        if self.trajectory is not None:
            self.trajectory.write(a)

    def dump(self, data):
        """ Method to save restart-file used if the search is
        restarted from some point in the search. 
        """
        if self.comm.rank == 0 and self.restart is not None:
            pickle.dump(data, open(self.restart, "wb"), protocol=2)

    def save_state(self):
        """ Saves the current state of the search, so the search can
        be continued after having finished or stoped prematurely.
        """
        #def func():
        #    return np.random.get_state()
        random_states_all_processes = parallel_function_eval(self.comm, np.random.get_state)
        self.dump((self.steps,
                   self.population,
                   self.gpr.get_state(),
                   random_states_all_processes,
                   self.traj_name))

    def read(self, restart=None):
        """ Method to restart a search from the restart-file and the
        trajectory-file containing all structures evaluated so far.
        """

        if restart is None:
            restart = self.restart
        self.steps, self.population, gpr_state, random_states_all_processes, structure_file = pickle.load(open(restart, "rb"))
        np.random.set_state(random_states_all_processes[ self.comm.rank % len(random_states_all_processes) ])
        training_structures = read(structure_file, index=':')

        # Salvage GPR model
        self.gpr = GPR(descriptor=gpr_state[0],
                       kernel=gpr_state[1],
                       prior=gpr_state[2],
                       n_restarts_optimizer=gpr_state[3],
                       scale_reg= gpr_state[4],
                       template_structure=gpr_state[5])
        self.save_structures(training_structures)
        if self.population_method != 'clustering':
            self.population.gpr = self.gpr

    def log(self):
        if self.logfile is not None:
            if self.steps == 0:
                msg = "GOFEE"
                self.logfile.write(msg)

            self.logfile.write(self.log_msg)
            self.logfile.flush()
        self.log_msg = ''

    def write_BFGS_stats(self, a, idx_relax):
        f"\nidx_relax 3: {idx_relax}\n"
        ### Just for testing ###
        if idx_relax is not None:
            a_closest, d2closest = self.get_nearest_structure(a)
            anew_decent = self.gpr.memory.structures[idx_relax]
            mean_BFGS_step = np.mean(self.BFGS_step_lengths)
            a.info['key_value_pairs']['BFGS_scale'] = mean_BFGS_step
            a.info['key_value_pairs']['d2closest'] = d2closest
            write(f'BFGS{self.steps}.traj', [a, a_closest, anew_decent])
            self.log_msg += f"\nBFGS_data {mean_BFGS_step, d2closest}\n"
            self.log_msg += f"BFGS_data {self.BFGS_step_lengths}\n"

    def get_featurespace_distance(self, a1, a2):
        ### Just for testing ###
        f1 = self.gpr.descriptor.get_feature(a1)
        f2 = self.gpr.descriptor.get_feature(a2)
        d = euclidean(f1,f2)
        return d

    def get_nearest_structure(self, a):
        ### Just for testing ###
        f = self.gpr.descriptor.get_feature(a)
        d = cdist(f.reshape(1,-1), self.gpr.memory.features, metric='euclidean').reshape(-1)
        index_closest = np.argmin(d)
        dmin = np.min(d)
        a_closest = self.gpr.memory.structures[index_closest]
        return a_closest, dmin

    def check_similarity_with_all_runs(self,a):
        #### 2022/11/18: Read all trajectory
        
        path = '../'
        list_directory = os.listdir(path)
        
        f1 = self.gpr.descriptor.get_feature(a)
        K0 = self.gpr.kernel.kernel_value(f1,f1)
        
        for i in list_directory:
            if i.find('run') != -1: # and self.steps > 10:
                    #try:
                    os.chdir(path)
                    os.chdir(i)
                    
                    try:
                        structures = read('structures.traj', index=':')
                        
                        os.chdir(self.current_path)
                        
                        for j in range(len(structures)):                        
                            f2 = self.gpr.descriptor.get_feature(structures[j])                        
                            similarity = self.gpr.kernel.kernel_value(f1,f2) / K0
                            if similarity > self.similarity_thr:
                                self.log_msg += f"Similarity= {similarity}\n"
                                self.log_msg += f"Structure of this step is taken from {i}.\n\n"
                            
                                #self.log_msg += "This is the newest version\n"
                                #write(f'{self.current_path}/structure_found{self.steps}.traj', a)
                                #write(f'{self.current_path}/structure_similar{self.steps}.traj', structures[j])
                            
                                #nearest = []
                                #nearest.append(structures[j-1])
                                #nearest.append(structures[j])
                                #nearest.append(structures[j+1])
                            
                                if j % 2 == 0:                      
                                    #self.save_structures([structures[j]])
                                    self.gpr.memory.save_data([structures[j]])
                                    self.population.add([structures[j]])
                                    self.write(structures[j])
                                          
                                    #self.save_structures([structures[j+1]])
                                    self.gpr.memory.save_data([structures[j+1]])
                                    self.population.add([structures[j+1]])
                                    self.write(structures[j+1])                                                            
                            
                                else:
                                    #self.save_structures([structures[j]])
                                    self.gpr.memory.save_data([structures[j]])
                                    self.population.add([structures[j]])
                                    self.write(structures[j])
                                
                                    #self.save_structures([structures[j-1]])
                                    self.gpr.memory.save_data([structures[j-1]])
                                    self.population.add([structures[j-1]])
                                    self.write(structures[j-1])
                            
                            #self.save_structures([structures[j]])
                            
                            #self.save_structures([structures[j+1]])
                            
                            #index_lowest = np.argmin([a.get_potential_energy() for a in nearest])
                            #self.population.add([nearest[index_lowest]])
                            #self.population.add([structures[j]])
                            
                            #self.write(nearest[index_lowest])
                            
                                return False
                    except:
                        pass         
            #return True
        #self.log_msg += f"Similarity= {similarity}\n"
        return True                            
                            
                            
        
        
        
        
