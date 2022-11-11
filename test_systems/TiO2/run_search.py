import numpy as np

from ase.calculators.dftb import Dftb
from ase.io import read

from candidate_operations.candidate_generation import CandidateGenerator, StartGenerator, OperationConstraint
from candidate_operations.basic_mutations import RattleMutation, RattleMutation2, PermutationMutation
from gofee import GOFEE

import sys

### Set up StartGenerator and mutations ###
# read slab
slab = read('slab.traj', index='0')

# Stoichiometry of atoms to be placed
stoichiometry = 5*[22]+10*[8]

# Box in which to place atoms
c = slab.get_cell()
c[2,2] = 25
slab.set_cell(c)
v = np.copy(c)
v[2,2] = 2.3
p0 = np.array((0.0,0.,8.))
box = [p0, v]

# initialize startgenerator
sg = StartGenerator(slab, stoichiometry, box)

# initialize rattle mutation
n_to_optimize = len(stoichiometry)
candidate_generator = CandidateGenerator([0.2, 0.2, 0.6],
                                        [sg,
                                         PermutationMutation(n_to_optimize, Npermute=2),
                                         RattleMutation(n_to_optimize, Nrattle=3, rattle_range=4)])

### Define calculator ###
calc = Dftb(label='TiO2_surface',
            Hamiltonian_SCC='No',
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_Ti='"d"',
            Hamiltonian_MaxAngularMomentum_O='"p"',
            Hamiltonian_Charge='0.000000',
            Hamiltonian_Filling ='Fermi {',
            Hamiltonian_Filling_empty= 'Temperature [Kelvin] = 0.000000',
            kpts=(2,1,1))

### Initialize and run search ###
search = GOFEE(calc=calc,
               startgenerator=sg,
               candidate_generator=candidate_generator,
               max_steps=200,
               max_relax_dist=4,
               population_size=5,
               kappa=1,
               dualpoint=True,
               restart='restart')

search.run()