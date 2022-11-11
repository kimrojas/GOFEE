import numpy as np

from ase.calculators.dftb import Dftb

from surrogate.gpr import GPR

from ase.io import read, write
from candidate_operations.candidate_generation import CandidateGenerator, StartGenerator, OperationConstraint
from candidate_operations.basic_mutations import RattleMutation, RattleMutation2, PermutationMutation

import sys

from gofee import GOFEE

### Set up StartGenerator and mutations ###
# read slab
slab = read('slab.traj', index='0')

# Stoichiometry of atoms to be placed
stoichiometry = 16*[14]

# Box in which to place atoms
c = slab.get_cell()
v = np.copy(c)
v[2,2] = 3.0
p0 = np.array([0,0,9.5])
box = [p0, v]

# initialize startgenerator
sg = StartGenerator(slab, stoichiometry, box)

# initialize rattle mutation
n_to_optimize = len(stoichiometry)

# Add position constraint to mutations
z_constraint = OperationConstraint(zlim=[8.5, 15.0])

candidate_generator = CandidateGenerator([0.2, 0.8],
                                         [sg,
                                          RattleMutation2(n_to_optimize, Nrattle=3,
                                                          constraints=z_constraint)])

### Define calculator ###
calc = Dftb(label='Si_surface',
            Hamiltonian_SCC='No',
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_Si='"d"',
            Hamiltonian_MaxAngularMomentum_H='"s"',
            Hamiltonian_Charge='0.000000',
            Hamiltonian_Filling ='Fermi {',
            Hamiltonian_Filling_empty= 'Temperature [Kelvin] = 0.000000',
            kpts=(2,2,1))

### Initialize and run search ###
search = GOFEE(structures=None,
               calc=calc,
               gpr=None,
               startgenerator=sg,
               candidate_generator=candidate_generator,
               max_steps=300,
               dmax_cov=1.5,
               population_size=10,
               kappa=2,
               dualpoint=True,
               restart='restart')

search.run()
