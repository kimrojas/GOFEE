import numpy as np

from ase.calculators.dftb import Dftb
from ase.io import read

from gofee.candidates import CandidateGenerator, StartGenerator
from gofee.candidates import RattleMutation, PermutationMutation
from gofee import GOFEE

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

### Set up StartGenerator and mutations ###
# read slab
slab = read('TiO2_slab.traj', index='0')

# Stoichiometry of atoms to be placed
stoichiometry = 5*[22]+10*[8]

# Box in which to place atoms
v = slab.get_cell()
v[2,2] = 2.5
p0 = np.array((0.0,0.,8.))
box = [p0, v]

# initialize startgenerator
sg = StartGenerator(slab, stoichiometry, box)

# initialize rattle and permutation mutations
n_to_optimize = len(stoichiometry)
permutation = PermutationMutation(n_to_optimize, Npermute=2)
rattle = RattleMutation(n_to_optimize, Nrattle=3, rattle_range=4)

candidate_generator = CandidateGenerator([0.2, 0.2, 0.6],
                                         [sg, permutation, rattle])

### Initialize and run search ###
search = GOFEE(calc=calc,
               startgenerator=sg,
               candidate_generator=candidate_generator,
               max_steps=100,
               population_size=5)
search.run()