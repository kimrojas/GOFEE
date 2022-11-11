import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms

from gofee.candidates import CandidateGenerator, StartGenerator
from gofee.candidates import RattleMutation, RattleMutation1, RattleMutation2, RattleMutationLocal, PermutationMutation

from time import time

np.random.seed(0)

a0 = read('Au5Al5_list.traj', index='9')
indices_fixed = []
for c in a0.constraints:
    if isinstance(c, FixAtoms):
        indices_fixed = c.get_indices()

n_to_optimize = a0.get_number_of_atoms() - len(indices_fixed)
print(n_to_optimize)

rattle = RattleMutation1(n_to_optimize, Nrattle=4, rattle_range=5, force_all_bonds_valid=False)

structures = []
for i in range(300):
    print(i)
    a = rattle1.get_new_candidate([a0])
    structures += [a0.copy(), a]

write('mutated.traj', structures)