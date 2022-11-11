import numpy as np
from abc import ABC, abstractmethod
from ase.data import covalent_radii
from ase.geometry import get_distances

from ase.visualize import view

from gofee.candidates.candidate_generation import OffspringOperation


def pos_add_sphere(rattle_strength):
    """Help function for rattling within a sphere
    """
    r = rattle_strength * np.random.rand()**(1/3)
    theta = np.random.uniform(low=0, high=2*np.pi)
    phi = np.random.uniform(low=0, high=np.pi)
    pos_add = r * np.array([np.cos(theta)*np.sin(phi),
                            np.sin(theta)*np.sin(phi),
                            np.cos(phi)])
    return pos_add

def pos_add_sphere_shell(rmin, rmax):
    """Help function for rattling atoms within a spherical
    shell.
    """
    r = np.random.uniform(rmin**3, rmax**3)**(1/3)
    theta = np.random.uniform(low=0, high=2*np.pi)
    phi = np.random.uniform(low=0, high=np.pi)
    pos_add = r * np.array([np.cos(theta)*np.sin(phi),
                            np.sin(theta)*np.sin(phi),
                            np.cos(phi)])
    return pos_add

class RattleMutation(OffspringOperation):
    """Class to perform rattle mutations on structures.

    Rattles a number of randomly selected atoms within a sphere 
    of radius 'rattle_range' of their original positions.
    - Moves one involved atom at a time.
    
    Parameters:

    n_top: int
        The number of atoms to optimize. Specifically the
        atoms with indices [-n_top:] are optimized.
    
    Nrattle: float
        The average number of atoms to rattle.

    rattle_range: float
        The maximum distance within witch to rattle the
        atoms. Atoms are rattled uniformly within a sphere of this
        radius.

    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.

    description: str
        Name of the operation, which will be saved in
        info-dict of structures, on which the operation is applied.    
    """
    def __init__(self, n_top, Nrattle=3, rattle_range=3,
                 description='RattleMutation', *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs)
        self.description = description
        self.n_top = n_top
        self.probability = Nrattle/n_top
        self.rattle_range = rattle_range

    def operation(self, parents):
        a = parents[0]
        a = self.rattle(a)
        return a

    def rattle(self, atoms):
        """ Rattles atoms one at a time within a sphere of radius
        self.rattle_range.
        """
        a = atoms.copy()
        Natoms = len(a)
        Nslab = Natoms - self.n_top

        # Randomly select indices of atoms to permute - in random order.
        indices_to_rattle = np.arange(Nslab,Natoms)[np.random.rand(self.n_top)
                                                     < self.probability]
        indices_to_rattle = np.random.permutation(indices_to_rattle)
        if len(indices_to_rattle) == 0:
            indices_to_rattle = [np.random.randint(Nslab,Natoms)]

        # Perform rattle operations in sequence.
        for i in indices_to_rattle:
            posi_0 = np.copy(a.positions[i])
            for _ in range(200):
                # Perform rattle
                pos_add = pos_add_sphere(self.rattle_range)
                a.positions[i] += pos_add
                
                # Check position constraint
                obey_constraint = self.check_constraints(a.positions[i])
                # Check if rattle was valid
                valid_bondlengths = self.check_bondlengths(a, indices=[i])

                valid_operation = valid_bondlengths and obey_constraint
                if not valid_operation:
                    a.positions[i] = posi_0
                else:
                    break
        if valid_operation:
            return a
        else:
            # If mutation is not successfull in supplied number
            # of trials, return initial structure.
            return None

class RattleMutation2(OffspringOperation):
    """Class to perform rattle mutations on structures.

    Rattles a number of randomly selected atom to the visinity
    of other candomly selected atoms random atom.
    - Moves one involved atom at a time.
    
    Parameters:

    n_top: int
        The number of atoms to optimize. Specifically the
        atoms with indices [-n_top:] are optimized.
    
    Nrattle: float
        The average number of atoms to rattle.

    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.

    description: str
        Name of the operation, which will be saved in
        info-dict of structures, on which the operation is applied.    
    """
    def __init__(self, n_top, Nrattle=3, description='RattleMutation',
                 cluster=False, *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs)
        self.description = description
        self.n_top = n_top
        self.probability = Nrattle/n_top
        self.cluster = cluster

    def operation(self, parents):
        a = parents[0]
        a = self.rattle(a)
        return a

    def rattle(self, atoms):
        """ Repeatedly rattles a random atom to the visinity of another
        random atom.
        """  
        a = atoms.copy()
        Natoms = len(a)
        Nslab = Natoms - self.n_top
        num = a.numbers

        # Randomly select indices of atoms to permute - in random order.
        indices_to_rattle = np.arange(Nslab,Natoms)[np.random.rand(self.n_top)
                                                     < self.probability]
        indices_to_rattle = np.random.permutation(indices_to_rattle)
        if len(indices_to_rattle) == 0:
            indices_to_rattle = [np.random.randint(Nslab,Natoms)]

        # Perform rattle operations in sequence.
        for i in indices_to_rattle:
            posi_0 = np.copy(a.positions[i])
            for _ in range(10):
                j = np.random.randint(Nslab,Natoms)

                # Perform rattle
                covalent_dist_ij = covalent_radii[num[i]] + covalent_radii[num[j]]
                rmin = self.blmin * covalent_dist_ij
                rmax = self.blmax * covalent_dist_ij
                pos_add = pos_add_sphere_shell(rmin, rmax)
                a.positions[i] = np.copy(a.positions[j]) + pos_add

                # Check position constraint 
                obey_constraint = self.check_constraints(a.positions[i])
                # Check if rattle was valid
                valid_bondlengths = self.check_bondlengths(a, indices=[i])
                if self.cluster:
                    not_too_close_all = self.check_bondlengths(a, indices=[i], check_isolated=False)
                    if self.n_top > 1:
                        #not_isolated_cluster = self.check_bondlengths(a, indices=np.arange(Nslab,Natoms), indices_placed=np.arange(Nslab,Natoms), check_too_close=False)
                        not_isolated_cluster = self.check_bondlengths(a, indices=[i], indices_placed=np.arange(Nslab,Natoms), check_too_close=False)
                    else:
                        not_isolated_cluster = True
                    valid_bondlengths = not_too_close_all and not_isolated_cluster
                else:
                    valid_bondlengths = self.check_bondlengths(a, indices=[i])
                
                valid_operation = valid_bondlengths and obey_constraint
                if not valid_operation:
                    a.positions[i] = posi_0
                else:
                    break
        if valid_operation:
            return a
        else:
            # If mutation is not successfull in supplied number
            # of trials, return initial structure.
            return None

class RattleMutationLocal(OffspringOperation):
    """Class to perform a local rattle mutations on structures.

    Randomly selects a central atom for the mutation, and rattles
    the closest "Nrattle" atoms, including the central one.
    Each atom is rattleed uniformly within a sphere of radius
    "rattle_range".
    - Removes all involved atoms before placing them one at a time.
    
    Parameters:

    n_top: int
        The number of atoms to optimize. Specifically the
        atoms with indices [-n_top:] are optimized.
    
    Nrattle: float
        The average number of atoms to rattle.

    rattle_range: float
        The maximum distance within witch to rattle the
        atoms. Atoms are rattled uniformly within a sphere of this
        radius.

    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.

    description: str
        Name of the operation, which will be saved in
        info-dict of structures, on which the operation is applied.    
    """
    def __init__(self, n_top, Nrattle=5, rattle_range=4,
                 description='RattleMutation', cluster=False, *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs)
        self.description = description
        self.n_top = n_top
        self.Nrattle = Nrattle
        self.rattle_range = rattle_range
        self.cluster = cluster

    def operation(self, parents):
        a = parents[0]
        a = self.rattle(a)
        return a

    def rattle(self, atoms):
        """ Performs the local rattle mutation.
        """
        a = atoms.copy()
        Natoms = len(a)
        Nslab = Natoms - self.n_top
        indices_slab = list(np.arange(Nslab))

        index_center = np.random.randint(Nslab, Natoms)
        dists2center = get_distances([atoms.positions[index_center]],
                                        atoms.positions[Nslab:Natoms],
                                        cell=atoms.get_cell(),
                                        pbc=atoms.get_pbc())[1].reshape(-1)
        indices_to_rattle = np.argsort(dists2center)[:self.Nrattle] + Nslab
        indices_to_rattle = np.random.permutation(indices_to_rattle)
        
        indices_placed = list(np.setdiff1d(np.arange(Nslab,Natoms), indices_to_rattle))
        # Perform rattle operations in sequence.
        for i in indices_to_rattle:
            posi_0 = np.copy(a.positions[i])
            for _ in range(100):
                # Perform rattle
                pos_add = pos_add_sphere(self.rattle_range)
                a.positions[i] += pos_add
                
                # Check position constraint
                obey_constraint = self.check_constraints(a.positions[i])
                # Check if rattle was valid
                if Nslab == 0 and len(indices_placed) == 0:
                    valid_bondlengths = True
                    # valid_bondlengths = self.check_bondlengths(a, indices=[i], indices_placed=indices_placed, check_isolated=False)
                else:
                    if self.cluster:
                        not_too_close_all = self.check_bondlengths(a, indices=[i], indices_placed=indices_slab+indices_placed, check_isolated=False)
                        if len(indices_placed) > 1:
                            not_isolated_cluster = self.check_bondlengths(a, indices=[i], indices_placed=indices_placed, check_too_close=False)
                        else:
                            not_isolated_cluster = True
                        valid_bondlengths = not_too_close_all and not_isolated_cluster
                    else:
                        valid_bondlengths = self.check_bondlengths(a, indices=[i], indices_placed=indices_slab+indices_placed)

                valid_operation = valid_bondlengths and obey_constraint
                if not valid_operation:
                    a.positions[i] = posi_0
                else:
                    indices_placed.append(i)
                    break
            else:
                pass #indices_placed.append(i)
        if valid_operation:
            return a
        else:
            # If mutation is not successfull in supplied number
            # of trials, return None.
            return None

class RattleMutation1(OffspringOperation):
    """Class to perform rattle mutations on structures.

    Rattles a number of randomly selected atoms within a sphere 
    of radius 'rattle_range' of their original positions.
    - Removes all involved atoms before placing them one at a time.
    
    Parameters:

    n_top: int
        The number of atoms to optimize. Specifically the
        atoms with indices [-n_top:] are optimized.
    
    Nrattle: float
        The average number of atoms to rattle.

    rattle_range: float
        The maximum distance within witch to rattle the
        atoms. Atoms are rattled uniformly within a sphere of this
        radius.

    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.

    description: str
        Name of the operation, which will be saved in
        info-dict of structures, on which the operation is applied.    
    """
    def __init__(self, n_top, Nrattle=5, rattle_range=4,
                 description='RattleMutation', cluster=False, *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs)
        self.description = description
        self.n_top = n_top
        self.probability = Nrattle/n_top
        self.rattle_range = rattle_range
        self.cluster = cluster
    def operation(self, parents):
        a = parents[0]
        a = self.rattle(a)
        return a

    def rattle(self, atoms):
        """ Performs the local rattle mutation.
        """
        a = atoms.copy()
        Natoms = len(a)
        Nslab = Natoms - self.n_top
        indices_slab = list(np.arange(Nslab))

        # Randomly select indices of atoms to permute - in random order.
        indices_to_rattle = np.arange(Nslab,Natoms)[np.random.rand(self.n_top)
                                                     < self.probability]
        indices_to_rattle = np.random.permutation(indices_to_rattle)
        if len(indices_to_rattle) == 0:
            indices_to_rattle = [np.random.randint(Nslab,Natoms)]

        indices_placed = list(np.setdiff1d(np.arange(Nslab,Natoms), indices_to_rattle))
        # Perform rattle operations in sequence.
        for i in indices_to_rattle:
            posi_0 = np.copy(a.positions[i])
            for _ in range(100):
                # Perform rattle
                pos_add = pos_add_sphere(self.rattle_range)
                a.positions[i] += pos_add
                
                # Check position constraint
                obey_constraint = self.check_constraints(a.positions[i])
                # Check if rattle was valid
                if Nslab == 0 and len(indices_placed) == 0:
                    valid_bondlengths = True
                    #valid_bondlengths = self.check_bondlengths(a, indices=[i], indices_placed=None, check_isolated=False)
                else:
                    if self.cluster:
                        not_too_close_all = self.check_bondlengths(a, indices=[i], indices_placed=indices_slab+indices_placed, check_isolated=False)
                        if len(indices_placed) > 1:
                            not_isolated_cluster = self.check_bondlengths(a, indices=[i], indices_placed=indices_placed, check_too_close=False)
                        else:
                            not_isolated_cluster = True
                        valid_bondlengths = not_too_close_all and not_isolated_cluster
                    else:
                        valid_bondlengths = self.check_bondlengths(a, indices=[i], indices_placed=indices_slab+indices_placed)

                valid_operation = valid_bondlengths and obey_constraint
                if not valid_operation:
                    a.positions[i] = posi_0
                else:
                    indices_placed.append(i)
                    break
            else:
                indices_placed.append(i)
        if valid_operation:
            return a
        else:
            # If mutation is not successfull in supplied number
            # of trials, return None.
            return None

class PermutationMutation(OffspringOperation):
    """Class to perform permutation mutations on structures.

    Swaps the positions of a number of pairs of unlike atoms.
    
    Parameters:

    n_top: int
        The number of atoms to optimize. Specifically the
        atoms with indices [-n_top:] are optimized.
    
    Npermute: float
        The average number of permutations to perform.
    
    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.
    
    description: str
        Name of the operation, which will be saved in
        info-dict of structures, on which the operation is applied.
    """

    def __init__(self, n_top, Npermute=3,
                 description='PermutationMutation', *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs)
        self.description = description
        self.n_top = n_top
        self.probability = Npermute/n_top

    def operation(self, parents):
        a = parents[0]
        a = self.permute(a)
        return a

    def permute(self, atoms):
        """ Permutes atoms of different type in structure.
        """
        a = atoms.copy()
        Natoms = len(a)
        Nslab = Natoms - self.n_top
        num = a.numbers

        # Check if permutation mutation is applicable to structure.
        num_unique_top = list(set(num[-self.n_top:]))
        assert len(num_unique_top) > 1, 'Permutations with one atomic type is not valid'

        # Randomly select indices of atoms to permute - in random order.
        indices_to_permute = np.arange(Nslab,Natoms)[np.random.rand(self.n_top)
                                                   < self.probability]
        indices_to_permute = np.random.permutation(indices_to_permute)
        if len(indices_to_permute) == 0:
            indices_to_permute = [np.random.randint(Nslab,Natoms)]

        # Perform permutations in sequence.
        for i in indices_to_permute:
            for _ in range(100):
                j = np.random.randint(Nslab,Natoms)
                while num[i] == num[j]:
                    j = np.random.randint(Nslab,Natoms)

                # Permute
                pos_i = np.copy(a.positions[i])
                pos_j = np.copy(a.positions[j])
                a.positions[i] = pos_j
                a.positions[j] = pos_i

                # Check if rattle was valid
                valid_bondlengths = self.check_bondlengths(a, indices=[i,j])
                
                if not valid_bondlengths:
                    a.positions[i] = pos_i
                    a.positions[j] = pos_j
                else:
                    break
        if valid_bondlengths:
            return a
        else:
            # If mutation is not successfull in supplied number
            # of trials, return initial structure.
            return None

