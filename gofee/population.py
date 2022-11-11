import numpy as np
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms

from copy import deepcopy

from gofee.utils import Kmeans, InteratomicDistanceComparator

class AbstractPopulation(ABC):

    def __init__(self, population_size, **kwargs):
        self.pop_size = population_size
        self.pop = []
        self.pop_MLrelaxed = []

    def add(self, structures):
        if isinstance(structures, Atoms):
            self.add_structure(structures)
        elif isinstance(structures, list):
            assert isinstance(structures[0], Atoms)
            for a in structures:
                self.add_structure(a)

    @abstractmethod
    def add_structure(self, a):
        """ Need to be implemented for populations that
        are maintained by adding the newest candidates based on
        their fitness.
        """
        pass

    @abstractmethod
    def get_refined_pop(self):
        pass

    def get_structure(self):
        t = np.random.randint(len(self.pop))
        return self.pop[t].copy()

    def get_structure_pair(self):
        if len(self.pop) >= 2:
            t1, t2 = np.random.permutation(len(self.pop))[:2]
        else:
            t1 = 0
            t2 = 0
        structure_pair = [self.pop[t1].copy(), self.pop[t2].copy()]
        return structure_pair

    @abstractmethod
    def update(self, structures, features):
        pass


def get_raw_score(a):
    return a.get_potential_energy()

class LVPopulation(AbstractPopulation):

    def __init__(self, population_size, **kwargs):
        AbstractPopulation.__init__(self, population_size, **kwargs)
        self.comparator = InteratomicDistanceComparator(mic=True)

    def add_structure(self, a):
        """ Adds a single candidate to the population. """
        assert 'energy' in a.get_calculator().results
        E = a.get_potential_energy()
        assert 'forces' in a.get_calculator().results
        F = a.get_forces()

        if len(self.pop) == 0:
            self.pop.append(a)
            return

        # check if the structure is too low in raw score
        raw_score_a = get_raw_score(a)
        raw_score_worst = get_raw_score(self.pop[-1])
        if raw_score_a > raw_score_worst \
                and len(self.pop) == self.pop_size:
            return

        # check if the new candidate should
        # replace a similar structure in the population
        for (i, b) in enumerate(self.pop):
            if len(self.pop) == len(self.pop_MLrelaxed):
                b_relaxed = self.pop_MLrelaxed[i]
            else:
                b_relaxed = b
            #if self.comparator.looks_like(a, b):
            if self.comparator.looks_like(a, b_relaxed):
                if get_raw_score(b) > raw_score_a:
                    del self.pop[i]
                    self.pop.append(a)
                    self.pop.sort(key=lambda x: get_raw_score(x))
                return

        # the new candidate needs to be added, so remove the highest
        # energy one
        if len(self.pop) == self.pop_size:
            del self.pop[-1]

        # add the new candidate
        self.pop.append(a)
        self.pop.sort(key=lambda x: get_raw_score(x))

    def get_refined_pop(self):
        """ Returns Model relaxed population.
        """
        return self.pop_MLrelaxed

    def update(self, structures, features):
        pass




class Population():

    def __init__(self, population_size, gpr, similarity2equal=0.9995, **kwargs):
        self.pop_size = population_size
        self.gpr = gpr
        self.similarity2equal = similarity2equal
        self.largest_energy = np.inf
        self.pop = []
        self.pop_MLrelaxed = []

    def __sort(self):
        E_pop = np.array([a.get_potential_energy() for a in self.pop])
        sorted_indices = np.argsort(E_pop)  # sort from lowest to highest
        self.pop = [self.pop[i] for i in sorted_indices]

    def __save_FandE(self, anew, E, F):
        a = anew.copy()
        results = {'energy': E, 'forces': F}
        calc = SinglePointCalculator(a, **results)
        a.set_calculator(calc)
        return a

    def remove_duplicate_structures(self):
        structures2remove = []
        for i, ai in enumerate(self.pop_MLrelaxed):
            for j, aj in enumerate(self.pop_MLrelaxed):
                if j <= i:
                    continue
                if self.looks_like(ai, aj):
                    # Always remove j index because it will be the one with the largest
                    # energy, since the population is sorted after energy.
                    if not j in structures2remove:
                        structures2remove.append(j)

        # Remove from the end of the population, so indices do not change
        for k in sorted(structures2remove)[::-1]:
            del self.pop[k]
            del self.pop_MLrelaxed[k]

    def add(self, structures):
        if isinstance(structures, Atoms):
            self.__add_structure(structures)
        elif isinstance(structures, list):
            assert isinstance(structures[0], Atoms)
            for a in structures:
                self.__add_structure(a)
            
    def __add_structure(self, a):
        self.remove_duplicate_structures()
        assert 'energy' in a.get_calculator().results
        E = a.get_potential_energy()
        assert 'forces' in a.get_calculator().results
        F = a.get_forces()
        
        if E > self.largest_energy and len(self.pop) == self.pop_size:
            return

        #a = self.__save_FandE(anew, E, F)
        
        # check for similar structure in pop - starting from the largest energy
        for i, ai in enumerate(self.pop_MLrelaxed[::-1]):
            if self.looks_like(a, ai):
                Ei = self.pop[-(i+1)].get_potential_energy()
                if E < Ei:  # if structure in pop is worse, replace.
                    del self.pop[-(i+1)]
                    self.pop.append(a)
                    
                    # sort and set largest energy
                    self.__sort()
                    self.largest_energy = self.pop[-1].get_potential_energy()
                    return
                else:  # If structure in pop is better, discart new structure.
                    return

        # if no similar structure was found in population
        # just add if population us not full
        if len(self.pop) < self.pop_size:
            self.pop.append(a)
        else:  # replace worst
            del self.pop[-1]
            self.pop.append(a)

        # sort and set largest energy
        self.__sort()
        self.largest_energy = self.pop[-1].get_potential_energy()

    def get_refined_pop(self):
        """ Returns Model relaxed population.
        """
        return self.pop_MLrelaxed

    def get_structure(self):
        t = np.random.randint(len(self.pop))
        return self.pop[t].copy()

    def get_structure_pair(self):
        if len(self.pop) >= 2:
            t1, t2 = np.random.permutation(len(self.pop))[:2]
        else:
            t1 = 0
            t2 = 0
        structure_pair = [self.pop[t1].copy(), self.pop[t2].copy()]
        return structure_pair

    def looks_like(self, a1, a2):
        f1 = self.gpr.descriptor.get_feature(a1)
        f2 = self.gpr.descriptor.get_feature(a2)
        K0 = self.gpr.K0
        similarity = self.gpr.kernel.kernel_value(f1,f2) / K0
        if similarity > self.similarity2equal:
            return True
        else:
            return False

    def update(self, structures, features):
        pass

class ClusteringPopulation():

    def __init__(self, population_size, dE=5, cluster_contributions=None, weight=0, **kwargs):
        """
        Parameters:

        population_size : int

        dE : float

        cluster_contributions : list
            List of fractions, representing the fraction of clusters should
            come from the best cluster, second best cluster, etc.
            These extra structures from a cluster are found by using clustering
            to further subdivide the original cluster.
            Ex. [0.4, 0.2] would result in 40% of structures from the best original
            cluster, 20% from the second best and 1 structure from each remaing cluster.

        """
        self.population_size = population_size
        self.dE = dE
        self.cluster_contributions = cluster_contributions
        self.weight = weight
        self.pop = None
        self.pop_MLrelaxed = None
        self.features_pop = None

    def update(self, structures, features):
        Ndata = len(structures)
        ### Exclude high energy structures ###
        # Extract all energies
        E = np.array([a.get_potential_energy() for a in structures])
        # Get lowest energy
        Emin = np.min(E)
        # Use only structures with energies below Emin+dE
        for i in range(5):
            filt = E <= Emin + self.dE * 2**i
            if np.sum(filt) >= 2*self.population_size:
                break
        else:
            filt = np.ones(Ndata, dtype=bool)
            index_sort = np.argsort(E)
            filt[index_sort[2*self.population_size:]] = False

        structures_filt = [structures[i] for i in range(len(structures)) if filt[i]]
        f_filt = features[filt]
        E_filt = E[filt]

        ### Cluster structures ###
        n_clusters = 1 + min(self.population_size-1, int(np.floor(len(E_filt)/5)))
        if self.cluster_contributions is not None:
            extra_cluster_numbers = np.array([np.floor(frac*n_clusters).astype(int) for frac in self.cluster_contributions])
            extra_cluster_numbers = extra_cluster_numbers[extra_cluster_numbers > 1]
            n_clusters = n_clusters - np.sum(extra_cluster_numbers) + len(extra_cluster_numbers)

        self.pop, self.features_pop, cluster_indices_all = self._get_population_data_from_clustering(structures_filt,
                                                                                                     f_filt,
                                                                                                     E_filt,
                                                                                                     n_clusters)

        index_sort = self._sort_population()

        # Make further clustering of best clusters
        if self.cluster_contributions is not None:
            N_resolved_clusters = len(extra_cluster_numbers)
            # Perform fine-grained clustering of the best clusters
            cluster_indices_all_sort = [cluster_indices_all[i] for i in index_sort[:N_resolved_clusters]]
            for cluster_indices, n_clusters in zip(cluster_indices_all_sort, extra_cluster_numbers):
                structures_sub = [structures_filt[i] for i in cluster_indices]
                f_sub = f_filt[cluster_indices]
                E_sub = E_filt[cluster_indices]
                n_clusters = 1 + min(n_clusters-1, int(np.floor(len(E_sub)/6)))

                if n_clusters > 1:
                    pop_add, features_pop_add, _ = self._get_population_data_from_clustering(structures_sub,
                                                                                            f_sub,
                                                                                            E_sub,
                                                                                            n_clusters)
                    # Remove current contributions to clusters to be further resolved
                    self.pop = self.pop[1:]
                    self.features_pop = self.features_pop[1:]
                    # Add new contribution at the end of list/array
                    self.pop = self.pop + pop_add
                    self.features_pop = np.r_[self.features_pop, features_pop_add]

            self._sort_population()

    def _get_population_data_from_clustering(self, structures, features, energies, n_clusters):
        kmeans = Kmeans(n_clusters=n_clusters).fit(features)
        cluster_labels = kmeans.labels_

        # Get population as minimum energy structure of each cluster.
        indices = np.arange(len(energies))
        pop = []
        features_pop = []
        cluster_indices_all = []
        for ic in range(n_clusters):
            filt_cluster = cluster_labels == ic
            cluster_indices = indices[filt_cluster]
            index_best_in_cluster = cluster_indices[np.argmin(energies[filt_cluster])]
            pop.append(structures[index_best_in_cluster])
            features_pop.append(features[index_best_in_cluster])
            cluster_indices_all.append(cluster_indices)
        return pop, features_pop, cluster_indices_all

    def _sort_population(self):
        # Sort population by energy
        index_sort = np.argsort([a.get_potential_energy() for a in self.pop])
        self.pop = [self.pop[i] for i in index_sort]
        self.features_pop = np.array([self.features_pop[i] for i in index_sort])
        return index_sort

    def add(self, structures):
        pass

    def get_refined_pop(self):
        return self.pop_MLrelaxed

    def get_structure(self):
        t = np.random.randint(len(self.pop))
        return self.pop[t].copy()

    def get_structure_pair(self):
        Npop = len(self.pop)
        pop_indices = np.arange(Npop)
        weights = self._get_weights(Npop)
        if len(self.pop) >= 2:
            t1, t2 = np.random.choice(pop_indices, size=2, p=weights)
            while t1 == t2:
                t1, t2 = np.random.choice(pop_indices, size=2, p=weights)
        else:
            t1 = 0
            t2 = 0
            
        structure_pair = [self.pop[t1].copy(), self.pop[t2].copy()]
        return structure_pair

    def _get_weights(self, N):
        x = np.arange(1,1+N)
        return 1/x**self.weight / np.sum(1/x**self.weight)