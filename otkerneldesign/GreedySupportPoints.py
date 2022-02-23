#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2022

@author: Elias Fekhari
"""
#%%
import numpy as np
import openturns as ot
from scipy.spatial import distance_matrix

class GreedySupportPoints:
    """
    Generate a SupportPoints sample

    Parameters
    ----------
    size: int
        Sample size
    distribution: ot.ComposedDistribution()
        Input random distribution including its dimension. If None then Uniform(0, 1).
    initial_design: np.array()
        Numpy array with the shape size x dimension
    candiate_points: np.array()
        Numpy array with the shape candidate_point_size x dimension
    my_seed: int 
        Pseudo-random seed used to generate the design

    Returns
    -------
    return a SupportPoints sample with a given sample size and dimension.
    """
    def __init__(
        self, 
        distribution=None,
        candidate_set_size=None,
        candidate_set=None,
        initial_design=None, 
    ):
        # Inconsistency
        if candidate_set_size is not None and candidate_set is not None:
            raise ValueError(
                "Since you provided a candidate set, you cannot specify its size."
            )

        # Dimension
        if distribution is None:
            if candidate_set is not None:
                candidate_set = ot.Sample(candidate_set)
                self._dimension = candidate_set.getDimension()
            else:
                raise ValueError("Either provide a distribution or a candidate set.")
        else:
            self._dimension = distribution.getDimension()

        # Candidate set
        if candidate_set is not None:
            self._candidate_set = ot.Sample(candidate_set)
            candidate_set_size = self._candidate_set.getSize()
            if self._candidate_set.getDimension() != self._dimension:
                raise ValueError(
                    "Candidate set dimension {} does not match distribution dimension {}".format(
                        self._candidate_set.getDimension(), self._dimension
                    )
                )
        else:
            if candidate_set_size is None:
                candidate_set_size = 2 ** 12

            sobol = ot.LowDiscrepancyExperiment(
                ot.SobolSequence(), distribution, candidate_set_size, True
            )
            sobol.setRandomize(False)
            self._candidate_set = sobol.generate()
      
        # Cast candidate set and initial design to fit with numpy implementation 
        self._candidate_set = np.array(self._candidate_set)
        if initial_design is None:
            self._initial_size = 0
            self._design_indices = np.array([], dtype='int32')
        else :
            self._candidate_set = np.vstack([candidate_set, np.array(initial_design)])#candidate_points
            self._initial_size = initial_design.shape[0]
            self._design_indices = np.arange(candidate_set_size, candidate_set_size + self._initial_size)#design_indexes
        # Compute distances
        self.distances = self.compute_distance_matrix()
        self._target_potential = self.compute_target_potential()
        
    def compute_distance_matrix(self, batch_nb=8):
        # Divide the candidate points in batches
        batch_size = self._candidate_set.shape[0] // batch_nb
        batches = []
        for batch_index in range(batch_nb):
            if batch_index==batch_nb - 1:
                batches.append(self._candidate_set[batch_size * batch_index : ])
            else:    
                batches.append(self._candidate_set[batch_size * batch_index : batch_size * (batch_index + 1)])
        # Build matrix of distances between all the couples of candidate points
        # Built block by block to avoid filling up memory
        distances = np.zeros([self._candidate_set.shape[0]] * 2, dtype='float16')
        for i, _ in enumerate(batches):
            for j in range(i+1):
                # raw i column j in the distances matrix
                batch_dist = distance_matrix(batches[i], batches[j])
                # Lower right corner block of the matrix has a different shape
                if (i==batch_nb - 1) and (j==batch_nb - 1):
                    distances[batch_size * i : , batch_size * j : ] = batch_dist
                # Lower left corner block of the matrix has a different shape
                elif (i==batch_nb - 1):
                    distances[batch_size * i : , 
                            batch_size * j : batch_size * (j + 1)] = batch_dist
                # Squared block
                else:
                    distances[batch_size * i : batch_size * (i + 1), 
                            batch_size * j : batch_size * (j + 1)] = batch_dist
        distances = np.tril(distances)
        distances += distances.T
        return distances

    def compute_target_potential(self):
        potentials = self.distances.mean(axis=0)
        return potentials

    def compute_current_potential(self, design_indices):
        distances_to_design = self.distances[:, design_indices]
        current_potential = distances_to_design.mean(axis=1)
        current_potential *= len(design_indices) / (len(design_indices) + 1)
        return current_potential
    
    # Change the declaration of the initial_size to make the code more robust
    def select_design(self, size):
        for _ in range(size):
            if len(self._design_indices)==0:
                criteria = self._target_potential
            else:
                current_potential = self.compute_current_potential(self._design_indices)
                criteria = self._target_potential - current_potential
            next_index = np.argmin(criteria)
            self._design_indices = np.append(self._design_indices, next_index)
        sample = self._candidate_set[self._design_indices[self._initial_size:]]
        return sample


# %%
size = 20
dimension = 2
distribution = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dimension)
sp = GreedySupportPoints(distribution=distribution, candidate_set_size=2**12)
sample = sp.select_design(size)

