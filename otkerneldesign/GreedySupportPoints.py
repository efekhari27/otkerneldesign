#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2022

@author: Elias Fekhari
"""
import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


class GreedySupportPoints:
    """
    Incrementally select new design points with greedy support points. Support points use a specific kernel, the energy-distance kernel.

    Parameters
    ----------
    distribution : :class:`openturns.Distribution`
        Distribution the design points must represent.
        If not specified, then *candidate_set* must be specified instead.
        Even if *candidate_set* is specified, can be useful if it allows the use of analytical formulas.
    candidate_set_size : positive int
        Size of the set of all candidate points.
        Unnecessary if *candidate_set* is specified. Otherwise, :math:`2^{12}` by default.
    candidate_set : 2-d list of float
        Large sample that empirically represents a distribution.
        If not specified, then *distribution* and *candidate_set_size* must be in order to generate it automatically.
    initial_design : 2-d list of float
        Sample of points that must be included in the design. Empty by default.

    Examples
    --------
    >>> import openturns as ot
    >>> import otkerneldesign as otkd
    >>> distribution = ot.ComposedDistribution([ot.Normal(0.5, 0.1)] * 2)
    >>> # Greedy support points design
    >>> sp = otkd.GreedySupportPoints(distribution=distribution)
    >>> sp_design = sp.select_design(20)
    """

    def __init__(
        self,
        distribution=None,
        candidate_set_size=None,
        candidate_set=None,
        initial_design=None,
    ):
        self._method_label = "support points"
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
                candidate_set_size = 2**12

            sobol = ot.LowDiscrepancyExperiment(
                ot.SobolSequence(), distribution, candidate_set_size, True
            )
            sobol.setRandomize(False)
            self._candidate_set = sobol.generate()

        # Cast candidate set and initial design to fit with numpy implementation
        self._candidate_set = np.array(self._candidate_set)
        if initial_design is None:
            self._initial_size = 0
            self._design_indices = []
        else:
            initial_design = np.array(initial_design)
            self._candidate_set = np.vstack([self._candidate_set, initial_design])  # candidate_points
            self._initial_size = initial_design.shape[0]
            self._design_indices = list(
                range(candidate_set_size, candidate_set_size + self._initial_size)
            )  # design_indexes
        # Compute distances
        self.distances = self.compute_distance_matrix()
        self._target_potential = self.compute_target_potential()
        self._target_energy = self.compute_target_energy()

    def compute_distance_matrix(self, batch_nb=8):
        """
        Compute the matrix of pair-wise Euclidean distances between all candidate points.
        To avoid saturating the memory, this symmetric matrix is computed by
        blocks and using half-precision floating-point format (e.g., `numpy.float16`).

        Parameters
        ----------
        batch_nb : positive int
                    Number of blocks used to compute the symmetric
                    matrix of distances. By default set to 8.

        Returns
        -------
        distances : 2-d numpy array
                    Squared and symmetric matrix of distances between all
                    the couples of points in the candidate set.
        """

        # Divide the candidate points in batches
        batch_size = self._candidate_set.shape[0] // batch_nb
        batches = []
        for batch_index in range(batch_nb):
            if batch_index == batch_nb - 1:
                batches.append(self._candidate_set[batch_size * batch_index :])
            else:
                batches.append(
                    self._candidate_set[
                        batch_size * batch_index : batch_size * (batch_index + 1)
                    ]
                )
        # Build matrix of distances between all the couples of candidate points
        # Built block by block to avoid filling up memory
        distances = np.zeros([self._candidate_set.shape[0]] * 2, dtype="float16")
        for i, _ in enumerate(batches):
            for j in range(i + 1):
                # raw i column j in the distances matrix
                batch_dist = distance_matrix(batches[i], batches[j])
                # Lower right corner block of the matrix has a different shape
                if (i == batch_nb - 1) and (j == batch_nb - 1):
                    distances[batch_size * i :, batch_size * j :] = batch_dist
                # Lower left corner block of the matrix has a different shape
                elif i == batch_nb - 1:
                    distances[
                        batch_size * i :, batch_size * j : batch_size * (j + 1)
                    ] = batch_dist
                # Squared block
                else:
                    distances[
                        batch_size * i : batch_size * (i + 1),
                        batch_size * j : batch_size * (j + 1),
                    ] = batch_dist
        distances = np.tril(distances)
        distances += distances.T
        return distances

    def compute_target_potential(self):
        """
        Compute the potential of the target probability measure :math:`\\mu`.

        Returns
        -------
        potential : numpy.array
                    Potential of the measure :math:`\\mu` computed over the N-sized
                    candidate set and defined for the characteristic energy-distance
                    kernel of Székely and Rizzo by

        .. math::
            P_{\\mu}(x) := \\int k(x, x') d \\mu(x')
                         = \\frac{1}{N} \\sum_{k=1}^N \\|\\vect{x}-\\vect{x}'^{(k)}\\|.
        """
        potentials = self.distances.mean(axis=0)
        return potentials

    def compute_target_energy(self):
        """
        Compute the energy of the target probability measure :math:`\\mu`.

        Returns
        -------
        potential : float
                    Energy of the measure :math:`\\mu` defined by

        .. math::
            E_{\\mu} := \\int \\int k(\\vect{x}, \\vect{x}') d \\mu(\\vect{x}) d \\mu(\\vect{x}').

        """
        target_energy = self._target_potential.mean()
        return target_energy

    def compute_current_potential(self, design_indices):
        """
        Compute the potential of the discrete measure (a.k.a, kernel mean embedding)
        defined by the design :math:`\\vect{X}_n`. Considering the discrete measure
        :math:`\\zeta_n = \\frac{1}{n} \\sum_{i=1}^{n} \\delta(\\vect{x}^{(i)})`,
        its potential is defined for the characteristic energy-distance kernel of Székely and Rizzo

        .. math::
            P_{\\zeta_n}(x) = \\frac{1}{n} \\sum_{i=1}^{n} k(\\vect{x}, \\vect{x}^{(i)})
                            = \\frac{1}{n} \\sum_{i=1}^{n} \\|\\vect{x}-\\vect{x}^{(i)}\\|.

        Parameters
        ----------
        design_indices : list of positive int
                         List of the indices of the selected points
                         in the Sample of candidate points

        Returns
        -------
        potential : numpy.array
                    Potential of the discrete measure defined by the design (a.k.a, kernel mean embedding)

        """
        if len(design_indices) == 0:
            return np.zeros(len(self._candidate_set))
        distances_to_design = self.distances[:, design_indices]
        current_potential = distances_to_design.mean(axis=1)
        current_potential *= len(design_indices) / (len(design_indices) + 1)
        return current_potential

    def compute_current_energy(self, design_indices):
        """
        Compute the energy of the discrete measure defined by the design :math:`\mat{X}_n`.
        Considering the discrete measure :math:`\\zeta_n = \\frac{1}{n} \\sum_{i=1}^{n} \\delta(\\vect{x}^{(i)})`, its energy is defined as

        .. math::
            E_{\\zeta_n} := \\frac{1}{n^2} \\sum_{i=1}^{n} \\sum_{j=1}^{n} k(\\vect{x}^{(i)}, \\vect{x}^{(j)}).

        Parameters
        ----------
        design_indices : list of positive int
                         List of the indices of the selected points
                         in the Sample of candidate points

        Returns
        -------
        potential : float
                    Energy of the discrete measure defined by the design
        """
        current_potential = self.compute_current_potential(design_indices)
        current_energy = current_potential[design_indices].mean()
        return current_energy

    def select_design(self, size):
        """
        Select a design with greedy support points.

        Parameters
        ----------
        size : positive int
            Number of points to be selected

        Returns
        -------
        design : :class:`openturns.Sample`
            Sample of all selected points
        """
        design_indices = [index for index in self._design_indices]
        for _ in range(size):
            current_potential = self.compute_current_potential(design_indices)
            criteria = self._target_potential - current_potential
            next_index = np.argmin(criteria)
            design_indices.append(next_index)
        design = self._candidate_set[design_indices[self._initial_size :]]
        return design

    def draw_energy_convergence(self, design_indices):
        """
        Draws the convergence of the energy for a set of points selected among the candidate set.

        Parameters
        ----------
        design_indices : list of positive int
                         List of the indices of the selected points
                         in the Sample of candidate points

        Returns
        -------
        fig : matplotlib.Figure
                    Energy convergence of the design of experiments

        plot_data : data used to plot the figure
        """
        energies = []
        sizes = range(10, len(design_indices))
        for i in sizes:
            energies.append(self.compute_current_energy(design_indices[:i]))
        fig, ax = plt.subplots(1, figsize=(9, 6))
        (plot_data,) = ax.plot(sizes, energies, label=self._method_label)
        ax.axhline(self._target_energy, color="k", label="target")
        ax.set_title("Energy convergence")
        ax.set_xlabel("design size ($n$)")
        ax.set_ylabel("Energy")
        ax.legend(loc="best")
        plt.close()
        return fig, plot_data
