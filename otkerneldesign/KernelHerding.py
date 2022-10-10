#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2022

@author: Joseph Muré
"""
import numpy as np
import openturns as ot
from scipy.special import erfc
from copy import deepcopy
import matplotlib.pyplot as plt


class KernelHerding:
    """
    Incrementally select new design points with kernel herding.

    Parameters
    ----------
    kernel : :class:`openturns.CovarianceModel`
        Covariance kernel used to define potentials.
        By default a product of Matern kernels with smoothness 5/2.
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
    is_greedy : Boolean 
        Set to False by default, then the criterion is the difference between the current and target potential. 
        When set to True, the MMD minimization is strictly greedy. In practice, the two criteria are very close, 
        only for the greedy one the current potential is multiplied by :math:`(\\frac{m}{m+1})`.

    Examples
    --------
    >>> import openturns as ot
    >>> import otkerneldesign as otkd
    >>> distribution = ot.ComposedDistribution([ot.Normal(0.5, 0.1)] * 2)
    >>> dimension = distribution.getDimension()
    >>> # Kernel definition
    >>> ker_list = [ot.MaternModel([0.1], [1.0], 2.5)] * dimension
    >>> kernel = ot.ProductCovarianceModel(ker_list)
    >>> # Kernel herding design
    >>> kh = otkd.KernelHerding(kernel=kernel, distribution=distribution)
    >>> kh_design, _ = kh.select_design(size=20)
    """

    def __init__(
        self,
        kernel=None,
        distribution=None,
        candidate_set_size=None,
        candidate_set=None,
        initial_design=None,
        is_greedy=False,
    ):
        self._method_label = "kernel herding"
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

        # Kernel
        if kernel is None:
            supposed_size = 50
            scale = supposed_size ** (-1 / self._dimension)
            ker_list = [ot.MaternModel([scale], 2.5)] * self._dimension
            self._kernel = ot.ProductCovarianceModel(ker_list)
        else:
            self._set_kernel(kernel)

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

        # Initial design
        if initial_design is None:
            self._design_indices = []
        else:
            self._design_indices = list(
                range(candidate_set_size, candidate_set_size + len(initial_design))
            )
            self._candidate_set.add(initial_design)
        self._initial_size = len(self._design_indices)

        # Analytical potential?
        if distribution is not None:
            self._examine_distribution(distribution)
        else:
            self._analytical = None

        # Covariance matrix
        if self._analytical is None:
            self._covmatrix_indices = list(range(self._candidate_set.getSize()))
            self._covmatrix = np.array(self._kernel.discretize(self._candidate_set))
        else:
            self._covmatrix_indices = [-1] * self._candidate_set.getSize()
            self._covmatrix = np.zeros((0, self._candidate_set.getSize()))
        self._target_potential = self.compute_target_potential()
        self._target_energy = self.compute_target_energy()
        self.is_greedy = is_greedy

    def _set_kernel(self, kernel):
        if kernel.getInputDimension() == self._dimension:
            self._kernel = kernel
        else:
            raise ValueError(
                "Incorrect dimension {}, should be {}".format(
                    kernel.getInputDimension(), self._dimension
                )
            )

    def _examine_distribution(self, distribution):
        self._analytical = None
        if distribution.getClassName() == "Normal":
            if distribution == ot.Normal(distribution.getDimension()):
                self._analytical = ["normal"] * distribution.getDimension()
        elif distribution.getClassName() == "Uniform":
            if distribution == ot.Uniform(0.0, 1.0):
                self._analytical = ["uniform"]
        elif distribution.getClassName() == "ComposedDistribution":
            dist_list = distribution.getDistributionCollection()
            dist_strings = ["unknown"] * len(dist_list)
            is_legit = [False] * len(dist_list)
            for num, dist in enumerate(dist_list):
                if dist == ot.Uniform(0.0, 1.0):
                    is_legit[num] = True
                    dist_strings[num] = "uniform"
                elif dist == ot.Normal(1):
                    is_legit[num] = True
                    dist_strings[num] = "normal"
            if np.prod(is_legit):
                self._analytical = dist_strings

    @staticmethod
    def _compute_uniform_aux(points, theta):
        points = np.array(points)
        expon = np.exp(-np.sqrt(5) / theta * points)
        quad = 5 * np.sqrt(5) * points ** 2
        lin = 25 * theta * points
        cons = 8 * np.sqrt(5) * theta ** 2
        return expon * (quad + lin + cons)

    @staticmethod
    def _compute_normal_aux(points, theta):
        points = np.array(points)
        root = np.sqrt(5) / theta
        term = (
            root / 3 / np.sqrt(2 * np.pi) * (3 - root ** 2) * np.exp(-(points ** 2) / 2)
        )
        factor_erfc = erfc((root - points) / np.sqrt(2))
        expon = np.exp(root ** 2 / 2 - root * points) / 6
        quad = root ** 2 * points ** 2
        lin = (3 - 2 * root ** 2) * root * points
        cons = root ** 2 * (root ** 2 - 2) + 3
        return term + expon * factor_erfc * (quad + lin + cons)

    def compute_target_potential(self):
        """
        Compute the potential of the target probability measure :math:`\\mu`.

        Returns
        -------
        potential : numpy.array
                    Potential of the measure :math:`\\mu` defined by
        
        .. math::
            P_{\\mu}(\\vect{x}) := \\int k(\\vect{x}, \\vect{x}') d \\mu(\\vect{x}').

        """
        if self._analytical is None:
            return self._covmatrix.mean(axis=0)

        # At this point, we know the potential is computed analytically.
        if self._kernel.getClassName() == "ProductCovarianceModel":
            if self._kernel.getNuggetFactor() > 1e-12:
                raise ValueError("No nugget factor allowed.")
            kernel_list = self._kernel.getCollection()
        elif self._kernel.getInputDimension() == 1:
            kernel_list = [self._kernel]
        else:
            raise ValueError("Multidimensional kernels must be products")

        for kernel in kernel_list:
            if kernel.getInputDimension() != 1:
                raise ValueError("1 kernel per input dimension")
            if kernel.getImplementation().getClassName() != "MaternModel":
                raise ValueError("Only Matern kernels allowed.")
            if kernel.getFullParameter()[-1] != 2.5:  # getNu() cannot be accessed
                raise ValueError("Nu must be 2.5, not {}".format(kernel.getNu()))
            if kernel.getNuggetFactor() > 1e-12:
                raise ValueError("No nugget factor allowed.")

        potential = np.ones(self._candidate_set.getSize())

        for ind, kernel in enumerate(kernel_list):
            theta = kernel.getScale()[0]
            points = np.reshape(self._candidate_set[:, ind], -1)
            distribution = self._analytical[ind]

            if distribution == "uniform":
                term = 16 * theta / 3 / np.sqrt(5)
                factor = -1 / 15 / theta
                aux1 = self._compute_uniform_aux(points, theta)
                aux2 = self._compute_uniform_aux(-points + 1.0, theta)
                potential *= term + factor * (aux1 + aux2)

            elif distribution == "normal":
                aux1 = self._compute_normal_aux(points, theta)
                aux2 = self._compute_normal_aux(-points, theta)
                potential *= aux1 + aux2

            else:
                raise ValueError("Distribution should be either 'uniform' or 'normal'")

        return potential

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
        Compute the potential of the discrete measure (a.k.a, kernel mean embedding) defined by the design :math:`\mat{X}_n`.
        Considering the discrete measure :math:`\\zeta_n = \\frac{1}{n} \\sum_{i=1}^{n} \\delta(\\vect{x}^{(i)})`, its potential is defined as 
        
        .. math::
            P_{\\zeta_n}(\\vect{x}) := \\frac{1}{n} \\sum_{i=1}^{n} k(\\vect{x}, \\vect{x}^{(i)}).

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
            return np.zeros(self._candidate_set.getSize())
        covmatrix_design_indices_rows = self._extract_from_covmatrix(design_indices)
        return covmatrix_design_indices_rows.mean(axis=0)
    
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

    def compute_mmd(self, design_indices):
        """
        Compute Maximum Mean Discrepancy between :math:`\\mu` and :math:`\\zeta_n = \\frac{1}{n} \\sum_{i=1}^{n} \\delta(\\vect{x}^{(i)})`.

        Parameters
        ----------
        design_indices : list of positive int
                         List of the indices of the selected points
                         in the Sample of candidate points

        Returns
        -------
        mmd : float
                Maximum Mean Discrepancy between target and current measure.
        """
        current_energy = self.compute_current_energy(design_indices)
        current_design = self._candidate_set[design_indices]
        cross_potential = np.array(self._kernel.computeCrossCovariance(current_design, self._candidate_set)).mean()
        mmd = current_energy + self._target_energy - 2 * cross_potential
        return mmd

    def compute_criterion(self, design_indices):
        """
        Compute the criterion on a design :math:`\mat{X}_n`. At any point of the candidate set, 
        this criterion is given by the difference between the target potential 
        and the potential of a discrete measure defined by the given design.

        Parameters
        ----------
        design_indices : list of positive int
            List of the indices of the selected points :math:`\mat{X}_n`
            in the Sample of candidate points

        Returns
        -------
        current_potential - target_potential : numpy.array
                                                Vector of the values taken by the criterion on all candidate points
        """
        current_potential = self.compute_current_potential(design_indices)
        m = len(design_indices)
        if self.is_greedy:
            return (m / (m + 1)) * current_potential - self._target_potential
        else:
            return current_potential - self._target_potential

    def select_design(self, size, initial_design_indices=[]):
        """
        Select a design with kernel herding. 

        Parameters
        ----------
        size : positive int
            Number of points to be selected
        initial_design_indices : list of positive int
            List of the indices of *already* selected points (empty by default)
            in the Sample of candidate points

        Returns
        -------
        design : :class:`openturns.Sample`
            Sample of all selected points
        design_indices : list of positive int or None
            List of the indices of the selected points
            in the Sample of candidate points

        """
        design_indices = deepcopy(self._design_indices)
        for _ in range(size):
            criteria = self.compute_criterion(design_indices)
            next_index = np.argmin(criteria)
            design_indices.append(next_index)
        design = self._candidate_set[design_indices[self._initial_size:]]
        return design

    def _extract_from_covmatrix(self, design_indices):
        """
        Smart extractor of lines of the covariance matrix.
        Computes only the missing lines of the matrix.

        Parameters
        ----------
        design_indices : list of positive int
            List of the indices of the lines of the covariance matrix to select.
            Each index corresponds to a Point in the Sample of candidate points.

        Returns
        -------
        block : matrix (numpy array)
            Block containing the required lines of the covariance matrix.

        """
        where_missing = np.array(self._covmatrix_indices)[design_indices] == -1
        indices_to_compute = np.array(design_indices)[where_missing]
        new_lines = np.nan * np.ones(
            (where_missing.sum(), self._candidate_set.getSize())
        )
        for num, index in enumerate(indices_to_compute):
            new_lines[num, :] = self._kernel.discretizeRow(
                self._candidate_set, int(index)
            ).asPoint()
            self._covmatrix_indices[index] = self._covmatrix.shape[0] + num
        self._covmatrix = np.vstack((self._covmatrix, new_lines))
        return self._covmatrix[np.array(self._covmatrix_indices)[design_indices], :]

    def get_candidate_set(self):
        """
        Accessor to the candidate set.

        Returns
        -------
        candidate_set : openturns.Sample
            A deepcopy of the candidate set.

        """
        return deepcopy(self._candidate_set)

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
        sizes = range(0, len(design_indices))
        for i in sizes:
            energies.append(self.compute_current_energy(design_indices[:i]))
        fig, ax = plt.subplots(1, figsize=(9, 6))
        plot_data, = ax.plot(sizes, energies, label=self._method_label)
        ax.axhline(self._target_energy, color='k', label='target')
        ax.set_title('Energy convergence')
        ax.set_xlabel('design size ($n$)')
        ax.set_ylabel('Energy')
        ax.legend(loc='best')
        plt.close()
        return fig, plot_data

    def draw_mmd_convergence(self, design_indices):
        """
        Draws the convergence of the MMD between a discrete measure and the target measure.

        Parameters
        ----------
        design_indices : list of positive int
                         List of the indices of the selected points
                         in the Sample of candidate points

        Returns
        -------
        fig : matplotlib.Figure
                    MMD convergence of the design of experiments
        
        plot_data : data used to plot the figure
        """
        mmds = []
        sizes = range(1, len(design_indices))
        for i in sizes:
            mmds.append(self.compute_mmd(design_indices[:i]))
        fig, ax = plt.subplots(1, figsize=(9, 6))
        plot_data, = ax.plot(sizes, mmds, label=self._method_label)
        ax.set_title('MMD convergence')
        ax.set_xlabel('design size ($n$)')
        ax.set_ylabel('MMD')
        ax.legend(loc='best')
        plt.close()
        return fig, plot_data

    def get_indices(self, sample):
        """
        When provided a subsample of the candidate set, returns the indices of its points in the candidate set.

        Parameters
        ----------
        sample : 2-d list of float
            A subsample of the candidate set.

        Returns
        -------
        indices : list of int
            Indices of the points of the sample within the candidate set.
        """
        sample = np.array(sample)
        if len(sample.shape) != 2:
            raise ValueError("Not a sample: shape is {} instead of 2.".format(len(sample.shape)))
        candidate_array = np.array(self._candidate_set) # convert to numpy array so np.where works
        indices = []
        for sample_index, pt in enumerate(sample):
            index = np.where((candidate_array==pt).prod(axis=1))[0]
            if len(index) != 1:
                raise ValueError("The point {}, with index {} in the sample, is not in the candidate set.".format(pt, sample_index))
            indices.extend(index)
        return indices