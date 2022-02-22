#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2022

@author: Joseph Muré
"""
import numpy as np
import openturns as ot
from copy import deepcopy


class KernelHerdingTensorized:
    """
    Incrementally select new design points.

    Parameters
    ----------
    kernel : :class:`openturns.CovarianceModel`
        Covariance kernel used to define potentials.
        By default a product of Matern kernels with smoothness 5/2.
    distribution : :class:`openturns.Distribution`
        Distribution of the set of candidate points.
        If not specified, then *candidate_set* must be specified instead.
        Even if *candidate_set* is specified, can be useful if it allows the use of tensorized formulas.
    candidate_set_size : positive int
        Size of the set of all candidate points.
        Unnecessary if *candidate_set* is specified. Otherwise, :math:`2^{12}` by default.
    candidate_set : 2-d list of float
        Large sample that represents a distribution.
        If not specified, then *distribution* and *candidate_set_size* must be in order to generate it automatically.
    initial_design : 2-d list of float
        Sample of points that must be included in the design. Empty by default.
    """

    def __init__(
        self,
        kernel=None,
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
            self._distribution = distribution

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

        # tensorized potential?
        if distribution is not None:
            self.examine_distribution(distribution)
        else:
            self._tensorized = False

        # Covariance matrix
        if self._tensorized:
            self._covmatrix_indices = [-1] * self._candidate_set.getSize()
            self._covmatrix = np.zeros((0, self._candidate_set.getSize()))
        else:
            self._covmatrix_indices = list(range(self._candidate_set.getSize()))
            self._covmatrix = np.array(self._kernel.discretize(self._candidate_set))
        self._target_potential = self.compute_target_potential()

    def _set_kernel(self, kernel):
        if kernel.getInputDimension() == self._dimension:
            self._kernel = kernel
        else:
            raise ValueError(
                "Incorrect dimension {}, should be {}".format(
                    kernel.getInputDimension(), self._dimension
                )
            )

    def examine_distribution(self, distribution):
        self._tensorized = False
        if distribution.getClassName() == "ComposedDistribution":
            if distribution.hasIndependentCopula():
                self._tensorized = True

    def compute_target_potential(self):
        if self._tensorized is None:
            return self._covmatrix.mean(axis=0)

        # At this point, we know the potential is a product of univariate potentials.
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
            if kernel.getNuggetFactor() > 1e-12:
                raise ValueError("No nugget factor allowed.")

        marginal_list = self._distribution.getDistributionCollection()
        marginal_potential_functions = []
        for marginal, kernel in zip(marginal_list, kernel_list):
            # regular grid
            uniform_nodes = ot.RegularGrid(0.01, 0.99, 99)
            # Apply quantile function
            nodes = marginal.computeQuantile(uniform_nodes.getValues())
            # Compute covariance matrix
            marginal_covmatrix = np.array(kernel.discretize(nodes))
            # Compute potentials
            marginal_potentials = marginal_covmatrix.mean(axis=0)
            # Create a piecewise linear function
            marginal_potential_function = ot.Function(ot.PiecewiseLinearEvaluation(nodes.asPoint(), marginal_potentials))
            marginal_potential_functions.append(marginal_potential_function)

        # Aggregate the functions
        aggregated_potential_functions = ot.AggregatedFunction(marginal_potential_functions)

        # Product
        inputs = ["x{}".format(i) for i in range(len(marginal_list))]
        formula = [" * ".join(inputs)]
        product = ot.SymbolicFunction(inputs, formula)

        # Potential function
        potential_function = ot.ComposedFunction(aggregated_potential_functions, product)

        return potential_function(self._candidate_set)


    def compute_current_potential(self, design_indices):
        """
        Compute the potential of the design and the weights of the design points.

        Parameters
        ----------
        design_indices : list of positive int
                         List of the indices of the selected points
                         in the Sample of candidate points

        Returns
        -------
        potential : potential of the measure defined by the design

        """
        if len(design_indices) == 0:
            return np.zeros(self._candidate_set.getSize())
        covmatrix_design_indices_rows = self._extract_from_covmatrix(design_indices)
        return covmatrix_design_indices_rows.mean(axis=0)

    def compute_criterion(self, design_indices):
        """
        Compute the criterion on a design.

        Parameters
        ----------
        design_indices : list of positive int
            List of the indices of the selected points
            in the Sample of candidate points

        Returns
        -------
        current_potential - target_potential : vector (numpy array) of the values taken by the criterion on all candidate points

        """
        current_potential = self.compute_current_potential(design_indices)
        return current_potential - self._target_potential

    def select_design(self, size, initial_design_indices=[]):
        """
        Select a design with kernel herding.

        Parameters
        ----------
        size : positive int
            Number of points to be selected
        design_indices : list of positive int
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
        current_design_indices = deepcopy(initial_design_indices)
        current_design_indices.extend(self._design_indices)
        current_design_indices = list(set(current_design_indices))

        design_indices_orig_len = len(current_design_indices)
        design = ot.Sample(size, self._dimension)

        for k in range(size):
            current_potential = self.compute_current_potential(current_design_indices)
            crit = current_potential - self._target_potential
            iopt = crit.argmin()
            design[k] = self._candidate_set[iopt]
            current_design_indices.append(iopt)

        return design, current_design_indices[design_indices_orig_len:]

    def _extract_from_covmatrix(self, design_indices):
        """
        Smart extractor of lines of the covariance matrix.
        Computes only the missing lines of the matrix.

        Parameters
        ----------
        design_indices : list of positive int
            List of the indices of the lines of the covariance matrix to select.
            Each index correponds to a Point in the Sample of candidate points.

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

size = 20
dimension = 2
distribution = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dimension)
ker_list = [ot.MaternModel([0.1], [1.0], 2.5)] * dimension
kernel = ot.ProductCovarianceModel(ker_list)

kh = KernelHerdingTensorized(
    kernel=kernel,
    candidate_set_size=2 ** 12,
    distribution=distribution
)
uniform_design_points, uniform_design_indexes = kh.select_design(size)