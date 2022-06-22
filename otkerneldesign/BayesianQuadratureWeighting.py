#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2022

@author: Elias Fekhari
"""
import openturns as ot
import numpy as np


class BayesianQuadratureWeighting:
    """
    Optimally-weighting a sample for probabilistic integration.

    Parameters
    ----------
    kernel : :class:`openturns.CovarianceModel`
        Covariance kernel used to define potentials.
        By default a product of Matern kernels with smoothness 5/2.
    distribution : :class:`openturns.Distribution`
        Distribution of the set of candidate set.
        If not specified, then *distribution_sample* must be specified instead.
        Even if *distribution_sample* is specified, can be useful if it allows the use of analytical formulas.
    distribution_sample_size : positive int
        Size of the set of all candidate points.
        Unnecessary if *distribution_sample* is specified. Otherwise, :math:`2^{12}` by default.
    distribution_sample : 2-d list of float
        Large sample that empirically represents a distribution.
        If not specified, then *distribution* and *distribution_sample_size* must be in order to generate it automatically.

    Example
    -------
    >>> import openturns as ot
    >>> import otkerneldesign as otkd
    >>> distribution = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * 2)
    >>> mc_sample = distribution.getSample(20)
    >>> ker_list = [ot.MaternModel([0.1], [1.0], 2.5)] * 2
    >>> kernel = ot.ProductCovarianceModel(ker_list)
    >>> qw = otkd.BayesianQuadratureWeighting(
    >>>     kernel=kernel,
    >>>     distribution=distribution
    >>> )
    >>> optimal_weights = qw.compute_bayesian_quadrature_weights(mc_sample)
    """

    def __init__(
        self,
        kernel=None,
        distribution=None,
        distribution_sample_size=None,
        distribution_sample=None,
    ):
        # Inconsistency
        if distribution_sample_size is not None and distribution_sample is not None:
            raise ValueError(
                "Since you provided a candidate set, you cannot specify its size."
            )

        # Dimension
        if distribution is None:
            if distribution_sample is not None:
                distribution_sample = ot.Sample(distribution_sample)
                self._dimension = distribution_sample.getDimension()
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
        if distribution_sample is not None:
            self._distribution_sample = ot.Sample(distribution_sample)
            distribution_sample_size = self._distribution_sample.getSize()
            if self._distribution_sample.getDimension() != self._dimension:
                raise ValueError(
                    "Candidate set dimension {} does not match distribution dimension {}".format(
                        self._distribution_sample.getDimension(), self._dimension
                    )
                )
        else:
            if distribution_sample_size is None:
                distribution_sample_size = 2 ** 12

            sobol = ot.LowDiscrepancyExperiment(
                ot.SobolSequence(), distribution, distribution_sample_size, True
            )
            sobol.setRandomize(False)
            self._distribution_sample = sobol.generate()

    def _set_kernel(self, kernel):
        if not kernel.isStationary():
            raise ValueError("Only stationary kernels allowed.")

        if kernel.getInputDimension() == self._dimension:
            self._kernel = kernel
        else:
            raise ValueError(
                "Incorrect dimension {}, should be {}".format(
                    kernel.getInputDimension(), self._dimension
                )
            )

    def compute_bayesian_quadrature_weights(self, input_sample):
        """
        Compute optimal weights for probabilistic integration using a given sample.

        Parameters
        ----------
            input_sample : :class:`openturns.Sample`
                    Sample of points to be optimally weighted.
        """
        # TODO: 
        # Test dimension consistency with init objects 
        size = input_sample.getSize()
        global_sample = ot.Sample(input_sample)
        global_sample.add(self._distribution_sample)
        cov_rows = np.zeros((size, global_sample.getSize()))
        for idx in range(size):
            cov_rows[idx, :] = self._kernel.discretizeRow(global_sample, int(idx)).asPoint()
        covmatrix = cov_rows[:, :size] + np.identity(size) * 1e-4
        # TODO: 
        # - add a test on the conditioning using np.cond(covmatrix)
        # - try inversion using Pytorch 
        potentials = cov_rows[:, size:].mean(axis=1)
        return np.linalg.solve(covmatrix.T, potentials.T).T


