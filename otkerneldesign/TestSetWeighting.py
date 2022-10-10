#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2022

@author: Joseph Muré
"""
import openturns as ot
import numpy as np


class TestSetWeighting:
    """
    Weighting of a test-set for optimal machine learning performance
    metric (e.g, Integrated Squared Error, predictivity coefficient) estimation.

    Parameters
    ----------
    train : :class:`openturns.Sample`
        Training set used for fitting a machine learning model.
    test : :class:`openturns.Sample`
        Training set used for validating a machine learning model.
    distribution_sample : 2-d list of float
        Large sample that empirically represents a distribution.
    kernel : :class:`openturns.CovarianceModel`
        Covariance kernel used to define potentials.
        By default a product of Matern kernels with smoothness 5/2.

    Examples
    --------
    TODO
    """
    def __init__(
        self,
        train,
        test,
        distribution_sample,
        kernel=None,
    ):

        # Dimension
        self._train = ot.Sample(train)
        self._dimension = self._train.getDimension()

        self._test = ot.Sample(test)
        if self._test.getDimension() != self._dimension:
            raise ValueError(
                "Test set dimension {} does not match train set dimension {}".format(
                    self._test.getDimension(), self._dimension
                )
            )

        self._distribution_sample = ot.Sample(distribution_sample)
        if self._distribution_sample.getDimension() != self._dimension:
            raise ValueError(
                "Distribution sample dimension {} does not match design dimension {}".format(
                    self._distribution_sample.getDimension(), self._dimension
                )
            )

        # Kernel
        if kernel is None:
            scale = self._test.getSize() ** (-1 / self._dimension)
            ker_list = [ot.MaternModel([scale], 2.5)] * self._dimension
            self._kernel = ot.ProductCovarianceModel(ker_list)
        else:
            self._set_kernel(kernel)

        # Concatenate train, test and distribution samples
        global_sample = ot.Sample(self._train)
        global_sample.add(self._test)
        global_sample.add(self._distribution_sample)
        # Compute covriance between train and all points
        train_size = self._train.getSize()
        self._train_rows = np.zeros((train_size, global_sample.getSize()))
        for train_index in range(train_size):
            self._train_rows[train_index, :] = self._kernel.discretizeRow(
                global_sample, int(train_index)
            ).asPoint()
        # Compute covariance between test and all points
        test_size = self._test.getSize()
        self._test_rows = np.zeros((test_size, global_sample.getSize()))
        for test_index in range(test_size):
            self._test_rows[test_index, :] = self._kernel.discretizeRow(
                global_sample, train_size + int(test_index)
            ).asPoint()

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

    def compute_weights(self, residuals=None):
        """
        Compute optimal-weights for better performance estimation of a machine learning model.

        Parameters
        ----------
        residuals: list 
                    List of residuals in the case of a non-interpolating machine learning model. 
                    By default set to `None` for an interpolating model (e.g., Gaussian process regression).

        Return
        ------
        weights : list 
                    List of weights to be associated with the test set.   

        """
        train_size = self._train_rows.shape[0]
        test_size = self._test_rows.shape[0]

        if residuals is not None:
            residuals_array = np.array(residuals).reshape(-1, 1)
            if len(residuals_array) != test_size:
                raise ValueError("Residuals size does not match test size.")

        test_covmatrix = self._test_rows[
            :, train_size : train_size + test_size
        ]  # 9 x 9

        train_covmatrix = self._train_rows[:, :train_size]  # 11 x 11 # Km
        train_test_covmatrix = self._train_rows[
            :, train_size : train_size + test_size
        ]  # 11 x 9
        apply_inverse = np.linalg.solve(
            train_covmatrix, self._train_rows
        )  # (11 x 11) (11 x 2^12)

        covcond = (
            self._test_rows - train_test_covmatrix.T @ apply_inverse
        )  # 9 x 2^12 + (9 x 11) (11 x 2^12)
        variance = self._kernel.getAmplitude()[0] ** 2
        squares = apply_inverse * self._train_rows  # 11 x 2^12
        quadratic = variance - squares.sum(axis=0).reshape(-1, 1)  # 2^12 x 1
        barbar_cov = (
            2.0 * covcond ** 2
            + quadratic[train_size : train_size + test_size] @ quadratic.T
        )  # 9 x 2^12 + (9 x 1)(1 x 2^12)
        if residuals is not None:  # if metamodel does not interpolate
            error_mean = apply_inverse.T @ residuals_array  # (2^12 x 11) (11 x 1)
            squared_error_mean = error_mean ** 2  # (2^12 x 1)
            error_mean_test = error_mean[train_size : train_size + test_size]  # (9 x 1)
            squared_error_mean_test = squared_error_mean[
                train_size : train_size + test_size
            ]  # (9 x 1)
            barbar_cov += squared_error_mean_test @ squared_error_mean.T  # (9 x 2^12)
            barbar_cov += squared_error_mean_test @ quadratic.T  # (9 x 2^12)
            barbar_cov += (
                quadratic[train_size : train_size + test_size] @ squared_error_mean.T
            )  # (9 x 2^12)
            barbar_cov += (
                4 * (error_mean_test @ error_mean.T) * covcond
            )  # (9 x 2^12) * (9 x 2^12)
        test_covmatrix = barbar_cov[:, train_size : train_size + test_size]
        test_potentials = barbar_cov.mean(axis=1)

        return np.linalg.pinv(test_covmatrix) @ test_potentials

